/**
 * @file        plume_renderer/plume_vr.cpp
 * @brief       VR (the vr branch): plume's frames to the headset through the SDK's OpenXR layer (vr_openxr.h).
 *
 * The SDK's layer (the xenos plugin's VR) makes the OpenXR session on a Vulkan device and shows "screens" (an image
 * with both eyes side by side) at the frame's head pose. Here it runs on plume's device: plume makes its instance and
 * device with the runtime's extensions and features and keeps a native graphics queue for the headset
 * (plume::VulkanVrHooks); each frame plume's finished picture is copied into a screen by a command buffer of ours on
 * plume's queue (after the frame's own, so in order), which signals the layer's timeline semaphore. The head pose:
 * the camera hooks (burnout_vr.cpp) turn the game's camera and note the pose (NoteCameraPose) as the frame is drawn;
 * the frame's present, on the same thread, takes it for the frame.
 */
#include "plume_renderer/plume_vr.h"
#include "plume_renderer/plume_draw.h"
#include "plume_renderer/plume_shader_cache.h"
#include "plume_renderer/plume_postfx.h"
#include "shader_cache.h"

#include <algorithm>
#include <array>
#include <span>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <plume_vulkan.h>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>  // GetEnvironmentVariableA (the seat offset)
#endif

#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/ui/vulkan/vr_openxr.h>

namespace rex::plume_renderer::vr {
namespace xr = rex::ui::vulkan::vr;

namespace {
struct FramePose {
  bool posed = false;
  float pose[16]{}, tangents[4]{};
  bool interpolated = false;  // the frame's logic state and moment between it and the one before (InterpolateDraw)
  double state = 0.0;
  float t = 1.f;
  uint64_t serial = 0;
  std::chrono::steady_clock::time_point taken{};  // when its pose was taken (the headset's look-ahead measure)
  bool cameras = false;  // the logic state's camera and the drawn one (NoteFrameCameras)
  float logic_camera[12]{}, drawn_camera[12]{};
};
std::mutex g_pose_mutex;
std::deque<FramePose> g_poses;  // presented frames' poses, in order, for the submissions

struct Submission {
  VkCommandPool pool = VK_NULL_HANDLE;
  std::array<VkCommandBuffer, 4> commands{};
  std::array<VkFence, 4> fences{};
  uint32_t next = 0;
} g_submission;
bool g_active = false;
FramePose g_frame;           // the frame being encoded and submitted (BeginFrame)
std::atomic<uint32_t> g_split_calls{0}, g_split_unposed{0}, g_split_refused{0};  // development
bool g_split = false;        // its scene went to its screen at its first interface draw (SplitAtInterface)
}  // namespace

bool Requested() {
  static const bool requested = [] {
    const char* v = std::getenv("XERENGE_VR_PLUME");
    return v && v[0] == '1';
  }();
  return requested;
}

bool Stereo() {
  static const bool mono = [] {
    const char* v = std::getenv("XERENGE_VR_MONO");
    return v && v[0] == '1';
  }();
  return g_active && !mono;
}

float EyeShift(int eye) { return xr::EyeShift(eye); }

void BeforeInterface() {
  if (!Requested()) {
    return;
  }
  plume::VulkanVrHooks& hooks = plume::vulkanVrHooks();
  // The OpenXR instance is made here (the first call); without a runtime or headset the list is empty and the layer
  // stays off.
  for (const char* extension : xr::InstanceExtensions()) {
    hooks.instanceExtensions.emplace_back(extension);
  }
  if (!xr::Active()) {
    REXLOG_WARN("plume VR: no OpenXR runtime or headset - flat");
    return;
  }
  hooks.deviceExtensions = [](VkInstance instance, VkPhysicalDevice physical_device) {
    std::vector<std::string> names;
    for (const char* extension : xr::DeviceExtensions(vkGetInstanceProcAddr, instance, physical_device)) {
      names.emplace_back(extension);
    }
    return names;
  };
  hooks.features = true;
  hooks.reserveDirectQueue = true;
}

void AfterDevice(plume::RenderInterface* render_interface, plume::RenderDevice* device) {
  if (!Requested() || !xr::Active() || !render_interface || !device) {
    return;
  }
  auto* vk_interface = static_cast<plume::VulkanInterface*>(render_interface);
  auto* vk_device = static_cast<plume::VulkanDevice*>(device);
  const uint32_t family = vk_device->queueFamilyIndices[0];  // DIRECT
  const plume::VulkanQueueFamily& queues = vk_device->queueFamilies[family];
  if (queues.reserved == UINT32_MAX) {
    REXLOG_ERROR("plume VR: no native queue of the graphics family kept for the headset - flat");
    return;
  }
  // The layer takes the queue at its `used` index: the one plume left out.
  xr::QueueCount(vkGetInstanceProcAddr, vk_interface->instance, vk_device->physicalDevice, family, queues.reserved);
  xr::OnDeviceCreated(nullptr, vkGetInstanceProcAddr, vk_interface->instance, vk_device->physicalDevice, vk_device->vk,
                      family);
  VkCommandPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pool_info.queueFamilyIndex = family;
  if (vkCreateCommandPool(vk_device->vk, &pool_info, nullptr, &g_submission.pool) != VK_SUCCESS) {
    REXLOG_ERROR("plume VR: vkCreateCommandPool failed - flat");
    return;
  }
  VkCommandBufferAllocateInfo alloc{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  alloc.commandPool = g_submission.pool;
  alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc.commandBufferCount = uint32_t(g_submission.commands.size());
  vkAllocateCommandBuffers(vk_device->vk, &alloc, g_submission.commands.data());
  VkFenceCreateInfo fence_info{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
  for (VkFence& fence : g_submission.fences) {
    vkCreateFence(vk_device->vk, &fence_info, nullptr, &fence);
  }
  g_active = true;
  REXLOG_INFO("plume VR: session on plume's device (graphics family {}, the headset's queue {})", family,
              queues.reserved);
}

namespace {
// The game frame just built: its pose and logic moment, taken once (by its Swap marker or its present, whichever
// comes first) and kept by serial, so the present that draws it uses its own (UseFramePoseOf). The queue the
// presents fill was the only link before: when plume's present ran behind, or took two game frames in one batch, the
// frame drawn got a pose three frames old (the stats' "3.0 poses waiting") - the markers and sparks trailed a head
// turn and caught up, and the headset was told the wrong pose (Matt 2026-10-31).
std::mutex g_frame_poses_mutex;
std::array<FramePose, 16> g_frame_poses;
FramePose g_last_built;
uint64_t g_last_built_swaps = ~0ull;
FramePose g_selected;
bool g_selected_valid = false;

FramePose TakeBuiltFrame() {
  std::scoped_lock lock{g_frame_poses_mutex};
  const uint64_t swaps = xr::GameSwapCount();
  if (swaps == g_last_built_swaps) {
    return g_last_built;
  }
  FramePose frame;
  frame.taken = xr::CameraPoseTime();
  frame.cameras = xr::TakeFrameCameras(frame.logic_camera, frame.drawn_camera);
  frame.posed = xr::TakeCameraPose(*reinterpret_cast<float(*)[16]>(frame.pose),
                                   *reinterpret_cast<float(*)[4]>(frame.tangents));
  // Named while the frame was built (burnout_vr.cpp InterpolateCamera, keyed by the swaps before its own).
  frame.interpolated = xr::FrameInterpolation(swaps, frame.state, frame.t) ||
                       (swaps && xr::FrameInterpolation(swaps - 1, frame.state, frame.t));
  static uint64_t serial = 0;
  frame.serial = ++serial;
  g_frame_poses[frame.serial % g_frame_poses.size()] = frame;
  g_last_built = frame;
  g_last_built_swaps = swaps;
  return frame;
}
}  // namespace

uint32_t NoteFrameEnd() {
  if (!g_active) {
    return 0;
  }
  return uint32_t(TakeBuiltFrame().serial);
}

void UseFramePose(uint32_t serial) {
  if (!g_active || serial == 0) {
    return;
  }
  std::scoped_lock lock{g_frame_poses_mutex};
  const FramePose& frame = g_frame_poses[serial % g_frame_poses.size()];
  if (frame.serial == serial) {
    g_selected = frame;
    g_selected_valid = true;
  }
}

void NoteFramePresented() {
  if (!g_active) {
    return;
  }
  const FramePose frame = TakeBuiltFrame();
  std::scoped_lock lock{g_pose_mutex};
  g_poses.push_back(frame);
  while (g_poses.size() > 4) {
    g_poses.pop_front();
  }
}

// The camera's axes as the seat's (s: -1 1 -1, burnout_vr.cpp CameraAxes), then the view space's (sigma)
// (XERENGE_VR_VIEW_SIGNS "s0 s1 s2 g0 g1 g2"; development: the VIEWSIGNS trigger file).
// sigma: x opposite RenderWare's right (the paused race turned 25 degrees and pitched 20: only x flipped, and so
// z's against it, keeps the streaks on the scene).
float g_view_signs[6] = {-1.f, 1.f, -1.f, -1.f, 1.f, 1.f};
const float* ViewSigns() {
  static const bool read = [] {
    if (const char* e = std::getenv("XERENGE_VR_VIEW_SIGNS")) {
      std::sscanf(e, "%f %f %f %f %f %f", &g_view_signs[0], &g_view_signs[1], &g_view_signs[2], &g_view_signs[3],
                  &g_view_signs[4], &g_view_signs[5]);
    }
    return true;
  }();
  (void)read;
  return g_view_signs;
}

bool ViewSpaceDraw(const float* c) {
  static const bool off = std::getenv("XERENGE_VR_TURN_VIEW") && std::getenv("XERENGE_VR_TURN_VIEW")[0] == '0';
  if (off || !Stereo() || !g_frame.posed) {
    return false;
  }
  // A perspective projection alone: rows (sx 0 ox 0) (0 sy oy 0) (0 0 a b) (0 0 1 0).
  const auto zero = [](float v) { return std::fabs(v) < 1e-4f; };
  return c[0] > 0.1f && c[5] > 0.1f && zero(c[1]) && zero(c[3]) && zero(c[4]) && zero(c[7]) && zero(c[8]) &&
         zero(c[9]) && zero(c[12]) && zero(c[13]) && std::fabs(c[14] - 1.f) < 1e-4f && zero(c[15]);
}

// The seat offset (burnout_vr.cpp: the arrow keys), in the headset's space like the pose's translation: the game's
// camera is moved by it, so the view-space draws are too. Read once a frame (BeginFrame).
float g_seat_offset[3] = {0.f, 0.f, 0.f};

void ReadSeatOffset() {
#ifdef _WIN32
  char text[96] = {};
  if (GetEnvironmentVariableA("XERENGE_VR_SEAT", text, sizeof(text)) > 0) {
    std::sscanf(text, "%f %f %f", &g_seat_offset[0], &g_seat_offset[1], &g_seat_offset[2]);
  }
#endif
}

void TurnViewSpaceDraw(float* c) {
  // The renderer's camera rows are turned as row'_r = s_r sum_k pose[r][k] s_k row_k and its position moved by
  // sum_k pose[3][k] s_k row_k (burnout_vr.cpp PoseRwMatrix). A point in the game's view space (axes sigma_i row_i)
  // is in the turned one at v' = A v + t: A_ij = sigma_i sigma_j s_i s_j pose[i][j],
  // t_i = -sigma_i s_i sum_k pose[3][k] pose[i][k]. The draw's projection C (clip_i = dot(c_i, v)) becomes C [A t].
  const float* signs = ViewSigns();
  const float* s = signs;
  const float* sigma = signs + 3;
  const float* pose = g_frame.pose;
  float a[3][3], t[3];
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      a[i][j] = sigma[i] * sigma[j] * s[i] * s[j] * pose[i * 4 + j];
    }
    const float px = pose[12] + g_seat_offset[0], py = pose[13] + g_seat_offset[1], pz = pose[14] + g_seat_offset[2];
    t[i] = -sigma[i] * s[i] * (px * pose[i * 4] + py * pose[i * 4 + 1] + pz * pose[i * 4 + 2]);
  }
  for (int row = 0; row < 4; ++row) {
    float* r = c + row * 4;
    const float x = r[0], y = r[1], z = r[2];
    for (int j = 0; j < 3; ++j) {
      r[j] = x * a[0][j] + y * a[1][j] + z * a[2][j];
    }
    r[3] += x * t[0] + y * t[1] + z * t[2];
  }
}

namespace {
bool CameraVertexShader(uint64_t vs_hash) {
  static const std::unordered_set<uint64_t> hashes = [] {
    std::unordered_set<uint64_t> set;
    const char* list = std::getenv("MCLA_VR_CAMERA_VS");
    for (const char* p = list ? list : ""; *p;) {
      char* end = nullptr;
      const uint64_t microcode = std::strtoull(p, &end, 16);
      if (end == p) {
        break;
      }
      set.insert(microcode);
      if (const ShaderMicrocodeEntry* entry = PlumeShaderCache::Instance().FindByMicrocode(microcode)) {
        set.insert(entry->shaderHash);
      }
      p = *end ? end + 1 : end;
    }
    return set;
  }();
  return vs_hash != 0 && hashes.count(vs_hash) != 0;
}

// m = a b (row-major 4x4), in double whatever the operands: a world taken out of WVP and put back loses millimetres in
// float far from the origin (Angel Valley is 4.8 km out: every part of every car jittered, Matt 2026-10-31).
template <typename A, typename B, typename M>
void Multiply4(const A* a, const B* b, M* m) {
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      m[r * 4 + c] = M(double(a[r * 4]) * b[c] + double(a[r * 4 + 1]) * b[4 + c] + double(a[r * 4 + 2]) * b[8 + c] +
                       double(a[r * 4 + 3]) * b[12 + c]);
    }
  }
}

// The inverse by cofactors (in double: a projection's rows differ by orders of magnitude).
bool Invert4(const double* f, double* out) {
  double m[16], inv[16];
  for (int i = 0; i < 16; ++i) {
    m[i] = f[i];
  }
  inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] +
           m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
  inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] -
           m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
  inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] +
           m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
  inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] -
            m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
  inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] -
           m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
  inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] +
           m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
  inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] -
           m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
  inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] +
            m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
  inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] +
           m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
  inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] -
           m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
  inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] +
            m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
  inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] -
            m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
  inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] -
           m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
  inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] +
           m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
  inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] -
            m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
  inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] +
            m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
  const double det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
  if (std::fabs(det) < 1e-30) {
    return false;
  }
  for (int i = 0; i < 16; ++i) {
    out[i] = inv[i] / det;
  }
  return true;
}

// A scene draw of a logic state: its world, its place relative to the state's camera, its travel a step (from its pair
// the state before, if it had one), and the frame that last paired it.
struct StepDraw {
  std::array<double, 16> m;
  double at[3]{};  // its mesh's centre in the world (InterpolateDraw's `centre`; its origin without one)
  double rel[3]{};
  double travel[3]{};
  double origin_rel[3]{}, origin_travel[3]{};  // the same for its matrix's origin
  bool moving = false;
  uint64_t used = 0;
};

// A draw's world (W, WVP = VP W) from its WVP, by VP's x, y and w rows: the z row is the w row but for its translation
// (z = w - near: an infinite far plane), and VP inverted whole was near singular - 4.8 km out (Angel Valley, VP's
// translations ~2000) float's rounding of those translations put a draw's world metres off, differently each frame,
// and its parts lost their pairs between steps: drawn some between steps, some not, the cars came apart (Matt
// 2026-10-31: wobbling cars, wheels coming and going). x, y and w alone are well conditioned and fix W.
bool InvertXyw(const double* vp, double* out) {
  const double a[9] = {vp[0], vp[1], vp[2], vp[4], vp[5], vp[6], vp[12], vp[13], vp[14]};
  const double det = a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) +
                     a[2] * (a[3] * a[7] - a[4] * a[6]);
  if (std::fabs(det) < 1e-12) {
    return false;
  }
  out[0] = (a[4] * a[8] - a[5] * a[7]) / det;
  out[1] = (a[2] * a[7] - a[1] * a[8]) / det;
  out[2] = (a[1] * a[5] - a[2] * a[4]) / det;
  out[3] = (a[5] * a[6] - a[3] * a[8]) / det;
  out[4] = (a[0] * a[8] - a[2] * a[6]) / det;
  out[5] = (a[2] * a[3] - a[0] * a[5]) / det;
  out[6] = (a[3] * a[7] - a[4] * a[6]) / det;
  out[7] = (a[1] * a[6] - a[0] * a[7]) / det;
  out[8] = (a[0] * a[4] - a[1] * a[3]) / det;
  return true;
}

void WorldOf(const double* vp, const double* xyw_inverse, const float* c, double* world) {
  const int rows[3] = {0, 1, 3};
  for (int j = 0; j < 4; ++j) {  // W's column j: (x, y, z, h), h 1 for the translation
    double rhs[3];
    for (int k = 0; k < 3; ++k) {
      rhs[k] = double(c[rows[k] * 4 + j]) - (j == 3 ? vp[rows[k] * 4 + 3] : 0.0);
    }
    for (int r = 0; r < 3; ++r) {
      world[r * 4 + j] = xyw_inverse[r * 3] * rhs[0] + xyw_inverse[r * 3 + 1] * rhs[1] + xyw_inverse[r * 3 + 2] * rhs[2];
    }
    world[12 + j] = j == 3 ? 1.0 : 0.0;
  }
}

// A draw drawn between steps with its own pair (InterpolateDraw): carries the draws near it that have none.
struct Carrier {
  std::array<double, 16> from, to;
  double at[3]{};  // its centre at `to`
  double state = 0.0, span = 1.0;
};

struct Interpolation {
  std::mutex mutex;
  uint64_t serial = 0;  // the frame (FramePose::serial)
  double state = -1.0;
  double prev_state = -1.0;  // the state prev holds (frames running several steps: more than one before)
  bool recording = false;
  std::unordered_map<uint64_t, std::vector<StepDraw>> prev, curr;
  bool have_vp = false;
  double vp[16]{};
  double xyw_inverse[9]{};  // VP's x, y and w rows' 3x3 inverted (WorldOf)
  uint32_t vp_width = 0;
  double eye[3]{};  // the logic state's camera (FramePose::logic_camera), else the frame's
  std::unordered_map<uint64_t, uint32_t> counters;  // this frame's draws so far by mesh
  std::vector<Carrier> carriers, last_carriers;
  // This frame's motion (drawn world over the state's) by world: the draws of one body share their world (a car's
  // panels, a wheel's tyre and rim), and are drawn as the first of them was.
  std::unordered_map<uint64_t, std::array<double, 16>> body_motion;  // this frame's and the frame before's moving draws (InterpolateDraw)
  uint32_t carried = 0;  // development
  uint32_t draws = 0, interpolated = 0, paired = 0, found = 0, states = 0;  // development
  float max_move = 0.f;
  std::vector<std::array<float, 3>> objects;  // this frame's scene draws' origins as drawn (ndc x, y, w): PlaceMarker
  float track_w = 0.f, track[4]{};  // development: the nearest moving draw's origin on screen (XERENGE_VR_INTERPTRACK)
  double sample[16]{};
} g_interpolation;
}  // namespace

bool g_marker_run = true;  // the frame's 2D draws so far are all markers (BeginFrame)
bool g_marker_started = false;  // and the first of them was a triangle

bool MarkerDraw(uint64_t vs_hash, uint64_t ps_hash, uint32_t count, const float* c) {
  static const bool off = std::getenv("XERENGE_VR_MARKERS") && std::getenv("XERENGE_VR_MARKERS")[0] == '0';
  if (off || !Stereo() || !g_marker_run || !HudVertexShader(vs_hash)) {
    return false;
  }
  // The frame's first 2D draws, flat colour (the triangles, the position label's shapes), each with its car's depth
  // value in gPosOffset.z (past 0.5, 2 m), starting with a triangle. The run ends at the first other 2D draw: the race
  // HUD (z ~0) or the pause menu (textured) - the game leaves z as the last marker set it, and the pause menu's panels
  // carried it.
  constexpr uint64_t kFlatColourPs = 0x6EB945B547506423ull;
  const bool marker = ps_hash == kFlatColourPs && std::fabs(c[3] - 1.f) < 1e-5f && c[2] > 0.5f && c[2] < 1.f &&
                      c[4] > 0.f && c[5] < 0.f && (g_marker_started || count == 3);
  g_marker_started = marker;
  g_marker_run = marker;
  return marker;
}

void PlaceMarker(float* c, const float* centre) {
  // The markers' depth value is the game's with a 1 m near plane, z = 1 - 1/depth: taken as the camera's (0.5 m) they
  // came out at half their cars' distance (simulator race, each marker against the object below it: 0.49-0.51 from 40
  // to 200 m) - the eyes put them nearer, toward the middle of the view, and a sway moved them off their cars (Matt
  // 2026-10-31).
  if (c[2] >= 1.f) {
    return;
  }
  float depth = 1.f / (1.f - c[2]);
  if (!(depth > 0.1f && depth < 100000.f)) {
    return;
  }
  // The game projects the markers with its own camera, not the head-turned one: its centre (as drawn) taken back to
  // the game's view space at that depth (the frame's symmetric field of view, as burnout_vr.cpp sets the camera's
  // view window), turned as the view-space draws are (TurnViewSpaceDraw), projected again; the marker moved by the
  // difference, its depth the turned one.
  const FramePose& frame = g_frame;
  static const bool turn = !(std::getenv("XERENGE_VR_MARKER_TURN") && std::getenv("XERENGE_VR_MARKER_TURN")[0] == '0');
  if (turn && centre && frame.posed && frame.tangents[1] > 0.f && frame.tangents[2] > 0.f) {
    const float sx = 1.f / frame.tangents[1], sy = 1.f / frame.tangents[2];
    const float nx = centre[0] * c[4] + c[0], ny = centre[1] * c[5] + c[1];
    float p[16] = {sx, 0.f, 0.f, 0.f, 0.f, sy, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f};
    TurnViewSpaceDraw(p);
    float v[4] = {nx * depth / sx, ny * depth / sy, depth, 1.f};
    // Projected with the logic state's camera; the frame is drawn between states (the chase camera swinging round a
    // corner: a far marker cars to the side of its car, Matt 2026-10-31). Through the world into the drawn camera's
    // view space (axis i = sigma_i row i). XERENGE_VR_MARKER_CAMERA=0: not.
    static const bool recamera =
        !(std::getenv("XERENGE_VR_MARKER_CAMERA") && std::getenv("XERENGE_VR_MARKER_CAMERA")[0] == '0');
    if (recamera && frame.cameras) {
      const float* sigma = ViewSigns() + 3;
      const float* from = frame.logic_camera;
      const float* to = frame.drawn_camera;
      float world[3];
      for (int k = 0; k < 3; ++k) {
        world[k] = from[9 + k] + sigma[0] * v[0] * from[k] + sigma[1] * v[1] * from[3 + k] + sigma[2] * v[2] * from[6 + k];
      }
      const float d[3] = {world[0] - to[9], world[1] - to[10], world[2] - to[11]};
      for (int i = 0; i < 3; ++i) {
        v[i] = sigma[i] * (d[0] * to[3 * i] + d[1] * to[3 * i + 1] + d[2] * to[3 * i + 2]);
      }
    }
    float clip[4];
    for (int r = 0; r < 4; ++r) {
      clip[r] = p[r * 4] * v[0] + p[r * 4 + 1] * v[1] + p[r * 4 + 2] * v[2] + p[r * 4 + 3] * v[3];
    }
    float fx = clip[3] > 0.1f ? clip[0] / clip[3] : nx;
    const float fy = clip[3] > 0.1f ? clip[1] / clip[3] : ny;
    // The game eases a marker's place on screen after its car: a car crossing the view (a corner) left its marker up
    // to 260 px (of 3840) beside it, catching up over ~8 frames (simulator race, each marker against its car: the
    // object below it at its distance; Matt 2026-10-31: "The car itself moving is what caused the marker to move off
    // the car"). Put over its car as drawn this frame: the nearest object origin below it within 15% of its distance.
    // XERENGE_VR_MARKER_SNAP=0: not.
    static const bool snap = !(std::getenv("XERENGE_VR_MARKER_SNAP") && std::getenv("XERENGE_VR_MARKER_SNAP")[0] == '0');
    if (snap && clip[3] > 0.1f) {
      float best = 0.15f;
      bool found = false;
      float car_x = 0.f;
      std::scoped_lock lock{g_interpolation.mutex};
      for (const auto& o : g_interpolation.objects) {
        const float dx = std::fabs(o[0] - fx), dy = o[1] - fy;
        if (dy < 0.f && dy > -0.3f && std::fabs(o[2] - clip[3]) < 0.15f * clip[3] && dx < best) {
          best = dx, car_x = o[0], found = true;
        }
      }
      if (found) {
        clip[0] = car_x * clip[3];
        fx = car_x;
      }
    }
    if (static const bool log = std::getenv("XERENGE_VR_MARKERTRACE") != nullptr; log) {  // development
      static uint64_t traced = 0;
      if (traced != frame.serial) {
        traced = frame.serial;
        // The nearest object below it on screen (its car): its origin's distance.
        float best = 1e9f, best_w = 0.f, bx = 0.f, by = 0.f;
        {
          std::scoped_lock lock{g_interpolation.mutex};
          for (const auto& o : g_interpolation.objects) {
            const float dx = o[0] - fx, dy = o[1] - fy;
            if (dy < 0.f && dy > -0.2f && std::fabs(dx) < best) {
              best = std::fabs(dx), best_w = o[2], bx = o[0], by = o[1];
            }
          }
        }
        REXLOG_INFO("VR marker trace: frame {} drawn ndc {:.4f} {:.4f} depth {:.1f} head yaw {:.2f}; nearest object below "
                    "at {:.4f} {:.4f} w {:.1f}", frame.serial, fx, fy, clip[3],
                    std::atan2(-frame.pose[8], frame.pose[10]) * 57.29578f, bx, by, best_w);
      }
    }
    if (clip[3] > 0.1f) {
      c[0] += clip[0] / clip[3] - nx;
      c[1] += clip[1] / clip[3] - ny;
      depth = clip[3];
    }
  } else if (static int unknown = 0; unknown < 4) {
    ++unknown;
    REXLOG_INFO("VR marker: no centre ({}), not turned with the head", centre ? "unposed" : "not unpacked here");
  }
  // Close by (the car beside you) the marker is between the eyes and the car, and filled the view: hidden, all its
  // vertices on one point past the far plane (Matt 2026-10-31: "When a car is very close to me the screen goes dark
  // and the marker breaks ... Hide the markers that are close to me"). XERENGE_VR_MARKER_MIN_M sets the distance.
  static const float min_depth = [] {
    const char* v = std::getenv("XERENGE_VR_MARKER_MIN_M");
    return v ? float(std::atof(v)) : 10.f;
  }();
  if (depth < min_depth) {
    c[0] = c[1] = 0.f;
    c[2] = 2.f;
    c[3] = 1.f;
    c[4] = c[5] = 0.f;
    return;
  }
  for (int i = 0; i < 4; ++i) {
    c[i] *= depth;
  }
  c[4] *= depth;
  c[5] *= depth;
  if (static int logged = 0; logged < 8 || std::getenv("XERENGE_VR_MARKERLOG")) {  // development
    ++logged;
    REXLOG_INFO("VR marker: depth value {:.5f} -> {:.1f} m", c[2] / depth, depth);
  }
}

bool InterpolateDraw(uint64_t vs_hash, uint64_t mesh_key, bool depth_tested, uint32_t target_width, const float* c,
                     const float* centre, float* wvp_out) {
  const FramePose& frame = g_frame;
  if (!Stereo() || !frame.posed || !frame.interpolated || !depth_tested) {
    return false;
  }
  // c0-c3: clip_i = dot(c_i, position), the rows of WVP (column vectors: WVP = VP W). A perspective draw's w row has
  // x, y or z; an orthographic one (the interface, full-screen passes) is (0 0 0 1).
  if (std::fabs(c[12]) + std::fabs(c[13]) + std::fabs(c[14]) < 1e-6f) {
    return false;
  }
  Interpolation& vi = g_interpolation;
  std::scoped_lock lock{vi.mutex};
  if (frame.serial != vi.serial) {  // the frame's first scene draw
    vi.objects.clear();
    if (static const bool track = std::getenv("XERENGE_VR_INTERPTRACK") != nullptr; track && vi.track_w > 0.f) {
      REXLOG_INFO("VR interpolation track: t {:.3f} screen {:.4f} {:.4f} (as drawn by the state {:.4f} {:.4f}) depth {:.2f}",
                  frame.t, vi.track[0], vi.track[1], vi.track[2], vi.track[3], vi.track_w);
    }
    vi.track_w = 0.f;
    vi.serial = frame.serial;
    vi.body_motion.clear();
    vi.counters.clear();
    if (!vi.carriers.empty()) {
      vi.last_carriers = std::move(vi.carriers);
      vi.carriers.clear();
    }
    vi.have_vp = false;
    vi.recording = frame.state != vi.state;
    if (vi.recording) {
      ++vi.states;
      if (frame.state > vi.state && frame.state - vi.state <= 4.0) {
        vi.prev = std::move(vi.curr);
        vi.prev_state = vi.state;
      } else {
        vi.prev.clear();  // the timer started over, or a long gap
      }
      vi.curr.clear();
      vi.state = frame.state;
    }
    if (static uint32_t frames = 0; ++frames % 600 == 0 && std::getenv("XERENGE_VR_INTERPLOG")) {  // development
      REXLOG_INFO("VR interpolation (plume): {} of {} scene draws moved between steps (paired {}, found {}, {} new "
                  "states), t {:.3f} state {}",
                  vi.interpolated, vi.draws, vi.paired, vi.found, vi.states, frame.t, frame.state);
      REXLOG_INFO("  max move {:.4f}, its world {:.3f} {:.3f} {:.3f} {:.3f} / {:.3f} {:.3f} {:.3f} {:.3f} / {:.3f} {:.3f} "
                  "{:.3f} {:.3f} / {:.3f} {:.3f} {:.3f} {:.3f}; vp {:.3f} {:.3f} {:.3f} {:.3f} / {:.3f} {:.3f} {:.3f} {:.3f}",
                  vi.max_move, vi.sample[0], vi.sample[1], vi.sample[2], vi.sample[3], vi.sample[4], vi.sample[5],
                  vi.sample[6], vi.sample[7], vi.sample[8], vi.sample[9], vi.sample[10], vi.sample[11], vi.sample[12],
                  vi.sample[13], vi.sample[14], vi.sample[15], vi.vp[0], vi.vp[1], vi.vp[2], vi.vp[3], vi.vp[12],
                  vi.vp[13], vi.vp[14], vi.vp[15]);
      vi.max_move = 0.f;
      vi.draws = vi.interpolated = vi.paired = vi.found = vi.states = 0;
    }
  }
  if (CameraVertexShader(vs_hash)) {
    if (!vi.have_vp) {
      for (int i = 0; i < 16; ++i) {
        vi.vp[i] = c[i];
      }
      vi.have_vp = InvertXyw(vi.vp, vi.xyw_inverse);
      vi.vp_width = target_width;
      if (frame.cameras) {
        for (int k = 0; k < 3; ++k) {
          vi.eye[k] = frame.logic_camera[9 + k];
        }
      } else if (vi.have_vp) {  // the point clip x, y and w are 0 at
        const double t[3] = {vi.vp[3], vi.vp[7], vi.vp[15]};
        for (int k = 0; k < 3; ++k) {
          vi.eye[k] = -(vi.xyw_inverse[k * 3] * t[0] + vi.xyw_inverse[k * 3 + 1] * t[1] + vi.xyw_inverse[k * 3 + 2] * t[2]);
        }
      }
    }
    return false;
  }
  if (!vi.have_vp || target_width != vi.vp_width) {
    return false;  // before the camera's first draw, or another target (shadows, reflections)
  }
  // Drawn about the camera (its origin at the eye: w 0 there - vs FDA2FB150FC94348, 495AAFB282D2A23E in Angel Valley):
  // its "world" is the camera's place, and blended between two steps it moved by the camera's travel, up to 2.4 m a
  // step with the car parked, and the parts drawn over it came and went (Matt 2026-10-31: wobbling cars, missing
  // wheels). Drawn as it is.
  if (c[15] < 0.5f) {
    return false;
  }
  double world[16];
  WorldOf(vi.vp, vi.xyw_inverse, c, world);
  ++vi.draws;
  const auto note_object = [&](const float* m) {  // its origin where it is drawn (the markers' cars)
    if (m[15] > 1.f && vi.objects.size() < 4096) {
      vi.objects.push_back({m[3] / m[15], m[7] / m[15], m[15]});
    }
  };
  // Where it is: its mesh's centre (a wheel's hub), not its origin - a wheel's origin is off its axle, and swings round
  // it metres a step as the wheel turns. And that relative to the logic state's camera (the frame's camera without it).
  double here[3] = {world[3], world[7], world[11]};
  if (centre) {
    for (int r = 0; r < 3; ++r) {
      here[r] = world[r * 4] * centre[0] + world[r * 4 + 1] * centre[1] + world[r * 4 + 2] * centre[2] + world[r * 4 + 3];
    }
  }
  const double rel[3] = {here[0] - vi.eye[0], here[1] - vi.eye[1], here[2] - vi.eye[2]};
  // The same draw the state before. A mesh is often drawn several times close together (Angel Valley's chase view: the
  // cars beside each other on the grid, the wheels, the panels), and its draws come and go between steps as the view
  // culls them: taking the nearest of the mesh's draws paired one with its neighbour when its own was missing, and it
  // was drawn sliding up to the spacing (Matt 2026-10-31: wobbling cars, wheels coming and going). A draw is paired for
  // sure only where it should be, within kNear: where it was (still), where its own travel the step before puts it, or
  // where it was relative to the camera (moving with it: the player's car) - by its centre, or by its matrix's origin;
  // each draw of the state before once. Else the nearest gives it a travel for the next step to pair it by, and it is
  // drawn as it is or carried (below).
  const double span = std::max(1.0, vi.state - vi.prev_state);
  const double origin_rel[3] = {world[3] - vi.eye[0], world[7] - vi.eye[1], world[11] - vi.eye[2]};
  // How far the matrix's origin is from where the state before's draw `d` puts it: still, on by its travel, or the
  // same relative to the camera.
  const auto origin_cost = [&](const StepDraw& d) {
    const auto l = [](double x, double y, double z) { return std::sqrt(x * x + y * y + z * z); };
    double cost = l(world[3] - d.m[3], world[7] - d.m[7], world[11] - d.m[11]);
    if (d.moving) {
      cost = std::min(cost, l(world[3] - d.m[3] - d.origin_travel[0] * span, world[7] - d.m[7] - d.origin_travel[1] * span,
                              world[11] - d.m[11] - d.origin_travel[2] * span));
    }
    return std::min(cost, l(origin_rel[0] - d.origin_rel[0], origin_rel[1] - d.origin_rel[1],
                            origin_rel[2] - d.origin_rel[2]));
  };
  constexpr double kNear = 0.12;  // metres: a step changes a car's speed by millimetres; its parts are metres apart
  const auto length = [](double x, double y, double z) { return std::sqrt(x * x + y * y + z * z); };
  const auto before = vi.prev.find(mesh_key);
  const uint32_t index = vi.counters[mesh_key]++;
  StepDraw* from = nullptr;
  bool sure = false;
  if (before != vi.prev.end()) {
    ++vi.paired;
    double best = kNear;
    // First the same place among the mesh's draws, if still there (a millimetre): most draws (the scenery, a mesh
    // drawn hundreds of times: posts, trees) are, and the search below goes through them all.
    if (index < before->second.size()) {
      StepDraw& d = before->second[index];
      if (d.used != frame.serial && length(here[0] - d.at[0], here[1] - d.at[1], here[2] - d.at[2]) < 0.001) {
        from = &d;
        best = 0.0;
      }
    }
    for (StepDraw& d : from ? std::span<StepDraw>{} : std::span<StepDraw>{before->second}) {
      if (d.used == frame.serial) {
        continue;
      }
      double cost = length(here[0] - d.at[0], here[1] - d.at[1], here[2] - d.at[2]);
      if (d.moving) {
        cost = std::min(cost, length(here[0] - d.at[0] - d.travel[0] * span, here[1] - d.at[1] - d.travel[1] * span,
                                     here[2] - d.at[2] - d.travel[2] * span));
      }
      cost = std::min(cost, length(rel[0] - d.rel[0], rel[1] - d.rel[1], rel[2] - d.rel[2]));
      // And by its matrix's origin: a mesh whose centre is off its matrix's turn (a part 2.5 m from it, on a car
      // turning more than 5 degrees a step) swings with it, while the origin goes straight.
      cost = std::min(cost, origin_cost(d));
      if (cost < best) {
        best = cost;
        from = &d;
      }
    }
    sure = from != nullptr;
    if (!from) {
      // Not sure: the nearest, for its travel only (not drawn between steps by it, and not taken from the draws that
      // pair by place) - right, the next step pairs it by that travel; wrong, it does not (dust, the sparks).
      double nearest = 4.0 * span;
      for (StepDraw& d : before->second) {
        if (const double l = length(here[0] - d.at[0], here[1] - d.at[1], here[2] - d.at[2]); l < nearest) {
          nearest = l;
          from = &d;
        }
      }
    } else {
      from->used = frame.serial;
    }
  }
  if (vi.recording) {
    StepDraw& d = vi.curr[mesh_key].emplace_back();
    std::memcpy(d.m.data(), world, sizeof(world));
    std::memcpy(d.at, here, sizeof(here));
    std::memcpy(d.rel, rel, sizeof(rel));
    std::memcpy(d.origin_rel, origin_rel, sizeof(origin_rel));
    if (from) {
      d.moving = true;
      for (int k = 0; k < 3; ++k) {
        d.travel[k] = (here[k] - from->at[k]) / span;
        d.origin_travel[k] = (world[3 + 4 * k] - from->m[3 + 4 * k]) / span;
      }
    }
  }
  // Development (XERENGE_VR_CARCHECK): the player car's draws (3-9 m ahead, bottom middle) by how they were drawn.
  struct CarTally {
    uint64_t serial = 0;
    uint32_t own = 0, carried = 0, still = 0, loose = 0;
    float t = 0.f;
    std::string shifts;
    bool have_ref = false;  // the player car's body (A64F) this frame: its motion and centre
    double ref_motion[16]{}, ref_at[3]{};
    std::string rigid;  // each draw near it: how far (mm) from where the body's motion puts it
  };
  static CarTally tally;
  static const bool check = std::getenv("XERENGE_VR_CARCHECK") != nullptr;
  const bool in_car = check && c[15] > 3.f && c[15] < 9.f && std::fabs(c[3] / c[15]) < 0.25f && c[7] / c[15] < 0.f;
  if (check && tally.serial != frame.serial) {
    if (tally.own + tally.carried + tally.loose) {
      REXLOG_INFO("VR carcheck: frame {} t {:.2f}: {} own, {} carried, {} still, {} loose;{}", tally.serial, tally.t,
                  tally.own, tally.carried, tally.still, tally.loose, tally.shifts);
      if (tally.have_ref) {
        REXLOG_INFO("VR rigid: frame {} t {:.2f}:{}", tally.serial, tally.t, tally.rigid);
      }
    }
    tally = {};
    tally.serial = frame.serial;
    tally.t = frame.t;
  }
  // Development (XERENGE_VR_CARCHECK): draws with the same world are one body's parts and must be drawn the same.
  struct GroupTally {
    uint64_t serial = 0;
    std::unordered_map<uint64_t, std::array<double, 4>> first;  // world -> drawn shift, w
    std::unordered_map<uint64_t, std::string> who;  // world -> the first's key and path
    int told = 0;
    uint32_t groups_checked = 0, split = 0;
    double worst = 0.0, worst_w = 0.0;
  };
  static GroupTally groups;
  if (check && groups.serial != frame.serial) {
    if (groups.groups_checked) {
      REXLOG_INFO("VR groupcheck: frame {} t {:.2f}: {} same-world draws checked, {} drawn apart, worst {:.0f} mm at "
                  "{:.1f} m", groups.serial, frame.t, groups.groups_checked, groups.split, groups.worst * 1000.0,
                  groups.worst_w);
    }
    groups.first.clear();
    groups.who.clear();
    groups.serial = frame.serial;
    groups.groups_checked = groups.split = 0;
    groups.worst = groups.worst_w = 0.0;
  }
  const auto group_check = [&](const double* drawn, char path) {
    if (!check || frame.t >= 0.999f || c[15] > 60.f) {
      return;
    }
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < 12; ++i) {
      h = (h ^ uint64_t(int64_t(std::llround(world[i] * 1000.0)))) * 1099511628211ull;
    }
    const double shift[3] = {drawn ? drawn[3] - world[3] : 0.0, drawn ? drawn[7] - world[7] : 0.0,
                             drawn ? drawn[11] - world[11] : 0.0};
    const auto [it, added] = groups.first.try_emplace(h, std::array<double, 4>{shift[0], shift[1], shift[2], c[15]});
    const std::string me = fmt::format("{:016X} {}{} ({:.0f} {:.0f} {:.0f} mm) centre {}", mesh_key, path,
                                       sure ? "s" : from ? "u" : "-", shift[0] * 1000.0, shift[1] * 1000.0,
                                       shift[2] * 1000.0,
                                       centre ? fmt::format("{:.3f} {:.3f} {:.3f}", centre[0], centre[1], centre[2])
                                              : std::string("none"));
    if (added) {
      groups.who[h] = me;
    } else {
      ++groups.groups_checked;
      const double d = length(shift[0] - it->second[0], shift[1] - it->second[1], shift[2] - it->second[2]);
      groups.split += d > 0.005;
      if (d > 0.005 && groups.told < 400 && frame.serial % 50 == 0) {
        ++groups.told;
        REXLOG_INFO("VR groupsplit: frame {} w {:.1f} vs {:016X}: {} against {}", frame.serial, c[15], vs_hash, me,
                    groups.who[h]);
      }
      if (d > groups.worst) {
        groups.worst = d;
        groups.worst_w = c[15];
      }
    }
  };
  const auto rigid_check = [&](const double* drawn, char path) {
    group_check(drawn, path);
    if (!check || !tally.have_ref || frame.t >= 0.999f ||
        length(here[0] - tally.ref_at[0], here[1] - tally.ref_at[1], here[2] - tally.ref_at[2]) > 3.0) {
      return;
    }
    const double* m = drawn ? drawn : world;
    double actual[3], expected[3];
    const double* r = tally.ref_motion;
    for (int k = 0; k < 3; ++k) {
      actual[k] = centre ? m[k * 4] * centre[0] + m[k * 4 + 1] * centre[1] + m[k * 4 + 2] * centre[2] + m[k * 4 + 3]
                         : m[k * 4 + 3];
      expected[k] = r[k * 4] * here[0] + r[k * 4 + 1] * here[1] + r[k * 4 + 2] * here[2] + r[k * 4 + 3];
    }
    tally.rigid += fmt::format(" {:04X}:{:.0f}", uint32_t(mesh_key & 0xFFFF),
                               length(actual[0] - expected[0], actual[1] - expected[1], actual[2] - expected[2]) * 1000.0);
  };
  static const bool draws =
      !(std::getenv("XERENGE_VR_INTERPOLATE_DRAWS") && std::getenv("XERENGE_VR_INTERPOLATE_DRAWS")[0] == '0');
  // Another part of a body already drawn this frame (the same world): drawn as it was, whatever this one's pair - one
  // part new to the step (a car 40 m out changing its detail) was drawn as it is and the rest between the steps, 15 cm
  // apart.
  uint64_t body = 1469598103934665603ull;
  for (int i = 0; i < 12; ++i) {
    body = (body ^ uint64_t(int64_t(std::llround(world[i] * 10000.0)))) * 1099511628211ull;
  }
  const auto drawn_as = [&](const double* drawn) {  // this body's motion, for its other parts
    double motion[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    if (double inverse[16]; drawn && Invert4(world, inverse)) {
      Multiply4(drawn, inverse, motion);
    }
    vi.body_motion.try_emplace(body).first->second = [&] {
      std::array<double, 16> m;
      std::memcpy(m.data(), motion, sizeof(motion));
      return m;
    }();
  };
  if (draws && frame.t < 0.999f) {
    if (const auto mate = vi.body_motion.find(body); mate != vi.body_motion.end()) {
      double as_mate[16];
      Multiply4(mate->second.data(), world, as_mate);
      rigid_check(as_mate, 'G');
      Multiply4(vi.vp, as_mate, wvp_out);
      note_object(wvp_out);
      return true;
    }
  }
  if (!draws || frame.t >= 0.999f || before == vi.prev.end()) {
    if (draws && frame.t < 0.999f) {
      drawn_as(nullptr);
    }
    tally.loose += in_car && frame.t < 0.999f;
    rigid_check(nullptr, 'N');
    if (in_car && frame.t < 0.999f) {
      tally.shifts += fmt::format(" N{:04X}:0,0", uint32_t(mesh_key & 0xFFFF));
    }
    note_object(c);
    return false;
  }
  // Turned since its pair, by more than a car turns in a step (5 degrees): a spinning wheel. A wheel's origin is off its
  // axle (Angel Valley's player car: up to 3.5 m a step as it spins, the body 0.5 m), and blended between steps it left
  // the car and shrank (two turns 90 degrees apart blended halfway are a flattened matrix): the wheels that came and
  // went. Not blended, below.
  const auto turn_cos = [](const double* a, const double* b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int r = 0; r < 3; ++r) {
      for (int k = 0; k < 3; ++k) {
        dot += a[r * 4 + k] * b[r * 4 + k];
        na += a[r * 4 + k] * a[r * 4 + k];
        nb += b[r * 4 + k] * b[r * 4 + k];
      }
    }
    // trace(A^T B) over the scales: 1 + 2 cos(angle) for rotations.
    return (3.0 * dot / std::sqrt(std::max(na * nb, 1e-30)) - 1.0) / 2.0;
  };
  const bool turned = sure && turn_cos(from->m.data(), world) <= 0.996;
  const bool own = sure && !turned;
  double between[16];
  if (turned) {
    // Turned as the step has it, and moved by a point of it between the steps: its matrix's origin where that goes
    // straight (every mesh on the matrix moved the same: a car turning hard), else its centre - a wheel's hub, its
    // origin off the axle swinging round it.
    const double u = 1.0 - (1.0 - double(frame.t)) / span;
    const bool by_origin = origin_cost(*from) < kNear;
    std::memcpy(between, world, sizeof(between));
    for (int k = 0; k < 3; ++k) {
      between[3 + 4 * k] -= (1.0 - u) * (by_origin ? world[3 + 4 * k] - from->m[3 + 4 * k] : here[k] - from->at[k]);
    }
    tally.carried += in_car;
    ++vi.carried;
  } else if (own) {
    // Moved: by more than the rounding of a world taken back out of WVP (a millimetre, 1e-4 of the axes).
    bool moved = false;
    for (int i = 0; i < 16 && !moved; ++i) {
      moved = std::fabs(from->m[i] - world[i]) > (i == 3 || i == 7 || i == 11 ? 1e-3f : 1e-4f);
    }
    ++vi.found;
    if (!moved) {
      rigid_check(nullptr, 'S');
      tally.still += in_car;
      note_object(c);
      return false;
    }
    tally.own += in_car;
    // The camera is drawn at t between the state before this one and this one; prev may be several states back (a
    // frame that ran more than one logic step - Angel Valley at 7680x5760: the car's parts were blended over 2-5 steps
    // by the camera's t, landing off the car and each other, Matt 2026-10-31). The same moment between prev's state and
    // this.
    const double u = 1.0 - (1.0 - double(frame.t)) / span;
    for (int i = 0; i < 16; ++i) {
      between[i] = from->m[i] + (world[i] - from->m[i]) * u;
    }
    // A carrier of the draws near it (moving, not only rounding).
    if (vi.carriers.size() < 4096 &&
        length(here[0] - from->at[0], here[1] - from->at[1], here[2] - from->at[2]) > 0.01) {
      Carrier& carrier = vi.carriers.emplace_back();
      carrier.from = from->m;
      std::memcpy(carrier.to.data(), world, sizeof(world));
      carrier.state = vi.state;
      std::memcpy(carrier.at, here, sizeof(here));
      carrier.span = span;
    }
  } else {
    // Not paired for sure: carried by the nearest moving draw within 4.5 m of its centre (its car) - with its car, its
    // turn as the step has it. Else as it is. The draws so far this frame, else the frame before's (a car's wheels
    // drawn before its body): the same state's as they were, a state before's on by their travel.
    const auto at = [&](const double* a) { return length(here[0] - a[0], here[1] - a[1], here[2] - a[2]); };
    std::array<double, 16> carrier_from{}, carrier_to{};
    double nearest = 4.5;
    bool carried = false;
    // And only one moving as it does (within 25 cm a step, by its nearest the step before): the dust and sparks about
    // a car are near it but not of it, and carried with it they went with the car between steps and back at each.
    const auto moves_with = [&](const std::array<double, 16>& k_from, const std::array<double, 16>& k_to, double k_span) {
      if (!from) {
        return false;
      }
      double d2 = 0.0;
      for (int i = 0; i < 3; ++i) {
        const double v = (here[i] - from->at[i]) / span - (k_to[3 + 4 * i] - k_from[3 + 4 * i]) / k_span;
        d2 += v * v;
      }
      return d2 < 0.25 * 0.25;
    };
    for (size_t i = 0; from && i < vi.carriers.size(); ++i) {
      const Carrier& k = vi.carriers[i];
      if (const double d = at(k.at); d < nearest && moves_with(k.from, k.to, k.span)) {
        nearest = d;
        carrier_from = k.from, carrier_to = k.to;
        carried = true;
      }
    }
    if (!carried && from) {
      for (const Carrier& k : vi.last_carriers) {
        std::array<double, 16> f = k.from, to = k.to;
        double where[3] = {k.at[0], k.at[1], k.at[2]};
        if (k.state != vi.state) {  // the step after: from where it was by the same travel
          if (vi.state <= k.state || vi.state - k.state > 4.0) {
            continue;
          }
          const double steps = span / k.span;
          f = k.to;
          for (int i = 0; i < 3; ++i) {
            to[3 + 4 * i] = k.to[3 + 4 * i] + (k.to[3 + 4 * i] - k.from[3 + 4 * i]) * steps;
            where[i] += (k.to[3 + 4 * i] - k.from[3 + 4 * i]) * steps;
          }
        }
        if (const double d = at(where); d < nearest && moves_with(k.from, k.to, k.span)) {
          nearest = d;
          carrier_from = f, carrier_to = to;
          carried = true;
        }
      }
    }
    double inverse[16];
    if (!carried || !Invert4(carrier_to.data(), inverse)) {
      tally.loose += in_car;
      drawn_as(nullptr);
      rigid_check(nullptr, 'L');
      if (in_car) {
        tally.shifts += fmt::format(" L{:04X}:{}", uint32_t(mesh_key & 0xFFFF), sure ? "s" : from ? "u" : "-");
      }
      note_object(c);
      return false;
    }
    // Its motion back to the frame's moment, applied to this draw.
    const double u = 1.0 - (1.0 - double(frame.t)) / span;
    double carrier_between[16], motion[16];
    for (int i = 0; i < 16; ++i) {
      carrier_between[i] = carrier_from[i] + (carrier_to[i] - carrier_from[i]) * u;
    }
    Multiply4(carrier_between, inverse, motion);
    Multiply4(motion, world, between);
    tally.carried += in_car;
    ++vi.carried;
  }
  if (in_car) {
    tally.shifts += fmt::format(" {}{:04X}:{:.0f},{:.0f}", own ? 'O' : 'C', uint32_t(mesh_key & 0xFFFF),
                                (between[3] - world[3]) * 1000.0, (between[11] - world[11]) * 1000.0);
  }
  if (check && own && in_car && !tally.have_ref && (mesh_key & 0xFFFF) == 0xA64F) {
    double inverse[16];
    if (Invert4(world, inverse)) {
      Multiply4(between, inverse, tally.ref_motion);
      std::memcpy(tally.ref_at, here, sizeof(here));
      tally.have_ref = true;
    }
  }
  drawn_as(between);
  rigid_check(between, own ? 'O' : turned ? 'T' : 'C');
  Multiply4(vi.vp, between, wvp_out);
  note_object(wvp_out);
  ++vi.interpolated;
  // The player car: nearest the bottom middle of the view, 3-12 m ahead.
  const auto off_centre = [](float x, float y) { return x * x + (y + 0.4f) * (y + 0.4f); };
  if (wvp_out[15] > 3.f && wvp_out[15] < 12.f &&
      (vi.track_w == 0.f || off_centre(wvp_out[3] / wvp_out[15], wvp_out[7] / wvp_out[15]) <
                                off_centre(vi.track[0], vi.track[1]))) {
    vi.track_w = wvp_out[15];
    vi.track[0] = wvp_out[3] / wvp_out[15];
    vi.track[1] = wvp_out[7] / wvp_out[15];
    vi.track[2] = c[3] / c[15];
    vi.track[3] = c[7] / c[15];
  }
  return true;
}

uint32_t HudBlendControl(uint32_t blend_control) {
  // RB_BLENDCONTROL: alpha_srcblend in bits 16-20 (xenos::BlendFactor: kSrcAlpha 6, kOne 1). The game blends its HUD for a
  // picture; over the cleared target the alpha it leaves is the layer's coverage. With SRC_ALPHA it was a squared: the
  // panels' dark backings (a 0.6: 0.36) and the tinted bars came out faint or washed (Burnout 2026-10-31: the race's
  // boost bar pale, the rating's OK and the speed panel's backing missing, the event card without its box).
  static const bool off = std::getenv("XERENGE_VR_HUD_COVERAGE") && std::getenv("XERENGE_VR_HUD_COVERAGE")[0] == '0';
  if (off || ((blend_control >> 16) & 0x1Fu) != 6u) {
    return blend_control;
  }
  return (blend_control & ~(0x1Fu << 16)) | (1u << 16);
}

uint32_t HudColorMask(uint32_t color_mask, uint32_t blend_control) {
  // Burnout's race HUD (2026-10-31): its bars, numbers and the panels' dark backings blend SRC_ALPHA /
  // ONE_MINUS_SRC_ALPHA with RGB written only (mask 7). Over the cleared target their alpha stayed 0 and the
  // compositor added their colour to the scene: the boost bar pale pink on the sky, the timer white, the backings
  // gone. Additive draws (dest ONE) stay as they are: alpha 0 is what adding is in a premultiplied layer.
  static const bool off = std::getenv("XERENGE_VR_HUD_COVERAGE") && std::getenv("XERENGE_VR_HUD_COVERAGE")[0] == '0';
  const uint32_t color_dest = (blend_control >> 8) & 0x1Fu;  // xenos::BlendFactor: kOneMinusSrcAlpha 7
  if (off || color_dest != 7u || (color_mask & 0x7u) == 0 || (color_mask & 0x8u) != 0) {
    return color_mask;
  }
  return color_mask | 0x8u;
}

namespace {
std::string DevDir() {
  const char* temp = std::getenv("TEMP");
  return std::string(temp ? temp : ".") + "\\xerenge_vr\\";
}
std::mutex g_dev_mutex;
std::vector<uint64_t> g_dev_skip;
std::atomic<int> g_dev_log{0};  // 1 armed for the next frame, 2 logging this frame
std::chrono::steady_clock::time_point g_dev_checked{};
}  // namespace

bool DevDraw(uint64_t vs_hash, uint64_t ps_hash, uint32_t blend_control, uint32_t color_mask, uint32_t prim_type,
             uint32_t count, bool hud) {
  static const bool enabled = std::getenv("XERENGE_VR_DEV") != nullptr;
  if (!enabled) {
    return false;
  }
  std::scoped_lock lock{g_dev_mutex};
  if (g_dev_log.load() == 2) {
    REXLOG_INFO("VR draw: vs {:016X} ps {:016X} blend {:08X} mask {:X} prim {} n {}{}", vs_hash, ps_hash, blend_control,
                color_mask, prim_type, count, hud ? " hud" : "");
  }
  for (const uint64_t h : g_dev_skip) {
    if (h == vs_hash || h == ps_hash) {
      return true;
    }
  }
  return false;
}

bool DevLogging() { return g_dev_log.load() == 2; }

void DevFrameEnd() {
  static const bool enabled = std::getenv("XERENGE_VR_DEV") != nullptr;
  if (!enabled) {
    return;
  }
  std::scoped_lock lock{g_dev_mutex};
  if (g_dev_log.load() == 2) {
    g_dev_log = 0;
    REXLOG_INFO("VR draw log: end of frame");
  } else if (g_dev_log.load() == 1) {
    g_dev_log = 2;
  }
  const auto now = std::chrono::steady_clock::now();
  if (now - g_dev_checked < std::chrono::milliseconds(250)) {
    return;
  }
  g_dev_checked = now;
  const std::string dir = DevDir();
  if (std::FILE* f = std::fopen((dir + "DRAWLOG").c_str(), "rb")) {
    std::fclose(f);
    std::remove((dir + "DRAWLOG").c_str());
    g_dev_log = 1;
  }
  if (std::FILE* f = std::fopen((dir + "VIEWSIGNS").c_str(), "rb")) {
    float v[6];
    if (std::fscanf(f, "%f %f %f %f %f %f", &v[0], &v[1], &v[2], &v[3], &v[4], &v[5]) == 6) {
      ViewSigns();
      std::memcpy(g_view_signs, v, sizeof(v));
      REXLOG_INFO("VR view signs {} {} {} {} {} {}", v[0], v[1], v[2], v[3], v[4], v[5]);
    }
    std::fclose(f);
    std::remove((dir + "VIEWSIGNS").c_str());
  }
  if (std::FILE* f = std::fopen((dir + "SKIP").c_str(), "rb")) {
    char text[4096] = {};
    const size_t n = std::fread(text, 1, sizeof(text) - 1, f);
    std::fclose(f);
    std::remove((dir + "SKIP").c_str());
    g_dev_skip.clear();
    for (char* p = text; p < text + n;) {
      char* end = nullptr;
      const uint64_t h = std::strtoull(p, &end, 16);
      if (end == p) {
        ++p;
        continue;
      }
      g_dev_skip.push_back(h);
      p = end;
    }
    REXLOG_INFO("VR skip: {} shaders", g_dev_skip.size());
  }
}

bool HudVertexShader(uint64_t vs_hash) {
  static const std::unordered_set<uint64_t> hashes = [] {
    std::unordered_set<uint64_t> set;
    const char* list = std::getenv("MCLA_VR_HUD_VS");
    for (const char* p = list ? list : ""; *p;) {
      char* end = nullptr;
      const uint64_t microcode = std::strtoull(p, &end, 16);
      if (end == p) {
        break;
      }
      set.insert(microcode);
      // The draws name plume's own shader hash when the microcode is in its cache.
      if (const ShaderMicrocodeEntry* entry = PlumeShaderCache::Instance().FindByMicrocode(microcode)) {
        set.insert(entry->shaderHash);
      }
      p = *end ? end + 1 : end;
    }
    REXLOG_INFO("plume VR: {} HUD vertex shaders", set.size());
    return set;
  }();
  return vs_hash != 0 && hashes.count(vs_hash) != 0;
}

void BeginFrame() {
  if (!g_active) {
    return;
  }
  ReadSeatOffset();
  g_frame = {};
  g_split = false;
  g_marker_run = true;
  g_marker_started = false;
  {
    std::scoped_lock lock{g_pose_mutex};
    if (!g_poses.empty()) {
      g_frame = g_poses.front();
      g_poses.pop_front();
    }
  }
  {  // the pose of the game frame this present draws, when its Swap marker named it
    std::scoped_lock lock{g_frame_poses_mutex};
    static uint32_t frames = 0, own = 0;  // development: how many found their own
    own += g_selected_valid ? 1 : 0;
    if (++frames == 900) {
      REXLOG_INFO("plume VR: {} of {} frames drawn with their own frame's pose (the rest: the presents' queue)", own,
                  frames);
      frames = own = 0;
    }
    if (g_selected_valid) {
      g_frame = g_selected;
      g_selected_valid = false;
      std::scoped_lock pose_lock{g_pose_mutex};
      g_poses.clear();
    }
  }
  xr::SetPresentPose(g_frame.posed, *reinterpret_cast<const float(*)[16]>(g_frame.pose),
                     *reinterpret_cast<const float(*)[4]>(g_frame.tangents));
  xr::SetPresentPoseTime(g_frame.taken);
  xr::SetFrameHasScene(g_frame.posed);
}

bool InterfaceSplit() { return g_active && g_split; }

void SplitAtInterface(plume::RenderCommandList* list, plume::RenderTexture* target, uint32_t width, uint32_t height) {
  static const bool off = std::getenv("XERENGE_VR_HUD_LAYER") && std::getenv("XERENGE_VR_HUD_LAYER")[0] == '0';
  ++g_split_calls;
  if (off || !g_active || !g_frame.posed || g_split || !list || !target || !width || !height) {
    g_split_unposed += g_frame.posed ? 0 : 1;
    return;
  }
  const int screen = xr::SplitFrame(width, height, float(height) / float(width));
  if (screen < 0) {
    ++g_split_refused;
    return;
  }
  g_split = true;
  VkCommandBuffer cmd = static_cast<plume::VulkanCommandList*>(list)->vk;
  auto* texture = static_cast<plume::VulkanTexture*>(target);
  const uint32_t layers = texture->desc.arraySize;
  xr::RecordLayeredFrameCopy(cmd, texture->vk, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, width, height, screen, layers);
  // The HUD over transparent black.
  VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
  barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  barrier.image = texture->vk;
  barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers};
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr,
                       1, &barrier);
  const VkClearColorValue clear{};
  vkCmdClearColorImage(cmd, texture->vk, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clear, 1, &barrier.subresourceRange);
  barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  barrier.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  barrier.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, 0, 0, nullptr,
                       0, nullptr, 1, &barrier);
}

void Submit(plume::RenderDevice* device, plume::RenderCommandQueue* queue, plume::RenderTexture* shown, uint32_t width,
            uint32_t height, uint32_t layers) {
  if (!g_active || !shown || !width || !height) {
    return;
  }
  const FramePose& frame = g_frame;
  if (g_split) {
    xr::HudCaptured();  // the HUD's picture, copied below, goes with the split frame
  }
  DevFrameEnd();  // development triggers (DevDraw)
  xr::SetSceneSharpening(GetSharpening());  // flat plume's sharpening (its own pass reads one layer), per eye
  // The flat screen's shape (title, menus) is the frame's: at the 16:9 default a 3840x4320 eye's frame was squashed to
  // half its height (Matt 2026-10-31: "The intro is too wide"). Its picture keeps a 16:9 strip (plume_draw.cpp's box).
  xr::SetScreenAspect(float(height) / float(width));
  int screen = xr::BeginFrame(width, height);
  const bool split = screen <= -2;  // its scene is on screen -2 - screen (SplitAtInterface), the HUD to copy
  if (split) {
    screen = -2 - screen;
  }
  {  // development: every 2 s, what reached the headset
    static uint32_t calls = 0, posed = 0, refused = 0, queued = 0, split_count = 0;
    static auto since = std::chrono::steady_clock::now();
    ++calls;
    posed += frame.posed ? 1 : 0;
    refused += screen < 0 ? 1 : 0;
    split_count += split ? 1 : 0;
    {
      std::scoped_lock lock{g_pose_mutex};
      queued += uint32_t(g_poses.size());
    }
    const auto now = std::chrono::steady_clock::now();
    if (now - since >= std::chrono::seconds(2)) {
      REXLOG_INFO("plume VR: {} frames to the headset in {:.1f} s ({} posed, {} HUD split, {} refused, {:.1f} poses waiting; HUD {} unposed {} refused {})", calls,
                  std::chrono::duration<double>(now - since).count(), posed, split_count, refused, double(queued) / calls, g_split_calls.exchange(0), g_split_unposed.exchange(0), g_split_refused.exchange(0));
      calls = posed = refused = queued = split_count = 0;
      since = now;
    }
  }
  if (screen < 0) {
    return;
  }
  auto* vk_device = static_cast<plume::VulkanDevice*>(device);
  auto* vk_queue = static_cast<plume::VulkanCommandQueue*>(queue);
  auto* vk_texture = static_cast<plume::VulkanTexture*>(shown);
  const uint32_t index = g_submission.next;
  g_submission.next = (index + 1) % uint32_t(g_submission.commands.size());
  vkWaitForFences(vk_device->vk, 1, &g_submission.fences[index], VK_TRUE, UINT64_MAX);
  vkResetFences(vk_device->vk, 1, &g_submission.fences[index]);
  VkCommandBuffer cmd = g_submission.commands[index];
  vkResetCommandBuffer(cmd, 0);
  VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &begin);
  if (split) {  // the target holds the HUD (the left eye's: both draw the same) for the HUD layer
    xr::RecordFrameCopy(cmd, vk_texture->vk, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, width, height, screen, xr::kCopyHud);
  } else {
    xr::RecordLayeredFrameCopy(cmd, vk_texture->vk, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, width, height, screen,
                               layers);
  }
  vkEndCommandBuffer(cmd);
  VkSemaphore timeline = VK_NULL_HANDLE;
  uint64_t value = 0;
  const bool signal = xr::TakeFrameSignal(timeline, value);
  VkTimelineSemaphoreSubmitInfo timeline_info{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
  timeline_info.signalSemaphoreValueCount = signal ? 1 : 0;
  timeline_info.pSignalSemaphoreValues = &value;
  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.pNext = &timeline_info;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &cmd;
  submit.signalSemaphoreCount = signal ? 1 : 0;
  submit.pSignalSemaphores = &timeline;
  std::scoped_lock lock{*vk_queue->queue->mutex};
  if (vkQueueSubmit(vk_queue->queue->vk, 1, &submit, g_submission.fences[index]) != VK_SUCCESS) {
    static bool logged = false;
    if (!logged) {
      logged = true;
      REXLOG_ERROR("plume VR: vkQueueSubmit failed");
    }
  }
}

}  // namespace rex::plume_renderer::vr
