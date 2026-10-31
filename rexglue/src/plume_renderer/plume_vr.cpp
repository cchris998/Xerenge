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
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <plume_vulkan.h>

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
    t[i] = -sigma[i] * s[i] * (pose[12] * pose[i * 4] + pose[13] * pose[i * 4 + 1] + pose[14] * pose[i * 4 + 2]);
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

// m = a b (row-major 4x4).
void Multiply4(const float* a, const float* b, float* m) {
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      m[r * 4 + c] = a[r * 4] * b[c] + a[r * 4 + 1] * b[4 + c] + a[r * 4 + 2] * b[8 + c] + a[r * 4 + 3] * b[12 + c];
    }
  }
}

// The inverse by cofactors (in double: a projection's rows differ by orders of magnitude).
bool Invert4(const float* f, float* out) {
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
    out[i] = float(inv[i] / det);
  }
  return true;
}

struct Interpolation {
  std::mutex mutex;
  uint64_t serial = 0;  // the frame (FramePose::serial)
  double state = -1.0;
  bool recording = false;
  std::unordered_map<uint64_t, std::vector<std::array<float, 16>>> prev, curr;
  std::unordered_map<uint64_t, uint32_t> counters;
  bool have_vp = false;
  float vp[16]{}, vp_inverse[16]{};
  uint32_t vp_width = 0;
  uint32_t draws = 0, interpolated = 0, paired = 0, found = 0, states = 0;  // development
  float max_move = 0.f;
  std::vector<std::array<float, 3>> objects;  // development: this frame's draws' origins (ndc x, y, w) (MARKERTRACE)
  float track_w = 0.f, track[4]{};  // development: the nearest moving draw's origin on screen (XERENGE_VR_INTERPTRACK)
  float sample[16]{};
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
    if (static const bool log = std::getenv("XERENGE_VR_MARKERTRACE") != nullptr; log) {  // development
      static uint64_t traced = 0;
      if (traced != frame.serial) {
        traced = frame.serial;
        // The nearest object below it on screen (its car): its origin's distance.
        float best = 1e9f, best_w = 0.f, bx = 0.f, by = 0.f;
        {
          std::scoped_lock lock{g_interpolation.mutex};
          for (const auto& o : g_interpolation.objects) {
            const float dx = o[0] - nx, dy = o[1] - ny;
            if (dy < 0.f && dy > -0.2f && std::fabs(dx) < best) {
              best = std::fabs(dx), best_w = o[2], bx = o[0], by = o[1];
            }
          }
        }
        REXLOG_INFO("VR marker trace: frame {} game ndc {:.4f} {:.4f} depth {:.1f} head yaw {:.2f}; nearest object below "
                    "at {:.4f} {:.4f} w {:.1f}", frame.serial, nx, ny, depth,
                    std::atan2(-frame.pose[8], frame.pose[10]) * 57.29578f, bx, by, best_w);
      }
    }
    float p[16] = {sx, 0.f, 0.f, 0.f, 0.f, sy, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f, 0.f};
    TurnViewSpaceDraw(p);
    const float v[4] = {nx * depth / sx, ny * depth / sy, depth, 1.f};
    float clip[4];
    for (int r = 0; r < 4; ++r) {
      clip[r] = p[r * 4] * v[0] + p[r * 4 + 1] * v[1] + p[r * 4 + 2] * v[2] + p[r * 4 + 3] * v[3];
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
                     float* wvp_out) {
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
    vi.counters.clear();
    vi.have_vp = false;
    vi.recording = frame.state != vi.state;
    if (vi.recording) {
      ++vi.states;
      if (frame.state > vi.state && frame.state - vi.state <= 4.0) {
        vi.prev = std::move(vi.curr);
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
      std::memcpy(vi.vp, c, sizeof(vi.vp));
      vi.have_vp = Invert4(vi.vp, vi.vp_inverse);
      vi.vp_width = target_width;
    }
    return false;
  }
  if (!vi.have_vp || target_width != vi.vp_width) {
    return false;  // before the camera's first draw, or another target (shadows, reflections)
  }
  if (static const bool trace = std::getenv("XERENGE_VR_MARKERTRACE") != nullptr; trace && c[15] > 1.f && vi.objects.size() < 4096) {
    vi.objects.push_back({c[3] / c[15], c[7] / c[15], c[15]});  // development
  }
  float world[16];
  Multiply4(vi.vp_inverse, c, world);
  const uint32_t index = vi.counters[mesh_key]++;
  if (vi.recording) {
    auto& list = vi.curr[mesh_key];
    list.emplace_back();
    std::memcpy(list.back().data(), world, sizeof(world));
  }
  ++vi.draws;
  const auto before = vi.prev.find(mesh_key);
  if (frame.t >= 0.999f || before == vi.prev.end()) {
    return false;
  }
  ++vi.paired;
  // The same draw the state before: the same place among the mesh's draws when it has not moved (a centimetre), else the
  // nearest within a step's travel (metres). Not the same place within the travel: identical props (posts, cones) swap
  // places in the draw order as the view culls them, and one was drawn sliding into its neighbour.
  const auto distance2 = [&](const std::array<float, 16>& m) {
    const float x = m[3] - world[3], y = m[7] - world[7], z = m[11] - world[11];
    return x * x + y * y + z * z;
  };
  constexpr float kStep2 = 4.f * 4.f;
  const std::array<float, 16>* from = nullptr;
  if (index < before->second.size() && distance2(before->second[index]) < 0.01f * 0.01f) {
    from = &before->second[index];
  } else {
    float nearest = kStep2;
    for (const auto& m : before->second) {
      if (const float d = distance2(m); d < nearest) {
        nearest = d;
        from = &m;
      }
    }
  }
  // Moved: by more than the rounding of a world taken back out of WVP (a millimetre, 1e-4 of the axes).
  bool moved = false;
  for (int i = 0; from && i < 16 && !moved; ++i) {
    moved = std::fabs((*from)[i] - world[i]) > (i == 3 || i == 7 || i == 11 ? 1e-3f : 1e-4f);
  }
  vi.found += from ? 1 : 0;
  if (from) {
    const float d = std::fabs((*from)[3] - world[3]) + std::fabs((*from)[7] - world[7]) + std::fabs((*from)[11] - world[11]);
    if (d >= vi.max_move) {
      vi.max_move = d;
      std::memcpy(vi.sample, world, sizeof(world));
    }
  }
  if (!moved) {
    return false;
  }
  float between[16];
  for (int i = 0; i < 16; ++i) {
    between[i] = (*from)[i] + (world[i] - (*from)[i]) * frame.t;
  }
  Multiply4(vi.vp, between, wvp_out);
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
