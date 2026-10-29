// VR (the vr branch): Burnout Revenge in a headset, through the SDK's OpenXR layer (rexglue-sdk, branch vr: the
// layer brought over from the Midnight Club: Los Angeles port - multiview stereo, DLSS, the HUD on its own layer, the
// headset's pacing). VR runs on the xenos plugin's Vulkan backend; plume (the default renderer) stays flat.
//
// Turned on by --vr=true (the SDK's cvar) or XERENGE_VR=1.
//
// The head: RenderWare cameras. CGtViewport::Update (sub_8235A080) builds the camera's world matrix on its stack
// (rows right, up, at, pos at r1+80, +96, +112, +128), copies it into the camera's frame (the matrix
// RwFrameGetMatrix, sub_820FBE60, returns) and derives the viewport's view matrix from the same copy
// (sub_821425A8), after setting the view window (sub_82353BA8 -> RwCameraSetViewWindow, sub_8234ABC0). At the
// RwFrameGetMatrix call the stack matrix is turned and moved by the head's pose (in the camera's own space), so the
// frame, the view matrix and the culling (from the view-projection matrix) all follow the head; at the view window
// the headset's field of view replaces the game's (a symmetric frustum covering both eyes).
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

#include <rex/cvar.h>
#include <rex/hook.h>
#include <rex/logging.h>
#include <rex/ppc.h>
#include <rex/system/gpu_plugin.h>
#include <rex/ui/vulkan/vr_openxr.h>

#include "burnout_vr.h"
#include "guest_memory.h"

extern "C" void __imp__sub_820FBE60(PPCContext& __restrict, uint8_t*);
extern "C" void __imp__sub_8234ABC0(PPCContext& __restrict, uint8_t*);

namespace vr = rex::ui::vulkan::vr;

namespace {

constexpr uint32_t kViewportUpdateMatrixCaller = 0x8235A1A4;  // CGtViewport::Update's RwFrameGetMatrix call
constexpr uint32_t kViewWindowCaller = 0x82353C14;            // sub_82353BA8's RwCameraSetViewWindow call

float LoadFloat(const uint8_t* base, uint32_t address) {
  const uint32_t bits = xerenge::LoadGuestU32(base, address);
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

void StoreFloat(uint8_t* base, uint32_t address, float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  xerenge::StoreGuestU32(base, address, bits);
}

// The pose the viewport being updated was turned by (CGtViewport::Update: the matrix, then the view window, on one
// thread).
thread_local bool t_posed = false;
thread_local float t_pose[16], t_tangents[4];

// Development: XERENGE_VR_AXES="sx sy sz" multiplies the head's x, y, z axes (and position) as taken into the
// camera's (right, up, at) - the handedness check.
struct Axes {
  // RenderWare's camera is left-handed (right x up = at: its right is the screen's left); OpenXR's seat x is right
  // and z back: x = -right, y = up, z = -at.
  float s[3] = {-1.f, 1.f, -1.f};
};
const Axes& CameraAxes() {
  static const Axes axes = [] {
    Axes a;
    if (const char* v = std::getenv("XERENGE_VR_AXES")) {
      std::sscanf(v, "%f %f %f", &a.s[0], &a.s[1], &a.s[2]);
    }
    return a;
  }();
  return axes;
}

}  // namespace

namespace xerenge::vr_port {

bool Requested() {
  const char* v = std::getenv("XERENGE_VR");
  return (v && v[0] == '1') || rex::cvar::Query<bool>("vr");
}

void ConfigureAtStart(rex::RuntimeConfig& config) {
  if (!Requested()) {
    // Development: XERENGE_XENOS_VULKAN=1 runs flat on the xenos plugin's Vulkan backend (the VR one), for comparing.
    if (const char* v = std::getenv("XERENGE_XENOS_VULKAN"); v && v[0] == '1') {
      config.graphics = rex::system::LoadGpuPlugin("xenos", "vulkan");
    }
    return;
  }
  // The SDK's VR layer is Vulkan's: the xenos plugin with its Vulkan backend (plume and D3D12 have none).
  rex::cvar::SetFlagByName("vr", "true");
  config.graphics = rex::system::LoadGpuPlugin("xenos", "vulkan");
  if (!config.graphics) {
    REXLOG_ERROR("VR: could not load the xenos GPU plugin with the Vulkan backend");
    return;
  }
  // (The plugin's cvars exist once it is loaded.) The SDK tells the head-tracked scene by its draws' matrices: Burnout's vertex shaders keep gObjToWorldMatrix at
  // c15 and gObjToProjMatrix at c0, transposed (clip.i = dot(c_i, position)).
  rex::cvar::SetFlagByName("vr_world_constant", "15");
  rex::cvar::SetFlagByName("vr_wvp_constant", "0");
  rex::cvar::SetFlagByName("vr_constants_transposed", "true");
  REXLOG_INFO("VR: the xenos plugin, Vulkan backend");
}

void PaceFrame() {
  // Once per swap: the logic timer (sub_82363E68) is called while loading too, many times between swaps; each call
  // waiting for the headset (which waits for a new game frame) slowed loading to a crawl (1 FPS, a minute and more).
  static uint64_t paced_at = ~0ull;
  if (vr::Active()) {
    const uint64_t swaps = vr::GameSwapCount();
    if (swaps != paced_at) {
      paced_at = swaps;
      vr::PaceGameFrame();
    }
  }
}

}  // namespace xerenge::vr_port

// RwFrameGetMatrix (r3 the frame): at CGtViewport::Update's call the camera's new world matrix is on the caller's
// stack (r1+80); turned and moved by the head's pose before it is copied into the frame.
REX_HOOK_RAW(sub_820FBE60) {
  if (uint32_t(ctx.lr) == kViewportUpdateMatrixCaller && vr::Active()) {
    t_posed = vr::HeadPose(t_pose, t_tangents);
    if (t_posed) {
      // A symmetric frustum covering both eyes.
      const float h = std::max(-t_tangents[0], t_tangents[1]), v = std::max(t_tangents[2], -t_tangents[3]);
      t_tangents[0] = -h;
      t_tangents[1] = h;
      t_tangents[2] = v;
      t_tangents[3] = -v;
      const uint32_t m = ctx.r1.u32 + 80;
      float rows[4][3];
      for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 3; ++c) {
          rows[r][c] = LoadFloat(base, m + 16 * r + 4 * c);
        }
      }
      const float* s = CameraAxes().s;
      // The camera's axes as OpenXR's seat axes (x right, y up, z back): right, up, -at.
      float seat[3][3];
      for (int c = 0; c < 3; ++c) {
        seat[0][c] = s[0] * rows[0][c];
        seat[1][c] = s[1] * rows[1][c];
        seat[2][c] = s[2] * rows[2][c];
      }
      float out[3][3];
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
          out[r][c] = t_pose[r * 4 + 0] * seat[0][c] + t_pose[r * 4 + 1] * seat[1][c] + t_pose[r * 4 + 2] * seat[2][c];
        }
      }
      for (int c = 0; c < 3; ++c) {
        StoreFloat(base, m + 0 + 4 * c, s[0] * out[0][c]);
        StoreFloat(base, m + 16 + 4 * c, s[1] * out[1][c]);
        StoreFloat(base, m + 32 + 4 * c, s[2] * out[2][c]);
        StoreFloat(base, m + 48 + 4 * c,
                   rows[3][c] + t_pose[12] * seat[0][c] + t_pose[13] * seat[1][c] + t_pose[14] * seat[2][c]);
      }
      vr::NoteCameraPose(t_pose, t_tangents);
      static int logged = 0;  // development: the first few cameras posed
      if (logged < 6 || std::getenv("XERENGE_VR_CAMLOG")) {
        ++logged;
        REXLOG_INFO("VR camera: viewport {:08X} right {:.3f} {:.3f} {:.3f} up {:.3f} {:.3f} {:.3f} at {:.3f} {:.3f} {:.3f} "
                    "pos {:.2f} {:.2f} {:.2f} -> at {:.3f} {:.3f} {:.3f}",
                    ctx.r30.u32, rows[0][0], rows[0][1], rows[0][2], rows[1][0], rows[1][1], rows[1][2], rows[2][0],
                    rows[2][1], rows[2][2], rows[3][0], rows[3][1], rows[3][2], LoadFloat(base, m + 32),
                    LoadFloat(base, m + 36), LoadFloat(base, m + 40));
      }
    }
  }
  __imp__sub_820FBE60(ctx, base);
}

// RwCameraSetViewWindow (r3 the camera, r4 the window: x, y at distance 1): the headset's field of view for a posed
// viewport, and the frame's pose for the headset.
REX_HOOK_RAW(sub_8234ABC0) {
  if (uint32_t(ctx.lr) == kViewWindowCaller && t_posed && vr::Active()) {
    t_posed = false;
    const uint32_t window = ctx.r4.u32;
    static int logged = 0;  // development
    if (logged < 6 || std::getenv("XERENGE_VR_CAMLOG")) {
      ++logged;
      REXLOG_INFO("VR camera: view window {:.4f} {:.4f} -> {:.4f} {:.4f}", LoadFloat(base, window),
                  LoadFloat(base, window + 4), t_tangents[1], t_tangents[2]);
    }
    StoreFloat(base, window + 0, t_tangents[1]);
    StoreFloat(base, window + 4, t_tangents[2]);
    vr::SetFramePose(t_pose, t_tangents);
  }
  __imp__sub_8234ABC0(ctx, base);
}
