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
#include <chrono>
#include <map>
#include <mutex>
#include <string>

#include <fmt/format.h>
#include <rex/cvar.h>
#include <rex/hook.h>
#include <rex/logging.h>
#include <rex/ppc.h>
#include <rex/system/gpu_plugin.h>
#include <rex/system/kernel_state.h>
#include <rex/ui/vulkan/vr_openxr.h>

#include "burnout_vr.h"
#include "guest_memory.h"
#include "xerenge_settings.h"

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
  // The VR layer's direct resolves (render target -> texture image, no EDRAM round trip) place each resolve at the
  // texture's origin: Burnout draws its scene in three tiles (predicated tiling), each resolved into its third of the
  // frame, and the headset showed three slices of different places. The plain resolve path until that is handled
  // (MCLA_VR_DIRECT_RESOLVE set by hand wins).
  if (!std::getenv("MCLA_VR_DIRECT_RESOLVE")) {
    _putenv_s("MCLA_VR_DIRECT_RESOLVE", "0");
  }
  // The HUD on its own layer: the SDK splits a head-tracked frame at its first draw by one of these vertex shaders
  // (the race HUD's sprites and text, drawn after the scene is composited into the 1280x720 target; a frame's draw
  // log, VR_LOG_FRAME_FILE). MCLA_VR_HUD_VS set by hand wins.
  if (!std::getenv("MCLA_VR_HUD_VS")) {
    _putenv_s("MCLA_VR_HUD_VS", "BCEC88072A5F344D,30B5339A49CB9B5B,B2352AFC8C60D6B7,D25CBA074BB11277");
  }
  // DLSS's inputs: the scene is drawn in three tiles at 2x MSAA (each tile's depth rows at its resolve, sample 0), and
  // the static world's vertex shaders draw with the camera's view-projection as their WVP (the world at the identity:
  // the camera, and the others' worlds from it for their own motion). Set by hand wins.
  if (!std::getenv("MCLA_VR_DLSS_TILES")) {
    _putenv_s("MCLA_VR_DLSS_TILES", "1");
  }
  if (!std::getenv("MCLA_VR_CAMERA_VS")) {
    _putenv_s("MCLA_VR_CAMERA_VS", "C3FF17490757589F,1A563CA9A421178A,ACB68BA117ABC107");
  }
  config.graphics = rex::system::LoadGpuPlugin("xenos", "vulkan");
  if (!config.graphics) {
    REXLOG_ERROR("VR: could not load the xenos GPU plugin with the Vulkan backend");
    return;
  }
  // (The plugin's cvars exist once it is loaded.) The SDK tells the head-tracked scene by its draws' matrices: Burnout's vertex shaders keep gObjToWorldMatrix at
  // c15 and gObjToProjMatrix at c0, transposed (clip.i = dot(c_i, position)).
  rex::cvar::SetFlagByName("vr_world_constant", "-1");  // c15 in only some shaders: rigid
  rex::cvar::SetFlagByName("vr_wvp_constant", "0");
  rex::cvar::SetFlagByName("vr_constants_transposed", "true");
  // Frames at the headset's rate: no 60 Hz vblank wait (the SDK paces the game to the headset), and the logic on the
  // wall clock (its sixtieth-of-a-second steps whatever the frame rate: game_timing.cpp). Measured on the simulator,
  // 2560x2160 an eye: 72.0 FPS at 72 Hz, 90.0 at 90 Hz (1% low 78), 0% late.
  const auto from_command_line = [](const char* name) {
    return rex::cvar::GetFlagSource(name) == rex::cvar::Source::kCommandLine;
  };
  if (!from_command_line("vsync")) {
    rex::cvar::SetFlagByName("vsync", "false");
  }
  if (!std::getenv("XERENGE_UNLOCK_FPS")) {
    _putenv_s("XERENGE_UNLOCK_FPS", "1");
  }
  // The eyes' render size: the draw resolution scale XxY on 1280x720 from the settings (Driver Details > Settings >
  // GRAPHICS > VR EYE SIZE, vr_resolution in xerenge.conf; 2x3, 2560x2160, by default) unless the command line sets it.
  if (!from_command_line("draw_resolution_scale_x") && !from_command_line("draw_resolution_scale_y")) {
    const std::string size = xerenge::settings::ConfString("vr_resolution", "2x3");
    const size_t cross = size.find('x');
    const int x = cross != std::string::npos ? std::clamp(std::atoi(size.c_str()), 1, 4) : 2;
    const int y = cross != std::string::npos ? std::clamp(std::atoi(size.c_str() + cross + 1), 1, 6) : 3;
    rex::cvar::SetFlagByName("draw_resolution_scale_x", std::to_string(x).c_str());
    rex::cvar::SetFlagByName("draw_resolution_scale_y", std::to_string(y).c_str());
  }
  // Bloom and motion blur as the flat game has them: the launcher sets these from xerenge.conf, and a VR start (run.sh
  // --vr, burnout.exe --vr) does not go through it. Without bloom the 3D frame comes out at under half its brightness
  // on this renderer (plume makes up for it; this one draws the composite as the console does), so it follows the
  // settings like the flat game's. Already set in the environment wins.
  if (!std::getenv("XERENGE_NO_BLOOM") && !xerenge::settings::ConfString("bloom", "true").starts_with("true")) {
    _putenv_s("XERENGE_NO_BLOOM", "1");
  }
  if (!std::getenv("XERENGE_NO_MOTION_BLUR") &&
      !xerenge::settings::ConfString("motion_blur", "true").starts_with("true")) {
    _putenv_s("XERENGE_NO_MOTION_BLUR", "1");
  }
  // DLSS (VR DLSS, vr_dlss: off, dlaa, or an upscale 1.5 / 2 of the eye size for the headset).
  if (!from_command_line("vr_dlss") && !from_command_line("vr_dlss_output_scale")) {
    const std::string dlss = xerenge::settings::ConfString("vr_dlss", "off");
    if (dlss == "dlaa" || dlss == "1.5" || dlss == "2") {
      rex::cvar::SetFlagByName("vr_dlss", "true");
      rex::cvar::SetFlagByName("vr_dlss_output_scale", dlss == "dlaa" ? "1.0" : dlss.c_str());
    }
  }
  // The frame cap (VR FRAME CAP, vr_frame_cap: 1 off, 2 half the refresh, 3 a third).
  if (!from_command_line("vr_frame_cap")) {
    const std::string cap = xerenge::settings::ConfString("vr_frame_cap", "1");
    rex::cvar::SetFlagByName("vr_frame_cap", cap == "2" || cap == "3" ? cap.c_str() : "1");
  }
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
// Development (XERENGE_VR_FRAMELOG=1): the callers of RwFrameGetMatrix on the posed camera's frame, every 2 s.
std::atomic<uint32_t> g_camera_frame{0};

// The head pose of the frame being built (CGtViewport::Update's thread), for the renderer's reads (another thread).
std::mutex g_pose_mutex;
bool g_posed = false;
float g_pose[16], g_tangents[4];

// The renderer reads the camera frame (sub_82175350, building the view matrix: RwFrameGetMatrix from 0x82175374,
// 0x821753A0 and 0x82175614) as the frame's matrix and, 64 bytes on, its LTM: given a posed copy of both. The
// frame itself keeps the game's matrix: posing it (the first try) fed the head back into the chase camera, which
// read it back and re-aimed at the car - the car stayed in the middle of the view whatever the head did.
bool RendererCameraRead(uint32_t lr) {
  return lr == 0x82175374u || lr == 0x821753A0u || lr == 0x82175614u;
}

// Rows right, up, at, pos (RenderWare, left-handed: right x up = at) turned and moved by the head's pose.
void PoseRwMatrix(uint8_t* base, uint32_t m, const float (&pose)[16]) {
  float rows[4][3];
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 3; ++c) {
      rows[r][c] = LoadFloat(base, m + 16 * r + 4 * c);
    }
  }
  const float* s = CameraAxes().s;
  float seat[3][3];  // the camera's axes as OpenXR's seat axes (x right, y up, z back)
  for (int c = 0; c < 3; ++c) {
    seat[0][c] = s[0] * rows[0][c];
    seat[1][c] = s[1] * rows[1][c];
    seat[2][c] = s[2] * rows[2][c];
  }
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      const float v = pose[r * 4 + 0] * seat[0][c] + pose[r * 4 + 1] * seat[1][c] + pose[r * 4 + 2] * seat[2][c];
      StoreFloat(base, m + 16 * r + 4 * c, s[r] * v);
    }
  }
  for (int c = 0; c < 3; ++c) {
    StoreFloat(base, m + 48 + 4 * c,
               rows[3][c] + pose[12] * seat[0][c] + pose[13] * seat[1][c] + pose[14] * seat[2][c]);
  }
}

REX_HOOK_RAW(sub_820FBE60) {
  const uint32_t lr = uint32_t(ctx.lr);
  static const bool frame_log = std::getenv("XERENGE_VR_FRAMELOG") != nullptr;
  if (frame_log && ctx.r3.u32 == g_camera_frame.load(std::memory_order_relaxed)) {
    static std::mutex m;
    static std::map<uint32_t, uint32_t> callers;
    static auto since = std::chrono::steady_clock::now();
    std::scoped_lock lock{m};
    ++callers[lr];
    if (std::chrono::steady_clock::now() - since > std::chrono::seconds(2)) {
      std::string text;
      for (const auto& [caller, n] : callers) {
        text += fmt::format(" {:08X}x{}", caller, n);
      }
      REXLOG_INFO("VR camera frame {:08X} read by{}", ctx.r3.u32, text);
      callers.clear();
      since = std::chrono::steady_clock::now();
    }
  }
  if (lr == kViewportUpdateMatrixCaller) {
    g_camera_frame.store(ctx.r3.u32, std::memory_order_relaxed);
    if (vr::Active()) {
      // The pose this frame is built with; the view window (next) takes the headset's field of view.
      t_posed = vr::HeadPose(t_pose, t_tangents);
      if (t_posed) {
        const float h = std::max(-t_tangents[0], t_tangents[1]), v = std::max(t_tangents[2], -t_tangents[3]);
        t_tangents[0] = -h;
        t_tangents[1] = h;
        t_tangents[2] = v;
        t_tangents[3] = -v;
        std::scoped_lock lock{g_pose_mutex};
        std::memcpy(g_pose, t_pose, sizeof(g_pose));
        std::memcpy(g_tangents, t_tangents, sizeof(g_tangents));
        g_posed = true;
      }
    }
    __imp__sub_820FBE60(ctx, base);
    return;
  }
  if (RendererCameraRead(lr) && ctx.r3.u32 == g_camera_frame.load(std::memory_order_relaxed) && vr::Active()) {
    float pose[16];
    bool posed;
    {
      std::scoped_lock lock{g_pose_mutex};
      posed = g_posed;
      std::memcpy(pose, g_pose, sizeof(pose));
    }
    __imp__sub_820FBE60(ctx, base);
    if (!posed) {
      return;
    }
    // A copy of the matrix and the LTM 64 bytes on, each turned by the head, in place of the frame's.
    static uint32_t copy = 0;
    if (!copy) {
      copy = REX_KERNEL_MEMORY()->SystemHeapAlloc(128, 16);
    }
    std::memcpy(xerenge::GuestPointer(base, copy), xerenge::GuestPointer(base, ctx.r3.u32), 128);
    PoseRwMatrix(base, copy, pose);
    PoseRwMatrix(base, copy + 64, pose);
    ctx.r3.u64 = copy;
    static int logged = 0;
    if (logged < 3 || std::getenv("XERENGE_VR_CAMLOG")) {
      ++logged;
      REXLOG_INFO("VR camera: renderer read {:08X} from {:08X}: at {:.3f} {:.3f} {:.3f} -> {:.3f} {:.3f} {:.3f}",
                  ctx.r3.u32, lr, LoadFloat(base, copy + 64 + 32), LoadFloat(base, copy + 64 + 36),
                  LoadFloat(base, copy + 64 + 40), LoadFloat(base, copy + 32), LoadFloat(base, copy + 36),
                  LoadFloat(base, copy + 40));
    }
    vr::NoteCameraPose(pose, g_tangents);
    return;
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
