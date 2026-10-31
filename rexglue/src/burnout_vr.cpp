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
#include <array>
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
#include <rex/input/input_system.h>
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
  rex::cvar::SetFlagByName("vr", "true");
  // Start recentres the view on the head (Matt 2026-10-31: "We need to recentre the headset when pressing start"): the
  // pause menu and the menus' Start, as F8 does. 0x0010 is X_INPUT_GAMEPAD_START.
  rex::input::SetButtonPressedHook([](uint16_t button) {
    if (button == 0x0010) {
      vr::Recenter();
    }
  });
  // The renderer (VR RENDERER, vr_renderer in xerenge.conf; XERENGE_VR_RENDERER wins): xenos (the SDK's VR layer on
  // the xenos plugin's Vulkan backend, the default) or plume (its own renderer, no EDRAM emulation: plume_vr.cpp runs
  // the same OpenXR layer on plume's device). The camera hooks below serve both.
  std::string renderer = xerenge::settings::ConfString("vr_renderer", "xenos");
  if (const char* v = std::getenv("XERENGE_VR_RENDERER"); v && *v) {
    renderer = v;
  }
  // The HUD on its own layer: either renderer splits a head-tracked frame at its first draw by one of these vertex shaders
  // (the race HUD's sprites and text, drawn after the scene is composited into the 1280x720 target; a frame's draw
  // log, VR_LOG_FRAME_FILE). MCLA_VR_HUD_VS set by hand wins.
  if (!std::getenv("MCLA_VR_HUD_VS")) {
    _putenv_s("MCLA_VR_HUD_VS", "BCEC88072A5F344D,30B5339A49CB9B5B,B2352AFC8C60D6B7,D25CBA074BB11277");
  }
  if (renderer == "plume") {
    _putenv_s("XERENGE_VR_PLUME", "1");
    // The sharpening from xerenge.conf, as the launcher sets it (burnout.exe started directly in VR had none, where the
    // flat game had the menu's MEDIUM): the headset's scene is sharpened per eye with it (plume_vr.cpp).
    if (!std::getenv("XERENGE_SHARPEN")) {
      const std::string sharpen = xerenge::settings::ConfString("sharpen", "");
      if (!sharpen.empty()) {
        _putenv_s("XERENGE_SHARPEN", sharpen.c_str());
      }
    }
    // plume's switches, as the launcher and run.sh set them for it.
    for (const char* name : {"XERENGE_D3D_TARGETS", "XERENGE_D3D_UI", "XERENGE_REAL_SHADERS", "XERENGE_D3D_DRAWS",
                             "XERENGE_SECONDARY_TICKS"}) {
      if (!std::getenv(name)) {
        _putenv_s(name, "1");
      }
    }
    // The eyes' render size: VR EYE SIZE (vr_resolution, XxY on 1280x720: 2x3, 2560x2160, by default), plume's render
    // resolution. XERENGE_RENDER_RESOLUTION set by hand wins.
    if (!std::getenv("XERENGE_RENDER_RESOLUTION")) {
      const std::string size = xerenge::settings::ConfString("vr_resolution", "2x3");
      const size_t cross = size.find('x');
      const int x = cross != std::string::npos ? std::clamp(std::atoi(size.c_str()), 1, 6) : 2;
      const int y = cross != std::string::npos ? std::clamp(std::atoi(size.c_str() + cross + 1), 1, 6) : 3;
      _putenv_s("XERENGE_RENDER_RESOLUTION", fmt::format("{}x{}", 1280 * x, 720 * y).c_str());
    }
    // Frames at the headset's rate: no desktop vsync, no 60 Hz vblank pacing (the logic on the wall clock).
    _putenv_s("XERENGE_NO_HOST_VSYNC", "1");
    if (!std::getenv("XERENGE_UNLOCK_FPS")) {
      _putenv_s("XERENGE_UNLOCK_FPS", "1");
    }
    // The headset's pacing by its timer, not its token: the token (a frame's publication, then the next swap) waits on
    // plume's present, a frame behind on its own thread - game, plume and headset in series: 56 FPS at 180 Hz, 180 with
    // the timer (simulator, race, 2560x2160 an eye). MCLA_VR_PACE_TOKEN set by hand wins.
    if (!std::getenv("MCLA_VR_PACE_TOKEN")) {
      _putenv_s("MCLA_VR_PACE_TOKEN", "0");
    }
    // Drawing between logic steps: the camera turned to the frame's moment here, the moving draws moved with it by
    // plume_vr.cpp InterpolateDraw (Matt 2026-10-30: the car selection's turning camera "very jerky" against the head:
    // 60 steps a second under 90 Hz frames). XERENGE_VR_INTERPOLATE=0: off.
    // The static world's vertex shaders (one WVP, the camera's VP): the other draws' worlds are taken from it.
    if (!std::getenv("MCLA_VR_CAMERA_VS")) {
      _putenv_s("MCLA_VR_CAMERA_VS", "C3FF17490757589F,1A563CA9A421178A,ACB68BA117ABC107");
    }
    REXLOG_INFO("VR: the plume renderer");
    return;
  }
  // The SDK's VR layer on the xenos plugin with its Vulkan backend.
  // The VR layer's direct resolves (render target -> texture image, no EDRAM round trip) in mode 1: Burnout draws its
  // scene in three tiles (predicated tiling), each resolved into its third of the frame; mode 1 writes each at its
  // offset and the memory too. Mode 2 (the SDK's VR default, the memory left out) showed a stale picture in a tile (the
  // menus' garage in the middle third); the plain path (0) costs a third of the frame rate: simulator, race, 3840x3600
  // an eye, uncapped at 180 Hz: mode 0 113 FPS, mode 1 163. MCLA_VR_DIRECT_RESOLVE set by hand wins.
  if (!std::getenv("MCLA_VR_DIRECT_RESOLVE")) {
    _putenv_s("MCLA_VR_DIRECT_RESOLVE", "1");
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
  // The SDK lowers the eye size when video memory looks short, by Midnight Club's estimate (1.05 GB a unit of scale
  // area): Burnout's render targets take far less (4.9 GB at 5x6), and its estimate held 4x6 (5120x4320, 90 FPS at
  // 90 Hz, 0% late on the simulator) down to 4x5. 0.3 GB a unit. MCLA_VR_VRAM_GB_PER_AREA set by hand wins.
  if (!std::getenv("MCLA_VR_VRAM_GB_PER_AREA")) {
    _putenv_s("MCLA_VR_VRAM_GB_PER_AREA", "0.3");
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

// Drawing between logic steps (60 a second, under 72-120 Hz frames): the camera of each logic state (CGtViewport::Update
// on the update thread), and at the renderer's read the frame's camera turned and moved to its moment between the
// state it draws and the one before (one step behind the wall clock, as plume's interpolation); the SDK moves the
// moving draws the same way (vr::SetFrameInterpolation). XERENGE_VR_INTERPOLATE=0: off.
bool XerengeLogicClock(double& state_steps, double& real_steps);
struct CameraState {
  double state = -1.0;
  float rows[4][3]{};
};
std::mutex g_camera_states_mutex;
std::array<CameraState, 4> g_camera_states;  // the latest last
bool InterpolationOn() {
  static const bool on = !(std::getenv("XERENGE_VR_INTERPOLATE") && std::getenv("XERENGE_VR_INTERPOLATE")[0] == '0');
  return on;
}

void NoteCameraState(uint8_t* base, uint32_t matrix) {
  double state, real;
  if (!InterpolationOn() || !XerengeLogicClock(state, real)) {
    return;
  }
  CameraState c;
  c.state = state;
  for (int r = 0; r < 4; ++r) {
    for (int k = 0; k < 3; ++k) {
      c.rows[r][k] = LoadFloat(base, matrix + 16 * r + 4 * k);
    }
  }
  std::scoped_lock lock{g_camera_states_mutex};
  if (g_camera_states.back().state == state) {
    g_camera_states.back() = c;  // the same state again (a frame without a step)
    return;
  }
  if (state < g_camera_states.back().state) {
    g_camera_states = {};  // the timer started over
  }
  std::rotate(g_camera_states.begin(), g_camera_states.begin() + 1, g_camera_states.end());
  g_camera_states.back() = c;
}

// The renderer's copy of the camera matrix (`m`, rows right, up, at, pos) moved to the frame's moment: the state it
// holds (by its rows) and the one before, t = where the clock less a step falls between them. The frame's state and t
// go to the SDK for the moving draws.
void InterpolateCamera(uint8_t* base, uint32_t m) {
  if (!InterpolationOn()) {
    return;
  }
  double state_now, real;
  if (!XerengeLogicClock(state_now, real)) {
    return;
  }
  float rows[4][3];
  for (int r = 0; r < 4; ++r) {
    for (int k = 0; k < 3; ++k) {
      rows[r][k] = LoadFloat(base, m + 16 * r + 4 * k);
    }
  }
  CameraState from, to;
  {
    std::scoped_lock lock{g_camera_states_mutex};
    int found = -1;
    for (int i = int(g_camera_states.size()) - 1; i >= 0 && found < 0; --i) {
      if (g_camera_states[i].state >= 0.0 && std::memcmp(g_camera_states[i].rows, rows, sizeof(rows)) == 0) {
        found = i;
      }
    }
    if (found <= 0 || g_camera_states[found - 1].state < 0.0) {
      return;
    }
    from = g_camera_states[found - 1];
    to = g_camera_states[found];
  }
  // Where the clock stands against the state it draws (real - state) cycles through a step's width, but where that
  // step sits moves (the timer runs a step a little early, or the drawn state trails by most of a step): measured, the
  // low end of the last 128 frames' (2nd percentile) is where t = 0 falls, so the frames' moments stay evenly spaced.
  static std::array<double, 128> offsets{};
  static uint32_t offset_count = 0;
  offsets[offset_count++ % offsets.size()] = real - to.state;
  double low = real - to.state;
  if (offset_count >= 16) {
    std::array<double, 128> sorted = offsets;
    const size_t n = std::min<size_t>(offset_count, sorted.size());
    std::sort(sorted.begin(), sorted.begin() + n);
    low = sorted[n / 50];
  }
  const double span = to.state - from.state;
  float t = span > 0.0 && span <= 4.0
                ? float(std::clamp((real - low - 1.0 - from.state) / span, 0.0, 1.0))
                : 1.f;
  // A cut (a replay's or a crash's cameras): drawn as it is.
  const float dx = to.rows[3][0] - from.rows[3][0], dy = to.rows[3][1] - from.rows[3][1],
              dz = to.rows[3][2] - from.rows[3][2];
  if (dx * dx + dy * dy + dz * dz > 10.f * 10.f) {
    t = 1.f;
  }
  vr::SetFrameInterpolation(vr::GameSwapCount(), to.state, t);
  if (static const bool log = std::getenv("XERENGE_VR_INTERPLOG") != nullptr; log) {  // development
    const float x = from.rows[3][0] + (to.rows[3][0] - from.rows[3][0]) * t,
                z = from.rows[3][2] + (to.rows[3][2] - from.rows[3][2]) * t;
    REXLOG_INFO("VR interpolation: frame {} state {} t {:.3f} real {:.3f} camera {:.3f} {:.3f} (state's {:.3f} {:.3f})",
                vr::GameSwapCount(), to.state, t, real, x, z, to.rows[3][0], to.rows[3][2]);
  }
  if (t >= 1.f) {
    return;
  }
  float out[4][3];
  for (int r = 0; r < 4; ++r) {
    for (int k = 0; k < 3; ++k) {
      out[r][k] = from.rows[r][k] + (to.rows[r][k] - from.rows[r][k]) * t;
    }
  }
  // The axes made orthonormal again (at, then up, then right), each keeping its length (1 for a camera).
  auto dot = [](const float* a, const float* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
  auto normalize = [&](float* a) {
    const float l = std::sqrt(dot(a, a));
    if (l > 1e-6f) {
      a[0] /= l, a[1] /= l, a[2] /= l;
    }
  };
  normalize(out[2]);
  const float ua = dot(out[1], out[2]);
  for (int k = 0; k < 3; ++k) {
    out[1][k] -= ua * out[2][k];
  }
  normalize(out[1]);
  const float ra = dot(out[0], out[2]), ru = dot(out[0], out[1]);
  for (int k = 0; k < 3; ++k) {
    out[0][k] -= ra * out[2][k] + ru * out[1][k];
  }
  normalize(out[0]);
  for (uint32_t matrix : {m, m + 64}) {  // the matrix and the LTM
    for (int r = 0; r < 4; ++r) {
      for (int k = 0; k < 3; ++k) {
        StoreFloat(base, matrix + 16 * r + 4 * k, out[r][k]);
      }
    }
  }
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
      NoteCameraState(base, ctx.r1.u32 + 80);  // the new matrix, on CGtViewport::Update's stack
    }
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
    {  // The head's pose taken again for each frame drawn (once a frame: its three reads), not only at CGtViewport::Update:
       // that runs once a logic step, 60 a second, and at 90 Hz every third frame was drawn with the last one's head
       // (simulator, smooth head sweep: per-frame turn 1.2, 0.6, 0 degrees; Matt 2026-10-31: "Turning my head still makes
       // the camera jerky in the car selection screen", in races the world moved with the head then snapped back).
      static uint64_t pose_frame = ~0ull;
      static float frame_pose[16];
      static bool frame_posed = false;
      if (pose_frame != vr::GameSwapCount()) {
        pose_frame = vr::GameSwapCount();
        float tangents[4];
        frame_posed = vr::HeadPoseNow(frame_pose, tangents);
      }
      if (frame_posed) {
        std::memcpy(pose, frame_pose, sizeof(pose));
        posed = true;
      }
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
    {  // once a frame (three reads): the same moment for all
      static uint64_t interpolated_frame = ~0ull;
      static uint8_t interpolated[128];
      if (interpolated_frame != vr::GameSwapCount()) {
        interpolated_frame = vr::GameSwapCount();
        float logic_camera[12], drawn_camera[12];  // rows right, up, at, pos: the state's, and as drawn (between)
        for (int i = 0; i < 12; ++i) {
          logic_camera[i] = LoadFloat(base, copy + 16 * (i / 3) + 4 * (i % 3));
        }
        InterpolateCamera(base, copy);
        for (int i = 0; i < 12; ++i) {
          drawn_camera[i] = LoadFloat(base, copy + 16 * (i / 3) + 4 * (i % 3));
        }
        vr::NoteFrameCameras(logic_camera, drawn_camera);  // the race's markers (plume_vr.cpp PlaceMarker)
        std::memcpy(interpolated, xerenge::GuestPointer(base, copy), 128);
      } else {
        std::memcpy(xerenge::GuestPointer(base, copy), interpolated, 128);
      }
    }
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
    if (static const bool pose_log = std::getenv("XERENGE_VR_POSELOG") != nullptr; pose_log) {  // development
      static uint64_t logged_frame = ~0ull;
      if (logged_frame != vr::GameSwapCount()) {
        logged_frame = vr::GameSwapCount();
        REXLOG_INFO("VR pose drawn: frame {} at {} us, yaw {:.3f} degrees", logged_frame,
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count(),
                    std::atan2(-pose[8], pose[10]) * 57.29578f);
      }
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
