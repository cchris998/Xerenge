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
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <tlhelp32.h>
#endif
#include <thread>
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
extern "C" void __imp__sub_8259C328(PPCContext& __restrict, uint8_t*);
extern "C" void __imp__sub_8259C400(PPCContext& __restrict, uint8_t*);
extern "C" void __imp__sub_825AFE90(PPCContext& __restrict, uint8_t*);

namespace vr = rex::ui::vulkan::vr;
bool XerengeLogicClock(double& state_steps, double& real_steps);

namespace {

// The seat: an offset added to the head's place in the headset's space (x right, y up, z back), moved with the arrow
// keys (left/right, up/down) and Ctrl+Up/Down (forward/back) while the game's window has the focus, kept in
// xerenge.conf as vr_offset "x y z" metres (Matt 2026-11-01: "arrow keys for moving the camera, and ctrl+up or down
// for the depth"). Added to the game's camera only (PoseRwMatrix) and, through XERENGE_VR_SEAT in the process's
// environment, to plume's view-space draws: not to the pose the headset is told the frame was drawn from - SteamVR
// reprojected each frame back to the real head, the image warped and the car came no closer (Matt 2026-11-01).
std::atomic<float> g_seat[3] = {0.f, 0.f, 0.f};
// Metres a second and the farthest each way: forward and back faster and farther, to reach into the car from the chase
// camera, 5-6 m behind it (Matt 2026-11-01: "it should move the camera forwards into the car").
constexpr float kSeatSpeed[3] = {0.25f, 0.25f, 1.f};
constexpr float kSeatRange[3] = {2.f, 2.f, 8.f};

// The cockpit view, Y in a race (CycleViews) or C chooses it (kept in xerenge.conf as vr_cockpit): the camera pinned in the player's car, facing its
// way (PinCamera), with a seat of its own (vr_cockpit_offset, in the car's axes from the car's origin) - the chase
// view's vr_offset is kept for it. g_seat is the view's in use. It comes on once the race has started (PinCamera: the
// GO), and while the game's camera is its chase camera: car selection, the race's introduction, the countdown and the
// game's own shots (takedowns, crashes) are the game's (Matt 2026-11-01).
std::atomic<bool> g_cockpit{false};      // the setting
std::atomic<bool> g_race_on{false};      // the race started (its GO)
std::atomic<bool> g_chase_view{false};   // the game's camera its chase camera
std::atomic<bool> g_cockpit_now{false};  // the view in use: both
// Y in a race goes round the game's two cameras with the cockpit between them: bumper (low on the track), cockpit,
// chase, bumper (Matt 2026-11-02: "From the low on track view, if you press Y it should go to cockpit view. From cockpit
// view Y goes to chase cam"). The game's camera last seen (a takedown's or a crash's shot keeps it): from the bumper the
// press goes to the game (to its chase camera) and turns the cockpit on; in the cockpit the game never sees it
// (CycleViews), and the cockpit goes off over the chase camera.
enum class GameView { kUnknown, kChase, kBumper };
std::atomic<GameView> g_game_view{GameView::kUnknown};
std::atomic<bool> g_cockpit_coming{false};  // the cockpit at the game's first chase update, not 5 on
std::atomic<int64_t> g_view_seen_ms{0};       // when the game's camera was last its chase or bumper camera (steady ms)
// The driver's seat from the car's origin (x right, y up, z back), Matt's, set in the headset (2026-11-01).
constexpr const char* kCockpitSeat = "-0.350 0.722 0.000";

void PublishSeat();

// The player's car's model in the cockpit (plume_vr.cpp, XERENGE_VR_CARMODEL; 0 before it is known): each has its own
// cockpit seat, vr_cockpit_offset_<model> - the cars' cabins are different heights (Matt 2026-11-01: "we need to store
// the user offset per car"). A car without one starts from vr_cockpit_offset, then kCockpitSeat.
std::atomic<uint64_t> g_car_model{0};

std::string SeatKey() {
  if (!g_cockpit_now.load(std::memory_order_relaxed)) {
    return "vr_offset";
  }
  const uint64_t model = g_car_model.load(std::memory_order_relaxed);
  return model ? fmt::format("vr_cockpit_offset_{:016X}", model) : std::string("vr_cockpit_offset");
}

void LoadSeat() {
  float x = 0.f, y = 0.f, z = 0.f;
  const std::string fallback = !g_cockpit_now.load() ? std::string("0 0 0")
                               : g_car_model.load() ? xerenge::settings::ConfString("vr_cockpit_offset", kCockpitSeat)
                                                    : std::string(kCockpitSeat);
  if (std::sscanf(xerenge::settings::ConfString(SeatKey(), fallback).c_str(), "%f %f %f", &x, &y, &z) == 3) {
    g_seat[0] = std::clamp(x, -kSeatRange[0], kSeatRange[0]);
    g_seat[1] = std::clamp(y, -kSeatRange[1], kSeatRange[1]);
    g_seat[2] = std::clamp(z, -kSeatRange[2], kSeatRange[2]);
  }
  PublishSeat();
}

// plume_vr.cpp CockpitGlass and NoteCockpitCar see it as XERENGE_VR_SEAT's fourth number.
bool InCockpit() { return g_cockpit_now.load(std::memory_order_relaxed); }

// The view in use from the setting and the race; on a change, its seat.
void UpdateView() {
  const bool now = g_cockpit.load() && g_race_on.load() && g_chase_view.load();
  if (g_cockpit_now.exchange(now) != now) {
    LoadSeat();
    REXLOG_INFO("VR: the {} camera (race {}, game's camera {})", now ? "cockpit" : "game's", g_race_on.load() ? "on" : "off",
                g_chase_view.load() ? "chase" : "its own");
  }
}

float LoadFloat(const uint8_t* base, uint32_t address);
void StoreFloat(uint8_t* base, uint32_t address, float value);

// The cockpit's camera pinned to the car (Matt 2026-11-01: "we need the camera to actually be pinned inside the
// vehicle"): the chase camera swings round the car on a spring, and moved inside it went through the bodywork. The car's
// matrix (right, up, at, pos, 16 bytes apart) is the first of the game's car table, a global in the image: the matrix the
// body is drawn with. The car's physics matrix (0x82B554E0) is another, but the body is drawn ahead of it by its speed -
// 22 cm at 36 m/s, 42 cm at 55 m/s - and pinned to that the view slid back through the car accelerating and forward
// braking (Matt: "We need to be pinned depth too").
constexpr uint32_t kPlayerCarMatrix = 0x82CE0040u;
// 1 while the player accelerates in the race: from the countdown's first shot (not before it), 0 again a second or two
// after letting go and while paused.
constexpr uint32_t kRaceAccelerating = 0x82846B80u;
constexpr uint32_t kRacePaused = 0x828466E4u;  // a byte: 1 while the race's pause menu is up
// The countdown: the game's own shots between the race's introduction (its chase camera, ~3.5 s) and the GO, 5.4 s
// (3, 2, 1: one shot each). Its chase camera back after this many updates of others (60 a second) is the GO.
constexpr int kCountdownUpdates = 120;

struct Pin {
  std::mutex mutex;
  bool have_relative = false;  // the camera's axes in the car's (rows 0-2), snapped to the car's: -1, 0 or 1 each
  float relative[3][3]{};
  float last_pos[3]{};
  // The game's own (chase) camera for each pinned one lately: it projects the race's markers with it (FrameMarkerCamera).
  std::array<std::array<float, 24>, 32> cameras{};  // pinned rows right, up, at, pos, then the game's
  size_t next_camera = 0;
  bool have_car = false;  // the car there the update before
  int chase_updates = 0;  // updates in a row the game's camera has been its chase camera
  int shot_updates = 0;   // ... and another camera (before the GO: the countdown's shots)
  uint64_t model_seen = 0;  // the car's model (plume), and for how many steps the same
  int model_steps = 0;
} g_pin;

void ReadCar(const uint8_t* base, uint32_t matrix, float (&car)[4][3]) {
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 3; ++c) {
      car[r][c] = LoadFloat(base, matrix + 16 * r + 4 * c);
    }
  }
}

// The camera's new matrix (on CGtViewport::Update's stack) put at the car's origin, facing the car's way: its axes the
// car's, matched to the chase camera's (which roughly faces the car's way) the first step. The seat (PoseRwMatrix) then
// moves it to the driver's place in those axes.
void PinCamera(uint8_t* base, uint32_t m) {
  std::scoped_lock lock{g_pin.mutex};
  float car[4][3];
  ReadCar(base, kPlayerCarMatrix, car);
  const float at2 = car[2][0] * car[2][0] + car[2][1] * car[2][1] + car[2][2] * car[2][2];
  const float jx = car[3][0] - g_pin.last_pos[0], jy = car[3][1] - g_pin.last_pos[1], jz = car[3][2] - g_pin.last_pos[2];
  // No car (a menu), or it jumped more than 50 m in a step (a new race, a restart): the race not on till its GO (below),
  // and the car's model taken again.
  const bool have_car = std::fabs(at2 - 1.f) < 0.05f;
  if (!have_car || !g_pin.have_car || !(jx * jx + jy * jy + jz * jz < 50.f * 50.f)) {
    g_race_on = false;
    g_cockpit_coming = false;
    g_game_view = GameView::kUnknown;
    g_pin.have_relative = false;
    g_pin.model_steps = 0;
    g_pin.shot_updates = 0;
#ifdef _WIN32
    SetEnvironmentVariableA("XERENGE_VR_CARMODEL", nullptr);  // the last race's car's
#endif
  }
  g_pin.have_car = have_car;
  // The game's camera its usual chase camera (1.18 m up and 4.6 m behind the car, looking its way; the spring swings it
  // half a metre aside) for 5 updates: the countdown's shots, a takedown's and a crash's are the game's own (Matt
  // 2026-11-01: "the camera usually zooms to the crashed car, we should respect that"). The race is on (latched till
  // the next) from its GO: the chase camera back after the countdown's shots - the introduction before them shows the
  // chase camera too (Matt: not "during the race countdown. Only once the race starts") - or the player accelerating
  // (an event without the shots).
  {
    float game[4][3];
    ReadCar(base, m, game);
    const float d[3] = {game[3][0] - car[3][0], game[3][1] - car[3][1], game[3][2] - car[3][2]};
    const auto along = [&](const float* v, int k) { return v[0] * car[k][0] + v[1] * car[k][1] + v[2] * car[k][2]; };
    const float x = along(d, 0), y = along(d, 1), z = along(d, 2);
    const bool chase = std::fabs(x) < 1.5f && y > 0.5f && y < 2.2f && z > -7.f && z < -3.f && along(game[2], 2) > 0.9f;
    // The bumper camera: 0.55 m up and 1.10 m forward, looking the car's way (a crash's shot from 1.14 up, 1 m back was
    // taken for it when looser).
    const bool bumper = std::fabs(x) < 0.5f && y > 0.2f && y < 0.9f && z > 0.5f && z < 2.f && along(game[2], 2) > 0.9f;
    g_pin.chase_updates = chase ? std::min(g_pin.chase_updates + 1, 5) : 0;
    if (chase && g_cockpit_coming.exchange(false)) {
      g_pin.chase_updates = 5;
    }
    g_chase_view = g_pin.chase_updates >= 5;
    g_pin.shot_updates = chase || bumper ? g_pin.shot_updates : std::min(g_pin.shot_updates + 1, kCountdownUpdates);
    if (have_car && ((g_chase_view && g_pin.shot_updates >= kCountdownUpdates) ||
                     xerenge::LoadGuestU32(base, kRaceAccelerating) == 1)) {
      g_race_on = true;
    }
    if (chase || bumper) {
      g_view_seen_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                           std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    const GameView view = g_chase_view ? GameView::kChase : bumper ? GameView::kBumper : g_game_view.load();
    if (g_game_view.exchange(view) != view) {
      REXLOG_INFO("VR: the game's {} camera ({:.2f} {:.2f} {:.2f} from the car)",
                  view == GameView::kChase ? "chase" : "bumper", x, y, z);
    }
  }
  std::memcpy(g_pin.last_pos, car[3], sizeof(g_pin.last_pos));
  UpdateView();
#ifdef _WIN32
  // The car's place for the last few logic states, for plume's model of it (plume_vr.cpp NoteCockpitCar).
  if (double state, real; InCockpit() && XerengeLogicClock(state, real)) {
    static std::array<std::array<double, 4>, 4> places{};
    if (places[0][0] != state) {
      std::rotate(places.rbegin(), places.rbegin() + 1, places.rend());
      places[0] = {state, car[3][0], car[3][1], car[3][2]};
      std::string text;
      for (const auto& p : places) {
        text += fmt::format("{:.0f} {:.4f} {:.4f} {:.4f} ", p[0], p[1], p[2], p[3]);
      }
      SetEnvironmentVariableA("XERENGE_VR_CAR0", text.c_str());
    }
  }
#endif
  if (!InCockpit()) {
    g_pin.have_relative = false;
    return;
  }
  // The car's model taken once it holds for half a second at the start (a part knocked off later changes the meshes
  // drawn with the body: kept till the next race), its seat loaded.
#ifdef _WIN32
  if (g_pin.model_steps < 30) {
    char text[32] = {};
    const uint64_t model = GetEnvironmentVariableA("XERENGE_VR_CARMODEL", text, sizeof(text)) > 0
                               ? std::strtoull(text, nullptr, 16)
                               : 0;
    g_pin.model_steps = model && model == g_pin.model_seen ? g_pin.model_steps + 1 : 0;
    g_pin.model_seen = model;
    if (g_pin.model_steps == 30 && model != g_car_model.load()) {
      g_car_model = model;
      LoadSeat();
      REXLOG_INFO("VR cockpit: car {:016X}, seat {:.3f} {:.3f} {:.3f}", model, g_seat[0].load(), g_seat[1].load(),
                  g_seat[2].load());
    }
  }
#endif
  float cam[4][3];
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 3; ++c) {
      cam[r][c] = LoadFloat(base, m + 16 * r + 4 * c);
    }
  }
  const auto dot = [](const float* a, const float* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
  if (!g_pin.have_relative) {
    // Each camera axis the car axis nearest it; taken when each is clearly one (the chase camera within ~35 degrees of
    // the car's way), and they make a turn, not a mirror.
    bool clear = true;
    for (int r = 0; r < 3; ++r) {
      int nearest = 0;
      float best = 0.f;
      for (int k = 0; k < 3; ++k) {
        const float d = dot(cam[r], car[k]);
        if (std::fabs(d) > std::fabs(best)) {
          best = d;
          nearest = k;
        }
      }
      clear = clear && std::fabs(best) > 0.8f;
      for (int k = 0; k < 3; ++k) {
        g_pin.relative[r][k] = k == nearest ? (best < 0.f ? -1.f : 1.f) : 0.f;
      }
    }
    const float(&q)[3][3] = g_pin.relative;
    const float det = q[0][0] * (q[1][1] * q[2][2] - q[1][2] * q[2][1]) -
                      q[0][1] * (q[1][0] * q[2][2] - q[1][2] * q[2][0]) +
                      q[0][2] * (q[1][0] * q[2][1] - q[1][1] * q[2][0]);
    if (!clear || std::fabs(det) < 0.5f) {
      return;
    }
    g_pin.have_relative = true;
  }
  std::array<float, 24> pair;
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 3; ++c) {
      float v = r == 3 ? car[3][c] : 0.f;
      for (int k = 0; k < 3 && r < 3; ++k) {
        v += g_pin.relative[r][k] * car[k][c];
      }
      StoreFloat(base, m + 16 * r + 4 * c, v);
      pair[r * 3 + c] = v;
      pair[12 + r * 3 + c] = cam[r][c];
    }
  }
  // Kept once each (the camera is updated more often than the logic steps, and the frame drawn is a step or two old).
  const auto& last = g_pin.cameras[(g_pin.next_camera + g_pin.cameras.size() - 1) % g_pin.cameras.size()];
  if (std::memcmp(last.data(), pair.data(), 12 * sizeof(float)) != 0) {
    g_pin.cameras[g_pin.next_camera] = pair;
    g_pin.next_camera = (g_pin.next_camera + 1) % g_pin.cameras.size();
  }
}

// The frame's cameras taken (the renderer's read): in the cockpit, the game's own camera for the frame's pinned one
// (`logic`, rows right, up, at, pos), for plume as XERENGE_VR_MARKERCAM - read on this thread when the frame ends
// (plume_vr.cpp TakeBuiltFrame). The game projects the race's markers with its chase camera, and turned to the pinned
// one they sat low and swung with its spring (Matt 2026-11-01: "HUD markers are very jumpy when in cockpit view").
void FrameMarkerCamera(const float (&logic)[12]) {
#ifdef _WIN32
  std::scoped_lock lock{g_pin.mutex};
  for (const auto& c : g_pin.cameras) {
    if (InCockpit() && std::memcmp(c.data(), logic, sizeof(logic)) == 0) {
      std::string text;
      for (int i = 12; i < 24; ++i) {
        text += fmt::format("{:.6f} ", c[i]);
      }
      SetEnvironmentVariableA("XERENGE_VR_MARKERCAM", text.c_str());
      return;
    }
  }
  if (static int missed = 0; InCockpit() && std::getenv("XERENGE_VR_CARCHECK") && ++missed % 60 == 1) {  // development
    REXLOG_INFO("VR markers: the frame's camera not among the pinned ones ({} times)", missed);
  }
  SetEnvironmentVariableA("XERENGE_VR_MARKERCAM", nullptr);
#endif
}

// The seat for plume (plume_vr.cpp TurnViewSpaceDraw): the process's environment, which both modules see.
void PublishSeat() {
#ifdef _WIN32
  SetEnvironmentVariableA("XERENGE_VR_SEAT", fmt::format("{:.4f} {:.4f} {:.4f} {}", g_seat[0].load(), g_seat[1].load(),
                                                         g_seat[2].load(), InCockpit() ? 1 : 0).c_str());
#endif
}

// The cockpit view chosen (C, or Y in a race), kept in xerenge.conf.
void SetCockpit(bool on) {
  g_cockpit = on;
  xerenge::settings::SetConfString("vr_cockpit", on ? "1" : "0");
  UpdateView();
  REXLOG_INFO("VR: {} view", on ? "cockpit" : "chase");
}

// The pad as the game reads it (`state`, an XINPUT_STATE: the buttons big-endian at +4, Y 0x8000) for player `user`:
// Y going down in a race (the game's race-running flag; not paused) goes round the views (g_game_view). A press the
// game is not to see is taken out of the state while it is held.
void CycleViews(uint8_t* base, uint32_t user, uint32_t state) {
  static bool held[4] = {}, hidden[4] = {};
  if (user >= 4 || !vr::Active()) {
    return;
  }
  uint8_t* high = xerenge::GuestPointer(base, state + 4);  // the buttons' high byte: Y is its 0x80
  const bool y = (*high & 0x80) != 0;
  if (y && !held[user]) {
    const int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now().time_since_epoch()).count();
    const bool paused = *xerenge::GuestPointer(base, kRacePaused) != 0;
    hidden[user] = false;
    // In a race (the game's camera its chase or bumper camera lately: not a menu, the results), not paused (Y does
    // nothing there).
    if (!paused && now - g_view_seen_ms.load() < 1500) {
      if (InCockpit()) {
        hidden[user] = true;  // cockpit -> chase: the game stays on its chase camera
        SetCockpit(false);
      } else if (g_game_view.load() == GameView::kBumper) {
        g_cockpit_coming = true;  // bumper -> the game's chase camera, and the cockpit over it
        if (!g_cockpit.load()) {
          SetCockpit(true);
        }
      }
    }
  }
  held[user] = y;
  if (y && hidden[user]) {
    *high &= 0x7F;
  }
}

// Once a game frame: the keys held move the seat (kSeatSpeed); let go, it is saved. C changes view.
void MoveSeat() {
#ifdef _WIN32
  static auto last = std::chrono::steady_clock::now();
  static bool moving = false;
  const auto now = std::chrono::steady_clock::now();
  const float dt = std::min(std::chrono::duration<float>(now - last).count(), 0.1f);
  last = now;
  DWORD process = 0;
  GetWindowThreadProcessId(GetForegroundWindow(), &process);
  const auto held = [](int key) { return (GetAsyncKeyState(key) & 0x8000) != 0; };
  float move[3] = {0.f, 0.f, 0.f};
  if (process == GetCurrentProcessId()) {
    const bool ctrl = held(VK_CONTROL);
    move[0] = (held(VK_RIGHT) ? 1.f : 0.f) - (held(VK_LEFT) ? 1.f : 0.f);
    const float up_down = (held(VK_UP) ? 1.f : 0.f) - (held(VK_DOWN) ? 1.f : 0.f);
    (ctrl ? move[2] : move[1]) = ctrl ? -up_down : up_down;  // forward is -z
    static bool c_was = true;  // a press counts once C has been seen up
    const bool c = held('C');
    if (c && !c_was) {
      SetCockpit(!g_cockpit.load());
    }
    c_was = c;
  }
  const bool any = move[0] != 0.f || move[1] != 0.f || move[2] != 0.f;
  for (int k = 0; k < 3 && any; ++k) {
    g_seat[k] = std::clamp(g_seat[k].load() + move[k] * kSeatSpeed[k] * dt, -kSeatRange[k], kSeatRange[k]);
  }
  if (any) {
    PublishSeat();
  }
  if (moving && !any) {
    xerenge::settings::SetConfString(
        SeatKey(), fmt::format("{:.3f} {:.3f} {:.3f}", g_seat[0].load(), g_seat[1].load(), g_seat[2].load()));
    REXLOG_INFO("VR: seat offset {:.3f} {:.3f} {:.3f} m", g_seat[0].load(), g_seat[1].load(), g_seat[2].load());
  }
  moving = any;
#endif
}

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
  g_cockpit = xerenge::settings::ConfString("vr_cockpit", "0") == "1";
  LoadSeat();
  // The pad while another window has the focus (SteamVR's VR View being recorded, Matt 2026-11-01: input stopped):
  // SDL's controller events reach an unfocused window only with this; in the headset the desktop's focus means nothing.
  if (!std::getenv("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS")) {
    _putenv_s("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1");
  }
  // Back recentres the view on the head, anywhere, as F8 does (Matt 2026-10-31: "We need to recentre the headset when
  // pressing start"; 2026-11-01: "change the recentre button from start to back"). The press still reaches the game.
  // 0x0020 is X_INPUT_GAMEPAD_BACK.
  rex::input::SetButtonPressedHook([](uint16_t button) {
    if (button == 0x0020) {
      vr::Recenter();
    }
  });
  // The renderer (vr_renderer in xerenge.conf; XERENGE_VR_RENDERER wins): plume (its own renderer, no EDRAM emulation:
  // plume_vr.cpp runs the OpenXR layer on plume's device; the default, the VR work is on it) or xenos (the SDK's VR
  // layer on the xenos plugin's Vulkan backend). The camera hooks below serve both.
  std::string renderer = xerenge::settings::ConfString("vr_renderer", "plume");
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
      const int y = cross != std::string::npos ? std::clamp(std::atoi(size.c_str() + cross + 1), 1, 8) : 3;
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
    // The race camera's shake, scaled 0 (none) to 1 by xerenge.conf camera_shake, 0.25 by default (RaceCameraRotation).
    if (!std::getenv("XERENGE_VR_CAMERA_SHAKE")) {
      _putenv_s("XERENGE_VR_CAMERA_SHAKE", xerenge::settings::ConfString("camera_shake", "0.25").c_str());
    }
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
  // The static world's vertex shaders draw with the camera's view-projection as their WVP (the world at the identity):
  // the camera, and the other draws' worlds from it for drawing between logic steps. Set by hand wins.
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
      MoveSeat();
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
  if (static const bool log = std::getenv("XERENGE_VR_CAMSTATELOG") != nullptr; log) {  // development
    REXLOG_INFO("VR camera state {} pos {:.4f} {:.4f} {:.4f} at {:.5f} {:.5f} {:.5f} up {:.5f} {:.5f} {:.5f}", state,
                c.rows[3][0], c.rows[3][1], c.rows[3][2], c.rows[2][0], c.rows[2][1], c.rows[2][2], c.rows[1][0],
                c.rows[1][1], c.rows[1][2]);
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
               rows[3][c] + (pose[12] + g_seat[0].load(std::memory_order_relaxed)) * seat[0][c] +
                   (pose[13] + g_seat[1].load(std::memory_order_relaxed)) * seat[1][c] +
                   (pose[14] + g_seat[2].load(std::memory_order_relaxed)) * seat[2][c]);
  }
}

// Development (XERENGE_VR_WATCH_CAMERA=1): which code writes the camera object CGtViewport::Update reads (its
// quaternion at +32, position at +48) - a CPU write breakpoint (DR0, DR1) on every thread, the writers' host addresses
// named by the guest function they belong to (PPCFuncMappings), logged every 2 s. To find the boost's shake.
#ifdef _WIN32
namespace camera_watch {
std::atomic<uint64_t> g_hits[32];
std::atomic<uint32_t> g_counts[32];
uint32_t g_armed_object = 0;

LONG CALLBACK Handler(EXCEPTION_POINTERS* info) {
  if (info->ExceptionRecord->ExceptionCode != EXCEPTION_SINGLE_STEP || !(info->ContextRecord->Dr6 & 3)) {
    return EXCEPTION_CONTINUE_SEARCH;
  }
  const uint64_t rip = info->ContextRecord->Rip;
  for (int i = 0; i < 32; ++i) {
    uint64_t seen = g_hits[i].load(std::memory_order_relaxed);
    if (seen == rip || (seen == 0 && g_hits[i].compare_exchange_strong(seen, rip))) {
      g_counts[i].fetch_add(1, std::memory_order_relaxed);
      break;
    }
  }
  info->ContextRecord->Dr6 = 0;
  return EXCEPTION_CONTINUE_EXECUTION;
}

void Arm(uint8_t* base, uint32_t object) {
  static bool handler = false;
  if (!handler) {
    AddVectoredExceptionHandler(1, Handler);
    handler = true;
  }
  const uint64_t quaternion = uint64_t(xerenge::GuestPointer(base, object + 32));
  const uint64_t position = uint64_t(xerenge::GuestPointer(base, object + 48));
  std::thread([quaternion, position] {
    const DWORD self = GetCurrentThreadId();
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    THREADENTRY32 entry{sizeof(entry)};
    int armed = 0;
    for (BOOL more = Thread32First(snapshot, &entry); more; more = Thread32Next(snapshot, &entry)) {
      if (entry.th32OwnerProcessID != GetCurrentProcessId() || entry.th32ThreadID == self) {
        continue;
      }
      HANDLE thread = OpenThread(THREAD_GET_CONTEXT | THREAD_SET_CONTEXT | THREAD_SUSPEND_RESUME, FALSE,
                                 entry.th32ThreadID);
      if (!thread) {
        continue;
      }
      SuspendThread(thread);
      CONTEXT context{};
      context.ContextFlags = CONTEXT_DEBUG_REGISTERS;
      if (GetThreadContext(thread, &context)) {
        context.Dr0 = quaternion;
        context.Dr1 = position;
        // L0, L1; RW write (01); LEN 4 bytes (11)
        context.Dr7 = (context.Dr7 & ~0xFF0003ull) | 1 | 4 | (1ull << 16) | (3ull << 18) | (1ull << 20) | (3ull << 22);
        armed += SetThreadContext(thread, &context) ? 1 : 0;
      }
      ResumeThread(thread);
      CloseHandle(thread);
    }
    CloseHandle(snapshot);
    REXLOG_INFO("VR camera watch: armed on {} threads", armed);
  }).join();
}

void Report() {
  static auto since = std::chrono::steady_clock::now();
  if (std::chrono::steady_clock::now() - since < std::chrono::seconds(2)) {
    return;
  }
  since = std::chrono::steady_clock::now();
  for (int i = 0; i < 32; ++i) {
    const uint64_t rip = g_hits[i].load();
    if (!rip) {
      break;
    }
    // The guest function: the mapping whose host code starts nearest below.
    uint64_t best = 0;
    size_t guest = 0;
    for (const PPCFuncMapping* m = PPCFuncMappings; m->host; ++m) {
      const uint64_t host = uint64_t(m->host);
      if (host <= rip && host > best) {
        best = host, guest = m->guest;
      }
    }
    REXLOG_INFO("VR camera watch: written from host {:X} = sub_{:08X}+{:X}, {} times", rip, guest, rip - best,
                g_counts[i].load());
  }
}
}  // namespace camera_watch
#endif

// The race camera's shake: the race camera (sub_822F5128) turns its view by three small axis-angle rotations, the
// sin (sub_8259C328) and cos (sub_8259C400) of each angle taken at the return addresses below - the shake (Matt's boost
// run 2026-10-31: each one near 0 driving, up to 0.02 rad boosting). Each angle is scaled by camera_shake in
// xerenge.conf from 0 (none) to 1 (the game's), 0.25 by default (Matt 2026-10-31: "a dampener option to reduce the
// shake, say 0 - 1"; "that works perfectly, lets have 0.25 be the default"). XERENGE_VR_CAMERA_SHAKE set by hand wins.
// GRAPHICS > VR CAMERA SHAKE (the menu's and the race's pause menu) changes it at once (XerengeSetCameraShake).
// Development: XERENGE_VR_ROTLOG logs each angle.
std::atomic<double> g_camera_shake{-1.0};  // below 0: not read yet
int RaceCameraRotation(uint32_t lr) {
  switch (lr) {
    case 0x822F56F8: case 0x822F5708: return 0;
    case 0x822F5A30: case 0x822F5A40: return 1;
    case 0x822F5D5C: case 0x822F5D6C: return 2;
    default: return -1;
  }
}

void TraceRaceCameraRotation(PPCContext& ctx, bool sine) {
  const int which = RaceCameraRotation(uint32_t(ctx.lr));
  if (which < 0) {
    return;
  }
  if (sine && std::getenv("XERENGE_VR_ROTLOG")) {
    REXLOG_INFO("VR race camera rotation {}: angle {:.5f}", which, ctx.f1.f64);
  }
  double shake = g_camera_shake.load(std::memory_order_relaxed);
  if (shake < 0.0) {
    const char* v = std::getenv("XERENGE_VR_CAMERA_SHAKE");
    shake = v ? std::clamp(std::atof(v), 0.0, 1.0) : 1.0;
    g_camera_shake.store(shake, std::memory_order_relaxed);
  }
  ctx.f1.f64 *= shake;
}

void XerengeSetCameraShake(double shake) {
  g_camera_shake.store(std::clamp(shake, 0.0, 1.0), std::memory_order_relaxed);
}

REX_HOOK_RAW(sub_8259C328) {
  TraceRaceCameraRotation(ctx, true);
  __imp__sub_8259C328(ctx, base);
}

REX_HOOK_RAW(sub_8259C400) {
  TraceRaceCameraRotation(ctx, false);
  __imp__sub_8259C400(ctx, base);
}

// The game's XamInputGetState (r3 the player, r4 the state): Y in a race goes round the views (CycleViews).
// The cockpit's steering wheel turns with the player's steering, the left stick as the game reads it (sThumbLX,
// big-endian at +8), not with the front wheels as drawn: those shook when the car hit a wall or another car (Matt
// 2026-11-02: "the steering wheel jitters a lot, can we remove that effect so its just affected by your turning").
// For plume (plume_vr.cpp SteeringFrame) as XERENGE_VR_STEER, -1 (left) to 1: the pad turned furthest of those read.
void PublishSteer(uint8_t* base, uint32_t user, uint32_t state) {
  static int16_t sticks[4] = {};
  static int16_t published = 0;
  if (user >= 4 || !vr::Active()) {
    return;
  }
  const uint8_t* x = xerenge::GuestPointer(base, state + 8);
  const bool paused = *xerenge::GuestPointer(base, kRacePaused) != 0;  // the stick moves through the pause menu
  sticks[user] = paused ? 0 : int16_t(uint16_t(x[0]) << 8 | x[1]);
  int16_t steer = 0;
  for (const int16_t v : sticks) {
    steer = std::abs(v) > std::abs(steer) ? v : steer;
  }
  if (steer != published) {
    published = steer;
#ifdef _WIN32
    SetEnvironmentVariableA("XERENGE_VR_STEER", fmt::format("{:.4f}", std::max(steer / 32767.f, -1.f)).c_str());
#endif
  }
}

REX_HOOK_RAW(sub_825AFE90) {
  const uint32_t user = ctx.r3.u32, state = ctx.r4.u32;
  __imp__sub_825AFE90(ctx, base);
  if (ctx.r3.u32 == 0) {
    CycleViews(base, user, state);
    PublishSteer(base, user, state);
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
      PinCamera(base, ctx.r1.u32 + 80);  // in the cockpit, where it sits in the car
      NoteCameraState(base, ctx.r1.u32 + 80);  // the new matrix, on CGtViewport::Update's stack
#ifdef _WIN32
      if (static const bool watch = std::getenv("XERENGE_VR_WATCH_CAMERA") != nullptr; watch) {  // development
        const uint32_t object = xerenge::LoadGuestU32(base, ctx.r30.u32 + 104);
        if (object && object != camera_watch::g_armed_object) {
          camera_watch::g_armed_object = object;
          camera_watch::Arm(base, object);
        }
        camera_watch::Report();
      }
#endif
      if (static int told = 0; told < 4 && std::getenv("XERENGE_VR_CAMSTATELOG")) {  // development
        ++told;
        const uint32_t object = xerenge::LoadGuestU32(base, ctx.r30.u32 + 104);
        REXLOG_INFO("VR camera: viewport {:08X}, its camera object {:08X} (quaternion +32, position +48)", ctx.r30.u32,
                    object);
      }
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
        FrameMarkerCamera(logic_camera);
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
