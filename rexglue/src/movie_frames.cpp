// Retire the movie renderer's in-flight frames.
//
// CCalVideoRenderer counts the frames it has handed to the GPU in a field at
// renderer+368, bumping it per blit, and the only thing that drops it is the
// callback the console runs from the Xenos completion interrupt. Until it
// falls, Render answers "still pending", CCalMoviePlayer never advances, and
// CB4FrontEnd::Update - which calls into the player from the title's main
// thread - never returns. That is the freeze: not a video defect but the whole
// guest stopping, with the player's own worker threads waiting alongside it.
//
// That is what happened on this title's previous runtime, which had no
// interrupt-driven retire at all, and draining the counter by hand after each
// Render was the fix there.
//
// Xenos delivers that interrupt. Plume walks PM4_INTERRUPT but the movie
// player still blocks: after BG1_P is cancelled for EAHD the guest never
// reaches player 28→29, and FrontEnd::Update stops. Drain the counter on
// plume the same way the previous runtime did. Leave xenos alone — a second
// drain there underflows to 0xFFFFFFFF. XERENGE_FRAME_RETIRE forces it on
// either backend.
#include <rex/logging.h>
#include <chrono>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include <rex/hook.h>
#include <rex/ppc.h>
#include <rex/runtime.h>
#include <rex/system/interfaces/graphics.h>
#include <rex/system/kernel_state.h>

#include "guest_memory.h"

extern "C" void __imp__sub_82482680(PPCContext& __restrict, uint8_t*);  // Render
extern "C" void __imp__sub_824823F0(PPCContext& __restrict, uint8_t*);  // retire callback
extern "C" void __imp__sub_824822C8(PPCContext& __restrict, uint8_t*);  // Stop
extern "C" void __imp__sub_82482430(PPCContext& __restrict, uint8_t*);  // Close

namespace {

constexpr uint32_t kInFlightOffset = 368;

bool PlumePresentationActive() {
  auto* kernel_state = REX_KERNEL_STATE();
  if (!kernel_state || !kernel_state->emulator()) {
    return false;
  }
  auto* graphics = kernel_state->emulator()->graphics_system();
  return graphics && graphics->uses_direct_presentation();
}

// VR branch, xenos: set once a retire interrupt has been missed (sub_82482680): from then on the counter is drained
// as on plume.
std::atomic<bool> g_interrupt_missed{false};

bool Enabled() {
  static const bool forced = std::getenv("XERENGE_FRAME_RETIRE") != nullptr;
  return forced || PlumePresentationActive() || g_interrupt_missed.load(std::memory_order_relaxed);
}

bool Tracing() {
  static const bool on = std::getenv("XERENGE_MOVIE_TRACE") != nullptr;
  return on;
}

uint32_t InFlight(const uint8_t* base, uint32_t renderer) {
  return xerenge::LoadGuestU32(base, renderer + kInFlightOffset);
}

// Bounded, so a counter that never falls - a renderer already torn down, or one
// misread here - cannot turn the drain into a hang of its own.
void Retire(PPCContext& ctx, uint8_t* base, uint32_t renderer, uint32_t leave_in_flight) {
  for (int guard = 0; guard < 8 && InFlight(base, renderer) > leave_in_flight; ++guard) {
    ctx.r3.u32 = renderer;
    __imp__sub_824823F0(ctx, base);
  }
}

// VR branch, xenos: whether the renderer's count has stood above 0, unchanged, for a quarter of a second (a missed retire
// interrupt: the player then waits for it for good - in Render at 3, in its flush and teardown for 0). Sets
// g_interrupt_missed, and from then on the counter is drained as on plume.
void WatchForMissedRetire(const uint8_t* base, uint32_t renderer) {
  static uint32_t watched = 0, value = 0;
  static std::chrono::steady_clock::time_point since;
  if (g_interrupt_missed.load(std::memory_order_relaxed) || PlumePresentationActive()) {
    return;
  }
  const uint32_t now_value = InFlight(base, renderer);
  const auto now = std::chrono::steady_clock::now();
  if (now_value == 0 || now_value >= 0x80000000u || renderer != watched || now_value != value) {
    watched = renderer;
    value = now_value;
    since = now;
    return;
  }
  if (now - since > std::chrono::milliseconds(250)) {
    REXLOG_WARN("movie: renderer {:08X} held {} frame(s) for 250 ms (a missed retire interrupt): drained from now on",
                renderer, now_value);
    g_interrupt_missed = true;
  }
}

}  // namespace

REX_HOOK_RAW(sub_82482680) {
  const uint32_t renderer = ctx.r3.u32;
  WatchForMissedRetire(base, renderer);  // VR branch, xenos
  if (g_interrupt_missed.load(std::memory_order_relaxed) && InFlight(base, renderer) >= 0x80000000u) {
    xerenge::StoreGuestU32(base, renderer + kInFlightOffset, 0);  // a late interrupt after a drain went below 0
  }
  if (Enabled()) {
    // Render refuses the blit when 3 frames are already outstanding. If the
    // CP interrupt never retires them, the player retries forever on the
    // main thread and the title stops.
    if (InFlight(base, renderer) >= 3) {
      Retire(ctx, base, renderer, 2);
      ctx.r3.u32 = renderer;
    }
  }
  __imp__sub_82482680(ctx, base);
  if (!Enabled()) {
    return;
  }
  // Keep the guest-visible result: the retire callback writes its own return
  // value to r3, and leaking that makes the caller read a successful frame
  // submission as a failure.
  const uint64_t render_result = ctx.r3.u64;
  if (Tracing()) {
    static std::atomic<uint32_t> calls{0};
    const uint32_t n = calls.fetch_add(1, std::memory_order_relaxed);
    if (n < 8) {
      std::cerr << "movie: render #" << n << " in flight " << InFlight(base, renderer) << '\n';
    }
  }
  Retire(ctx, base, renderer, 1);
  ctx.r3.u64 = render_result;
}

REX_HOOK_RAW(sub_824822C8) {
  const uint32_t renderer = ctx.r3.u32;
  WatchForMissedRetire(base, renderer);  // VR branch, xenos
  if (Enabled()) {
    Retire(ctx, base, renderer, 0);
    ctx.r3.u32 = renderer;
  }
  __imp__sub_824822C8(ctx, base);
}

REX_HOOK_RAW(sub_82482430) {
  const uint32_t renderer = ctx.r3.u32;
  WatchForMissedRetire(base, renderer);  // VR branch, xenos
  if (Enabled()) {
    Retire(ctx, base, renderer, 0);
    ctx.r3.u32 = renderer;
  }
  __imp__sub_82482430(ctx, base);
}

// VR branch, xenos: the movie player's worker and the title's main thread hand frames to each other through the
// player's synchronization events (sub_8248D1E0..sub_8248D240 wait without a timeout on the events at +84, +100,
// +116, +132, +148). On xenos a wakeup was now and then lost: the worker waiting on +100 (sub_8248D1F8, called from
// 0x8248DAC4) and the main thread on +132 (sub_8248D228) for good - the main menu froze on 3 of 4 boots, flat and in
// VR, never on plume. The worker's wait gets a quarter-second limit there, after which it carries on as if woken (it
// then signals the main thread). XERENGE_NO_MOVIE_WAIT_LIMIT=1 leaves it as it was.
extern "C" void __imp__sub_8248D1F8(PPCContext& __restrict, uint8_t*);
extern "C" void __imp__sub_8248D228(PPCContext& __restrict, uint8_t*);
extern "C" void __imp__KeWaitForSingleObject(PPCContext& __restrict, uint8_t*);

namespace {

// The player's wait on its event at `offset` with a quarter-second limit; false when it was not limited (plume, or
// XERENGE_NO_MOVIE_WAIT_LIMIT=1: the caller runs the original).
bool LimitedPlayerWait(PPCContext& ctx, uint8_t* base, uint32_t offset, const char* who) {
  static const bool off = std::getenv("XERENGE_NO_MOVIE_WAIT_LIMIT") != nullptr;
  if (off || PlumePresentationActive()) {
    return false;
  }
  static uint32_t timeout = 0;
  if (!timeout) {
    timeout = REX_KERNEL_MEMORY()->SystemHeapAlloc(8, 8);
    // A relative timeout: -250 ms in 100 ns units, big-endian.
    const uint64_t bits = uint64_t(int64_t(-2500000));
    xerenge::StoreGuestU32(base, timeout, uint32_t(bits >> 32));
    xerenge::StoreGuestU32(base, timeout + 4, uint32_t(bits));
  }
  const uint32_t player = ctx.r3.u32;
  const uint64_t lr = ctx.lr;
  ctx.r3.u64 = player + offset;
  ctx.r4.u64 = 3;
  ctx.r5.u64 = 1;
  ctx.r6.u64 = 0;
  ctx.r7.u64 = timeout;
  __imp__KeWaitForSingleObject(ctx, base);
  ctx.lr = lr;
  if (ctx.r3.u32 == 0x00000102u) {  // STATUS_TIMEOUT: a lost wakeup
    static std::atomic<uint32_t> told{0};
    if (told.fetch_add(1, std::memory_order_relaxed) < 20) {
      REXLOG_WARN("movie: the player's {} waited 250 ms on {:08X}+{} (a lost wakeup): carried on", who, player,
                  offset);
    }
    ctx.r3.u64 = 0;
  }
  return true;
}

}  // namespace

REX_HOOK_RAW(sub_8248D1F8) {
  if (!LimitedPlayerWait(ctx, base, 100, "worker")) {
    __imp__sub_8248D1F8(ctx, base);
  }
}

REX_HOOK_RAW(sub_8248D228) {
  if (!LimitedPlayerWait(ctx, base, 132, "main thread")) {
    __imp__sub_8248D228(ctx, base);
  }
}
