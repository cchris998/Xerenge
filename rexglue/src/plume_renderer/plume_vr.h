/**
 * @file        plume_renderer/plume_vr.h
 * @brief       VR (the vr branch): plume's frames to the headset through the SDK's OpenXR layer (vr_openxr.h).
 */
#pragma once

#include <cstdint>
#include <vector>

namespace rex::plume_renderer {
struct GuestDrawSnapshot;
}  // namespace rex::plume_renderer

namespace plume {
struct RenderInterface;
struct RenderDevice;
struct RenderCommandQueue;
struct RenderTexture;
struct RenderCommandList;
}  // namespace plume

namespace rex::plume_renderer::vr {

/// XERENGE_VR_PLUME=1 (burnout_vr.cpp, when VR runs on plume): the headset's session on plume's own Vulkan device.
bool Requested();
/// Both eyes in one pass (multiview): the stereo shaders, 2-layer scene targets. False without a session, or with
/// XERENGE_VR_MONO=1 (the same picture to both eyes).
bool Stereo();
/// The eye's (0 left, 1 right) clip shift of perspective geometry (x += it), for the stereo shaders.
float EyeShift(int eye);
/// The shaders' multiview variants are to be used (Stereo, and decided before the first shader is made).
inline bool StereoShaders() { return Stereo(); }
/// Before plume's Vulkan interface is made: the runtime's instance and device extensions and features, and a native
/// queue kept for it.
void BeforeInterface();
/// After the device: the headset's session on it.
void AfterDevice(plume::RenderInterface* render_interface, plume::RenderDevice* device);
/// The game's thread, at a frame's present (it drew the frame): the pose its camera was read with.
void NoteFramePresented();
/// The game's thread, at its Swap: the frame's pose kept under a serial for the Swap marker (GuestDrawSnapshot
/// vr_pose_serial; 0 when VR is off).
uint32_t NoteFrameEnd();
/// The present, taking a Swap marker into its batch: the frame it draws (its last) gets that frame's pose (BeginFrame).
void UseFramePose(uint32_t serial);
/// Whether the vertex shader (plume's shader hash, or the microcode's) is one of the HUD's (MCLA_VR_HUD_VS, the
/// microcode hashes the SDK's xenos path splits at; burnout_vr.cpp sets the title's).
bool HudVertexShader(uint64_t vs_hash);

// Development: the DRAWLOG trigger's frame is being logged.
bool DevLogging();

// Whether this frame's HUD split has happened (SplitAtInterface): everything drawn after it is 2D, kept in the 16:9 box
// like the marked interface - the pause menu, not always marked, was stretched over a 3840x4320 eye (Matt 2026-10-31:
// "The pause menu is too tall").
bool InterfaceSplit();

// The race's markers over the other cars (blue triangles and their outlines: the 2D shader, gPosOffset c0 with the car's
// depth-buffer value in z where the interface has ~0) were on the HUD sheet, at its distance whatever the car's (Matt
// 2026-10-31: "they should be the depth of each car they relate to"). They stay in the scene: PlaceMarker scales c0 and
// c1.xy by the car's view depth (from the game camera's projection), the same place on screen with w = that depth, so
// the eyes' clip shift puts them at it.
bool MarkerDraw(uint64_t vs_hash, uint64_t ps_hash, uint32_t count, const float* vs_constants);
void PlaceMarker(float* vs_constants, const float* centre_pixels);

// A HUD draw's blend control (after the split, over transparent black): alpha SRC_ALPHA becomes ONE, so the layer's
// alpha is the HUD's coverage (a + (1 - a) under), as the xenos path does. Else as given.
uint32_t HudBlendControl(uint32_t blend_control);

// A draw drawn in the game's view space: its first four vertex constants (gObjToProjMatrix) a perspective projection
// alone. The particle system builds the sparks and the speed streaks there on the CPU from the game's own camera, not
// the renderer's head-turned one, so they followed the head (Burnout 2026-10-31, the paused race turned 25 degrees: the
// streaks kept their place in the view). TurnViewSpaceDraw puts the head's turn into that projection.
bool ViewSpaceDraw(const float* vs_constants);
void TurnViewSpaceDraw(float* vs_constants);

// Drawing between logic steps (logic at 60 steps a second under the headset's faster frames; burnout_vr.cpp turns the
// camera to the frame's moment and names the frame's state and t, vr::SetFrameInterpolation): each moving draw of the
// scene goes from where it was at the state before to where it is, as the SDK's xenos path does
// (command_processor.cpp). Its world is its WVP (c0-c3) over the frame's camera, the WVP of MCLA_VR_CAMERA_VS's draws.
// True with the draw's c0-c3 to use in `wvp_out`. Every depth-tested draw of the camera's target is recorded.
bool InterpolateDraw(uint64_t vs_hash, uint64_t mesh_key, bool depth_tested, uint32_t target_width,
                     const float* vs_constants, float* wvp_out);

// Development (VR): the trigger files %TEMP%\xerenge_vr\DRAWLOG (log the next frame's draws: shaders, blend, write
// mask, primitive and count, and whether in the HUD) and SKIP (a list of shader hashes, hex: their draws are left out
// while it lists them). True: leave this draw out.
bool DevDraw(uint64_t vs_hash, uint64_t ps_hash, uint32_t blend_control, uint32_t color_mask, uint32_t prim_type,
             uint32_t count, bool hud);
// Development: a frame ends (the draw log's frame).
void DevFrameEnd();

// A HUD draw's colour write mask: one that blends over (colour dest ONE_MINUS_SRC_ALPHA) but leaves alpha out writes
// alpha too, so it covers what is under it in the layer (else it is added on top: bars washed out, dark backings gone).
uint32_t HudColorMask(uint32_t color_mask, uint32_t blend_control);
/// The present worker, before the frame's draws are encoded: the frame's pose (NoteFramePresented's, in order).
void BeginFrame();
/// Encoding the frame's draws, at its first interface draw: the scene (`target`, both eyes, in COLOR_WRITE, outside a
/// pass) to the frame's screen, and the target cleared to transparent black for the HUD (its own layer).
void SplitAtInterface(plume::RenderCommandList* list, plume::RenderTexture* target, uint32_t width, uint32_t height);
/// After the frame's submission: `shown` (the finished picture, in COPY_SOURCE, `layers` layers) to the headset's
/// eyes, on the same queue, with the session's timeline signal.
void Submit(plume::RenderDevice* device, plume::RenderCommandQueue* queue, plume::RenderTexture* shown, uint32_t width,
            uint32_t height, uint32_t layers);

}  // namespace rex::plume_renderer::vr
