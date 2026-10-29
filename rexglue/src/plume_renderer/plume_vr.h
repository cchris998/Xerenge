/**
 * @file        plume_renderer/plume_vr.h
 * @brief       VR (the vr branch): plume's frames to the headset through the SDK's OpenXR layer (vr_openxr.h).
 */
#pragma once

#include <cstdint>

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
/// Whether the vertex shader (plume's shader hash, or the microcode's) is one of the HUD's (MCLA_VR_HUD_VS, the
/// microcode hashes the SDK's xenos path splits at; burnout_vr.cpp sets the title's).
bool HudVertexShader(uint64_t vs_hash);
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
