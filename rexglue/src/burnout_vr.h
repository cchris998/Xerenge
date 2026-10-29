// VR (the vr branch): see burnout_vr.cpp.
#pragma once

#include <rex/runtime.h>

namespace xerenge::vr_port {

/// --vr=true (the SDK's cvar) or XERENGE_VR=1.
bool Requested();

/// OnPreSetup: in VR, the xenos GPU plugin with its Vulkan backend.
void ConfigureAtStart(rex::RuntimeConfig& config);

/// Once a frame, the game's frame timer: one game frame per headset frame while the game outpaces it.
void PaceFrame();

}  // namespace xerenge::vr_port
