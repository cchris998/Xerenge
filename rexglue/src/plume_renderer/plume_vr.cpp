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
#include "plume_renderer/plume_shader_cache.h"
#include "shader_cache.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
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

void NoteFramePresented() {
  if (!g_active) {
    return;
  }
  FramePose frame;
  frame.posed = xr::TakeCameraPose(*reinterpret_cast<float(*)[16]>(frame.pose),
                                   *reinterpret_cast<float(*)[4]>(frame.tangents));
  std::scoped_lock lock{g_pose_mutex};
  g_poses.push_back(frame);
  while (g_poses.size() > 4) {
    g_poses.pop_front();
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
  {
    std::scoped_lock lock{g_pose_mutex};
    if (!g_poses.empty()) {
      g_frame = g_poses.front();
      g_poses.pop_front();
    }
  }
  xr::SetPresentPose(g_frame.posed, *reinterpret_cast<const float(*)[16]>(g_frame.pose),
                     *reinterpret_cast<const float(*)[4]>(g_frame.tangents));
  xr::SetFrameHasScene(g_frame.posed);
}

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
