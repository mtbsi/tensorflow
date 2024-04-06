/* Copyright 2019 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_BACKENDS_PROFILER_GPU_CUPTI_TRACER_H_
#define XLA_BACKENDS_PROFILER_GPU_CUPTI_TRACER_H_

#include <functional>
#include <memory>

#include "absl/status/status.h"
#include "absl/types/optional.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti.h"
#include "third_party/gpus/cuda/include/nvtx3/nvToolsExt.h"
#include "xla/backends/profiler/gpu/cupti_collector.h"
#include "xla/backends/profiler/gpu/cupti_interface.h"
#include "tsl/platform/types.h"
#include "tsl/profiler/utils/buffer_pool.h"

namespace xla {
namespace profiler {

struct CuptiTracerOptions {
  bool required_callback_api_events = true;
  // The callback ids that will be enabled and monitored, if empty, all
  // Callback ids to be enabled using Callback API.
  // We only care CUPTI_CB_DOMAIN_DRIVER_API domain for now. It is kind of
  // redundant to have both CUPTI_CB_DOMAIN_DRIVER_API and
  // CUPTI_CB_DOMAIN_RUNTIME_API.
  std::vector<CUpti_driver_api_trace_cbid_enum> cbids_selected;
  // Activity kinds to be collected using Activity API. If empty, the Activity
  // API is disable.
  std::vector<CUpti_ActivityKind> activities_selected;
  // Whether to call cuptiFinalize.
  bool cupti_finalize = false;
  // Whether to call cuCtxSynchronize for each device before Stop().
  bool sync_devices_before_stop = false;
  // Whether to enable NVTX tracking, we need this for TensorRT tracking.
  bool enable_nvtx_tracking = false;
};

class CuptiTracer;
class CallbackAnnotationsEventsWeakPtr;
class CallbackAnnotationsAndEvents;

class CuptiDriverApiHook {
 public:
  virtual ~CuptiDriverApiHook() {}

  virtual absl::Status OnDriverApiEnter(
      int device_id, CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
      const CUpti_CallbackData* callback_info) = 0;
  virtual absl::Status OnDriverApiExit(
      int device_id, CUpti_CallbackDomain domain, CUpti_CallbackId cbid,
      const CUpti_CallbackData* callback_info) = 0;
  virtual absl::Status SyncAndFlush() = 0;

 protected:
  static absl::Status AddDriverApiCallbackEvent(
      CuptiTracer* collector, CuptiInterface* cupti_interface, int device_id,
      uint64_t start_tsc, uint64_t end_tsc, CUpti_CallbackDomain domain,
      CUpti_CallbackId cbid, const CUpti_CallbackData* callback_info);
};

// The class use to enable cupti callback/activity API and forward the collected
// trace events to CuptiTraceCollector. There should be only one CuptiTracer
// per process.
class CuptiTracer {
 public:
  // Not copyable or movable
  CuptiTracer(const CuptiTracer&) = delete;
  CuptiTracer& operator=(const CuptiTracer&) = delete;

  // Returns a pointer to singleton CuptiTracer.
  static CuptiTracer* GetCuptiTracerSingleton();

  // Only one profile session can be live in the same time.
  bool IsAvailable() const;
  bool NeedRootAccess() const { return need_root_access_; }

  void Enable(const CuptiTracerOptions& option, CuptiTraceCollector* collector);
  void Disable();

  absl::Status HandleCallback(CUpti_CallbackDomain domain,
                              CUpti_CallbackId cbid,
                              const CUpti_CallbackData* callback_info);

  // Returns a buffer and its size for CUPTI to store activities. This buffer
  // will be reclaimed when CUPTI makes a callback to ProcessActivityBuffer.
  void RequestActivityBuffer(uint8_t** buffer, size_t* size);

  // Parses CUPTI activity events from activity buffer, and emits events for
  // CuptiTraceCollector. This function is public because called from registered
  // callback. This just cache the buffer in the collector_.
  absl::Status ProcessActivityBuffer(CUcontext context, uint32_t stream_id,
                                     uint8_t* buffer, size_t size);

  static uint64_t GetTimestamp();
  static int NumGpus();
  // Returns the error (if any) when using libcupti.
  static std::string ErrorIfAny();

  // Return the last event in per-thread call back event buffer or nullptr.
  CuptiTracerEvent* LastCallbackEvent();

 protected:
  // protected constructor for injecting mock cupti interface for testing.
  explicit CuptiTracer(CuptiInterface* cupti_interface);

 private:
  // Gather all per-thread callback events and annotations.
  // Merged annotation map (correltionId->annotation) cross per-thread data.
  // Empty per-thread callback annotations and events.
  void GatherAllCallbackAnnotationsAndEvents();

  // Clear all gathered callback events and annotations cross all threads.
  // Clear the merged annotation map.
  // Also empty per-thread callback annotations and events.
  void ClearAllAnnotatedEvents();

  // Right before profiling, setting options which impact per-thread callback
  // events collections.
  void PrepareOptionSettings();

  // Process cached activity buffer, add event into collector.
  absl::Status ConvertActivityBuffer(uint8_t* buffer, size_t size);

  void FinalizeActivityBuffers();

  void FinalizeApiCallbackBuffers();

  static thread_local CallbackAnnotationsEventsWeakPtr
      callback_annotations_and_events_;

  // collected together at the end of profiling from all threads.
  std::list<std::shared_ptr<CallbackAnnotationsAndEvents>>
      collected_annotation_and_events_;

  // merged correlation_id to annotation from raw collected annotations above
  AnnotationMap merged_annotation_map_;

  struct ActivityBufferAndSize {
    std::shared_ptr<uint8_t> buffer;
    size_t size;
    ActivityBufferAndSize(uint8_t* p = nullptr, size_t sz = 0);
  };

  // Mutex maybe not needed, need to check cupti implementations.
  // Yet it is of low overhead.
  absl::Mutex activity_buffers_mutex_;
  std::list<ActivityBufferAndSize> activity_buffers_
      ABSL_GUARDED_BY(activity_buffers_mutex_);
  size_t estimated_num_dropped_activity_events_ = 0;
  std::atomic<size_t> estimated_num_activity_events_ = 0;
  std::atomic<size_t> cupti_dropped_activity_event_count_ = 0;

  std::atomic<size_t> num_callback_events_ = 0;
  std::atomic<size_t> dropped_callback_event_count_ = 0;

  // Buffer size and alignment, 32K and 8 as in CUPTI samples.
  static constexpr size_t kBufferSizeInBytes = 32 * 1024;

  absl::Status EnableApiTracing();
  absl::Status EnableActivityTracing();
  absl::Status DisableApiTracing();
  absl::Status DisableActivityTracing();
  absl::Status Finalize();
  void ConfigureActivityUnifiedMemoryCounter(bool enable);
  absl::Status HandleNVTXCallback(CUpti_CallbackId cbid,
                                  const CUpti_CallbackData* cbdata);

  int num_gpus_;
  std::optional<CuptiTracerOptions> option_;
  CuptiInterface* cupti_interface_ = nullptr;
  CuptiTraceCollector* collector_ = nullptr;

  // CUPTI 10.1 and higher need root access to profile.
  bool need_root_access_ = false;

  bool api_tracing_enabled_ = false;
  // Cupti handle for driver or runtime API callbacks. Cupti permits a single
  // subscriber to be active at any time and can be used to trace Cuda runtime
  // as and driver calls for all contexts and devices.
  CUpti_SubscriberHandle subscriber_;  // valid when api_tracing_enabled_.

  bool activity_tracing_enabled_ = false;

  std::unique_ptr<CuptiDriverApiHook> cupti_driver_api_hook_;

  tsl::profiler::BufferPool buffer_pool_;
};

}  // namespace profiler
}  // namespace xla

#endif  // XLA_BACKENDS_PROFILER_GPU_CUPTI_TRACER_H_
