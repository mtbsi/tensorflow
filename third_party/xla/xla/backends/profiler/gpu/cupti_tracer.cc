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

#include "xla/backends/profiler/gpu/cupti_tracer.h"

#include "absl/base/call_once.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/container/node_hash_set.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/cupti_activity.h"
#include "third_party/gpus/cuda/extras/CUPTI/include/generated_nvtx_meta.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/backends/profiler/gpu/cupti_collector.h"
#include "xla/backends/profiler/gpu/nvtx_utils.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/host_info.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/mem.h"
#include "tsl/profiler/backends/cpu/annotation_stack.h"

namespace xla {
namespace profiler {

namespace {

using absl::OkStatus;
using absl::Status;
using tsl::Env;
using tsl::profiler::AnnotationStack;

// CUPTI from CUDA 11.6 adds information about the hardware channel that ops
// run on; this makes its way into the channel_id and channel_type fields in the
// structs we export.
//
// Define some type aliases so we can access the hardware channel id if it's
// available.
#if CUDA_VERSION >= 12000  // CUDA 12.0
#define TF_CUPTI_HAS_CHANNEL_ID 1
using CuptiActivityKernelTy = CUpti_ActivityKernel9;
using CuptiActivityMemcpyTy = CUpti_ActivityMemcpy5;
using CuptiActivityMemcpyP2PTy = CUpti_ActivityMemcpyPtoP4;
using CuptiActivityMemsetTy = CUpti_ActivityMemset4;
#elif CUDA_VERSION >= 11060  // CUDA 11.6
#define TF_CUPTI_HAS_CHANNEL_ID 1
using CuptiActivityKernelTy = CUpti_ActivityKernel7;
using CuptiActivityMemcpyTy = CUpti_ActivityMemcpy5;
using CuptiActivityMemcpyP2PTy = CUpti_ActivityMemcpyPtoP4;
using CuptiActivityMemsetTy = CUpti_ActivityMemset4;
#else
using CuptiActivityKernelTy = CUpti_ActivityKernel4;
using CuptiActivityMemcpyTy = CUpti_ActivityMemcpy;
using CuptiActivityMemcpyP2PTy = CUpti_ActivityMemcpy2;
using CuptiActivityMemsetTy = CUpti_ActivityMemset;
#endif

static thread_local int internalCuCall = 0;

// Temporary disable cupti api tracing for this thread during the life scope of
// this class. Used for the API calls that initiated by us.
class CuptiApiTracingDisabler {
 public:
  CuptiApiTracingDisabler() { internalCuCall++; }
  ~CuptiApiTracingDisabler() { internalCuCall--; }
};

Status ToStatus(CUptiResult result) {
  if (result == CUPTI_SUCCESS) {
    return OkStatus();
  }
  const char *str = nullptr;
  cuptiGetResultString(result, &str);
  return tsl::errors::Unavailable("CUPTI error: ", str ? str : "<unknown>");
}

Status ToStatus(CUresult result) {
  if (result == CUDA_SUCCESS) {
    return OkStatus();
  }
  const char *str = nullptr;
  cuGetErrorName(result, &str);
  return tsl::errors::Unavailable("CUDA error: ", str ? str : "<unknown>");
}

inline void LogIfError(const Status &status) {
  if (status.ok()) return;
  LOG(ERROR) << status.message();
}

// Maps an OverheadKind enum to a const string.
const char *getActivityOverheadKindString(CUpti_ActivityOverheadKind kind) {
  switch (kind) {
    case CUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER:
      return "COMPILER";
    case CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH:
      return "BUFFER_FLUSH";
    case CUPTI_ACTIVITY_OVERHEAD_CUPTI_INSTRUMENTATION:
      return "INSTRUMENTATION";
    case CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE:
      return "RESOURCE";
    default:
      break;
  }
  return "<UNKNOWN>";
}

const char *getActivityUnifiedMemoryKindString(
    CUpti_ActivityUnifiedMemoryCounterKind kind) {
  switch (kind) {
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD:
      return "UM_BYTES_TRANSFER_HTOD";
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH:
      return "UM_BYTES_TRANSFER_DTOH";
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT:
      return "UM_CPU_PAGE_FAULT";
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_GPU_PAGE_FAULT:
      return "UM_GPU_PAGE_FAULT";
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING:
      return "UM_THRASHING";
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THROTTLING:
      return "UM_THROTTLING";
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_REMOTE_MAP:
      return "UM_REMOTE_MAP";
    case CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOD:
      return "UM_BYTES_TRANSFER_DTOD";
    default:
      break;
  }
  return "<UNKNOWN>";
}

// CUPTI_ERROR_INSUFFICIENT_PRIVILEGES is introduced at CUDA 10.1.
#if CUDA_VERSION <= 10000
#define CUPTI_ERROR_INSUFFICIENT_PRIVILEGES 35
#endif

#define RETURN_IF_CUPTI_ERROR(expr)                                         \
  do {                                                                      \
    CUptiResult status = expr;                                              \
    if (ABSL_PREDICT_FALSE(status != CUPTI_SUCCESS)) {                      \
      const char *errstr = "";                                              \
      cupti_interface_->GetResultString(status, &errstr);                   \
      LOG(ERROR) << "function " << #expr << "failed with error " << errstr; \
      if (status == CUPTI_ERROR_INSUFFICIENT_PRIVILEGES) {                  \
        return tsl::errors::PermissionDenied("CUPTI need root access!");    \
      } else {                                                              \
        return tsl::errors::Internal("CUPTI call error", errstr);           \
      }                                                                     \
    }                                                                       \
  } while (false)

size_t Bytes2D(const CUDA_MEMCPY2D *p) { return p->Height * p->WidthInBytes; }

size_t Bytes3D(const CUDA_MEMCPY3D *p) {
  return p->Depth * p->Height * p->WidthInBytes;
}

template <typename CudaMemcpy>
CuptiTracerEventType MemcpyKind(const CudaMemcpy *p) {
  if (p->srcMemoryType == CU_MEMORYTYPE_HOST &&
      p->dstMemoryType == CU_MEMORYTYPE_DEVICE) {
    return CuptiTracerEventType::MemcpyH2D;
  }
  if (p->srcMemoryType == CU_MEMORYTYPE_DEVICE &&
      p->dstMemoryType == CU_MEMORYTYPE_HOST) {
    return CuptiTracerEventType::MemcpyD2H;
  }
  if (p->srcMemoryType == CU_MEMORYTYPE_DEVICE &&
      p->dstMemoryType == CU_MEMORYTYPE_DEVICE) {
    return CuptiTracerEventType::MemcpyD2D;
  }
  return CuptiTracerEventType::Unsupported;
}

std::tuple<size_t /*bytes*/, CuptiTracerEventType, bool /*async*/>
DecodeDriverMemcpy(CUpti_CallbackId cbid, const void *params) {
  switch (cbid) {
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2: {
      const auto *p = reinterpret_cast<const cuMemcpyHtoD_v2_params *>(params);
      return std::make_tuple(p->ByteCount, CuptiTracerEventType::MemcpyH2D,
                             false);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2: {
      const auto *p =
          reinterpret_cast<const cuMemcpyHtoDAsync_v2_params *>(params);
      return std::make_tuple(p->ByteCount, CuptiTracerEventType::MemcpyH2D,
                             true);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2: {
      const auto *p = reinterpret_cast<const cuMemcpyDtoH_v2_params *>(params);
      return std::make_tuple(p->ByteCount, CuptiTracerEventType::MemcpyD2H,
                             false);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2: {
      const auto *p =
          reinterpret_cast<const cuMemcpyDtoHAsync_v2_params *>(params);
      return std::make_tuple(p->ByteCount, CuptiTracerEventType::MemcpyD2H,
                             true);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2: {
      const auto *p = reinterpret_cast<const cuMemcpyDtoD_v2_params *>(params);
      return std::make_tuple(p->ByteCount, CuptiTracerEventType::MemcpyD2D,
                             false);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2: {
      const auto *p =
          reinterpret_cast<const cuMemcpyDtoDAsync_v2_params *>(params);
      return std::make_tuple(p->ByteCount, CuptiTracerEventType::MemcpyD2D,
                             true);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpy: {
      const auto *p = reinterpret_cast<const cuMemcpy_params *>(params);
      return std::make_tuple(p->ByteCount, CuptiTracerEventType::MemcpyOther,
                             false);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync: {
      const auto *p = reinterpret_cast<const cuMemcpyAsync_params *>(params);
      return std::make_tuple(p->ByteCount, CuptiTracerEventType::MemcpyOther,
                             true);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpy2D_v2: {
      const auto *p = reinterpret_cast<const cuMemcpy2D_v2_params *>(params);
      return std::make_tuple(Bytes2D(p->pCopy), MemcpyKind(p->pCopy), false);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DAsync_v2: {
      const auto *p =
          reinterpret_cast<const cuMemcpy2DAsync_v2_params *>(params);
      return std::make_tuple(Bytes2D(p->pCopy), MemcpyKind(p->pCopy), true);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpy3D_v2: {
      const auto *p = reinterpret_cast<const cuMemcpy3D_v2_params *>(params);
      return std::make_tuple(Bytes3D(p->pCopy), MemcpyKind(p->pCopy), true);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DAsync_v2: {
      const auto *p =
          reinterpret_cast<const cuMemcpy3DAsync_v2_params *>(params);
      return std::make_tuple(Bytes3D(p->pCopy), MemcpyKind(p->pCopy), true);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyPeer: {
      const auto *p2p_params =
          reinterpret_cast<const cuMemcpyPeer_params *>(params);
      return std::make_tuple(p2p_params->ByteCount,
                             CuptiTracerEventType::MemcpyP2P, false);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyPeerAsync: {
      const auto *p2p_params =
          reinterpret_cast<const cuMemcpyPeerAsync_params *>(params);
      return std::make_tuple(p2p_params->ByteCount,
                             CuptiTracerEventType::MemcpyP2P, true);
    }
    default: {
      LOG(ERROR) << "Unsupported memcpy activity observed: " << cbid;
      return std::make_tuple(0, CuptiTracerEventType::Unsupported, false);
    }
  }
}

std::tuple<size_t /*bytes*/, CuptiTracerEventType, bool /*async*/>
DecodeDriverMemset(CUpti_CallbackId cbid, const void *params) {
  switch (cbid) {
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD8_v2: {
      const auto *p = reinterpret_cast<const cuMemsetD8_v2_params *>(params);
      return std::make_tuple(p->N, CuptiTracerEventType::Memset, false);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD16_v2: {
      const auto *p = reinterpret_cast<const cuMemsetD16_v2_params *>(params);
      return std::make_tuple(p->N, CuptiTracerEventType::Memset, false);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD32_v2: {
      const auto *p = reinterpret_cast<const cuMemsetD32_v2_params *>(params);
      return std::make_tuple(p->N, CuptiTracerEventType::Memset, false);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D8_v2: {
      const auto *p = reinterpret_cast<const cuMemsetD2D8_v2_params *>(params);
      return std::make_tuple(p->dstPitch * p->Height,
                             CuptiTracerEventType::Memset, false);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D16_v2: {
      const auto *p = reinterpret_cast<const cuMemsetD2D16_v2_params *>(params);
      return std::make_tuple(p->dstPitch * p->Height,
                             CuptiTracerEventType::Memset, false);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D32_v2: {
      const auto *p = reinterpret_cast<const cuMemsetD2D32_v2_params *>(params);
      return std::make_tuple(p->dstPitch * p->Height,
                             CuptiTracerEventType::Memset, false);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD8Async: {
      const auto *p = reinterpret_cast<const cuMemsetD8Async_params *>(params);
      return std::make_tuple(p->N, CuptiTracerEventType::Memset, true);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD16Async: {
      const auto *p = reinterpret_cast<const cuMemsetD16Async_params *>(params);
      return std::make_tuple(p->N, CuptiTracerEventType::Memset, true);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD32Async: {
      const auto *p = reinterpret_cast<const cuMemsetD32Async_params *>(params);
      return std::make_tuple(p->N, CuptiTracerEventType::Memset, true);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D8Async: {
      const auto *p =
          reinterpret_cast<const cuMemsetD2D8Async_params *>(params);
      return std::make_tuple(p->dstPitch * p->Height,
                             CuptiTracerEventType::Memset, true);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D16Async: {
      const auto *p =
          reinterpret_cast<const cuMemsetD2D16Async_params *>(params);
      return std::make_tuple(p->dstPitch * p->Height,
                             CuptiTracerEventType::Memset, true);
    }
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D32Async: {
      const auto *p =
          reinterpret_cast<const cuMemsetD2D32Async_params *>(params);
      return std::make_tuple(p->dstPitch * p->Height,
                             CuptiTracerEventType::Memset, true);
    }
    default: {
      LOG(ERROR) << "Unsupported memset activity observed: " << cbid;
      return std::make_tuple(0, CuptiTracerEventType::Unsupported, false);
    }
  }
}

// Cupti callback corresponding to a driver or runtime API. This global function
// is invoked twice for each API: at entry and at exit. The cbdata
// parameter is guaranteed by Cupti to be thread-safe. Most invocations are
// dropped to the floor and entry/exit is tracked for the APIs we deem
// performance-relevant.
void CUPTIAPI ApiCallback(void *user_data, CUpti_CallbackDomain domain,
                          CUpti_CallbackId cbid,
                          const CUpti_CallbackData *cbdata) {
  CuptiTracer *tracer = reinterpret_cast<CuptiTracer *>(user_data);
  tracer->HandleCallback(domain, cbid, cbdata).IgnoreError();
}

// Callback which is invoked when an empty buffer is requested by CUPTI.
// Allocates an empty aligned-memory buffer. The buffer is used by CUPTI as a
// ring buffer where device maintains activity profiles that have been
// collected.
void CUPTIAPI RequestCuptiActivityBuffer(uint8_t **buffer, size_t *size,
                                         size_t *maxNumRecords) {
  CuptiTracer::GetCuptiTracerSingleton()->RequestActivityBuffer(buffer, size);
  VLOG(3) << "Requested CUPTI Buffer, buffer=" << std::hex
          << reinterpret_cast<uintptr_t>(*buffer) << std::dec
          << " size=" << *size;
  // Request CUPTI to fill as many records as possible in the buffer.
  *maxNumRecords = 0;
}

// Callback which is invoked when a buffer containing activity records is
// available from CUPTI. Processes the buffer after reading activity records
// from it.
void CUPTIAPI ProcessCuptiActivityBuffer(CUcontext context, uint32_t stream_id,
                                         uint8_t *buffer, size_t size,
                                         size_t valid_size) {
  VLOG(3) << "Processing CUPTI Buffer, buffer:" << std::hex
          << reinterpret_cast<uintptr_t>(buffer) << std::dec
          << " size: " << size << " valid_size: " << valid_size;
  VLOG(3) << "Activity profile for stream " << stream_id;

  Status status = CuptiTracer::GetCuptiTracerSingleton()->ProcessActivityBuffer(
      context, stream_id, buffer, valid_size);
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
}

void AddKernelEventUponApiExit(CuptiTracer *tracer, uint32_t device_id,
                               const CUpti_CallbackData *cbdata,
                               uint64_t start_time, uint64_t end_time) {
  CuptiTracerEvent *event_ptr = tracer->LastCallbackEvent();
  if (event_ptr == nullptr) return;
  CuptiTracerEvent &event = *event_ptr;
  event.type = CuptiTracerEventType::Kernel;
  event.source = CuptiTracerEventSource::DriverCallback;
  event.name = cbdata->symbolName ? cbdata->symbolName : cbdata->functionName;
  event.start_time_ns = start_time;
  event.end_time_ns = end_time;
  event.thread_id = Env::Default()->GetCurrentThreadId();
  event.device_id = device_id;
  event.context_id = cbdata->contextUid;
  event.correlation_id = cbdata->correlationId;
  VLOG(3) << "Cuda Kernel launch API exit. name=" << event.name;
}

// Performs the actual callback for both normal and P2P memcpy operations.
CuptiTracerEvent PopulateMemcpyCallbackEvent(
    CuptiTracerEventType type, const CUpti_CallbackData *cbdata,
    size_t num_bytes, uint32_t src_device, uint32_t dst_device, bool async,
    uint64_t start_time, uint64_t end_time) {
  CuptiTracerEvent event{};
  event.type = type;
  event.source = CuptiTracerEventSource::DriverCallback;
  event.start_time_ns = start_time;
  event.end_time_ns = end_time;
  event.thread_id = Env::Default()->GetCurrentThreadId();
  event.device_id = src_device;
  event.context_id = cbdata->contextUid;
  event.correlation_id = cbdata->correlationId;
  event.memcpy_info.num_bytes = num_bytes;
  event.memcpy_info.destination = dst_device;
  event.memcpy_info.async = async;
  // These are not populated during callback for API activities.
  event.memcpy_info.copy_kind = CUPTI_ACTIVITY_MEMCPY_KIND_UNKNOWN;
  event.memcpy_info.dst_mem_kind = CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN;
  event.memcpy_info.src_mem_kind = CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN;
  return event;
}

void AddNormalMemcpyEventUponApiExit(CuptiTracer *tracer, uint32_t device_id,
                                     CUpti_CallbackId cbid,
                                     const CUpti_CallbackData *cbdata,
                                     uint64_t start_time, uint64_t end_time) {
  CuptiTracerEvent *event_ptr = tracer->LastCallbackEvent();
  if (event_ptr == nullptr) return;
  size_t num_bytes;
  CuptiTracerEventType type;
  bool async;
  std::tie(num_bytes, type, async) =
      DecodeDriverMemcpy(cbid, cbdata->functionParams);

  VLOG(3) << "Cuda Memcpy API exit. sz=" << num_bytes;
  CuptiTracerEvent event =
      PopulateMemcpyCallbackEvent(type, cbdata, num_bytes, device_id, device_id,
                                  async, start_time, end_time);
  *event_ptr = std::move(event);
}

void AddCuMemsetEventUponApiExit(CuptiTracer *tracer, uint32_t device_id,
                                 CUpti_CallbackId cbid,
                                 const CUpti_CallbackData *cbdata,
                                 uint64_t start_time, uint64_t end_time) {
  CuptiTracerEvent *event_ptr = tracer->LastCallbackEvent();
  if (event_ptr == nullptr) return;
  CuptiTracerEvent &event = *event_ptr;
  // We are casting all variants of cuMemset to cuMemsetD8 for accessing the
  // first member attribute, a CUdeviceptr.
  const auto *params =
      static_cast<const cuMemsetD8_v2_params *>(cbdata->functionParams);
  size_t num_bytes;
  bool async;
  CuptiTracerEventType type;
  std::tie(num_bytes, type, async) =
      DecodeDriverMemset(cbid, cbdata->functionParams);

  event.type = type;
  event.source = CuptiTracerEventSource::DriverCallback;
  event.start_time_ns = start_time;
  event.end_time_ns = end_time;
  event.thread_id = Env::Default()->GetCurrentThreadId();
  event.device_id = device_id;
  event.context_id = cbdata->contextUid;
  event.correlation_id = cbdata->correlationId;
  event.memset_info.num_bytes = num_bytes;
  // memset_info.kind cannot be determined from API.
  event.memset_info.async = async;
  VLOG(3) << "Cuda Memset API exit."
          << " dptr=" << reinterpret_cast<void *>(params->dstDevice)
          << " sz=" << num_bytes;
}

void AddP2PMemcpyEventUponApiExit(CuptiTracer *tracer,
                                  CuptiInterface *cupti_interface,
                                  uint32_t device_id, CUpti_CallbackId cbid,
                                  const CUpti_CallbackData *cbdata,
                                  uint64_t start_time, uint64_t end_time) {
  CuptiTracerEvent *event_ptr = tracer->LastCallbackEvent();
  if (event_ptr == nullptr) return;
  size_t num_bytes;
  CuptiTracerEventType type;
  bool async;
  std::tie(num_bytes, type, async) =
      DecodeDriverMemcpy(cbid, cbdata->functionParams);

  uint32_t dst_device = -1, src_device = -1;
  const auto *p2p_params =
      static_cast<const cuMemcpyPeer_params *>(cbdata->functionParams);
  cupti_interface->GetDeviceId(p2p_params->srcContext, &src_device);
  cupti_interface->GetDeviceId(p2p_params->dstContext, &dst_device);
  VLOG(3) << "Cuda P2P Memcpy API exit, src: " << src_device
          << " dst: " << dst_device << " size:" << num_bytes;
  CuptiTracerEvent event =
      PopulateMemcpyCallbackEvent(type, cbdata, num_bytes, src_device,
                                  dst_device, async, start_time, end_time);
  *event_ptr = std::move(event);
}

void AddCuMemAllocEventUponApiExit(CuptiTracer *tracer, uint32_t device_id,
                                   CUpti_CallbackId cbid,
                                   const CUpti_CallbackData *cbdata,
                                   uint64_t start_time, uint64_t end_time) {
  CuptiTracerEvent *event_ptr = tracer->LastCallbackEvent();
  if (event_ptr == nullptr) return;
  CuptiTracerEvent &event = *event_ptr;
  const auto *params =
      static_cast<const cuMemAlloc_v2_params *>(cbdata->functionParams);
  const void *dptr = reinterpret_cast<void *>(*params->dptr);
  event.type = CuptiTracerEventType::MemoryAlloc;
  event.source = CuptiTracerEventSource::DriverCallback;
  event.name = cbdata->functionName;
  event.start_time_ns = start_time;
  event.end_time_ns = end_time;
  event.thread_id = Env::Default()->GetCurrentThreadId();
  event.device_id = device_id;
  event.context_id = cbdata->contextUid;
  event.correlation_id = cbdata->correlationId;
  event.memalloc_info.address = reinterpret_cast<uintptr_t>(dptr);
  event.memalloc_info.num_bytes = params->bytesize;
  VLOG(3) << "Cuda MemAlloc API exit." << " dptr=" << dptr
          << " sz=" << params->bytesize;
}

void AddCuMemAllocPitchEventUponApiExit(CuptiTracer *tracer, uint32_t device_id,
                                        CUpti_CallbackId cbid,
                                        const CUpti_CallbackData *cbdata,
                                        uint64_t start_time,
                                        uint64_t end_time) {
  CuptiTracerEvent *event_ptr = tracer->LastCallbackEvent();
  if (event_ptr == nullptr) return;
  CuptiTracerEvent &event = *event_ptr;
  const auto *params =
      static_cast<const cuMemAllocPitch_v2_params *>(cbdata->functionParams);
  const void *dptr = reinterpret_cast<void *>(*params->dptr);
  event.type = CuptiTracerEventType::MemoryAlloc;
  event.source = CuptiTracerEventSource::DriverCallback;
  event.name = cbdata->functionName;
  event.start_time_ns = start_time;
  event.end_time_ns = end_time;
  event.thread_id = Env::Default()->GetCurrentThreadId();
  event.device_id = device_id;
  event.context_id = cbdata->contextUid;
  event.correlation_id = cbdata->correlationId;
  const size_t size_in_bytes = *params->pPitch * params->Height;
  event.memalloc_info.address = reinterpret_cast<uintptr_t>(dptr);
  event.memalloc_info.num_bytes = size_in_bytes;
  VLOG(3) << "Cuda MemAllocPitch API exit." << " dptr=" << dptr
          << " sz=" << size_in_bytes;
}

void AddCuMemAllocManagedEventUponApiExit(
    CuptiTracer *tracer, uint32_t device_id, CUpti_CallbackId cbid,
    const CUpti_CallbackData *cbdata, uint64_t start_time, uint64_t end_time) {
  CuptiTracerEvent *event_ptr = tracer->LastCallbackEvent();
  if (event_ptr == nullptr) return;
  CuptiTracerEvent &event = *event_ptr;
  const auto *params =
      static_cast<const cuMemAllocManaged_params *>(cbdata->functionParams);
  const void *dptr = reinterpret_cast<void *>(*params->dptr);
  event.type = CuptiTracerEventType::MemoryAlloc;
  event.source = CuptiTracerEventSource::DriverCallback;
  event.name = cbdata->functionName;
  event.start_time_ns = start_time;
  event.end_time_ns = end_time;
  event.thread_id = Env::Default()->GetCurrentThreadId();
  event.device_id = device_id;
  event.context_id = cbdata->contextUid;
  event.correlation_id = cbdata->correlationId;
  event.memalloc_info.address = reinterpret_cast<uintptr_t>(dptr);
  event.memalloc_info.num_bytes = params->bytesize;
  VLOG(3) << "Cuda MemAllocManaged API exit." << " dptr=" << dptr
          << " sz=" << params->bytesize;
}

void AddCuMemAllocHostEventUponApiExit(CuptiTracer *tracer, uint32_t device_id,
                                       CUpti_CallbackId cbid,
                                       const CUpti_CallbackData *cbdata,
                                       uint64_t start_time, uint64_t end_time) {
  CuptiTracerEvent *event_ptr = tracer->LastCallbackEvent();
  if (event_ptr == nullptr) return;
  CuptiTracerEvent &event = *event_ptr;
  const auto *params =
      static_cast<const cuMemAllocHost_v2_params *>(cbdata->functionParams);
  event.type = CuptiTracerEventType::MemoryAlloc;
  event.source = CuptiTracerEventSource::DriverCallback;
  event.name = cbdata->functionName;
  event.start_time_ns = start_time;
  event.end_time_ns = end_time;
  event.thread_id = Env::Default()->GetCurrentThreadId();
  event.device_id = device_id;
  event.context_id = cbdata->contextUid;
  event.correlation_id = cbdata->correlationId;
  event.memalloc_info.address = reinterpret_cast<uintptr_t>(*params->pp);
  event.memalloc_info.num_bytes = params->bytesize;
  VLOG(3) << "Cuda MemAllocHost API exit." << " pp=" << *params->pp
          << " sz=" << params->bytesize;
}

void AddCuMemHostAllocEventUponApiExit(CuptiTracer *tracer, uint32_t device_id,
                                       CUpti_CallbackId cbid,
                                       const CUpti_CallbackData *cbdata,
                                       uint64_t start_time, uint64_t end_time) {
  CuptiTracerEvent *event_ptr = tracer->LastCallbackEvent();
  if (event_ptr == nullptr) return;
  CuptiTracerEvent &event = *event_ptr;
  const auto *params =
      static_cast<const cuMemHostAlloc_params *>(cbdata->functionParams);
  event.type = CuptiTracerEventType::MemoryAlloc;
  event.source = CuptiTracerEventSource::DriverCallback;
  event.name = cbdata->functionName;
  event.start_time_ns = start_time;
  event.end_time_ns = end_time;
  event.thread_id = Env::Default()->GetCurrentThreadId();
  event.device_id = device_id;
  event.context_id = cbdata->contextUid;
  event.correlation_id = cbdata->correlationId;
  event.memalloc_info.address = reinterpret_cast<uintptr_t>(*params->pp);
  event.memalloc_info.num_bytes = params->bytesize;
  VLOG(3) << "Cuda MemHostAlloc API exit." << " pp=" << *params->pp
          << " sz=" << params->bytesize << " Flags=" << params->Flags;
}

void AddCuMemFreeEventUponApiExit(CuptiTracer *tracer, uint32_t device_id,
                                  CUpti_CallbackId cbid,
                                  const CUpti_CallbackData *cbdata,
                                  uint64_t start_time, uint64_t end_time) {
  CuptiTracerEvent *event_ptr = tracer->LastCallbackEvent();
  if (event_ptr == nullptr) return;
  CuptiTracerEvent &event = *event_ptr;
  const auto *params =
      static_cast<const cuMemFree_v2_params *>(cbdata->functionParams);
  const void *dptr = reinterpret_cast<void *>(params->dptr);
  event.type = CuptiTracerEventType::MemoryFree;
  event.source = CuptiTracerEventSource::DriverCallback;
  event.name = cbdata->functionName;
  event.start_time_ns = start_time;
  event.end_time_ns = end_time;
  event.thread_id = Env::Default()->GetCurrentThreadId();
  event.device_id = device_id;
  event.context_id = cbdata->contextUid;
  event.correlation_id = cbdata->correlationId;
  event.memfree_info.address = reinterpret_cast<uintptr_t>(dptr);
  VLOG(3) << "Cuda MemFree API exit." << " dptr=" << dptr;
}

void AddCuMemFreeHostEventUponApiExit(CuptiTracer *tracer, uint32_t device_id,
                                      CUpti_CallbackId cbid,
                                      const CUpti_CallbackData *cbdata,
                                      uint64_t start_time, uint64_t end_time) {
  CuptiTracerEvent *event_ptr = tracer->LastCallbackEvent();
  if (event_ptr == nullptr) return;
  CuptiTracerEvent &event = *event_ptr;
  const auto *params =
      static_cast<const cuMemFreeHost_params *>(cbdata->functionParams);
  event.type = CuptiTracerEventType::MemoryFree;
  event.source = CuptiTracerEventSource::DriverCallback;
  event.name = cbdata->functionName;
  event.start_time_ns = start_time;
  event.end_time_ns = end_time;
  event.thread_id = Env::Default()->GetCurrentThreadId();
  event.device_id = device_id;
  event.context_id = cbdata->contextUid;
  event.correlation_id = cbdata->correlationId;
  event.memfree_info.address = reinterpret_cast<uintptr_t>(params->p);
  VLOG(3) << "Cuda MemFreeHost API exit." << " p=" << params->p;
}

void AddCuMemHostRegisterEventUponApiExit(
    CuptiTracer *tracer, uint32_t device_id, CUpti_CallbackId cbid,
    const CUpti_CallbackData *cbdata, uint64_t start_time, uint64_t end_time) {
  CuptiTracerEvent *event_ptr = tracer->LastCallbackEvent();
  if (event_ptr == nullptr) return;
  CuptiTracerEvent &event = *event_ptr;
  const auto *params =
      static_cast<const cuMemHostRegister_v2_params *>(cbdata->functionParams);
  event.type = CuptiTracerEventType::HostRegister;
  event.source = CuptiTracerEventSource::DriverCallback;
  event.name = cbdata->functionName;
  event.start_time_ns = start_time;
  event.end_time_ns = end_time;
  event.thread_id = Env::Default()->GetCurrentThreadId();
  event.device_id = device_id;
  event.context_id = cbdata->contextUid;
  event.correlation_id = cbdata->correlationId;
  event.host_register_info.address = reinterpret_cast<uintptr_t>(params->p);
  event.host_register_info.num_bytes = params->bytesize;
  event.host_register_info.flags = params->Flags;
  VLOG(3) << "Cuda HostRegister API exit." << " p=" << params->p
          << " bytesize=" << params->bytesize << " flags=" << params->Flags;
}

void AddCuMemHostUnregisterEventUponApiExit(
    CuptiTracer *tracer, uint32_t device_id, CUpti_CallbackId cbid,
    const CUpti_CallbackData *cbdata, uint64_t start_time, uint64_t end_time) {
  CuptiTracerEvent *event_ptr = tracer->LastCallbackEvent();
  if (event_ptr == nullptr) return;
  CuptiTracerEvent &event = *event_ptr;
  const auto *params =
      static_cast<const cuMemHostUnregister_params *>(cbdata->functionParams);
  event.type = CuptiTracerEventType::HostUnregister;
  event.source = CuptiTracerEventSource::DriverCallback;
  event.name = cbdata->functionName;
  event.start_time_ns = start_time;
  event.end_time_ns = end_time;
  event.thread_id = Env::Default()->GetCurrentThreadId();
  event.device_id = device_id;
  event.context_id = cbdata->contextUid;
  event.correlation_id = cbdata->correlationId;
  event.host_unregister_info.address = reinterpret_cast<uintptr_t>(params->p);
  VLOG(3) << "Cuda HostUnregister API exit." << " p=" << params->p;
}

void AddGenericEventUponApiExit(CuptiTracer *tracer, uint32_t device_id,
                                CUpti_CallbackId cbid,
                                const CUpti_CallbackData *cbdata,
                                uint64_t start_time, uint64_t end_time) {
  CuptiTracerEvent *event_ptr = tracer->LastCallbackEvent();
  if (event_ptr == nullptr) return;
  CuptiTracerEvent &event = *event_ptr;
  event.type = CuptiTracerEventType::Generic;
  event.source = CuptiTracerEventSource::DriverCallback;
  event.name = cbdata->functionName;
  event.start_time_ns = start_time;
  event.end_time_ns = end_time;
  event.thread_id = Env::Default()->GetCurrentThreadId();
  event.device_id = device_id;
  event.context_id = cbdata->contextUid;
  event.correlation_id = cbdata->correlationId;
  VLOG(3) << "Observed generic API exit." << " name=" << cbdata->functionName;
}

template <bool cupti_has_channel_id, typename CuptiActivityKernel>
void AddKernelActivityEvent(CuptiTraceCollector *collector,
                            const CuptiActivityKernel *kernel) {
  CuptiTracerEvent event{};
  event.type = CuptiTracerEventType::Kernel;
  event.source = CuptiTracerEventSource::Activity;
  event.name = kernel->name;
  event.start_time_ns = kernel->start;
  event.end_time_ns = kernel->end;
  event.device_id = kernel->deviceId;
  event.context_id = kernel->contextId;
  event.stream_id = kernel->streamId;
  event.correlation_id = kernel->correlationId;
  AnnotationInfo info =
      collector->LookUpAnnotation(event.device_id, event.correlation_id);
  event.annotation = info.annotation;
  event.nvtx_range = info.nvtx_range;
  event.kernel_info.registers_per_thread = kernel->registersPerThread;
  event.kernel_info.static_shared_memory_usage = kernel->staticSharedMemory;
  event.kernel_info.dynamic_shared_memory_usage = kernel->dynamicSharedMemory;
  event.kernel_info.block_x = kernel->blockX;
  event.kernel_info.block_y = kernel->blockY;
  event.kernel_info.block_z = kernel->blockZ;
  event.kernel_info.grid_x = kernel->gridX;
  event.kernel_info.grid_y = kernel->gridY;
  event.kernel_info.grid_z = kernel->gridZ;
  if constexpr (cupti_has_channel_id) {
    event.kernel_info.channel_id = kernel->channelID;
    event.kernel_info.channel_type = kernel->channelType;
  }
  collector->AddEvent(std::move(event));
}

void AddMemcpyActivityEvent(CuptiTraceCollector *collector,
                            const CuptiActivityMemcpyTy *memcpy) {
  CuptiTracerEvent event{};
  switch (memcpy->copyKind) {
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
      event.type = CuptiTracerEventType::MemcpyH2D;
      event.name = "MemcpyH2D";
      break;
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
      event.type = CuptiTracerEventType::MemcpyD2H;
      event.name = "MemcpyD2H";
      break;
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
      event.type = CuptiTracerEventType::MemcpyD2D;
      event.name = "MemcpyD2D";
      break;
    case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP:
      event.type = CuptiTracerEventType::MemcpyP2P;
      event.name = "MemcpyP2P";
      break;
    default:
      event.type = CuptiTracerEventType::MemcpyOther;
      event.name = "MemcpyOther";
      break;
  }

  event.source = CuptiTracerEventSource::Activity;
  event.start_time_ns = memcpy->start;
  event.end_time_ns = memcpy->end;
  event.device_id = memcpy->deviceId;
  event.context_id = memcpy->contextId;
  event.stream_id = memcpy->streamId;
  event.correlation_id = memcpy->correlationId;
  AnnotationInfo info =
      collector->LookUpAnnotation(event.device_id, event.correlation_id);
  event.annotation = info.annotation;
  event.memcpy_info.copy_kind = memcpy->copyKind;
  event.memcpy_info.num_bytes = memcpy->bytes;
  event.memcpy_info.destination = memcpy->deviceId;
  event.memcpy_info.async = memcpy->flags & CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC;
  event.memcpy_info.src_mem_kind = memcpy->srcKind;
  event.memcpy_info.dst_mem_kind = memcpy->dstKind;
#if TF_CUPTI_HAS_CHANNEL_ID
  event.memcpy_info.channel_id = memcpy->channelID;
  event.memcpy_info.channel_type = memcpy->channelType;
#endif
  collector->AddEvent(std::move(event));
}

// Invokes callback upon peer-2-peer memcpy between different GPU devices.
void AddMemcpyP2PActivityEvent(CuptiTraceCollector *collector,
                               const CuptiActivityMemcpyP2PTy *memcpy) {
  CuptiTracerEvent event{};
  event.type = CuptiTracerEventType::MemcpyP2P;
  event.name = "MemcpyP2P";
  event.source = CuptiTracerEventSource::Activity;
  event.start_time_ns = memcpy->start;
  event.end_time_ns = memcpy->end;
  event.device_id = memcpy->srcDeviceId;
  event.context_id = memcpy->contextId;
  event.stream_id = memcpy->streamId;
  event.correlation_id = memcpy->correlationId;
  AnnotationInfo info =
      collector->LookUpAnnotation(event.device_id, event.correlation_id);
  event.annotation = info.annotation;
  event.memcpy_info.copy_kind = CUPTI_ACTIVITY_MEMCPY_KIND_PTOP;
  event.memcpy_info.num_bytes = memcpy->bytes;
  event.memcpy_info.destination = memcpy->dstDeviceId;
  event.memcpy_info.async = memcpy->flags & CUPTI_ACTIVITY_FLAG_MEMCPY_ASYNC;
  event.memcpy_info.src_mem_kind = memcpy->srcKind;
  event.memcpy_info.dst_mem_kind = memcpy->dstKind;
#if TF_CUPTI_HAS_CHANNEL_ID
  event.memcpy_info.channel_id = memcpy->channelID;
  event.memcpy_info.channel_type = memcpy->channelType;
#endif
  collector->AddEvent(std::move(event));
}

void AddCuptiOverheadActivityEvent(CuptiTraceCollector *collector,
                                   const CUpti_ActivityOverhead *overhead) {
  CuptiTracerEvent event{};
  event.type = CuptiTracerEventType::Overhead;
  event.name = getActivityOverheadKindString(overhead->overheadKind);
  event.source = CuptiTracerEventSource::Activity;
  event.start_time_ns = overhead->start;
  event.end_time_ns = overhead->end;
  // If the overhead is not related to a device, we assign it to device 0.
  event.device_id = 0;
  // NOTE: no correlation id.
  switch (overhead->objectKind) {
    case CUPTI_ACTIVITY_OBJECT_UNKNOWN:
      // Don't know how to deal with such activities because of we need either
      // attribute it to a GPU stream or a CPU thread.
      return;

    case CUPTI_ACTIVITY_OBJECT_THREAD:
    case CUPTI_ACTIVITY_OBJECT_PROCESS:
      event.thread_id = overhead->objectId.pt.threadId;
      break;
    case CUPTI_ACTIVITY_OBJECT_STREAM:
      event.stream_id = overhead->objectId.dcs.streamId;
      TF_FALLTHROUGH_INTENDED;
    case CUPTI_ACTIVITY_OBJECT_DEVICE:
    case CUPTI_ACTIVITY_OBJECT_CONTEXT:
      event.device_id = overhead->objectId.dcs.deviceId;
      break;
    default:
      LOG(ERROR) << "Unexpected object kind: " << overhead->objectKind;
      return;
  }
  collector->AddEvent(std::move(event));
}

void AddUnifiedMemoryActivityEvent(
    CuptiTraceCollector *collector,
    const CUpti_ActivityUnifiedMemoryCounter2 *record) {
  VLOG(3) << "Cuda Unified Memory Activity, kind: " << record->counterKind
          << " src: " << record->srcId << " dst: " << record->dstId;
  CuptiTracerEvent event{};
  event.type = CuptiTracerEventType::UnifiedMemory;
  event.name = getActivityUnifiedMemoryKindString(record->counterKind);
  event.source = CuptiTracerEventSource::Activity;
  event.start_time_ns = record->start;
  if (record->counterKind ==
          CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_CPU_PAGE_FAULT_COUNT ||
      record->counterKind ==
          CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_THRASHING ||
      record->counterKind ==
          CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_REMOTE_MAP ||
      record->end <= record->start) {
    // If the end time is not valid, trim it so that it can be shown on the UI.
    event.end_time_ns = record->start + 1;
  } else {
    event.end_time_ns = record->end;
  }
  event.device_id = record->srcId;
  // NOTE: not context id and correlation id.

  // For visualization purpose, we assign a pseudo stream id for each
  // record->counterKind of unified memory related events.
  constexpr int kPseudoStreamId = 0x10000000;
  event.stream_id = kPseudoStreamId + record->counterKind;
  event.memcpy_info.copy_kind = CUPTI_ACTIVITY_MEMCPY_KIND_UNKNOWN;
  // Check whether the activity is byte transfer.
  if (record->counterKind ==
          CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD ||
      record->counterKind ==
          CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH ||
      record->counterKind ==
          CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOD) {
    event.memcpy_info.num_bytes = record->value;
  } else {
    event.memcpy_info.num_bytes = 0;
  }
  event.memcpy_info.destination = record->dstId;
  event.memcpy_info.async = false;
  collector->AddEvent(std::move(event));
}

void AddMemoryActivityEvent(CuptiTraceCollector *collector,
                            const CUpti_ActivityMemory *memory) {
  CuptiTracerEvent event{};
  event.name = absl::StrCat("Memory ", GetMemoryKindName(memory->memoryKind));
  event.type = CuptiTracerEventType::MemoryResidency;
  event.source = CuptiTracerEventSource::Activity;
  event.start_time_ns = memory->start;
  event.end_time_ns = std::max(memory->end, memory->start + 1);
  event.device_id = memory->deviceId;
  event.context_id = memory->contextId;
  // Assign to default stream (0) so that event is included during Flush().
  event.stream_id = 0;
  event.memory_residency_info.num_bytes = memory->bytes;
  event.memory_residency_info.mem_kind = memory->memoryKind;
  event.memory_residency_info.address = memory->address;
  VLOG(5) << "Cuda activity " << event.name
          << " addr: " << reinterpret_cast<void *>(memory->address)
          << " bytes: " << memory->bytes;
  collector->AddEvent(std::move(event));
}

void AddMemsetActivityEvent(CuptiTraceCollector *collector,
                            const CuptiActivityMemsetTy *memset) {
  auto mem_kind = memset->memoryKind;
  CuptiTracerEvent event{};
  event.type = CuptiTracerEventType::Memset;
  event.source = CuptiTracerEventSource::Activity;
  event.name = absl::StrCat("Memset ", mem_kind);
  event.start_time_ns = memset->start;
  event.end_time_ns = std::max(memset->end, memset->start + 1);
  event.device_id = memset->deviceId;
  event.correlation_id = memset->correlationId;
  event.context_id = memset->contextId;
  event.stream_id = memset->streamId;
  event.memset_info.num_bytes = memset->bytes;
  event.memset_info.mem_kind = mem_kind;
  event.memset_info.async = (memset->flags & CUPTI_ACTIVITY_FLAG_MEMSET_ASYNC);
#if TF_CUPTI_HAS_CHANNEL_ID
  event.memset_info.channel_id = memset->channelID;
  event.memset_info.channel_type = memset->channelType;
#endif
  VLOG(5) << "Cuda activity " << event.name << " bytes: " << memset->bytes
          << " async: " << event.memset_info.async;
  collector->AddEvent(std::move(event));
}

void AddSynchronizationActivityEvent(
    CuptiTraceCollector *collector, const CUpti_ActivitySynchronization *sync) {
  CuptiTracerEvent event{};
  event.type = CuptiTracerEventType::Generic;
  event.source = CuptiTracerEventSource::Activity;
  switch (sync->type) {
    case CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_EVENT_SYNCHRONIZE:
      event.name = "cuEventSynchronize";
      break;
    case CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_WAIT_EVENT:
      event.name = "cuStreamWaitEvent";
      break;
    case CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_SYNCHRONIZE:
      event.name = "cuStreamSynchronize";
      break;
    case CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_CONTEXT_SYNCHRONIZE:
      event.name = "cuCtxSynchronize";
      break;
    default:
      event.name = "unknown synchronization event";
      break;
  }
  event.start_time_ns = sync->start;
  event.end_time_ns = std::max(sync->end, sync->start + 1);
  event.correlation_id = sync->correlationId;
  event.context_id = sync->contextId;
  VLOG(5) << "Cuda activity " << event.name;
  collector->AddEvent(std::move(event));
}

// This hook uses cupti activity api to measure device side activities.
class CuptiDriverApiHookWithActivityApi : public CuptiDriverApiHook {
 public:
  CuptiDriverApiHookWithActivityApi(const CuptiTracerOptions &option,
                                    CuptiInterface *cupti_interface,
                                    CuptiTracer *tracer)
      : option_(option), cupti_interface_(cupti_interface), tracer_(tracer) {}

  Status OnDriverApiEnter(int device_id, CUpti_CallbackDomain domain,
                          CUpti_CallbackId cbid,
                          const CUpti_CallbackData *cbdata) override {
    // Stash away the current Cupti timestamp into cbdata.
    *cbdata->correlationData =
        option_.required_callback_api_events ? CuptiTracer::GetTimestamp() : 0;
    return OkStatus();
  }
  Status OnDriverApiExit(int device_id, CUpti_CallbackDomain domain,
                         CUpti_CallbackId cbid,
                         const CUpti_CallbackData *cbdata) override {
    // If we are not collecting CPU events from Callback API, we can return now.
    if (!option_.required_callback_api_events) {
      return OkStatus();
    }

    // Grab timestamp for API exit. API entry timestamp saved in cbdata.
    uint64_t end_tsc = CuptiTracer::GetTimestamp();
    uint64_t start_tsc = *cbdata->correlationData;
    TrackContext(cbid, cbdata->context);
    return AddDriverApiCallbackEvent(tracer_, cupti_interface_, device_id,
                                     start_tsc, end_tsc, domain, cbid, cbdata);
  }
  Status SyncAndFlush() override {
    if (option_.sync_devices_before_stop) {
      CuptiApiTracingDisabler disabler;
      absl::MutexLock lock(&mutex_);
      for (auto &ctx : contexts_) {
        cuCtxPushCurrent(ctx);
        cuCtxSynchronize();  // Ignore error here for best effort.
        CUcontext current;
        cuCtxPopCurrent(&current);
      }
    }
    return OkStatus();
  }

 private:
  void TrackContext(CUpti_CallbackId cbid, CUcontext ctx) {
    if (!option_.sync_devices_before_stop) return;
    if (ctx == nullptr) return;
    absl::MutexLock lock(&mutex_);
    if (cbid == CUPTI_DRIVER_TRACE_CBID_cuCtxDestroy_v2 ||
        cbid == CUPTI_DRIVER_TRACE_CBID_cuCtxDestroy) {
      contexts_.erase(ctx);
    } else {
      contexts_.emplace(ctx);
    }
  }

  const CuptiTracerOptions option_;
  CuptiInterface *cupti_interface_;
  CuptiTracer *tracer_;
  absl::Mutex mutex_;
  absl::flat_hash_set<CUcontext> contexts_ TF_GUARDED_BY(mutex_);

  CuptiDriverApiHookWithActivityApi(const CuptiDriverApiHookWithActivityApi &) =
      delete;
  void operator=(const CuptiDriverApiHookWithActivityApi &) = delete;
};

/*static*/ std::string ErrorWithHostname(absl::string_view error_message) {
  return absl::StrCat(tsl::port::Hostname(), ": ", error_message);
}

}  // namespace

template <typename T>
class AppendOnlyBuffer {
 public:
  static constexpr size_t kBlockSize = 32768;
  typedef std::vector<T> Block;
  typedef std::list<Block> BlockList;

  explicit AppendOnlyBuffer(size_t block_size = kBlockSize)
      : block_size_(std::max((size_t)1024, block_size)) {
    Clear();
  }

  void Clear() {
    block_list_.clear();
    size_ = 0;
    last_block_ = &(block_list_.emplace_back());
    last_block_->reserve(block_size_);
  }

  BlockList &GetBlocks() { return block_list_; }

  void Append(const T &value) {
    if (last_block_->size() >= block_size_) {
      last_block_ = &(block_list_.emplace_back());
      last_block_->reserve(block_size_);
    }
    last_block_->push_back(value);
    size_++;
  }

  template <typename... Params>
  void Emplace(Params... params) {
    if (last_block_->size() >= block_size_) {
      last_block_ = &(block_list_.emplace_back());
      last_block_->reserve(block_size_);
    }
    last_block_->emplace_back(std::forward<Params>(params)...);
    size_++;
  }

  size_t Size() const { return size_; }
  T *LastElement() {
    return (size_ > 0) ? (&(last_block_->back())) : (nullptr);
  }

  AppendOnlyBuffer &operator=(AppendOnlyBuffer &&another) {
    block_size_ = another.block_size_;
    block_list_ = std::move(another.block_list_);
    last_block_ = &(block_list_.back());
    size_ = another.size_;
    another.Clear();
    return *this;
  }

 private:
  size_t block_size_ = kBlockSize;
  size_t size_ = 0;
  BlockList block_list_;
  Block *last_block_ = nullptr;

  AppendOnlyBuffer(AppendOnlyBuffer &&) = delete;
};

struct CallbackAnnotationsAndEvents {
  // If atomic counter still cause serious overhead, we need change
  // the max semantic to per thread level in the future.
  static std::atomic<size_t> s_callback_api_event_count;

  // Following need to be static no matter atomic counter is use or not.
  static size_t s_max_annotation_strings;
  static size_t s_max_callback_api_events;

  struct EventWithAnnotation {
    uint32_t correlation_id;
    absl::string_view annotation;
    absl::string_view nvtx_range;
    CuptiTracerEvent event;

    EventWithAnnotation() = default;
    EventWithAnnotation(uint32_t corr_id, absl::string_view ann,
                        absl::string_view nvtx)
        : correlation_id(corr_id), annotation(ann), nvtx_range(nvtx), event{} {}
  };

  // Annotation tends to be repetitive, use a hash_set to store the strings,
  // an use the reference to the string in the map.
  absl::node_hash_set<std::string> annotations;
  absl::node_hash_set<std::string> nvtx_ranges;
  AppendOnlyBuffer<EventWithAnnotation> event_annotation_buffer;
  size_t num_dropped_events = 0;

  CallbackAnnotationsAndEvents() = default;
  CallbackAnnotationsAndEvents(CallbackAnnotationsAndEvents &&another) {
    annotations = std::move(another.annotations);
    nvtx_ranges = std::move(another.nvtx_ranges);
    event_annotation_buffer = std::move(another.event_annotation_buffer);
    num_dropped_events = another.num_dropped_events;
    another.num_dropped_events = 0;
  }
  // Add an empty event with annotation and nvtx_range to the buffer.
  // return true if added, or false if the event is dropped
  bool Add(uint32_t device_id, uint32_t correlation_id,
           const absl::string_view annotation,
           const absl::string_view nvtx_range) {
    if (s_max_callback_api_events == 0 ||
        s_callback_api_event_count < s_max_callback_api_events) {
      s_callback_api_event_count++;
      // Some logic change as no cross thread string comparision should be
      // make here. The max_annotation_string is used to limit per-thread
      // annotation string count. And annotation string is not collected
      // if total callback event could overflow.
      bool too_many_annotations =
          (s_max_annotation_strings > 0) &&
          (annotations.size() >= s_max_annotation_strings);
      event_annotation_buffer.Emplace(
          correlation_id,
          (too_many_annotations || annotation.empty())
              ? absl::string_view()
              : *annotations.emplace(annotation).first,
          (too_many_annotations || nvtx_range.empty())
              ? absl::string_view()
              : *nvtx_ranges.emplace(nvtx_range).first);
    } else {
      num_dropped_events++;
      return false;
    }
    return true;
  }
  void clear() {
    annotations.clear();
    nvtx_ranges.clear();
    event_annotation_buffer.Clear();
    num_dropped_events = 0;
  }

 private:
  CallbackAnnotationsAndEvents(const CallbackAnnotationsAndEvents &) = delete;
  CallbackAnnotationsAndEvents &operator=(
      const CallbackAnnotationsAndEvents &) = delete;
  CallbackAnnotationsAndEvents &operator=(
      CallbackAnnotationsAndEvents &&another) = delete;
};

size_t CallbackAnnotationsAndEvents::s_max_annotation_strings = 1024 * 1024;
size_t CallbackAnnotationsAndEvents::s_max_callback_api_events =
    2 * 1024 * 1024;
std::atomic<size_t> CallbackAnnotationsAndEvents::s_callback_api_event_count =
    0;

// All active or in-active per-thread callback annotations and events
// buffers collected together. Due to the thread creating/destroying of
// the api callback events and annotations buffer is not under our control,
// this collection keep track of the per-thread data usage cross all
// related threads, and handle their life cycles.
class CallbackAnnotationsAndEventsCollection {
 public:
  static CallbackAnnotationsAndEventsCollection *Instance() {
    static absl::once_flag create_once;
    static CallbackAnnotationsAndEventsCollection *singleton = nullptr;
    absl::call_once(create_once, [&]() {
      singleton = new CallbackAnnotationsAndEventsCollection();
    });
    return singleton;
  }
  std::shared_ptr<CallbackAnnotationsAndEvents> CreateNew() {
    tsl::mutex_lock lock(mutex_);
    auto data = std::shared_ptr<CallbackAnnotationsAndEvents>(
        new CallbackAnnotationsAndEvents());
    active_set_.insert(data);
    return data;
  }

  // when thread_local is destroyed due to thread exit, this method
  // will be called to let this collection know the callback buffer
  // is not owning by an active thread.
  void Deactivate(std::shared_ptr<CallbackAnnotationsAndEvents> data) {
    tsl::mutex_lock lock(mutex_);
    if (data.get() != nullptr && active_set_.count(data)) {
      active_set_.erase(data);
      deactived_list_.emplace_back(data);
    }
  }

  // Thread local data could to be aggregated by this.
  // Yet it is caller's duty to avoid error from the parallel execution.
  // i.e., caller must be sure that there is no active thread will update
  // its data when calling this function.
  std::list<std::shared_ptr<CallbackAnnotationsAndEvents>> CollectAll(
      bool use_active = true, bool use_deactived = true) {
    std::list<std::shared_ptr<CallbackAnnotationsAndEvents>> result;
    tsl::mutex_lock lock(mutex_);
    if (use_active) {
      // Just move the data out, but keep the active data ptr valid
      // It use move constructor to swap the original buffer.
      for (auto t : active_set_) {
        result.emplace_back(new CallbackAnnotationsAndEvents(std::move(*t)));
      }
    }
    if (use_deactived) {
      while (!deactived_list_.empty()) {
        result.emplace_back(std::move(deactived_list_.front()));
        deactived_list_.pop_front();
      }
    }
    return result;
  }

 private:
  tsl::mutex mutex_;

  // data in active set is using by some active thread, so if this container
  // is destroyed first, it means child thread is not correctly joined,
  // data in active_set_ are not destroyed as only ptr stored in set. This may
  // report expected memory/resource leaks, yet it is better than possible
  // random crash in such case.
  absl::flat_hash_set<std::shared_ptr<CallbackAnnotationsAndEvents>>
      active_set_;
  std::list<std::shared_ptr<CallbackAnnotationsAndEvents>> deactived_list_;

  CallbackAnnotationsAndEventsCollection() = default;
  CallbackAnnotationsAndEventsCollection(
      const CallbackAnnotationsAndEventsCollection &) = delete;
  CallbackAnnotationsAndEventsCollection &operator=(
      const CallbackAnnotationsAndEventsCollection &) = delete;
};

// Perthread callback annotations and events buffer in shared_ptr.
// While the thread own its life cycle, the data also shared owner with
// CallbackAnnotationsAndEventsCollection singleton. So, when thread
// destroyed, it will also notify the Collection singleton.
class CallbackAnnotationsEventsWeakPtr {
 public:
  static CallbackAnnotationsAndEventsCollection *GeCollection() {
    return CallbackAnnotationsAndEventsCollection::Instance();
  }
  explicit CallbackAnnotationsEventsWeakPtr() {
    ptr_ = GeCollection()->CreateNew();
  }
  ~CallbackAnnotationsEventsWeakPtr() { GeCollection()->Deactivate(ptr_); }
  CallbackAnnotationsAndEvents *get() { return ptr_.get(); }
  CallbackAnnotationsAndEvents *operator->() { return ptr_.get(); }

 private:
  std::shared_ptr<CallbackAnnotationsAndEvents> ptr_;
  void operator=(const CallbackAnnotationsEventsWeakPtr &) = delete;
  CallbackAnnotationsEventsWeakPtr(const CallbackAnnotationsEventsWeakPtr &) =
      delete;
};

/*static*/ Status CuptiDriverApiHook::AddDriverApiCallbackEvent(
    CuptiTracer *tracer, CuptiInterface *cupti_interface, int device_id,
    uint64_t start_tsc, uint64_t end_tsc, CUpti_CallbackDomain domain,
    CUpti_CallbackId cbid, const CUpti_CallbackData *cbdata) {
  switch (cbid) {
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel:
#if CUDA_VERSION >= 11080  // CUDA 11.8
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchKernelEx:
#endif  // CUDA_VERSION >= 11080
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernel:
    case CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernelMultiDevice:
      AddKernelEventUponApiExit(tracer, device_id, cbdata, start_tsc, end_tsc);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpy:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAsync:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoH_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoHAsync_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoD_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoA_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyAtoA_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpy2D_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DUnaligned_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpy2DAsync_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpy3D_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpy3DAsync_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoA_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoAAsync_v2:
      // This would be the place to populate the memcpy API activity's src and
      // dst memory kind by casting cbdata->functionParams. However, we are not
      // doing that because that will incur significant overhead to get the
      // memory aperture of each argument.
      AddNormalMemcpyEventUponApiExit(tracer, device_id, cbid, cbdata,
                                      start_tsc, end_tsc);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyPeer:
    case CUPTI_DRIVER_TRACE_CBID_cuMemcpyPeerAsync:
      AddP2PMemcpyEventUponApiExit(tracer, cupti_interface, device_id, cbid,
                                   cbdata, start_tsc, end_tsc);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuMemAlloc_v2:
      AddCuMemAllocEventUponApiExit(tracer, device_id, cbid, cbdata, start_tsc,
                                    end_tsc);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuMemAllocPitch_v2:
      AddCuMemAllocPitchEventUponApiExit(tracer, device_id, cbid, cbdata,
                                         start_tsc, end_tsc);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuMemAllocManaged:
      AddCuMemAllocManagedEventUponApiExit(tracer, device_id, cbid, cbdata,
                                           start_tsc, end_tsc);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuMemAllocHost_v2:
      AddCuMemAllocHostEventUponApiExit(tracer, device_id, cbid, cbdata,
                                        start_tsc, end_tsc);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuMemHostAlloc:
      AddCuMemHostAllocEventUponApiExit(tracer, device_id, cbid, cbdata,
                                        start_tsc, end_tsc);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuMemFree_v2:
      AddCuMemFreeEventUponApiExit(tracer, device_id, cbid, cbdata, start_tsc,
                                   end_tsc);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuMemFreeHost:
      AddCuMemFreeHostEventUponApiExit(tracer, device_id, cbid, cbdata,
                                       start_tsc, end_tsc);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuMemHostRegister_v2:
      AddCuMemHostRegisterEventUponApiExit(tracer, device_id, cbid, cbdata,
                                           start_tsc, end_tsc);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuMemHostUnregister:
      AddCuMemHostUnregisterEventUponApiExit(tracer, device_id, cbid, cbdata,
                                             start_tsc, end_tsc);
      break;
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD8_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD16_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD32_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D8_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D16_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D32_v2:
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD8Async:
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD16Async:
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD32Async:
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D8Async:
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D16Async:
    case CUPTI_DRIVER_TRACE_CBID_cuMemsetD2D32Async:
      AddCuMemsetEventUponApiExit(tracer, device_id, cbid, cbdata, start_tsc,
                                  end_tsc);
      break;
    default:
      AddGenericEventUponApiExit(tracer, device_id, cbid, cbdata, start_tsc,
                                 end_tsc);
      break;
  }
  return OkStatus();
}

const char *GetTraceEventTypeName(const CuptiTracerEventType &type) {
  // Do not use a default so that this gives a build error when
  // CuptiTracerEventType is extended but this is not.
  switch (type) {
    case CuptiTracerEventType::MemcpyH2D:
      return "MemcpyH2D";
    case CuptiTracerEventType::MemcpyD2H:
      return "MemcpyD2H";
    case CuptiTracerEventType::MemcpyD2D:
      return "MemcpyD2D";
    case CuptiTracerEventType::MemcpyP2P:
      return "MemcpyP2P";
    case CuptiTracerEventType::MemcpyOther:
      return "MemcpyOther";
    case CuptiTracerEventType::Kernel:
      return "Compute";
    case CuptiTracerEventType::MemoryAlloc:
      return "MemoryAlloc";
    case CuptiTracerEventType::MemoryFree:
      return "MemoryFree";
    case CuptiTracerEventType::Memset:
      return "Memset";
    case CuptiTracerEventType::Overhead:
      return "Overhead";
    case CuptiTracerEventType::UnifiedMemory:
      return "UnifiedMemory";
    case CuptiTracerEventType::Generic:
      return "Generic";
    case CuptiTracerEventType::MemoryResidency:
      return "MemoryResidency";
    case CuptiTracerEventType::HostRegister:
      return "HostRegister";
    case CuptiTracerEventType::HostUnregister:
      return "HostUnregister";
    case CuptiTracerEventType::Unsupported:
      return "";
  }
}

CuptiTracer::CuptiTracer(CuptiInterface *cupti_interface)
    : num_gpus_(NumGpus()),
      cupti_interface_(cupti_interface),
      buffer_pool_(kBufferSizeInBytes) {}

/* static */ CuptiTracer *CuptiTracer::GetCuptiTracerSingleton() {
  static auto *singleton = new CuptiTracer(GetCuptiInterface());
  return singleton;
}

bool CuptiTracer::IsAvailable() const {
  return NumGpus() && !activity_tracing_enabled_ && !api_tracing_enabled_;
}

int CuptiTracer::NumGpus() {
  static int num_gpus = []() -> int {
    if (cuInit(0) != CUDA_SUCCESS) {
      return 0;
    }
    int gpu_count;
    if (cuDeviceGetCount(&gpu_count) != CUDA_SUCCESS) {
      return 0;
    }
    LOG(INFO) << "Profiler found " << gpu_count << " GPUs";
    return gpu_count;
  }();
  return num_gpus;
}

void CuptiTracer::Enable(const CuptiTracerOptions &option,
                         CuptiTraceCollector *collector) {
  option_ = option;
  collector_ = collector;

  cupti_driver_api_hook_.reset(
      new CuptiDriverApiHookWithActivityApi(option, cupti_interface_, this));

  Status status = EnableApiTracing();
  need_root_access_ |= status.code() == tsl::error::PERMISSION_DENIED;
  if (!status.ok()) return;

  EnableActivityTracing().IgnoreError();
  tsl::profiler::AnnotationStack::Enable(true);
}

void CuptiTracer::Disable() {
  DisableApiTracing().IgnoreError();
  DisableActivityTracing().IgnoreError();
  cupti_interface_->CleanUp();
  Finalize().IgnoreError();
  cupti_driver_api_hook_->SyncAndFlush().IgnoreError();

  // Processing cached activity buffer and cached callback/annotations
  // and add them into collector.
  GatherAllCallbackAnnotationsAndEvents();
  FinalizeApiCallbackBuffers();
  FinalizeActivityBuffers();

  collector_->Flush();
  collector_ = nullptr;
  option_.reset();
  cupti_driver_api_hook_.reset();
  tsl::profiler::AnnotationStack::Enable(false);
}

Status CuptiTracer::EnableApiTracing() {
  if (api_tracing_enabled_) return OkStatus();

  // Clear all per-thread annotation and events for API callback
  ClearAllAnnotatedEvents();
  PrepareOptionSettings();

  VLOG(1) << "Enable subscriber";
  // Subscribe can return CUPTI_ERROR_MAX_LIMIT_REACHED.
  // The application which calls CUPTI APIs cannot be used with Nvidia tools
  // like nvprof, Nvidia Visual Profiler, Nsight Compute, Nsight Systems.
  RETURN_IF_CUPTI_ERROR(cupti_interface_->Subscribe(
      &subscriber_, (CUpti_CallbackFunc)ApiCallback, this));
  api_tracing_enabled_ = true;

  if (!option_->cbids_selected.empty()) {
    for (auto cbid : option_->cbids_selected) {
      RETURN_IF_CUPTI_ERROR(cupti_interface_->EnableCallback(
          1 /* ENABLE */, subscriber_, CUPTI_CB_DOMAIN_DRIVER_API, cbid));
    }
  } else {  // select all callback ids.
    RETURN_IF_CUPTI_ERROR(cupti_interface_->EnableDomain(
        1 /* ENABLE */, subscriber_, CUPTI_CB_DOMAIN_DRIVER_API));
  }

  if (option_->enable_nvtx_tracking) {
    RETURN_IF_CUPTI_ERROR(cupti_interface_->EnableDomain(
        1 /* ENABLE */, subscriber_, CUPTI_CB_DOMAIN_NVTX));
  }
  return OkStatus();
}

Status CuptiTracer::DisableApiTracing() {
  if (!api_tracing_enabled_) return OkStatus();

  api_tracing_enabled_ = false;

  if (!option_->cbids_selected.empty()) {
    for (auto cbid : option_->cbids_selected) {
      RETURN_IF_CUPTI_ERROR(cupti_interface_->EnableCallback(
          0 /* DISABLE */, subscriber_, CUPTI_CB_DOMAIN_DRIVER_API, cbid));
    }
  } else {
    RETURN_IF_CUPTI_ERROR(cupti_interface_->EnableDomain(
        0 /* DISABLE */, subscriber_, CUPTI_CB_DOMAIN_DRIVER_API));
  }

  if (option_->enable_nvtx_tracking) {
    RETURN_IF_CUPTI_ERROR(cupti_interface_->EnableDomain(
        0 /* DISABLE */, subscriber_, CUPTI_CB_DOMAIN_NVTX));
  }

  VLOG(1) << "Disable subscriber";
  RETURN_IF_CUPTI_ERROR(cupti_interface_->Unsubscribe(subscriber_));
  return OkStatus();
}

Status CuptiTracer::EnableActivityTracing() {
  if (!option_->activities_selected.empty()) {
    // Initialize callback functions for Cupti Activity API.
    VLOG(1) << "Registering CUPTI activity callbacks";
    RETURN_IF_CUPTI_ERROR(cupti_interface_->ActivityRegisterCallbacks(
        RequestCuptiActivityBuffer, ProcessCuptiActivityBuffer));

    VLOG(1) << "Enabling activity tracing for "
            << option_->activities_selected.size() << " activities";
    for (auto activity : option_->activities_selected) {
      VLOG(1) << "Enabling activity tracing for: " << activity;
      if (activity == CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER) {
        ConfigureActivityUnifiedMemoryCounter(true);
      }
      RETURN_IF_CUPTI_ERROR(cupti_interface_->ActivityEnable(activity));
    }
  }
  activity_tracing_enabled_ = true;
  return OkStatus();
}

Status CuptiTracer::DisableActivityTracing() {
  if (activity_tracing_enabled_) {
    VLOG(1) << "Disabling activity tracing for "
            << option_->activities_selected.size() << " activities";
    for (auto activity : option_->activities_selected) {
      VLOG(1) << "Disabling activity tracing for: " << activity;
      if (activity == CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER) {
        ConfigureActivityUnifiedMemoryCounter(false);
      }
      RETURN_IF_CUPTI_ERROR(cupti_interface_->ActivityDisable(activity));
    }
    option_->activities_selected.clear();

    VLOG(1) << "Flushing CUPTI activity buffer";
    RETURN_IF_CUPTI_ERROR(
        cupti_interface_->ActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
    LOG(INFO) << "CUPTI activity buffer flushed";
  }
  activity_tracing_enabled_ = false;
  return OkStatus();
}

Status CuptiTracer::Finalize() {
  if (option_->cupti_finalize) {
    VLOG(1) << "CuptiFinalize";
    RETURN_IF_CUPTI_ERROR(cupti_interface_->Finalize());
  }
  return OkStatus();
}

/*static*/ uint64_t CuptiTracer::GetTimestamp() {
  uint64_t tsc;
  CuptiInterface *cupti_interface = GetCuptiInterface();
  if (cupti_interface && cupti_interface->GetTimestamp(&tsc) == CUPTI_SUCCESS) {
    return tsc;
  }
  // Return 0 on error. If an activity timestamp is 0, the activity will be
  // dropped during time normalization.
  return 0;
}

Status CuptiTracer::HandleNVTXCallback(CUpti_CallbackId cbid,
                                       const CUpti_CallbackData *cbdata) {
  const CUpti_NvtxData *pdata =
      reinterpret_cast<const CUpti_NvtxData *>(cbdata);
  if (cbid == CUPTI_CBID_NVTX_nvtxDomainRangePushEx) {
    const nvtxDomainRangePushEx_params *params =
        reinterpret_cast<const nvtxDomainRangePushEx_params *>(
            pdata->functionParams);
    // TODO(profiler): The messageType is actually NVTX_MESSAGE_TYPE_REGISTERED
    // (which is 3), However it seems to me that we can not get the registered
    // string from nvtxDomainRegisterStringA_params. If we reinterpret the
    // payload as ascii, it happen to work.
    NVTXRangeTracker::EnterRange(params->core.eventAttrib->message.ascii);
  } else if (cbid == CUPTI_CBID_NVTX_nvtxDomainRangePop) {
    NVTXRangeTracker::ExitRange();
  }
  return OkStatus();
}

Status CuptiTracer::HandleCallback(CUpti_CallbackDomain domain,
                                   CUpti_CallbackId cbid,
                                   const CUpti_CallbackData *cbdata) {
  if (!api_tracing_enabled_) return OkStatus();    // already unsubscribed.
  if (!cupti_driver_api_hook_) return OkStatus();  // already unsubscribed.
  if (domain == CUPTI_CB_DOMAIN_NVTX) return HandleNVTXCallback(cbid, cbdata);
  if (domain != CUPTI_CB_DOMAIN_DRIVER_API) return OkStatus();
  if (internalCuCall) return OkStatus();

  if (cbdata->context == nullptr) {
    // API callback is called before any CUDA context is created.
    // This is expected to be rare, and we ignore this case.
    VLOG(3) << "API callback received before creation of CUDA context\n";
    return tsl::errors::Internal("cutpi callback without context");
  }

  // Grab a correct device ID.
  uint32_t device_id = -1;
  RETURN_IF_CUPTI_ERROR(
      cupti_interface_->GetDeviceId(cbdata->context, &device_id));
  if (device_id >= num_gpus_) {
    return tsl::errors::Internal("Invalid device id:", device_id);
  }

  if (cbdata->callbackSite == CUPTI_API_ENTER) {
    TF_RETURN_IF_ERROR(cupti_driver_api_hook_->OnDriverApiEnter(
        device_id, domain, cbid, cbdata));
  } else if (cbdata->callbackSite == CUPTI_API_EXIT) {
    // Set up the map from correlation id to annotation string.
    const auto &annotation = AnnotationStack::Get();
    absl::string_view nvtx_range = NVTXRangeTracker::CurrentRange();

    if (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchCooperativeKernelMultiDevice) {
      // Kernels are launched on different devices by this API call, therefore
      // we need to populate per device annotation map respectively.
      nvtx_range = "";
    }
    bool appended = callback_annotations_and_events_->Add(
        device_id, cbdata->correlationId, annotation, nvtx_range);
    if (appended) {
      TF_RETURN_IF_ERROR(cupti_driver_api_hook_->OnDriverApiExit(
          device_id, domain, cbid, cbdata));
    }
  }
  return OkStatus();
}

void CuptiTracer::ConfigureActivityUnifiedMemoryCounter(bool enable) {
  CUpti_ActivityUnifiedMemoryCounterConfig config[2];
  // By experiments, currently only measurements from these two activities are
  // trustworthy. Others like GPU page fault may be problematic.
  config[0].kind =
      CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_HTOD;
  config[1].kind =
      CUPTI_ACTIVITY_UNIFIED_MEMORY_COUNTER_KIND_BYTES_TRANSFER_DTOH;

  for (size_t i = 0; i < 2; i++) {
    config[i].enable = enable;
  }

  CUptiResult res;

  res = cupti_interface_->ActivityConfigureUnifiedMemoryCounter(config, 2);
  if (res == CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED) {
    LOG(ERROR) << "Unified memory is not supported on the "
                  "underlying platform.\n";
  } else if (res == CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_DEVICE) {
    LOG(ERROR) << "Unified memory is not supported on the device.\n";
  } else if (res == CUPTI_ERROR_UM_PROFILING_NOT_SUPPORTED_ON_NON_P2P_DEVICES) {
    LOG(ERROR) << "Unified memory is not supported on the "
                  "non-P2P multi-gpu setup.\n";
  } else if (res != CUPTI_SUCCESS) {
    const char *errstr = "";
    cuptiGetResultString(res, &errstr);
    LOG(ERROR) << "Error while enabling unified memory profiling: " << errstr;
  } else {
    VLOG(1) << "Configuring Unified memory profiling: " << res;
  }
}

void CuptiTracer::RequestActivityBuffer(uint8_t **buffer, size_t *size) {
  // Keep the buffer pool here, as when estimated activity events is larger
  // than the max allowed, process activity buffer just return new flushing
  // buffer to the pool, so that no endless memory allocation happens after
  // enough event collected, since under such case, buffer comes from
  // the pool.
  *buffer = buffer_pool_.GetOrCreateBuffer();
  if (*buffer == nullptr) {
    LOG(WARNING)
        << "CUPTI Buffer not allocated, activity records will be dropped";
    *size = 0;
    return;
  }

  if (*buffer == nullptr) {
    LOG(WARNING)
        << "CUPTI Buffer not allocated, activity records will be dropped";
    *size = 0;
    return;
  }
  *size = kBufferSizeInBytes;
}

absl::Status CuptiTracer::ConvertActivityBuffer(uint8_t *buffer, size_t size) {
  CuptiInterface *cupti_interface = GetCuptiInterface();
  CUpti_Activity *record = nullptr;
  size_t event_count = 0;
  while (true) {
    CUptiResult status =
        cupti_interface->ActivityGetNextRecord(buffer, size, &record);
    if (status == CUPTI_SUCCESS) {
      event_count++;
      switch (record->kind) {
        case CUPTI_ACTIVITY_KIND_KERNEL:  // sequential
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
          AddKernelActivityEvent<TF_CUPTI_HAS_CHANNEL_ID>(
              collector_, reinterpret_cast<CuptiActivityKernelTy *>(record));
          break;
        case CUPTI_ACTIVITY_KIND_CDP_KERNEL:
          AddKernelActivityEvent<false>(
              collector_, reinterpret_cast<CUpti_ActivityCdpKernel *>(record));
          break;
        case CUPTI_ACTIVITY_KIND_MEMCPY:
          AddMemcpyActivityEvent(
              collector_, reinterpret_cast<CuptiActivityMemcpyTy *>(record));
          break;
        case CUPTI_ACTIVITY_KIND_MEMCPY2:
          AddMemcpyP2PActivityEvent(
              collector_, reinterpret_cast<CuptiActivityMemcpyP2PTy *>(record));
          break;
        case CUPTI_ACTIVITY_KIND_OVERHEAD:
          AddCuptiOverheadActivityEvent(
              collector_, reinterpret_cast<CUpti_ActivityOverhead *>(record));
          break;
        case CUPTI_ACTIVITY_KIND_UNIFIED_MEMORY_COUNTER:
          AddUnifiedMemoryActivityEvent(
              collector_,
              reinterpret_cast<CUpti_ActivityUnifiedMemoryCounter2 *>(record));
          break;
        case CUPTI_ACTIVITY_KIND_MEMORY: {
          AddMemoryActivityEvent(
              collector_, reinterpret_cast<CUpti_ActivityMemory *>(record));
        } break;
        case CUPTI_ACTIVITY_KIND_MEMSET:
          AddMemsetActivityEvent(
              collector_, reinterpret_cast<CuptiActivityMemsetTy *>(record));
          break;
        case CUPTI_ACTIVITY_KIND_SYNCHRONIZATION:
          AddSynchronizationActivityEvent(
              collector_,
              reinterpret_cast<CUpti_ActivitySynchronization *>(record));
          break;
        default:
          VLOG(3) << "Activity type " << record->kind << " is not supported.";
          break;
      }
    } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
      // Normal, just reach the end of the buffer.
      break;
    } else {
      LOG(WARNING) << "CUPTI parse ACTIVITY buffer error: " << status;
      return tsl::errors::Internal("Parse cupti activity buffer error.");
    }
  }
  VLOG(3) << "CUPTI Collector post-process one ACTIVITY buffer of size: "
          << size << ", total events count:" << event_count;
  return OkStatus();
}

CuptiTracer::ActivityBufferAndSize::ActivityBufferAndSize(uint8_t *p, size_t sz)
    : buffer(p,
             [](uint8_t *x) {
               if (x != nullptr) tsl::port::AlignedFree(x);
             }),
      size(sz) {}

Status CuptiTracer::ProcessActivityBuffer(CUcontext context, uint32_t stream_id,
                                          uint8_t *buffer, size_t size) {
  absl::Cleanup buffer_cleanup = [&]() {
    if (buffer) buffer_pool_.ReclaimBuffer(buffer);
  };
  if (size == 0) {
    return OkStatus();
  }
  if (!activity_tracing_enabled_) {
    LOG(WARNING) << "CUPTI activity buffer is reclaimed after flush.";
    return OkStatus();
  }
  if (cupti_interface_->Disabled()) return tsl::errors::Internal("Disabled.");

  // Report dropped records.
  size_t dropped = 0;
  if (CUPTI_SUCCESS == cupti_interface_->ActivityGetNumDroppedRecords(
                           context, stream_id, &dropped)) {
    cupti_dropped_activity_event_count_ += dropped;
  }

  // TODO: ensure this
  static constexpr size_t kMaxCuptiActivityEventSize = 64;
  size_t estimated_event_count =
      (size + kMaxCuptiActivityEventSize - 1) / kMaxCuptiActivityEventSize;
  if (estimated_num_activity_events_ >=
      collector_->options_.max_activity_api_events) {
    LOG(WARNING) << "Already too many activity events, drop the buffer of "
                 << size << "bytes of event to reuse.";
    estimated_num_dropped_activity_events_ += estimated_event_count;
    return OkStatus();
  }
  estimated_num_activity_events_ += estimated_event_count;

  // When cupti activity buffer is required to flush, save the buffer and its
  // valid size some where. All the saved activity buffer will be handled
  // after the profiling is stopped.
  VLOG(3) << "Caching CUPTI activity buffer of size:" << size;
  absl::MutexLock lock(&activity_buffers_mutex_);
  activity_buffers_.emplace_back(buffer, size);
  buffer = nullptr;  // So cleanup will not free it as it was saved already

  return OkStatus();
}

/*static*/ std::string CuptiTracer::ErrorIfAny() {
  if (CuptiTracer::NumGpus() == 0) {
    return ErrorWithHostname("No GPU detected.");
  } else if (CuptiTracer::GetCuptiTracerSingleton()->NeedRootAccess()) {
    return ErrorWithHostname(
        "Insufficient privilege to run libcupti (you need root permission).");
  } else if (CuptiTracer::GetTimestamp() == 0) {
    return ErrorWithHostname(
        "Failed to load libcupti (is it installed and accessible?)");
  }
  return "";
}

// void CuptiTraceCollector::Flush() {
// }

CuptiTracerEvent *CuptiTracer::LastCallbackEvent() {
  auto *last_event =
      callback_annotations_and_events_->event_annotation_buffer.LastElement();
  return last_event ? &last_event->event : nullptr;
}

// Gather all per-thread callback events and annotations.
// Merged annotation map (correltionId->annotation) cross per-thread data.
// Empty per-thread callback annotations and events.
void CuptiTracer::GatherAllCallbackAnnotationsAndEvents() {
  collected_annotation_and_events_ =
      CallbackAnnotationsAndEventsCollection::Instance()->CollectAll();
  VLOG(3) << "Total grabbed per thread annotated events: "
          << collected_annotation_and_events_.size();
  merged_annotation_map_.clear();
  dropped_callback_event_count_ = 0;
  for (const auto &annotations_events : collected_annotation_and_events_) {
    auto &buffer = annotations_events->event_annotation_buffer;
    for (auto &block : buffer.GetBlocks()) {
      for (auto &event_with_annotation : block) {
        if (!event_with_annotation.annotation.empty() ||
            !event_with_annotation.nvtx_range.empty()) {
          merged_annotation_map_.emplace(
              event_with_annotation.correlation_id,
              AnnotationInfo{event_with_annotation.annotation,
                             event_with_annotation.nvtx_range});
        }
      }
    }
    dropped_callback_event_count_ += annotations_events->num_dropped_events;
  }
  VLOG(3) << "Total merged annotation map: " << merged_annotation_map_.size();
  collector_->SetAnnotationMap(std::move(merged_annotation_map_));
}

// Clear all gathered callback events and annotations cross all threads.
// Clear the merged annotation map.
// Also empty per-thread callback annotations and events.
void CuptiTracer::ClearAllAnnotatedEvents() {
  VLOG(3) << "Cupti Tracer is clearing per-thread and collected data!";
  collected_annotation_and_events_.clear();
  merged_annotation_map_.clear();
  CallbackAnnotationsAndEventsCollection::Instance()->CollectAll().clear();
  dropped_callback_event_count_ = 0;
}

// Right before profiling, setting options which impact per-thread callback
// events collections.
void CuptiTracer::PrepareOptionSettings() {
  CallbackAnnotationsAndEvents::s_max_annotation_strings =
      collector_->options_.max_annotation_strings;
  CallbackAnnotationsAndEvents::s_max_callback_api_events =
      collector_->options_.max_callback_api_events;
  CallbackAnnotationsAndEvents::s_callback_api_event_count = 0;
}

void CuptiTracer::FinalizeActivityBuffers() {
  // dropped_activity_event_count_ = 0;
  while (true) {
    ActivityBufferAndSize buffer_and_size;
    {
      absl::MutexLock lock(&this->activity_buffers_mutex_);
      if (activity_buffers_.empty()) break;
      buffer_and_size = activity_buffers_.front();
      activity_buffers_.pop_front();
    }
    ConvertActivityBuffer(buffer_and_size.buffer.get(), buffer_and_size.size)
        .IgnoreError();
  }
  // if (dropped_activity_event_count_ > 0) {
  //   collector_->OnEventsDropped("total device(activity) events reaches max",
  //                   dropped_activity_event_count_);
  // }
}

void CuptiTracer::FinalizeApiCallbackBuffers() {
  for (auto &annotations_and_events : collected_annotation_and_events_) {
    auto &buffer = annotations_and_events->event_annotation_buffer;
    for (auto &block : buffer.GetBlocks()) {
      for (auto &event_with_annotation : block) {
        collector_->AddEvent(std::move(event_with_annotation.event));
      }
    }
  }
  // if (dropped_callback_event_count_ > 0) {
  //   OnEventsDropped("total driver(callback) events reaches max",
  //                   dropped_callback_event_count_);
  // }
}

thread_local CallbackAnnotationsEventsWeakPtr
    CuptiTracer::callback_annotations_and_events_;

}  // namespace profiler
}  // namespace xla
