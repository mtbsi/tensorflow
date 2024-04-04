/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/core/data/captured_function.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/name_utils.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/dataset_options.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace data {
namespace {

constexpr const char kDatasetType[] = "IndexFlatMap";
constexpr const char kIndexFlatMapDataset[] = "IndexFlatMapDataset";
constexpr const char kMapFn[] = "map_func";
constexpr const char kMapFuncTargs[] = "Tmap_func_args";
constexpr const char kMapFuncOtherArgs[] = "map_func_other_args";
constexpr const char kIndexMapFn[] = "index_map_func";
constexpr const char kIndexMapFuncTargs[] = "Tindex_map_func_args";
constexpr const char kIndexMapFuncOtherArgs[] = "index_map_func_other_args";
constexpr const char kOutputTypes[] = "output_types";
constexpr const char kOutputShapes[] = "output_shapes";

std::string ToDebugString(const std::vector<Tensor>& tensors) {
  std::vector<std::string> tensor_strs;
  tensor_strs.reserve(tensors.size());
  for (const Tensor& tensor : tensors) {
    tensor_strs.push_back(tensor.DebugString());
  }
  return absl::StrCat("{", absl::StrJoin(tensor_strs, ", "), "}");
}

absl::StatusOr<size_t> GetValue(const Tensor& tensor) {
  switch (tensor.dtype()) {
    case DT_UINT64:
      return tensor.scalar<uint64_t>()();
    case DT_UINT32:
      return tensor.scalar<uint32_t>()();
    case DT_INT64:
      return tensor.scalar<int64_t>()();
    case DT_INT32:
      return tensor.scalar<int32_t>()();
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "The `index_map_fn` for `index_flat_map` is expected to return two "
          "int32/int64 values representing the element index and an offset "
          "within the element. Got: ",
          tensor.DebugString()));
  }
}

class IndexFlatMapDatasetOp : public UnaryDatasetOpKernel {
 public:
  explicit IndexFlatMapDatasetOp(OpKernelConstruction* ctx);

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override;

 private:
  class Dataset;
  std::shared_ptr<FunctionMetadata> map_func_metadata_ = nullptr;
  std::shared_ptr<FunctionMetadata> index_map_func_metadata_ = nullptr;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

class IndexFlatMapDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input,
          std::unique_ptr<CapturedFunction> captured_map_func,
          std::unique_ptr<CapturedFunction> captured_index_map_func,
          const DataTypeVector& output_types,
          const std::vector<PartialTensorShape>& output_shapes)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        captured_map_func_(std::move(captured_map_func)),
        captured_index_map_func_(std::move(captured_index_map_func)),
        output_types_(output_types),
        output_shapes_(output_shapes) {
    input_->Ref();
  }

  ~Dataset() override { input_->Unref(); }

  const DataTypeVector& output_dtypes() const override { return output_types_; }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return output_shapes_;
  }

  std::string DebugString() const override {
    return name_utils::DatasetDebugString(kDatasetType);
  }

  int64_t CardinalityInternal(CardinalityOptions options) const override {
    // TODO(b/325112575): Implement this.
    return kUnknownCardinality;
  }

  absl::Status InputDatasets(
      std::vector<const DatasetBase*>* inputs) const override {
    inputs->push_back(input_);
    return absl::OkStatus();
  }

  absl::Status CheckExternalState() const override {
    return input_->CheckExternalState();
  }

  absl::Status RandomIndexingCompatible() const override {
    return absl::OkStatus();
  }

 protected:
  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const std::string& prefix) const override;

  absl::Status AsGraphDefInternal(SerializationContext* ctx,
                                  DatasetGraphDefBuilder* b,
                                  Node** output) const override {
    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));

    std::vector<Node*> map_func_other_args;
    DataTypeVector map_func_other_args_types;
    TF_RETURN_IF_ERROR(captured_map_func_->AddToGraph(
        ctx, b, &map_func_other_args, &map_func_other_args_types));

    std::vector<Node*> index_map_func_other_args;
    DataTypeVector index_map_func_other_args_types;
    TF_RETURN_IF_ERROR(captured_index_map_func_->AddToGraph(
        ctx, b, &index_map_func_other_args, &index_map_func_other_args_types));

    AttrValue map_func_attr;
    b->BuildAttrValue(captured_map_func_->func(), &map_func_attr);

    AttrValue map_func_arguments_types_attr;
    b->BuildAttrValue(map_func_other_args_types,
                      &map_func_arguments_types_attr);

    AttrValue index_map_func_attr;
    b->BuildAttrValue(captured_index_map_func_->func(), &index_map_func_attr);

    AttrValue index_map_func_arguments_types_attr;
    b->BuildAttrValue(index_map_func_other_args_types,
                      &index_map_func_arguments_types_attr);

    return b->AddDataset(
        this,
        /*inputs=*/
        {std::make_pair(0, input_graph_node)},
        /*list_inputs=*/
        {std::make_pair(1, map_func_other_args),
         std::make_pair(2, index_map_func_other_args)},
        /*attrs=*/
        {{kMapFn, map_func_attr},
         {kMapFuncTargs, map_func_arguments_types_attr},
         {kIndexMapFn, index_map_func_attr},
         {kIndexMapFuncTargs, index_map_func_arguments_types_attr}},
        output);
  }

 private:
  class Iterator;
  const DatasetBase* const input_;
  const std::unique_ptr<CapturedFunction> captured_map_func_;
  const std::unique_ptr<CapturedFunction> captured_index_map_func_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
};

class IndexFlatMapDatasetOp::Dataset::Iterator
    : public DatasetIterator<Dataset> {
 public:
  explicit Iterator(const Params& params) : DatasetIterator<Dataset>(params) {}

  absl::Status Initialize(IteratorContext* ctx) override
      ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock l(&mu_);
    TF_RETURN_IF_ERROR(
        dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_));
    TF_RETURN_IF_ERROR(dataset()->captured_map_func_->Instantiate(
        ctx, &instantiated_map_func_));
    TF_RETURN_IF_ERROR(dataset()->captured_index_map_func_->Instantiate(
        ctx, &instantiated_index_map_func_));
    return absl::OkStatus();
  }

  absl::Status GetNextInternal(IteratorContext* ctx,
                               std::vector<Tensor>* out_tensors,
                               bool* end_of_sequence) override
      ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock l(&mu_);
    // TODO(b/325112575): Make it easier to return multiple values from
    // IndexMapperFn.
    size_t offset = 0;
    IteratorContext ctx_with_index_mapper =
        GetContextWithIndexMapper(ctx, offset);
    std::vector<Tensor> input_tensors;
    TF_RETURN_IF_ERROR(input_impl_->GetNext(&ctx_with_index_mapper,
                                            &input_tensors, end_of_sequence));
    ctx->MergeCheckpoint(ctx_with_index_mapper.checkpoint());
    if (*end_of_sequence) {
      return absl::OkStatus();
    }

    std::vector<Tensor> mapped_tensors;
    TF_RETURN_IF_ERROR(instantiated_map_func_->Run(
        ctx, {std::move(input_tensors)}, &mapped_tensors));
    for (int i = 0; i < mapped_tensors.size(); ++i) {
      if (mapped_tensors[i].dims() == 0) {  // Scalar.
        out_tensors->push_back(std::move(mapped_tensors[i]));
      } else {
        out_tensors->push_back(MaybeCopySubSlice(mapped_tensors[i], offset));
      }
    }
    return absl::OkStatus();
  }

  IteratorContext GetContextWithIndexMapper(IteratorContext* ctx,
                                            size_t& offset) const {
    IteratorContext::Params params(ctx);
    params.index_mapper = GetFlatMapIndexMapper(ctx, offset);
    return IteratorContext(params);
  }

  IndexMapperFn GetFlatMapIndexMapper(IteratorContext* ctx,
                                      size_t& offset) const {
    return [this, ctx, &offset](size_t element_position) {
      size_t shuffled_index = ctx->index_mapper()
                                  ? ctx->index_mapper()(element_position)
                                  : element_position;
      absl::StatusOr<std::tuple<size_t, size_t>> unflattened_index =
          GetUnflattenedIndex(ctx, shuffled_index);
      // TODO(b/325112575): Update the index mapper API to return a `StatusOr`.
      offset = std::get<1>(*unflattened_index);
      return std::get<0>(*unflattened_index);
    };
  }

  // Given an index in the flattened dataset, returns a tuple of
  // (element index, offset within element) in the unflattend dataset.
  absl::StatusOr<std::tuple<size_t, size_t>> GetUnflattenedIndex(
      IteratorContext* ctx, size_t flattened_index) const {
    Tensor flattened_index_tensor(ctx->allocator({}), DT_INT64,
                                  TensorShape({}));
    flattened_index_tensor.scalar<int64_t>()() = flattened_index;

    std::vector<Tensor> unflattened_index;
    TF_RETURN_IF_ERROR(instantiated_index_map_func_->Run(
        ctx, {std::move(flattened_index_tensor)}, &unflattened_index));
    if (unflattened_index.size() != 2) {
      return absl::InvalidArgumentError(absl::StrCat(
          "The `index_map_fn` for `index_flat_map` is expected to return two "
          "int values representing the element index and an offset within the "
          "element. Got: ",
          ToDebugString(unflattened_index)));
    }
    TF_ASSIGN_OR_RETURN(size_t element_index, GetValue(unflattened_index[0]));
    TF_ASSIGN_OR_RETURN(size_t offset, GetValue(unflattened_index[1]));
    return std::tuple<size_t, size_t>{element_index, offset};
  }

  // TODO(b/325112575): Support save/load for index_flat_map.
  // TODO(b/325112575): Support symbolic checkpoints.
  absl::Status SaveInternal(SerializationContext* ctx,
                            IteratorStateWriter* writer) override {
    return absl::UnimplementedError(
        "TODO(b/325112575): Support save/load for index_flat_map.");
  }

  absl::Status RestoreInternal(IteratorContext* ctx,
                               IteratorStateReader* reader) override {
    return absl::UnimplementedError(
        "TODO(b/325112575): Support save/load for index_flat_map.");
  }

 private:
  mutable absl::Mutex mu_;
  std::unique_ptr<IteratorBase> input_impl_ ABSL_GUARDED_BY(mu_);
  std::unique_ptr<InstantiatedCapturedFunction> instantiated_map_func_;
  std::unique_ptr<InstantiatedCapturedFunction> instantiated_index_map_func_;
};

IndexFlatMapDatasetOp::IndexFlatMapDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {
  OP_REQUIRES_OK(ctx, FunctionMetadata::Create(ctx, kMapFn, /*params=*/{},
                                               &map_func_metadata_));
  OP_REQUIRES_OK(ctx, FunctionMetadata::Create(ctx, kIndexMapFn, /*params=*/{},
                                               &index_map_func_metadata_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
  OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
}

void IndexFlatMapDatasetOp::MakeDataset(OpKernelContext* ctx,
                                        DatasetBase* input,
                                        DatasetBase** output) {
  OP_REQUIRES(ctx, input->RandomIndexingCompatible().ok(),
              absl::FailedPreconditionError(absl::StrCat(
                  "`index_flat_map` requires all upstream transformations be "
                  "compatible with random access. Got: ",
                  input->RandomIndexingCompatible().ToString())));

  std::unique_ptr<CapturedFunction> captured_map_func;
  OP_REQUIRES_OK(
      ctx, CapturedFunction::Create(ctx, map_func_metadata_, kMapFuncOtherArgs,
                                    &captured_map_func));

  std::unique_ptr<CapturedFunction> captured_index_map_func;
  OP_REQUIRES_OK(ctx, CapturedFunction::Create(ctx, index_map_func_metadata_,
                                               kIndexMapFuncOtherArgs,
                                               &captured_index_map_func));
  *output = new Dataset(ctx, input, std::move(captured_map_func),
                        std::move(captured_index_map_func), output_types_,
                        output_shapes_);
}

std::unique_ptr<IteratorBase>
IndexFlatMapDatasetOp::Dataset::MakeIteratorInternal(
    const std::string& prefix) const {
  return std::make_unique<IndexFlatMapDatasetOp::Dataset::Iterator>(
      Iterator::Params{this, name_utils::IteratorPrefix(kDatasetType, prefix)});
}

REGISTER_KERNEL_BUILDER(Name(kIndexFlatMapDataset).Device(DEVICE_CPU),
                        IndexFlatMapDatasetOp);

}  // namespace
}  // namespace data
}  // namespace tensorflow
