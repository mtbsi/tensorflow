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

#include "xla/translate/mhlo_to_hlo/mlir_hlo_to_hlo.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/span.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/UseDefLists.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "xla/array.h"
#include "xla/client/lib/approx_topk.h"
#include "xla/client/lib/approx_topk_shape.h"
#include "xla/client/lib/matrix.h"
#include "xla/client/lib/quantize.h"
#include "xla/client/lib/slicing.h"
#include "xla/client/xla_builder.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/dynamic_parameter_binding.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/mlir/utils/error_util.h"
#include "xla/mlir/utils/type_util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_parser.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/status_macros.h"
#include "xla/translate/mhlo_to_hlo/attribute_exporter.h"
#include "xla/translate/mhlo_to_hlo/location_exporter.h"
#include "xla/translate/mhlo_to_hlo/stack_frame_index_builder.h"
#include "xla/translate/mhlo_to_hlo/type_to_shape.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/ml_dtypes.h"
#include "tsl/platform/statusor.h"

using ::int64_t;
using ::tsl::int16;
using ::tsl::int32;
using ::tsl::int8;
using ::tsl::StatusOr;  // TENSORFLOW_STATUS_OK
using ::tsl::uint16;
using ::tsl::uint32;
using ::tsl::uint64;
using ::tsl::uint8;

constexpr char kShapeIndicesAttr[] = "shape_indices";
constexpr char kPaddingArgIndicesAttr[] = "padding_arg_indices";
constexpr char kShardingAttr[] = "mhlo.sharding";
constexpr char kFrontendAttributesAttr[] = "mhlo.frontend_attributes";
constexpr char kReplicationAttr[] = "mhlo.is_same_data_across_replicas";
constexpr char kParameterReplicationAttr[] = "mhlo.parameter_replication";
constexpr char kLiteralAttr[] = "mhlo.literal";

// Array attribute. Same shape as infeed result, but contains a
// minor_to_major array for every tensor.
constexpr char kLayoutAttr[] = "layout";
constexpr char kDefaultLayoutAttrName[] = "xla_shape";

// Passes through everything except for unique_ptr, on which it calls get().
// This exists to allow the generated code to call XLA functions that take a raw
// pointer. In particular, PrecisionConfig is passed to xla::Dot and xla::Conv
// as a pointer and there is otherwise no way to avoid a memory leak.
template <typename T>
T Unwrap(T t) {
  return t;
}

template <typename T>
T* Unwrap(const std::unique_ptr<T>& t) {
  return t.get();
}

static mlir::LogicalResult GetXlaOp(
    mlir::Value val, const llvm::DenseMap<mlir::Value, xla::XlaOp>& val_map,
    xla::XlaOp* result, mlir::Operation* op) {
  auto iter = val_map.find(val);
  if (iter == val_map.end()) {
    return op->emitOpError(
        "requires all operands to be defined in the parent region for export");
  }
  *result = iter->second;
  return mlir::success();
}

bool IsBoundedOrStatic(mlir::Type ty) {
  auto ranked_ty = ty.dyn_cast_or_null<mlir::RankedTensorType>();
  if (!ranked_ty) return false;

  if (ranked_ty.hasStaticShape()) return true;

  auto encoding = ranked_ty.getEncoding()
                      .dyn_cast_or_null<mlir::mhlo::TypeExtensionsAttr>();
  if (!encoding || encoding.getBounds().empty()) return false;

  int64_t rank = ranked_ty.getRank();
  for (int64_t dim = 0; dim < rank; ++dim) {
    if (ranked_ty.isDynamicDim(dim) &&
        encoding.getBounds()[dim] == mlir::ShapedType::kDynamic)
      return false;
  }
  return true;
}

template <typename T>
xla::Array<T> ArrayFromDenseElementsAttr(mlir::DenseElementsAttr dense_attr) {
  constexpr xla::PrimitiveType type =
      xla::primitive_util::NativeToPrimitiveType<T>();
  xla::Shape shape = xla::TypeToShape(dense_attr.getType());
  xla::Array<T> array(shape.dimensions());
  if constexpr (!xla::primitive_util::Is4BitType(type)) {
    array.SetValues(dense_attr.getValues<T>());
  } else {
    // The only way to get subbyte integers from getValues() is to get them as
    // APInts.
    auto values = dense_attr.getValues<llvm::APInt>();
    for (int i = 0; i < values.size(); i++) {
      if constexpr (type == xla::U4) {
        array.data()[i] = xla::u4{values[i].getZExtValue()};
      } else {
        static_assert(type == xla::S4);
        array.data()[i] = xla::s4(values[i].getSExtValue());
      }
    }
  }
  return array;
}

absl::StatusOr<xla::Literal> CreateArrayLiteralFromAttr(mlir::ElementsAttr attr,
                                                        xla::Layout layout) {
  auto dense_attr = attr.dyn_cast<mlir::DenseElementsAttr>();
  if (!dense_attr)
    return tsl::errors::Unimplemented("Only dense elements attr are supported");

  xla::Shape shape = xla::TypeToShape(dense_attr.getType());

  return xla::primitive_util::PrimitiveTypeSwitch<StatusOr<xla::Literal>>(
      [&](auto primitive_type_constant) -> absl::StatusOr<xla::Literal> {
        if constexpr (xla::primitive_util::IsArrayType(
                          primitive_type_constant)) {
          using cpp_type =
              xla::primitive_util::NativeTypeOf<primitive_type_constant>;
          xla::Array<cpp_type> source_data =
              ArrayFromDenseElementsAttr<cpp_type>(dense_attr);
          return xla::LiteralUtil::CreateFromArrayWithLayout(source_data,
                                                             layout);
        }
        return tsl::errors::Internal(absl::StrCat(  // NOLINT
            "Unsupported type: ",
            xla::PrimitiveType_Name(shape.element_type())));
      },
      shape.element_type());
}

// Convert APInt into an int.
// TODO(hpucha): This should be consolidated into a general place.
static int ConvertAPInt(llvm::APInt i) { return i.getSExtValue(); }

static uint32_t Convertuint32_t(uint32_t i) { return i; }
static uint64_t Convertuint64_t(uint64_t i) { return i; }

// Convert APFloat to double.
static double ConvertAPFloat(llvm::APFloat value) {
  const auto& semantics = value.getSemantics();
  bool losesInfo = false;
  if (&semantics != &llvm::APFloat::IEEEdouble())
    value.convert(llvm::APFloat::IEEEdouble(),
                  llvm::APFloat::rmNearestTiesToEven, &losesInfo);
  return value.convertToDouble();
}

static inline bool Convertbool(bool value) { return value; }

static absl::string_view ConvertStringRef(mlir::StringRef value) {
  return {value.data(), value.size()};
}

static std::vector<int64_t> ConvertDenseIntAttr(
    mlir::DenseIntElementsAttr attr) {
  auto values = attr.getValues<int64_t>();
  return {values.begin(), values.end()};
}

static std::vector<int64_t> ConvertDenseIntAttr(
    std::optional<mlir::DenseIntElementsAttr> attr) {
  if (!attr) return {};
  return ConvertDenseIntAttr(*attr);
}

// Converts the broadcast_dimensions attribute into a vector of dimension
// numbers (empty if the attribute is absent).
static std::vector<int64_t> Convert_broadcast_dimensions(
    std::optional<mlir::DenseIntElementsAttr> broadcast_dimensions) {
  if (!broadcast_dimensions.has_value()) return {};

  return ConvertDenseIntAttr(*broadcast_dimensions);
}

static std::vector<xla::CrossProgramPrefetch> Convert_cross_program_prefetches(
    mlir::ArrayAttr prefetches) {
  std::vector<xla::CrossProgramPrefetch> cross_program_prefetches;
  for (auto prefetch : prefetches) {
    auto cpp = prefetch.cast<mlir::mhlo::CrossProgramPrefetchAttr>();
    xla::CrossProgramPrefetch xla_cpp;
    xla_cpp.set_parameter(cpp.getParameter());
    for (auto index : cpp.getIndices()) xla_cpp.add_index(index);
    cross_program_prefetches.push_back(xla_cpp);
  }
  return cross_program_prefetches;
}

// Converts StringRef to xla FftType enum
static xla::FftType Convert_fft_type(mlir::mhlo::FftType fft_type) {
  xla::FftType fft_type_enum;
  // Illegal fft_type string would be caught by the verifier, so 'FftType_Parse'
  // call below should never return false.
  if (!FftType_Parse(std::string(mlir::mhlo::stringifyFftType(fft_type)),
                     &fft_type_enum))
    return xla::FftType::FFT;
  return fft_type_enum;
}

static std::vector<std::pair<int64_t, int64_t>> Convert_padding(
    std::optional<mlir::DenseIntElementsAttr> padding) {
  return xla::ConvertNx2Attribute(padding).value();
}

static std::optional<bool> Convert_use_global_device_ids(
    std::optional<bool> use_global_device_ids) {
  if (!use_global_device_ids) return {};
  return *use_global_device_ids;
}

static std::vector<std::pair<int64_t, int64_t>> Convert_source_target_pairs(
    std::optional<mlir::DenseIntElementsAttr> source_target_pairs) {
  return xla::ConvertNx2Attribute(source_target_pairs).value();
}

static std::vector<xla::ReplicaGroup> Convert_replica_groups(
    mlir::DenseIntElementsAttr groups) {
  return xla::ConvertReplicaGroups(groups).value();
}

static void SetLayout(xla::Shape& shape, mlir::DenseIntElementsAttr layout) {
  if (shape.IsArray()) {
    shape.mutable_layout()->clear_minor_to_major();
    for (auto l : layout) {
      shape.mutable_layout()->mutable_minor_to_major()->push_back(
          l.getSExtValue());
    }
  } else if (shape.IsToken()) {
    assert(layout.empty() && "Invalid layout for token type");
  } else {
    assert(!shape.IsTuple() &&
           "Exporting layout for tuples is not implemented yet");
    assert(false && "Exporting unknown type with layout");
  }
}

static void SetLayout(xla::Shape& shape, mlir::ArrayAttr layouts) {
  if (shape.IsTuple()) {
    for (int i = 0; i < shape.tuple_shapes_size(); ++i) {
      SetLayout(*shape.mutable_tuple_shapes(i),
                layouts[i].cast<mlir::DenseIntElementsAttr>());
    }
  } else {
    assert(layouts.size() == 1);
    SetLayout(shape, layouts[0].cast<mlir::DenseIntElementsAttr>());
  }
}

// Converts types and corresponding layouts into xla shapes with layouts.
static std::vector<xla::Shape> ConvertTypesToShapesWithLayout(
    mlir::TypeRange value_types, mlir::ArrayAttr layouts) {
  std::vector<xla::Shape> shapes_with_layout;
  for (auto [type, layout] : llvm::zip(value_types, layouts)) {
    xla::Shape shape = xla::TypeToShape(type);
    SetLayout(shape, layout.cast<mlir::DenseIntElementsAttr>());
    shapes_with_layout.push_back(std::move(shape));
  }
  return shapes_with_layout;
}

// Converts StringRef to xla Transpose enum.
static xla::TriangularSolveOptions::Transpose Convert_transpose_a(
    mlir::mhlo::Transpose transpose) {
  return xla::ConvertTranspose(mlir::mhlo::stringifyTranspose(transpose))
      .value();
}

static xla::Layout ExtractLayout(
    mlir::Operation* op, int rank,
    llvm::StringRef attr_name = kDefaultLayoutAttrName) {
  if (auto attr = op->getAttrOfType<mlir::DenseIntElementsAttr>(attr_name)) {
    llvm::SmallVector<int64_t, 4> minor_to_major;
    DCHECK_EQ(rank, attr.size());
    minor_to_major.reserve(attr.size());
    for (const llvm::APInt& i : attr) {
      minor_to_major.push_back(i.getZExtValue());
    }
    return xla::LayoutUtil::MakeLayout(minor_to_major);
  }
  return xla::LayoutUtil::MakeDescendingLayout(rank);
}

// Returns a failure or a valid XLA shape corresponding to the given op's
// results.
static mlir::FailureOr<xla::Shape> ExtractXlaShape(mlir::Operation* op) {
  if (auto attr = op->getAttrOfType<mlir::StringAttr>(kDefaultLayoutAttrName)) {
    return *xla::ParseShape(
        absl::string_view(attr.getValue().data(), attr.getValue().size()));
  } else {
    std::vector<xla::Shape> subshapes;
    for (auto [index, result] : llvm::enumerate(op->getResults())) {
      subshapes.push_back(xla::TypeToShape(result.getType()));
      if (subshapes.back().element_type() == xla::PRIMITIVE_TYPE_INVALID) {
        return op->emitError()
               << "result #" << index << " type is not supported";
      }
    }
    if (subshapes.size() > 1) {
      return xla::ShapeUtil::MakeTupleShape(subshapes);
    }
    return subshapes[0];
  }
}

#define I64_ELEMENTS_ATTR_TO_VECTOR(attribute)               \
  static std::vector<int64_t> Convert_##attribute(           \
      std::optional<mlir::DenseIntElementsAttr> attribute) { \
    return ConvertDenseIntAttr(attribute);                   \
  }

I64_ELEMENTS_ATTR_TO_VECTOR(broadcast_sizes);
I64_ELEMENTS_ATTR_TO_VECTOR(permutation);
I64_ELEMENTS_ATTR_TO_VECTOR(start_indices);
I64_ELEMENTS_ATTR_TO_VECTOR(limit_indices);
I64_ELEMENTS_ATTR_TO_VECTOR(strides);
I64_ELEMENTS_ATTR_TO_VECTOR(slice_sizes);
I64_ELEMENTS_ATTR_TO_VECTOR(fft_length);
I64_ELEMENTS_ATTR_TO_VECTOR(dimensions);
I64_ELEMENTS_ATTR_TO_VECTOR(window_strides);
I64_ELEMENTS_ATTR_TO_VECTOR(lhs_dilation);
I64_ELEMENTS_ATTR_TO_VECTOR(rhs_dilation);

#undef I64_ELEMENTS_ATTR_TO_VECTOR

#define BOOL_ELEMENTS_ATTR_TO_VECTOR(attribute)           \
  static std::vector<bool> Convert_##attribute(           \
      std::optional<mlir::DenseElementsAttr> attribute) { \
    if (!attribute) return {};                            \
    auto values = attribute->getValues<bool>();           \
    return {values.begin(), values.end()};                \
  }

BOOL_ELEMENTS_ATTR_TO_VECTOR(window_reversal);

#undef BOOL_ELEMENTS_ATTR_TO_VECTOR

static std::vector<int64_t> Convert_ArrayRef(llvm::ArrayRef<int64_t> values) {
  return {values.begin(), values.end()};
}

// Converts the precision config array of strings attribute into the
// corresponding XLA proto. All the strings are assumed to be valid names of the
// Precision enum. This should have been checked in the op verify method.
static std::unique_ptr<xla::PrecisionConfig> Convert_precision_config(
    std::optional<mlir::ArrayAttr> optional_precision_config_attr) {
  if (!optional_precision_config_attr.has_value()) return nullptr;

  auto precision_config = std::make_unique<xla::PrecisionConfig>();
  for (auto attr : optional_precision_config_attr.value()) {
    xla::PrecisionConfig::Precision p;
    auto operand_precision =
        mlir::mhlo::stringifyPrecision(
            attr.cast<mlir::mhlo::PrecisionAttr>().getValue())
            .str();
    // TODO(jpienaar): Update this to ensure this is captured by verify.
    if (xla::PrecisionConfig::Precision_Parse(operand_precision, &p)) {
      precision_config->add_operand_precision(p);
    } else {
      auto* context = attr.getContext();
      mlir::emitError(mlir::UnknownLoc::get(context))
          << "unexpected operand precision " << operand_precision;
      return nullptr;
    }
  }

  return precision_config;
}

static xla::DotDimensionNumbers Convert_dot_dimension_numbers(
    mlir::mhlo::DotDimensionNumbersAttr dot_dimension_numbers_attr) {
  xla::DotDimensionNumbers dot_dimension_numbers;

  auto rhs_contracting_dimensions =
      dot_dimension_numbers_attr.getRhsContractingDimensions();
  auto lhs_contracting_dimensions =
      dot_dimension_numbers_attr.getLhsContractingDimensions();
  auto rhs_batch_dimensions =
      dot_dimension_numbers_attr.getRhsBatchingDimensions();
  auto lhs_batch_dimensions =
      dot_dimension_numbers_attr.getLhsBatchingDimensions();

  for (const auto& val : rhs_contracting_dimensions) {
    dot_dimension_numbers.add_rhs_contracting_dimensions(val);
  }
  for (const auto& val : lhs_contracting_dimensions) {
    dot_dimension_numbers.add_lhs_contracting_dimensions(val);
  }

  for (const auto& val : rhs_batch_dimensions) {
    dot_dimension_numbers.add_rhs_batch_dimensions(val);
  }

  for (const auto& val : lhs_batch_dimensions) {
    dot_dimension_numbers.add_lhs_batch_dimensions(val);
  }

  return dot_dimension_numbers;
}

static xla::ConvolutionDimensionNumbers Convert_dimension_numbers(
    mlir::mhlo::ConvDimensionNumbersAttr input) {
  return xla::ConvertConvDimensionNumbers(input);
}

static xla::SparsityDescriptor Convert_sparsity_descriptor(
    mlir::mhlo::SparsityDescriptorAttr sparsity_attr, bool is_lhs) {
  xla::SparsityDescriptor sparsity_descriptor;
  sparsity_descriptor.set_type(xla::SPARSITY_STRUCTURED_N_M);
  sparsity_descriptor.set_index(is_lhs ? 0 : 1);
  sparsity_descriptor.set_dimension(sparsity_attr.getDimension());
  sparsity_descriptor.set_n(sparsity_attr.getN());
  sparsity_descriptor.set_m(sparsity_attr.getM());
  return sparsity_descriptor;
}

xla::ChannelHandle Convert_channel_handle(mlir::mhlo::ChannelHandleAttr attr) {
  xla::ChannelHandle channel_handle;
  channel_handle.set_handle(attr.getHandle());
  channel_handle.set_type(
      static_cast<xla::ChannelHandle::ChannelType>(attr.getType()));
  return channel_handle;
}

std::optional<xla::ChannelHandle> Convert_channel_handle(
    std::optional<mlir::mhlo::ChannelHandleAttr> attr) {
  if (!attr.has_value()) return std::nullopt;
  return Convert_channel_handle(attr.value());
}

// Converts the comparison_direction string attribute into the XLA enum. The
// string is assumed to correspond to exactly one of the allowed strings
// representing the enum. This should have been checked in the op verify method.
static xla::ComparisonDirection Convert_comparison_direction(
    llvm::StringRef comparison_direction_string) {
  return xla::StringToComparisonDirection(comparison_direction_string.str())
      .value();
}

static xla::GatherDimensionNumbers Convert_dimension_numbers(
    mlir::mhlo::GatherDimensionNumbersAttr input) {
  xla::GatherDimensionNumbers output;

  auto offset_dims = input.getOffsetDims();
  std::copy(
      offset_dims.begin(), offset_dims.end(),
      tsl::protobuf::RepeatedFieldBackInserter(output.mutable_offset_dims()));

  auto collapsed_slice_dims = input.getCollapsedSliceDims();
  std::copy(collapsed_slice_dims.begin(), collapsed_slice_dims.end(),
            tsl::protobuf::RepeatedFieldBackInserter(
                output.mutable_collapsed_slice_dims()));

  auto start_index_map = input.getStartIndexMap();
  std::copy(start_index_map.begin(), start_index_map.end(),
            tsl::protobuf::RepeatedFieldBackInserter(
                output.mutable_start_index_map()));

  output.set_index_vector_dim(input.getIndexVectorDim());
  return output;
}

static xla::ScatterDimensionNumbers Convert_scatter_dimension_numbers(
    mlir::mhlo::ScatterDimensionNumbersAttr input) {
  xla::ScatterDimensionNumbers output;

  auto update_window_dims = input.getUpdateWindowDims();
  std::copy(update_window_dims.begin(), update_window_dims.end(),
            tsl::protobuf::RepeatedFieldBackInserter(
                output.mutable_update_window_dims()));

  auto inserted_window_dims = input.getInsertedWindowDims();
  std::copy(inserted_window_dims.begin(), inserted_window_dims.end(),
            tsl::protobuf::RepeatedFieldBackInserter(
                output.mutable_inserted_window_dims()));

  auto scatter_dims_to_operand_dims = input.getScatterDimsToOperandDims();
  std::copy(scatter_dims_to_operand_dims.begin(),
            scatter_dims_to_operand_dims.end(),
            tsl::protobuf::RepeatedFieldBackInserter(
                output.mutable_scatter_dims_to_operand_dims()));

  output.set_index_vector_dim(input.getIndexVectorDim());
  return output;
}

// Returns an OpSharding proto from the "sharding" attribute of the op. If the
// op doesn't have a sharding attribute or the sharding attribute is invalid,
// returns std::nullopt.
static std::optional<xla::OpSharding> CreateOpShardingFromAttribute(
    mlir::Operation* op) {
  auto shardingAttr = op->getAttrOfType<mlir::StringAttr>(kShardingAttr);
  if (!shardingAttr) return std::nullopt;
  return xla::ConvertSharding(shardingAttr.getValue());
}

// Returns a FrontendAttributes proto from the "frontend_attributes" attribute
// of the op. An empty FrontendAttributes proto is returned if an op does not
// have frontend attributes.
void ConstructFrontendAttributesFromAttribute(
    const mlir::DictionaryAttr& frontend_attributes_dict,
    xla::FrontendAttributes& frontend_attributes) {
  for (const auto& attr : frontend_attributes_dict)
    if (auto value_str_attr = attr.getValue().dyn_cast<mlir::StringAttr>())
      frontend_attributes.mutable_map()->insert(
          {attr.getName().str(), value_str_attr.getValue().str()});
}

static xla::FrontendAttributes CreateXlaFrontendAttributesFromOp(
    mlir::Operation* op) {
  xla::FrontendAttributes frontend_attributes;
  auto frontend_attributes_dict =
      op->getAttrOfType<mlir::DictionaryAttr>(kFrontendAttributesAttr);
  if (!frontend_attributes_dict) return frontend_attributes;
  ConstructFrontendAttributesFromAttribute(frontend_attributes_dict,
                                           frontend_attributes);
  return frontend_attributes;
}

static void ExtractFrontendAttributesFromFunction(
    mlir::func::FuncOp function,
    llvm::SmallVectorImpl<std::optional<xla::FrontendAttributes>>* fe_attrs) {
  fe_attrs->resize(function.getNumArguments(), std::nullopt);
  for (int i = 0, end = function.getNumArguments(); i < end; ++i)
    if (auto fe_attr = function.getArgAttrOfType<mlir::DictionaryAttr>(
            i, kFrontendAttributesAttr)) {
      xla::FrontendAttributes frontend_attributes;
      ConstructFrontendAttributesFromAttribute(fe_attr, frontend_attributes);
      (*fe_attrs)[i] = frontend_attributes;
    }
}

static bool SomeOptionalShardingsAreSet(
    llvm::ArrayRef<std::optional<xla::OpSharding>> shardings) {
  return llvm::any_of(shardings,
                      [](const std::optional<xla::OpSharding>& sharding) {
                        return sharding.has_value();
                      });
}

// Extracts argument and result shardings from function.
static void ExtractShardingsFromFunction(
    mlir::func::FuncOp function,
    llvm::SmallVectorImpl<std::optional<xla::OpSharding>>* arg_shardings,
    llvm::SmallVectorImpl<std::optional<xla::OpSharding>>* ret_shardings) {
  arg_shardings->resize(function.getNumArguments(),
                        std::optional<xla::OpSharding>());
  for (int i = 0, end = function.getNumArguments(); i < end; ++i)
    if (auto sharding =
            function.getArgAttrOfType<mlir::StringAttr>(i, kShardingAttr))
      (*arg_shardings)[i] = xla::ConvertSharding(sharding.getValue());

  ret_shardings->resize(function.getNumResults(),
                        std::optional<xla::OpSharding>());
  for (int i = 0, end = function.getNumResults(); i < end; ++i)
    if (auto sharding =
            function.getResultAttrOfType<mlir::StringAttr>(i, kShardingAttr))
      (*ret_shardings)[i] = xla::ConvertSharding(sharding.getValue());
}

// Creates a tuple sharding with the given shardings if at least one is present.
//
// Adds replicated shardings for any missing tuple shardings.
std::optional<xla::OpSharding> CreateTupleSharding(
    llvm::ArrayRef<std::optional<xla::OpSharding>> tuple_shardings) {
  if (tuple_shardings.empty() ||
      !SomeOptionalShardingsAreSet(tuple_shardings)) {
    return std::nullopt;
  }
  xla::OpSharding sharding;
  sharding.set_type(xla::OpSharding::TUPLE);
  for (const std::optional<xla::OpSharding>& tuple_sharding : tuple_shardings) {
    if (tuple_sharding) {
      *sharding.add_tuple_shardings() = *tuple_sharding;
    } else {
      xla::OpSharding fallback_sharding;
      fallback_sharding.set_type(xla::OpSharding::REPLICATED);
      *sharding.add_tuple_shardings() = fallback_sharding;
    }
  }

  return sharding;
}

// Returns the flattened result shardings of the given `op_sharding`, i.e.,
// either:
// - an empty vector if `sharding` is not present.
// - the tuple shardings in `sharding` if it has type TUPLE.
// - otherwise, returns a vector with `sharding` itself.
llvm::SmallVector<std::optional<xla::OpSharding>> GetResultShardings(
    std::optional<xla::OpSharding> op_sharding) {
  if (!op_sharding) {
    return {};
  }
  llvm::SmallVector<std::optional<xla::OpSharding>> res_shardings;
  if (op_sharding->type() == xla::OpSharding::TUPLE) {
    res_shardings.assign(op_sharding->tuple_shardings().begin(),
                         op_sharding->tuple_shardings().end());
  } else {
    res_shardings.push_back(op_sharding);
  }
  return res_shardings;
}

namespace mlir {
namespace {
class ConvertToHloModule {
 public:
  using ValueLoweringMap = llvm::DenseMap<Value, xla::XlaOp>;
  using FunctionLoweringMap =
      llvm::DenseMap<mlir::func::FuncOp, xla::XlaComputation>;

  // If use_tuple_args is true, then the entry function's arguments are
  // converted to a tuple and passed as a single parameter.
  // Similarly, if return tuple is true, then the entry function's return values
  // are converted to a tuple even when there is only a single return value.
  // Multiple return values are always converted to a tuple and returned as a
  // single value.
  explicit ConvertToHloModule(mlir::ModuleOp module,
                              xla::XlaBuilder& module_builder,
                              bool use_tuple_args, bool return_tuple,
                              MlirToHloConversionOptions options)
      : module_(module),
        module_builder_(module_builder),
        use_tuple_args_(use_tuple_args),
        return_tuple_(return_tuple),
        options_(options) {}

  // Perform the lowering to XLA. This function returns failure if an error was
  // encountered.
  //
  // TODO(hinsu): Check for dynamic shapes and exit instead of crashing.
  LogicalResult Run() {
    auto main = module_.lookupSymbol<mlir::func::FuncOp>("main");
    if (!main)
      return module_.emitError(
          "conversion requires module with `main` function");

    for (auto func : module_.getOps<func::FuncOp>()) {
      if (func.empty()) continue;
      if (failed(RunOnFunction(func))) return failure();
    }
    return success();
  }

  // Lower a specific function to HLO.
  LogicalResult RunOnFunction(mlir::func::FuncOp f);

  // Lower a `mlir::Region` to a `XlaComputation`
  LogicalResult LowerRegionAsComputation(
      mlir::Region* region, xla::XlaComputation* func,
      std::optional<llvm::ArrayRef<mlir::Value>> implicit_operands =
          std::nullopt,
      bool ensure_single_arg = false,
      llvm::ArrayRef<std::optional<xla::OpSharding>> arg_shardings = {},
      llvm::ArrayRef<std::optional<xla::OpSharding>> ret_shardings = {});

  // Lower a single `Block` to a `XlaComputation`
  LogicalResult LowerBasicBlockAsFunction(
      Block* block, xla::XlaBuilder* builder, bool is_entry_function,
      bool ensure_single_arg,
      const std::vector<bool>& entry_args_same_across_replicas,
      llvm::ArrayRef<std::optional<xla::OpSharding>> arg_shardings,
      llvm::ArrayRef<std::optional<xla::OpSharding>> ret_shardings,
      llvm::ArrayRef<std::optional<xla::FrontendAttributes>> fe_attrs,
      xla::XlaComputation* result,
      std::optional<llvm::ArrayRef<mlir::Value>> implicit_operands =
          std::nullopt);

  ::xla::HloModuleProto ConsumeMainProto() {
    auto main = module_.lookupSymbol<mlir::func::FuncOp>("main");
    // This is an invariant check as Run returns failure if there is no main
    // function and so the main proto shouldn't be consumed in that case.
    CHECK(main) << "requires module to have main function";  // Crash Ok.
    return lowered_computation_[main].proto();
  }

  // Lower function call to HLO call instruction
  LogicalResult LowerFunctionCall(
      mlir::func::CallOp call_op, xla::XlaBuilder* builder,
      ConvertToHloModule::ValueLoweringMap* value_lowering);

  // Look up a symbol with the specified name, returning null if no such name
  // exists.
  func::FuncOp LookUpSymbol(FlatSymbolRefAttr symbol) {
    return module_.lookupSymbol<mlir::func::FuncOp>(symbol);
  }

  // Get Reference to lowered XLA computation for a function.
  xla::XlaComputation& GetLoweredComputation(func::FuncOp func) {
    return lowered_computation_[func];
  }

  LogicalResult Lower(
      mlir::Operation* inst, bool is_entry_function,
      llvm::ArrayRef<std::optional<xla::OpSharding>> ret_shardings,
      xla::XlaBuilder* builder,
      ConvertToHloModule::ValueLoweringMap* value_lowering,
      xla::XlaOp* return_value);

  const MlirToHloConversionOptions& GetOptions() const { return options_; }

  xla::StackFrameIndexProto BuildStackFramesIndexProto() {
    return stack_frame_indexes_builder_.Build();
  }

 private:
  LogicalResult SetEntryTupleShapesAndLeafReplication(
      Block* block, const std::vector<bool>& entry_args_same_across_replicas,
      llvm::SmallVectorImpl<xla::Shape>* arg_shapes,
      std::vector<bool>* leaf_replication);

  LogicalResult SetEntryTupleShardings(
      Block* block, xla::XlaBuilder* builder,
      llvm::ArrayRef<std::optional<xla::OpSharding>> arg_shardings,
      llvm::SmallVectorImpl<xla::Shape>* arg_shapes);

  // The module being lowered.
  mlir::ModuleOp module_;

  // The top-level XlaBuilder.
  xla::XlaBuilder& module_builder_;

  // Common stack frame index builder.
  mlir::StackFrameIndexBuilder stack_frame_indexes_builder_;

  // Map between function and lowered computation.
  FunctionLoweringMap lowered_computation_;

  // Whether the entry function should take a single tuple as input.
  bool use_tuple_args_;

  // Whether to always return a tuple.
  bool return_tuple_;

  // Unique suffix to give to the name of the next lowered region.
  size_t region_id_ = 0;

  MlirToHloConversionOptions options_;
};

}  // namespace
}  // namespace mlir

namespace {

struct OpLoweringContext {
  llvm::DenseMap<mlir::Value, xla::XlaOp>* values;
  mlir::ConvertToHloModule* converter;
  xla::XlaBuilder* builder;
  mlir::StackFrameIndexBuilder* frame_index_builder;
};

mlir::LogicalResult GetTuple(mlir::Operation* op,
                             mlir::Operation::operand_range values,
                             OpLoweringContext ctx,
                             llvm::SmallVectorImpl<xla::XlaOp>& results) {
  results.reserve(values.size());
  for (mlir::Value value : values) {
    if (failed(GetXlaOp(value, *ctx.values, &results.emplace_back(), op)))
      return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult GetXlaOps(mlir::Operation* op,
                              llvm::ArrayRef<mlir::Value> values,
                              OpLoweringContext ctx,
                              llvm::SmallVectorImpl<xla::XlaOp>& results) {
  results.reserve(values.size());
  for (mlir::Value value : values) {
    if (failed(GetXlaOp(value, *ctx.values, &results.emplace_back(), op)))
      return mlir::failure();
  }
  return mlir::success();
}

// Checks that the results of `op` are simply returned at the end of this
// function rather than used by other ops in the same function.
//
// Used to check that new-style async ops on computations that contain sync
// versions of old-style async ops can be exported by downgrading to old-style
// async ops.
bool SimplyReturnedOp(mlir::Operation* op) {
  for (auto operand : op->getOperands()) {
    if (!llvm::isa<mlir::BlockArgument>(operand)) return false;
  }

  auto users = op->getUsers();
  if (users.empty()) return false;

  auto first_user = *users.begin();
  for (auto user : users) {
    if (first_user != user) return false;
  }

  if (llvm::isa<mlir::func::ReturnOp>(first_user)) return true;
  return false;
}

void BuildGetTupleElementsForTupleResults(mlir::Operation* op, xla::XlaOp tuple,
                                          OpLoweringContext ctx) {
  const std::optional<xla::OpSharding>& tuple_sharding =
      ctx.builder->sharding();
  if (tuple_sharding.has_value()) {
    assert(op->getNumResults() == tuple_sharding->tuple_shardings_size());
    for (auto [index, result] : llvm::enumerate(op->getResults())) {
      xla::XlaScopedShardingAssignment scoped_sharding(
          ctx.builder, tuple_sharding->tuple_shardings(index));
      (*ctx.values)[result] = xla::GetTupleElement(tuple, index);
    }
  } else {
    xla::XlaScopedShardingAssignment scoped_sharding(ctx.builder, std::nullopt);
    for (auto [index, result] : llvm::enumerate(op->getResults())) {
      (*ctx.values)[result] = xla::GetTupleElement(tuple, index);
    }
  }
}

}  // namespace

namespace mlir {
namespace mhlo {
namespace {
LogicalResult ExportXlaOp(CollectiveBroadcastOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp operand;
  if (failed(GetXlaOp(op.getOperand(), value_map, &operand, op)))
    return failure();
  value_map[op->getResult(0)] = xla::CollectiveBroadcast(
      operand, Convert_replica_groups(op.getReplicaGroups()),
      Convert_channel_handle(op.getChannelHandle()));

  return success();
}

LogicalResult ExportXlaOp(CompositeOp, OpLoweringContext) {
  // TODO: b/328526226 - Implement MHLO export for CompositeOp.
  return failure();
}

LogicalResult ExportXlaOp(ComputeReshapeShapeOp, OpLoweringContext) {
  // This op should've been removed during PrepareForExport.
  return failure();
}

LogicalResult ExportXlaOp(CstrReshapableOp, OpLoweringContext) {
  // This op should've been removed during PrepareForExport.
  return failure();
}

LogicalResult ExportXlaOp(DynamicBroadcastInDimOp op, OpLoweringContext ctx) {
  // This op has no expression in the legacy export format.
  return failure();
}

LogicalResult ExportXlaOp(DynamicConvOp op, OpLoweringContext ctx) {
  // TODO(b/264240901): Implement MHLO export for DynamicConvOp.
  return failure();
}

LogicalResult ExportXlaOp(DynamicGatherOp op, OpLoweringContext ctx) {
  // TODO(b/264240901): Implement MHLO export for DynamicGatherOp.
  return failure();
}

LogicalResult ExportXlaOp(DynamicIotaOp op, OpLoweringContext ctx) {
  // TODO(b/264240901): Implement MHLO export for DynamicIotaOp.
  return failure();
}

LogicalResult ExportXlaOp(DynamicPadOp op, OpLoweringContext ctx) {
  // TODO(b/264240901): Implement MHLO export for DynamicPadOp.
  return failure();
}

LogicalResult ExportXlaOp(DynamicReshapeOp op, OpLoweringContext ctx) {
  auto resultType = op.getResult().getType().dyn_cast<RankedTensorType>();
  if (!resultType) return op->emitOpError() << "expected ranked result";
  auto resultBounds = hlo::encodingToBounds(resultType.getEncoding());
  if (resultBounds.empty())
    return op->emitOpError() << "expected bounded result";
  auto shapeType = op.getOutputShape().getType().dyn_cast<RankedTensorType>();
  if (!shapeType || !shapeType.getElementType().isInteger(32))
    return op->emitOpError() << "expected output shape to be tensor<Nxi32>";

  auto& value_map = *ctx.values;
  xla::XlaOp operand;
  xla::XlaOp outputShape;
  if (failed(GetXlaOp(op.getOperand(), value_map, &operand, op)))
    return failure();
  if (failed(GetXlaOp(op.getOutputShape(), value_map, &outputShape, op)))
    return failure();

  SmallVector<xla::XlaOp> dimSizes;
  SmallVector<int64_t> newSizeBounds;
  std::vector<bool> dimsAreDynamic;
  for (auto i = 0; i < resultType.getRank(); ++i) {
    auto runtimeSizeX1 = xla::Slice(outputShape, {i}, {i + 1}, {1});
    dimSizes.push_back(xla::Reshape(runtimeSizeX1, {}));

    auto dimSize = resultType.getDimSize(i);
    auto dimBound = resultBounds[i];
    if (!hlo::isStaticDimSize(dimSize) && !hlo::isStaticDimSize(dimBound))
      return op->emitOpError() << "unbounded dynamism is not supported";
    newSizeBounds.push_back(hlo::isStaticDimSize(dimSize) ? dimSize : dimBound);
    dimsAreDynamic.push_back(!hlo::isStaticDimSize(dimSize));
  }
  value_map[op] =
      xla::DynamicReshape(operand, dimSizes, newSizeBounds, dimsAreDynamic);
  return success();
}

LogicalResult ExportXlaOp(RealDynamicSliceOp op, OpLoweringContext ctx) {
  // TODO(b/264240901): Implement MHLO export for RealDynamicSliceOp.
  return failure();
}

mlir::LogicalResult ExportXlaOp(mlir::mhlo::CopyOp op, OpLoweringContext ctx) {
  // If it's the only thing in a function we assume it's part of an async copy
  // op
  if (op.getCrossProgramPrefetchIndex() && !SimplyReturnedOp(op))
    return op->emitOpError() << "synchronous CopyOp should not include "
                                "cross_program_prefetch_index attribute.";
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp xla_arg_0;
  if (failed(
          GetXlaOp(*op.getODSOperands(0).begin(), value_map, &xla_arg_0, op)))
    return mlir::failure();
  auto xla_result = xla::Copy(Unwrap(xla_arg_0));
  value_map[result] = xla_result;
  return mlir::success();
}

LogicalResult ExportXlaOp(AddDependencyOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp token;
  xla::XlaOp operand;
  if (failed(GetXlaOp(op.getToken(), value_map, &token, op))) return failure();
  if (failed(GetXlaOp(op.getOperand(), value_map, &operand, op)))
    return failure();
  auto operand_shape = ctx.builder->GetShape(operand).value();
  value_map[op] = xla::internal::XlaBuilderFriend::BuildAddDependency(
      ctx.builder, operand, token, operand_shape);
  return success();
}

LogicalResult ExportXlaOp(AllGatherOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;

  SmallVector<xla::XlaOp> operands;
  if (failed(GetTuple(op.getOperation(), op.getOperands(), ctx, operands))) {
    return failure();
  }

  mlir::FailureOr<xla::Shape> shape_or = ExtractXlaShape(op.getOperation());
  if (failed(shape_or)) return failure();

  auto all_gather_dim = op.getAllGatherDim();
  int64_t shard_count = 0;
  for (size_t i = 0; i < operands.size(); ++i) {
    TensorType operand_type = op.getOperand(i).getType().cast<TensorType>();
    TensorType result_type = op.getType(i).cast<TensorType>();
    if (!operand_type.hasStaticShape() || !result_type.hasStaticShape())
      return failure();
    if (i == 0) {
      shard_count = result_type.getDimSize(all_gather_dim) /
                    operand_type.getDimSize(all_gather_dim);
    }
  }

  if (shape_or->IsTuple()) {
    std::optional<xla::Layout> layout = std::nullopt;
    if (shape_or->has_layout()) {
      layout = shape_or->layout();
    }
    auto tuple = xla::AllGatherTuple(
        operands, all_gather_dim, shard_count,
        Convert_replica_groups(op.getReplicaGroups()),
        Convert_channel_handle(op.getChannelHandle()), layout,
        Convert_use_global_device_ids(op.getUseGlobalDeviceIds()));
    BuildGetTupleElementsForTupleResults(op, tuple, ctx);
  } else {
    value_map[op->getResults()[0]] = xla::AllGather(
        operands[0], all_gather_dim, shard_count,
        Convert_replica_groups(op.getReplicaGroups()),
        Convert_channel_handle(op.getChannelHandle()), std::nullopt,
        Convert_use_global_device_ids(op.getUseGlobalDeviceIds()));
  }

  return success();
}

LogicalResult ExportXlaOp(AllReduceOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaComputation computation;
  if (failed(ctx.converter->LowerRegionAsComputation(&op.getComputation(),
                                                     &computation))) {
    return failure();
  }

  SmallVector<xla::XlaOp> operands;
  if (failed(GetTuple(op.getOperation(), op.getOperands(), ctx, operands)))
    return failure();

  mlir::FailureOr<xla::Shape> shape_or = ExtractXlaShape(op.getOperation());
  if (failed(shape_or)) return failure();
  if (shape_or->IsTuple()) {
    std::optional<xla::Shape> shape_with_layout = std::nullopt;
    if (shape_or->has_layout()) shape_with_layout = shape_or.value();
    auto tuple = xla::AllReduceTuple(
        operands, computation, Convert_replica_groups(op.getReplicaGroups()),
        Convert_channel_handle(op.getChannelHandle()), shape_with_layout,
        Convert_use_global_device_ids(op.getUseGlobalDeviceIds()));
    BuildGetTupleElementsForTupleResults(op, tuple, ctx);
  } else {
    value_map[op->getResults()[0]] = xla::AllReduce(
        operands[0], computation, Convert_replica_groups(op.getReplicaGroups()),
        Convert_channel_handle(op.getChannelHandle()), std::nullopt,
        Convert_use_global_device_ids(op.getUseGlobalDeviceIds()));
  }

  return success();
}

LogicalResult ExportXlaOp(AllToAllOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;

  SmallVector<xla::XlaOp> operands;
  if (failed(GetTuple(op.getOperation(), op.getOperands(), ctx, operands))) {
    return failure();
  }

  mlir::FailureOr<xla::Shape> shape_or = ExtractXlaShape(op.getOperation());
  if (failed(shape_or)) return failure();
  if (shape_or->IsTuple()) {
    std::optional<xla::Layout> layout = std::nullopt;
    if (shape_or->has_layout()) {
      layout = shape_or->layout();
    }
    auto tuple = xla::AllToAllTuple(
        operands, Convert_replica_groups(op.getReplicaGroups()), layout,
        Convert_channel_handle(op.getChannelHandle()));
    BuildGetTupleElementsForTupleResults(op, tuple, ctx);
  } else {
    // ArrayAllToAll always has exactly one operand (checked in the verifier).
    value_map[op->getResults()[0]] = xla::AllToAll(
        operands[0], *op.getSplitDimension(), *op.getConcatDimension(),
        *op.getSplitCount(), Convert_replica_groups(op.getReplicaGroups()),
        /*layout=*/std::nullopt, Convert_channel_handle(op.getChannelHandle()));
  }

  return success();
}

LogicalResult ExportXlaOp(ReduceScatterOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp operand;
  if (failed(GetXlaOp(op.getOperand(), value_map, &operand, op)))
    return failure();
  TensorType operand_type = op.getOperand().getType().cast<TensorType>();
  TensorType result_type = op.getType();
  if (!operand_type.hasStaticShape() || !result_type.hasStaticShape())
    return failure();
  auto scatter_dim = op.getScatterDimension();
  int64_t shard_count = operand_type.getDimSize(scatter_dim) /
                        result_type.getDimSize(scatter_dim);

  xla::XlaComputation computation;
  if (failed(ctx.converter->LowerRegionAsComputation(&op.getComputation(),
                                                     &computation))) {
    return failure();
  }

  value_map[op] = xla::ReduceScatter(
      operand, computation, scatter_dim, shard_count,
      Convert_replica_groups(op.getReplicaGroups()),
      Convert_channel_handle(op.getChannelHandle()), std::nullopt,
      Convert_use_global_device_ids(op.getUseGlobalDeviceIds()));
  return success();
}

LogicalResult ExportXlaOp(AsyncStartOp op, OpLoweringContext ctx) {
  for (auto* user : op.getResult().getUsers()) {
    if (auto asyncOp = dyn_cast_or_null<AsyncDoneOp>(user)) {
      if (asyncOp.getCalledComputation() != op.getCalledComputation()) {
        return op.emitOpError()
               << "Users of AsyncStart's return value must have "
                  "the same called_computation";
      }
    } else if (auto asyncOp = dyn_cast_or_null<AsyncUpdateOp>(user)) {
      if (asyncOp.getCalledComputation() != op.getCalledComputation()) {
        return op.emitOpError()
               << "Users of AsyncStart's return value must have "
                  "the same called_computation";
      }
    } else {
      return op.emitOpError() << "Users of AsyncStart's return value must be "
                              << "async_update or async_done";
    }
  }

  auto& value_map = *ctx.values;

  Value result = op.getResult();
  llvm::SmallVector<xla::XlaOp> operands;
  if (failed(GetTuple(op, op.getInputs(), ctx, operands))) return failure();

  mlir::func::FuncOp callee = ctx.converter->LookUpSymbol(
      FlatSymbolRefAttr::get(op->getContext(), op.getCalledComputation()));

  auto all_gather_op =
      dyn_cast_or_null<AllGatherOp>(callee.getBody().front().front());
  if (all_gather_op && SimplyReturnedOp(all_gather_op)) {
    TensorType operand_type =
        all_gather_op.getOperand(0).getType().cast<TensorType>();
    TensorType result_type = all_gather_op.getType(0).cast<TensorType>();
    if (!operand_type.hasStaticShape() || !result_type.hasStaticShape())
      return failure();
    if (operands.size() != 1) return failure();
    auto all_gather_dim = all_gather_op.getAllGatherDim();
    int64_t shard_count = result_type.getDimSize(all_gather_dim) /
                          operand_type.getDimSize(all_gather_dim);
    value_map[result] = xla::internal::XlaBuilderFriend::BuildAllGatherStart(
        ctx.builder, operands[0], all_gather_dim, shard_count,
        Convert_replica_groups(all_gather_op.getReplicaGroups()),
        Convert_channel_handle(all_gather_op.getChannelHandle()),
        ExtractLayout(all_gather_op,
                      result_type.cast<RankedTensorType>().getRank()),
        Convert_use_global_device_ids(all_gather_op.getUseGlobalDeviceIds()));
    return success();
  }
  auto all_reduce_op =
      dyn_cast_or_null<AllReduceOp>(callee.getBody().front().front());
  if (all_reduce_op && SimplyReturnedOp(all_reduce_op)) {
    xla::XlaComputation computation;
    if (failed(ctx.converter->LowerRegionAsComputation(
            &all_reduce_op.getComputation(), &computation))) {
      return failure();
    }
    if (operands.size() != 1) return failure();
    value_map[result] = xla::internal::XlaBuilderFriend::BuildAllReduceStart(
        ctx.builder, operands[0], computation,
        Convert_replica_groups(all_reduce_op.getReplicaGroups()),
        Convert_channel_handle(all_reduce_op.getChannelHandle()), std::nullopt,
        Convert_use_global_device_ids(all_reduce_op.getUseGlobalDeviceIds()));
    return success();
  }
  auto collective_permute_op =
      dyn_cast_or_null<CollectivePermuteOp>(callee.getBody().front().front());
  if (collective_permute_op && SimplyReturnedOp(collective_permute_op)) {
    value_map[result] =
        xla::internal::XlaBuilderFriend::BuildCollectivePermuteStart(
            ctx.builder, operands[0],
            Convert_source_target_pairs(
                collective_permute_op.getSourceTargetPairs()),
            Convert_channel_handle(collective_permute_op.getChannelHandle()));
    return mlir::success();
  }
  auto copy_op = dyn_cast_or_null<CopyOp>(callee.getBody().front().front());
  if (copy_op && SimplyReturnedOp(copy_op)) {
    std::optional<int> cross_program_prefetch_index =
        copy_op.getCrossProgramPrefetchIndex()
            ? std::make_optional(*copy_op.getCrossProgramPrefetchIndex())
            : std::nullopt;
    value_map[result] = xla::internal::XlaBuilderFriend::BuildCopyStart(
        ctx.builder, operands[0], cross_program_prefetch_index);
    return mlir::success();
  }
  auto send_op = dyn_cast_or_null<SendOp>(callee.getBody().front().front());
  if (send_op && SimplyReturnedOp(send_op)) {
    xla::XlaOp operand;
    if (operands.size() == 2)
      operand = operands[0];
    else
      operand =
          Tuple(ctx.builder, absl::Span<const xla::XlaOp>(operands).subspan(
                                 0, operands.size() - 1));
    xla::XlaOp token = operands[operands.size() - 1];

    value_map[result] = xla::internal::XlaBuilderFriend::BuildSend(
        ctx.builder, operand, token,
        Convert_channel_handle(send_op.getChannelHandle()),
        send_op.getIsHostTransfer());
    return mlir::success();
  }
  auto recv_op = dyn_cast_or_null<RecvOp>(callee.getBody().front().front());
  if (recv_op && SimplyReturnedOp(recv_op)) {
    auto result_types = result.getType().cast<AsyncBundleType>().getTypes()[1];

    mlir::Type received_type = mlir::TupleType::get(op->getContext(), {});
    if (isa<TupleType>(result_types)) {
      received_type = result_types.cast<TupleType>().getType(0);
    }

    value_map[result] = xla::internal::XlaBuilderFriend::BuildRecv(
        ctx.builder, operands[0], xla::TypeToShape(received_type),
        Convert_channel_handle(recv_op.getChannelHandle()),
        recv_op.getIsHostTransfer());
    return mlir::success();
  }

  if (failed(ctx.converter->RunOnFunction(callee))) return failure();
  xla::XlaComputation& computation =
      ctx.converter->GetLoweredComputation(callee);
  computation.mutable_proto()->mutable_computations(0)->set_execution_thread(
      op.getExecutionThread().str());
  auto [xla_op, computation_id] =
      xla::internal::XlaBuilderFriend::BuildAsyncStart(
          ctx.builder, operands, op.getExecutionThread().str(), computation,
          xla::TypeToShape(result.getType()));
  value_map[result] = xla_op;
  computation.mutable_proto()->mutable_computations(0)->set_id(computation_id);
  return success();
}

LogicalResult ExportXlaOp(AsyncUpdateOp op, OpLoweringContext ctx) {
  if (!isa<AsyncStartOp, AsyncUpdateOp>(op.getBundle().getDefiningOp())) {
    auto theerror = op.emitError()
                    << "Defining op of AsyncUpdate's operand must be "
                    << "async_start or async_update";
    if (op.getBundle().getDefiningOp()) {
      return theerror << ", but got "
                      << op.getBundle().getDefiningOp()->getName();
    } else {
      return theerror << ".";
    }
  }

  for (auto* user : op.getResult().getUsers()) {
    if (auto asyncOp = dyn_cast_or_null<AsyncDoneOp>(user)) {
      if (asyncOp.getCalledComputation() != op.getCalledComputation()) {
        return op.emitOpError()
               << "Users of AsyncUpdate's return value must have "
                  "the same group_id and called_computation";
      }
    } else if (auto asyncOp = dyn_cast_or_null<AsyncUpdateOp>(user)) {
      if (asyncOp.getCalledComputation() != op.getCalledComputation()) {
        return op.emitOpError()
               << "Users of AsyncUpdate's return value must have "
                  "the same group_id and called_computation";
      }
    } else {
      return op.emitOpError() << "Users of AsyncUpdate's return value must be "
                              << "async_update or async_done";
    }
  }
  auto& value_map = *ctx.values;

  Value result = op.getResult();
  xla::XlaOp operand;
  if (failed(GetXlaOp(op.getBundle(), value_map, &operand, op)))
    return failure();

  mlir::func::FuncOp callee = ctx.converter->LookUpSymbol(
      FlatSymbolRefAttr::get(op->getContext(), op.getCalledComputation()));
  xla::XlaComputation& computation =
      ctx.converter->GetLoweredComputation(callee);
  value_map[result] = xla::internal::XlaBuilderFriend::BuildAsyncUpdate(
      ctx.builder, operand, op.getExecutionThread().str(),
      computation.proto().computations(0).id(),
      xla::TypeToShape(result.getType()));
  return success();
}

LogicalResult ExportXlaOp(AsyncDoneOp op, OpLoweringContext ctx) {
  if (!isa<AsyncStartOp, AsyncUpdateOp>(op.getBundle().getDefiningOp())) {
    auto theerror = op.emitError()
                    << "Defining op of AsyncDone's operand must be "
                    << "async_start or async_update";
    if (op.getBundle().getDefiningOp())
      return theerror << ", but got "
                      << op.getBundle().getDefiningOp()->getName();
    return theerror << ".";
  }

  auto& value_map = *ctx.values;

  xla::XlaOp operand;
  if (failed(GetXlaOp(op.getBundle(), value_map, &operand, op)))
    return failure();

  mlir::func::FuncOp callee = ctx.converter->LookUpSymbol(
      FlatSymbolRefAttr::get(op->getContext(), op.getCalledComputation()));

  auto all_gather_op =
      dyn_cast_or_null<AllGatherOp>(callee.getBody().front().front());
  if (all_gather_op && SimplyReturnedOp(all_gather_op)) {
    value_map[op.getResult(0)] =
        xla::internal::XlaBuilderFriend::BuildAllGatherDone(
            ctx.builder, operand, xla::TypeToShape(all_gather_op.getType(0)));
    return success();
  }
  auto all_reduce_op =
      dyn_cast_or_null<AllReduceOp>(callee.getBody().front().front());
  if (all_reduce_op && SimplyReturnedOp(all_reduce_op)) {
    value_map[op.getResult(0)] =
        xla::internal::XlaBuilderFriend::BuildAllReduceDone(
            ctx.builder, operand, xla::TypeToShape(all_reduce_op.getType(0)));
    return success();
  }
  auto collective_permute_op =
      dyn_cast_or_null<CollectivePermuteOp>(callee.getBody().front().front());
  if (collective_permute_op && SimplyReturnedOp(collective_permute_op)) {
    value_map[op.getResult(0)] =
        xla::internal::XlaBuilderFriend::BuildCollectivePermuteDone(
            ctx.builder, operand,
            xla::TypeToShape(collective_permute_op.getType()));
    return success();
  }
  auto copy_op = dyn_cast_or_null<CopyOp>(callee.getBody().front().front());
  if (copy_op && SimplyReturnedOp(copy_op)) {
    value_map[op.getResult(0)] = xla::internal::XlaBuilderFriend::BuildCopyDone(
        ctx.builder, operand, xla::TypeToShape(copy_op.getType()));
    return success();
  }
  auto send_op = dyn_cast_or_null<SendOp>(callee.getBody().front().front());
  if (send_op && SimplyReturnedOp(send_op)) {
    value_map[op.getResult(0)] = xla::internal::XlaBuilderFriend::BuildSendDone(
        ctx.builder, operand,
        Convert_channel_handle(send_op.getChannelHandle()),
        send_op.getIsHostTransfer());
    return success();
  }
  auto recv_op = dyn_cast_or_null<RecvOp>(callee.getBody().front().front());
  if (recv_op && SimplyReturnedOp(recv_op)) {
    auto result_types =
        op.getBundle().getType().cast<AsyncBundleType>().getTypes()[1];

    mlir::Type received_type = mlir::TupleType::get(op->getContext(), {});
    if (isa<TupleType>(result_types)) {
      received_type = result_types.cast<TupleType>().getType(0);
    }

    xla::XlaOp xla_recv = xla::internal::XlaBuilderFriend::BuildRecvDone(
        ctx.builder, operand, xla::TypeToShape(received_type),
        Convert_channel_handle(recv_op.getChannelHandle()),
        recv_op.getIsHostTransfer());
    if (op.getNumResults() == 1) {
      value_map[op.getResult(0)] = xla_recv;
    } else {
      BuildGetTupleElementsForTupleResults(op, xla_recv, ctx);
    }
    return success();
  }

  xla::XlaComputation& computation =
      ctx.converter->GetLoweredComputation(callee);
  std::vector<xla::Shape> subshapes;
  for (const auto& item : op.getResults().getType()) {
    subshapes.push_back(xla::TypeToShape(item));
  }
  xla::Shape data_shape = xla::ShapeUtil::MakeTupleShape(subshapes);

  xla::XlaOp exportedOp = xla::internal::XlaBuilderFriend::BuildAsyncDone(
      ctx.builder, operand, op.getExecutionThread().str(),
      computation.proto().computations(0).id(), data_shape);
  if (op.getNumResults() == 1) {
    value_map[op.getResult(0)] = exportedOp;
  } else {
    BuildGetTupleElementsForTupleResults(op, exportedOp, ctx);
  }
  return success();
}

LogicalResult ExportXlaOp(BitcastConvertOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp operand;
  if (failed(GetXlaOp(op.getOperand(), value_map, &operand, op)))
    return failure();

  value_map[op] = xla::BitcastConvertType(
      operand,
      xla::ConvertMlirTypeToPrimitiveType(getElementTypeOrSelf(op.getType())));
  return success();
}

LogicalResult ExportXlaOp(BroadcastInDimOp op, OpLoweringContext ctx) {
  auto type = op.getType().dyn_cast<RankedTensorType>();
  if (!type) return failure();
  auto& value_map = *ctx.values;
  xla::XlaOp operand;
  if (failed(GetXlaOp(op.getOperand(), value_map, &operand, op)))
    return failure();

  value_map[op] =
      BroadcastInDim(operand, Convert_ArrayRef(type.getShape()),
                     Convert_broadcast_dimensions(op.getBroadcastDimensions()));
  return success();
}

LogicalResult ExportXlaOp(StochasticConvertOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp operand, random;
  if (failed(GetXlaOp(op.getOperand(), value_map, &operand, op)))
    return failure();
  if (failed(GetXlaOp(op.getRandom(), value_map, &random, op)))
    return failure();

  value_map[op] = xla::StochasticConvertType(
      operand, random,
      xla::ConvertMlirTypeToPrimitiveType(getElementTypeOrSelf(op.getType())));
  return success();
}

LogicalResult ExportXlaOp(CosineOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp arg;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &arg, op)))
    return mlir::failure();
  auto xla_result = xla::Cos(Unwrap(arg));
  value_map[result] = xla_result;
  return mlir::success();
}

LogicalResult ExportXlaOp(TanOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp arg;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &arg, op)))
    return mlir::failure();
  auto xla_result = xla::Tan(Unwrap(arg));
  value_map[result] = xla_result;
  return mlir::success();
}

LogicalResult ExportXlaOp(DotOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp lhs, rhs;
  if (failed(GetXlaOp(op.getLhs(), value_map, &lhs, op)))
    return mlir::failure();
  if (failed(GetXlaOp(op.getRhs(), value_map, &rhs, op)))
    return mlir::failure();
  xla::PrimitiveType preferred_element_type =
      xla::ConvertMlirTypeToPrimitiveType(getElementTypeOrSelf(op.getType()));
  value_map[op] = xla::Dot(
      lhs, rhs, Unwrap(Convert_precision_config(op.getPrecisionConfig())),
      preferred_element_type);
  return mlir::success();
}

LogicalResult ExportXlaOp(DotGeneralOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp lhs, rhs;
  if (failed(GetXlaOp(op.getLhs(), value_map, &lhs, op)))
    return mlir::failure();
  if (failed(GetXlaOp(op.getRhs(), value_map, &rhs, op)))
    return mlir::failure();
  xla::PrimitiveType preferred_element_type =
      xla::ConvertMlirTypeToPrimitiveType(getElementTypeOrSelf(op.getType()));
  value_map[op] = xla::DotGeneral(
      lhs, rhs, Convert_dot_dimension_numbers(op.getDotDimensionNumbers()),
      Unwrap(Convert_precision_config(op.getPrecisionConfig())),
      preferred_element_type);
  return mlir::success();
}

LogicalResult ExportXlaOp(SparseDotOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp lhs, rhs;
  if (failed(GetXlaOp(op.getLhs(), value_map, &lhs, op)))
    return mlir::failure();
  if (failed(GetXlaOp(op.getRhs(), value_map, &rhs, op)))
    return mlir::failure();
  xla::PrimitiveType preferred_element_type =
      xla::ConvertMlirTypeToPrimitiveType(getElementTypeOrSelf(op.getType()));

  llvm::SmallVector<xla::XlaOp> sparse_meta;
  if (failed(GetTuple(op, op.getMeta(), ctx, sparse_meta))) return failure();
  std::vector<xla::SparsityDescriptor> sparsity;
  if (op.getLhsSparsity().has_value()) {
    sparsity.push_back(
        Convert_sparsity_descriptor(*op.getLhsSparsity(), /*is_lhs=*/true));
  }
  if (op.getRhsSparsity().has_value()) {
    sparsity.push_back(
        Convert_sparsity_descriptor(*op.getRhsSparsity(), /*is_lhs=*/false));
  }

  value_map[op] =
      xla::SparseDot(lhs, rhs, absl::MakeSpan(sparse_meta), sparsity,
                     Convert_dot_dimension_numbers(op.getDotDimensionNumbers()),
                     Unwrap(Convert_precision_config(op.getPrecisionConfig())),
                     preferred_element_type);
  return mlir::success();
}

LogicalResult ExportXlaOp(DomainOp op, OpLoweringContext ctx) {
  auto& valueMap = *ctx.values;

  xla::Shape shape = xla::TypeToShape(op.getResult().getType());
  xla::XlaOp operand;
  if (failed(GetXlaOp(op.getOperand(), valueMap, &operand, op)))
    return failure();

  auto entry = xla::ConvertSharding(op.getEntryMetadata());
  if (!entry) return failure();
  auto exit = xla::ConvertSharding(op.getExitMetadata());
  if (!exit) return failure();

  valueMap[op] = xla::internal::XlaBuilderFriend::BuildDomain(
      ctx.builder, operand, *exit, *entry, shape);
  return success();
}

LogicalResult ExportXlaOp(IfOp op, OpLoweringContext ctx) {
  xla::XlaComputation true_branch;
  xla::XlaComputation false_branch;
  auto& value_map = *ctx.values;

  // mhlo.IfOp does not have any operands or blocks-arguments. The computation
  // inside the region-blocks use implicit captures of values defined above.
  // In order to create the xla parameters for functions corresponding to
  // IfOp regions, we need to infer the a region-block's arguments, using all
  // the values used in the region but defined above. Note that in case there
  // are zero implicit capture for a region, we use an empty tuple as the xla
  // parameter.
  //
  // Note that the implicit values used in true and false branch regions might
  // be different and, as a result, the xla parameters for the corresponding
  // regions could have different shapes.
  llvm::SetVector<mlir::Value> implicit_true_operand_set,
      implicit_false_operand_set;
  getUsedValuesDefinedAbove(op.getTrueBranch(), op.getTrueBranch(),
                            implicit_true_operand_set);
  getUsedValuesDefinedAbove(op.getFalseBranch(), op.getFalseBranch(),
                            implicit_false_operand_set);

  llvm::SmallVector<mlir::Value> implicit_true_operands(
      implicit_true_operand_set.begin(), implicit_true_operand_set.end());
  llvm::SmallVector<mlir::Value> implicit_false_operands(
      implicit_false_operand_set.begin(), implicit_false_operand_set.end());

  // Create xla parameters for functions corresponding to ifOp regions using the
  // implicit captures operands. Also export the instructions within those
  // regions.
  if (failed(ctx.converter->LowerRegionAsComputation(
          &op.getTrueBranch(), &true_branch,
          llvm::ArrayRef(implicit_true_operands),
          /*ensure_single_arg*/ true)) ||
      failed(ctx.converter->LowerRegionAsComputation(
          &op.getFalseBranch(), &false_branch,
          llvm::ArrayRef(implicit_false_operands),
          /*ensure_single_arg*/ true))) {
    return failure();
  }

  // Create the Xla pred argument.
  xla::XlaOp pred;
  if (failed(GetXlaOp(op.getPred(), value_map, &pred, op))) return failure();

  // Create the true branch Xla argument.
  llvm::SmallVector<xla::XlaOp> true_args;
  if (failed(GetXlaOps(op, implicit_true_operands, ctx, true_args)))
    return failure();
  xla::XlaOp true_arg =
      true_args.size() == 1 ? true_args[0] : Tuple(ctx.builder, true_args);

  // Create the false branch Xla argument.
  llvm::SmallVector<xla::XlaOp> false_args;
  if (failed(GetXlaOps(op, implicit_false_operands, ctx, false_args)))
    return failure();
  xla::XlaOp false_arg =
      false_args.size() == 1 ? false_args[0] : Tuple(ctx.builder, false_args);

  // Create XLA Conditional op.
  auto ifop =
      xla::Conditional(pred, true_arg, true_branch, false_arg, false_branch);

  // mhlo.IfOp have multiple returns, untuple all the results of XLA's.
  if (op.getNumResults() == 1) {
    value_map[op.getResult(0)] = ifop;
  } else {
    BuildGetTupleElementsForTupleResults(op, ifop, ctx);
  }

  return success();
}

LogicalResult ExportXlaOp(CaseOp op, OpLoweringContext ctx) {
  llvm::DenseMap<mlir::Value, xla::XlaOp>& value_map = *ctx.values;
  // OperandRange operands = op.branch_operands();
  MutableArrayRef<Region> branches = op.getBranches();
  llvm::SmallVector<xla::XlaOp, 4> branch_operands(branches.size());
  std::vector<xla::XlaComputation> computations(branches.size());
  std::vector<xla::XlaComputation*> computations_p(branches.size());

  // mhlo.CaseOp does not have any operands or blocks-arguments. The computation
  // inside the region-blocks use implicit captures of values defined above.
  // In order to create the xla parameters for functions corresponding to
  // CaseOp regions, we need to infer the a region-block's arguments, using all
  // the values used in the region but defined above. Note that in case there
  // are zero implicit captures for a region, we use an empty tuple as the xla
  // parameter.
  //
  // Note that the implicit values used in the regions might
  // be different and, as a result, the xla parameters for the corresponding
  // regions could have different shapes.
  for (unsigned i = 0; i < branches.size(); ++i) {
    llvm::SetVector<mlir::Value> implicit_operand_set;
    getUsedValuesDefinedAbove(branches[i], branches[i], implicit_operand_set);
    llvm::SmallVector<mlir::Value> implicit_operands(
        implicit_operand_set.begin(), implicit_operand_set.end());

    // Create the branches[i]'s Xla argument.
    llvm::SmallVector<xla::XlaOp> args;
    if (failed(GetXlaOps(op, implicit_operands, ctx, args))) return failure();
    branch_operands[i] = args.size() == 1 ? args[0] : Tuple(ctx.builder, args);

    // Create xla parameters for functions corresponding to region branches[i]
    // using the implicit captures operands. Also export the instructions within
    // that region.
    computations_p[i] = &computations[i];
    if (failed(ctx.converter->LowerRegionAsComputation(
            &branches[i], computations_p[i], llvm::ArrayRef(implicit_operands),
            /*ensure_single_arg*/ true)))
      return failure();
  }

  xla::XlaOp index;
  if (failed(GetXlaOp(op.getIndex(), value_map, &index, op))) return failure();

  xla::XlaOp caseop = xla::Conditional(index, computations_p, branch_operands);

  // mhlo.CaseOp have multiple returns, untuple all the results of XLA's.
  if (op.getNumResults() == 1) {
    value_map[op.getResult(0)] = caseop;
  } else {
    BuildGetTupleElementsForTupleResults(op, caseop, ctx);
  }
  return success();
}

// Specialize CompareOp export to set broadcast_dimensions argument.
mlir::LogicalResult ExportXlaOp(mlir::mhlo::CompareOp op,
                                OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp lhs, rhs;
  if (failed(GetXlaOp(op.getLhs(), value_map, &lhs, op)))
    return mlir::failure();
  if (failed(GetXlaOp(op.getRhs(), value_map, &rhs, op)))
    return mlir::failure();
  auto dir = Convert_comparison_direction(
      mlir::mhlo::stringifyComparisonDirection(op.getComparisonDirection()));
  auto type_attr = op.getCompareTypeAttr();

  xla::XlaOp xla_result;
  if (type_attr && type_attr.getValue() != mlir::mhlo::ComparisonType::NOTYPE) {
    auto type = xla::StringToComparisonType(
                    stringifyComparisonType(type_attr.getValue()).str())
                    .value();
    xla_result = xla::Compare(lhs, rhs, /*broadcast_dimensions=*/{}, dir, type);
  } else {
    xla_result = xla::Compare(lhs, rhs, dir);
  }
  value_map[op] = xla_result;
  return mlir::success();
}

LogicalResult ExportXlaOp(ConstantOp op, OpLoweringContext ctx) {
  return failure();
}

LogicalResult ExportXlaOp(mlir::mhlo::ConvolutionOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp lhs, rhs;
  if (failed(GetXlaOp(op.getLhs(), value_map, &lhs, op)))
    return mlir::failure();
  if (failed(GetXlaOp(op.getRhs(), value_map, &rhs, op)))
    return mlir::failure();
  xla::PrimitiveType preferred_element_type =
      xla::ConvertMlirTypeToPrimitiveType(getElementTypeOrSelf(op.getType()));
  xla::XlaOp xla_result = xla::ConvGeneralDilated(
      lhs, rhs, Convert_window_strides(op.getWindowStrides()),
      Convert_padding(op.getPadding()),
      Convert_lhs_dilation(op.getLhsDilation()),
      Convert_rhs_dilation(op.getRhsDilation()),
      xla::ConvertConvDimensionNumbers(op.getDimensionNumbers()),
      Convertuint64_t(op.getFeatureGroupCount()),
      Convertuint64_t(op.getBatchGroupCount()),
      Unwrap(Convert_precision_config(op.getPrecisionConfig())),
      preferred_element_type, Convert_window_reversal(op.getWindowReversal()));
  value_map[op] = xla_result;
  return mlir::success();
}

LogicalResult ExportXlaOp(ConvertOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp operand;
  if (failed(GetXlaOp(op.getOperand(), value_map, &operand, op)))
    return failure();

  value_map[op] = xla::ConvertElementType(
      operand,
      xla::ConvertMlirTypeToPrimitiveType(getElementTypeOrSelf(op.getType())));
  return success();
}

LogicalResult ExportXlaOp(CustomCallOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  llvm::SmallVector<xla::XlaOp> args;
  if (failed(GetTuple(op, op.getInputs(), ctx, args))) return failure();

  // Specially handle custom_calls from StableHLO that need stability guarantees
  // that XLA doesn't provide at the moment.
  //
  // In particular, we need 6mo backward compat and 1mo forward compat. This
  // will be provided by the StableHLO team by updating the following lowering.
  // This lowering provides that compatibility guarantee, lowering to the
  // appropriate HLO as the HLO implementing this custom_call may change.
  //
  // The only custom_call covered by the guarantee right now is ApproxTopK.
  // This means that any custom_call with call_target_name = "ApproxTopK"
  // written against the specification below will continue to behave as
  // described within the compatibility window.
  //
  // The attributes supported by the ApproxTopK custom_call are:
  //
  //  - called_computation : This indicates the comparator for scoring entries
  //  - has_side_effect: always False
  //  - api_version : always 4, the typed FFI API
  //  - backend_config : The actual arguments to ApproxTopK. This includes
  //    + top_k:i64 : the number of results to return
  //    + reduction_dim:i64 : which dimension to search for the top k elements
  //    + recall_target:f32: the expected number of top-k entries returned,
  //        divided by k.
  //    + aggregate_to_topk:bool : When true, aggregates approximate results to
  //        top-k. When false, returns the approximate results. The number of
  //        the approximate results is implementation defined and is greater
  //        equals to the specified `k`.
  //    + reduction_input_size_override:i64 : When set to a nonnegative value,
  //        it overrides the size determined by `input[reduction_dim]` for
  //        evaluating the recall. This option is useful when the given
  //        `input` is only a subset of the overall computation in SPMD or
  //        distributed pipelines, where the true input size cannot be deferred
  //        by the `input` shape.
  //    + is_fallback:bool : use the CPU/GPU fallback instead of the TPU
  //        implementation that uses PartialReduce (optional)
  //
  // The operands are a sequence of inputs over which to search, followed
  // by a list of initial values for each tensor in the first
  // list. Thus, we must have an even number of operands consisting of a
  // sequence of tensors with the same shape followed by the same number of
  // rank-0 tensors with the same element types as the corresponding inputs.
  // NB. Here, We mean "shape" in the StableHLO/MHLO sense of the dimensions of
  // a the tensor, excluding the element type, not the the HLO sense, which
  // includes it.
  //
  // Given the above operands and attributes, the custom_call returns tensors
  // with the same shapes as the inputs (i.e. the first half of the operands),
  // save for reduction_dim, which may have changed in accordance with the
  // values of aggregate_to_topk, recall_target, and
  // reduction_input_size_override above. These tensors will contain slices of
  // the input tensors perpendicular to that axis, which have approximately the
  // top values of the comparator along that axis to within recall_target.
  //
  // The operands and attributes must obey the following constraints:
  //
  // (C1) size(inputs) = size(init_values) = size(results)
  // (C2) All inputs have the same shape.
  // (C3) element_type(inputs[i]) = element_type(init_values[i])
  //                              = element_type(results[i]) for all i in [0, N)
  // (C4) shape(results[i]) = shape(inputs[i]) except that the dimension size
  //      of inputs[i] corresponding to reduction_dim_are replaced with a
  //      value >=k, which can be determined using ApproxTopKReductionOutputSize
  // (C5) called_computation has type
  //      (tensor<E0>, tensor<E0>, ..., tensor<EN-1>, tensor<EN-1>) ->
  //      tensor<i1>
  //        where Ei = element_type(inputs[i])
  // (C6) 0 <= reduction_dim < rank(inputs[0])
  // (C7) 0 < recall_target <= 1.0
  // (C8) dim(inputs[0],reduction_dim) < reduction_input_size_override
  //        || reduction_input_size_override < 0
  //
  // See arxiv:2206.14286 for more details.
  //
  // This feature is at time of writing only used by JAX, and is tested in the
  // jax2tf backwards compatibility tests.

  if (op.getCallTargetName() == "ApproxTopK") {
    auto isSupportedAttrName = [](NamedAttribute attr) {
      auto name = attr.getName();
      return name == "call_target_name" || name == "backend_config" ||
             name == "api_version" || name == "called_computations" ||
             name == "has_side_effect";
    };
    for (const auto& attr : op->getAttrs()) {
      if (!isSupportedAttrName(attr))
        return op.emitOpError()
               << attr.getName().getValue()
               << " is not a supported attribute for ApproxTopK";
    }
    auto backend_config =
        op.getBackendConfigAttr().dyn_cast_or_null<mlir::DictionaryAttr>();
    if (!backend_config)
      return op.emitOpError() << "Missing backend_config attribute";

    for (auto attr : backend_config) {
      auto name = attr.getName();
      if (!(name == "top_k" || name == "reduction_dim" ||
            name == "recall_target" || name == "aggregate_to_topk" ||
            name == "reduction_input_size_override" || name == "is_fallback"))
        return op.emitOpError()
               << name.getValue() << " is not a supported backend_config"
               << " attribute for ApproxTopK";
    }

    auto checkI64Attr =
        [&](const std::string& attr_name) -> mlir::LogicalResult {
      if (!backend_config.contains(attr_name))
        return op.emitOpError()
               << "Missing " << attr_name << " attribute in backend_config";
      auto attr = backend_config.getAs<IntegerAttr>(attr_name);
      if (!attr || !attr.getType().isInteger(64))
        return op.emitOpError()
               << attr_name
               << " attribute in backend_config must be of i64 type";
      return success();
    };
    auto checkF32Attr =
        [&](const std::string& attr_name) -> mlir::LogicalResult {
      if (!backend_config.contains(attr_name))
        return op.emitOpError()
               << "Missing " << attr_name << " attribute in backend_config";
      auto attr = backend_config.getAs<FloatAttr>(attr_name);
      if (!attr || !attr.getType().isF32())
        return op.emitOpError()
               << attr_name
               << " attribute in backend_config must be of f32 type";
      return success();
    };
    auto checkBoolAttr =
        [&](const std::string& attr_name) -> mlir::LogicalResult {
      if (!backend_config.contains(attr_name))
        return op.emitOpError()
               << "Missing " << attr_name << " attribute in backend_config";
      auto attr = backend_config.getAs<BoolAttr>(attr_name);
      if (!attr)
        return op.emitOpError()
               << attr_name
               << " attribute in backend_config must be of bool type";
      return success();
    };
    if (failed(checkI64Attr("top_k"))) return failure();
    if (failed(checkI64Attr("reduction_dim"))) return failure();
    if (failed(checkF32Attr("recall_target"))) return failure();
    if (failed(checkBoolAttr("aggregate_to_topk"))) return failure();
    if (failed(checkI64Attr("reduction_input_size_override"))) return failure();
    bool has_is_fallback = backend_config.contains("is_fallback");
    if (has_is_fallback && !backend_config.getAs<BoolAttr>("is_fallback"))
      return op.emitOpError()
             << "is_fallback attribute in backend_config must be of bool type";

    int64_t top_k = backend_config.getAs<IntegerAttr>("top_k").getInt();
    int64_t reduction_dim =
        backend_config.getAs<IntegerAttr>("reduction_dim").getInt();
    float recall_target = backend_config.getAs<FloatAttr>("recall_target")
                              .getValue()
                              .convertToFloat();
    bool aggregate_to_topk =
        backend_config.getAs<BoolAttr>("aggregate_to_topk").getValue();
    int64_t reduction_input_size_override =
        backend_config.getAs<IntegerAttr>("reduction_input_size_override")
            .getInt();
    bool is_fallback = has_is_fallback &&
                       backend_config.getAs<BoolAttr>("is_fallback").getValue();

    // (C1)
    if (args.size() % 2 != 0) {
      return op.emitOpError() << "ApproxTopK takes an even number of operands.";
    }
    auto num_inputs = args.size() / 2;
    absl::Span<const xla::XlaOp> inputs(args.begin(), num_inputs);
    absl::Span<const xla::XlaOp> init_values(args.begin() + num_inputs,
                                             num_inputs);
    if (num_inputs != op.getNumResults()) {
      return op.emitOpError() << "num_results does not match num_inputs";
    }

    SmallVector<RankedTensorType> input_types, init_value_types, result_types;
    for (size_t i = 0; i < num_inputs; ++i) {
      auto input_type = op.getOperand(i).getType().dyn_cast<RankedTensorType>();
      if (!input_type) return failure();
      input_types.push_back(input_type);
      auto init_value_type =
          op.getOperand(num_inputs + i).getType().dyn_cast<RankedTensorType>();
      if (!init_value_type) return failure();
      init_value_types.push_back(init_value_type);
      auto result_type = op.getResult(i).getType().dyn_cast<RankedTensorType>();
      if (!result_type) return failure();
      result_types.push_back(result_type);
    }

    for (size_t i = 0; i < inputs.size(); ++i) {
      // (C2)
      if (input_types[0].getShape() != input_types[i].getShape()) {
        return op.emitOpError() << "input shape mismatch at position " << i;
      }

      // (C3)
      if (init_value_types[i].getElementType() !=
          input_types[i].getElementType()) {
        return op.emitOpError()
               << "input and init_value element type mismatch at position "
               << i;
      }
      if (input_types[i].getElementType() != result_types[i].getElementType()) {
        return op.emitOpError()
               << "result element type mismatch at position " << i;
      }

      // (C4)
      for (size_t j = 0; j < input_types[i].getRank(); ++j) {
        if (j == reduction_dim) {
          auto reduction_output_size = xla::ApproxTopKReductionOutputSize(
              input_types[i].getShape()[j], input_types[i].getRank(), top_k,
              recall_target, aggregate_to_topk, reduction_input_size_override);
          if (!reduction_output_size.ok()) return failure();
          if (result_types[i].getShape()[j] != reduction_output_size->first)
            return op.emitOpError()
                   << "ApproxTopK aggregates to k="
                   << reduction_output_size->first << ", but got "
                   << result_types[i].getShape()[j];
          continue;
        }
        if (input_types[i].getShape()[j] != result_types[i].getShape()[j]) {
          return op.emitOpError() << "result shape mismatch at position " << i
                                  << ", index " << j;
        }
      }
    }

    // (C5)
    auto called_computations = op.getCalledComputations();
    if (called_computations.size() != 1) {
      return op.emitOpError()
             << "ApproxTopK takes exactly 1 called_computation.";
    }
    mlir::func::FuncOp callee = ctx.converter->LookUpSymbol(
        op.getCalledComputations()[0].cast<FlatSymbolRefAttr>());
    mlir::FunctionType callee_type = callee.getFunctionType();
    SmallVector<Type, 4> expected_callee_input_types;
    for (unsigned i = 0; i < num_inputs; ++i) {
      auto scalar = RankedTensorType::get({}, input_types[i].getElementType());
      expected_callee_input_types.push_back(scalar);
      expected_callee_input_types.push_back(scalar);
    }
    FunctionType expected_callee_type = mlir::FunctionType::get(
        op->getContext(), expected_callee_input_types,
        RankedTensorType::get({}, IntegerType::get(op->getContext(), 1)));
    if (callee_type != expected_callee_type) {
      return op.emitOpError()
             << "called_computation type does not match the expected type. Got "
             << callee_type << " expected " << expected_callee_type;
    }

    if (failed(ctx.converter->RunOnFunction(callee))) return failure();
    xla::XlaComputation& comparator =
        ctx.converter->GetLoweredComputation(callee);

    // (C6)
    if (reduction_dim < 0 || reduction_dim > input_types[0].getRank())
      return op.emitOpError() << "reduction_dim out of range";
    // (C7)
    if (recall_target <= 0 || recall_target > 1.0)
      return op.emitOpError() << "recall_target out of range";
    // (C8)
    if (reduction_input_size_override >= 0 &&
        reduction_input_size_override <
            input_types[0].getShape()[reduction_dim])
      return op.emitOpError() << "reduction_input_size_override out of range";

    xla::XlaOp cc_op;
    if (is_fallback) {
      cc_op = xla::ApproxTopKFallback(
          ctx.builder, inputs, init_values, top_k, reduction_dim, comparator,
          recall_target, aggregate_to_topk, reduction_input_size_override);
    } else {
      cc_op = xla::ApproxTopK(ctx.builder, inputs, init_values, top_k,
                              reduction_dim, comparator, recall_target,
                              aggregate_to_topk, reduction_input_size_override);
    }
    BuildGetTupleElementsForTupleResults(op, cc_op, ctx);
    return success();
  }

  if (op.getCalledComputations().size() > 1)
    return op.emitOpError()
           << "cannot export with more than one called computations";

  // Custom call can be exported either with called computation or with layout
  // attributes. The XlaBuilder API does not allow both.
  if (!op.getCalledComputations().empty() && op.getOperandLayouts() &&
      op.getResultLayouts()) {
    return op.emitOpError() << "cannot export if both called computation and "
                               "layouts are specified";
  }

  auto xla_api_version = xla::ConvertCustomCallApiVersion(op.getApiVersion());
  if (!xla_api_version.ok()) return failure();

  // CustomCallOp backend config can be either a string if we use any of the
  // older custom call API versions, or a dictionary attribute if we use typed
  // FFI. We always pass it as a string to the HLO instruction. If it was a
  // dictionary attribute we rely on MLIR printing to convert it to string.
  std::string backend_config;

  if (*xla_api_version == xla::CustomCallApiVersion::API_VERSION_TYPED_FFI) {
    // Serialize backend config dictionary as a string.
    if (auto dict = op.getBackendConfig()
                        .value_or(mlir::Attribute())
                        .dyn_cast_or_null<mlir::DictionaryAttr>()) {
      llvm::raw_string_ostream(backend_config) << dict;
    }
  } else {
    // Forward backend config string to the HLO instruction.
    if (auto str = op.getBackendConfig()
                       .value_or(mlir::Attribute())
                       .dyn_cast_or_null<mlir::StringAttr>()) {
      llvm::raw_string_ostream(backend_config) << str.strref();
    }
  }

  absl::StatusOr<xla::Literal> literal;
  const xla::Literal* literal_ptr = nullptr;
  auto literal_attr = op->getAttrOfType<DenseElementsAttr>(kLiteralAttr);
  if (literal_attr) {
    literal = CreateArrayLiteralFromAttr(literal_attr, {});
    if (!literal.ok()) return failure();
    literal_ptr = &*literal;
  }

  auto aliasInfo =
      xla::ConvertOutputOperandAliasing(op.getOutputOperandAliases());
  auto output_operand_aliasing = absl::MakeSpan(*aliasInfo);
  auto custom_call_schedule =
      xla::ConvertCustomCallSchedule(op.getCustomCallSchedule());
  if (!custom_call_schedule.ok()) return failure();

  std::string call_target_name(op.getCallTargetName());
  xla::Shape result_shape;
  if (op->getNumResults() == 1) {
    result_shape = xla::TypeToShape(op.getResult(0).getType());
  } else {
    std::vector<xla::Shape> subshapes;
    for (const auto& item : op.getResults().getType()) {
      subshapes.push_back(xla::TypeToShape(item));
    }
    result_shape = xla::ShapeUtil::MakeTupleShape(subshapes);
  }

  xla::XlaOp custom_call;
  if (op.getCalledComputations().size() == 1) {
    mlir::func::FuncOp callee = ctx.converter->LookUpSymbol(
        op.getCalledComputations()[0].cast<FlatSymbolRefAttr>());
    if (failed(ctx.converter->RunOnFunction(callee))) return failure();
    xla::XlaComputation& computation =
        ctx.converter->GetLoweredComputation(callee);
    custom_call = xla::CustomCallWithComputation(
        ctx.builder, call_target_name, args, computation, result_shape,
        backend_config, op.getHasSideEffect(), output_operand_aliasing,
        literal_ptr, *custom_call_schedule, *xla_api_version);
  } else if (op.getOperandLayouts() && op.getResultLayouts()) {
    auto operand_shapes_with_layout = ConvertTypesToShapesWithLayout(
        op.getOperandTypes(), op.getOperandLayouts().value());
    SetLayout(result_shape, op.getResultLayouts().value());

    custom_call = xla::CustomCallWithLayout(
        ctx.builder, call_target_name, args, result_shape,
        operand_shapes_with_layout, backend_config, op.getHasSideEffect(),
        output_operand_aliasing, literal_ptr, *custom_call_schedule,
        *xla_api_version);
  } else {
    custom_call = xla::CustomCall(
        ctx.builder, call_target_name, args, result_shape, backend_config,
        op.getHasSideEffect(), output_operand_aliasing, literal_ptr,
        *custom_call_schedule, *xla_api_version);
  }

  if (op->getNumResults() == 1) {
    value_map[op.getResult(0)] = custom_call;
  } else {
    BuildGetTupleElementsForTupleResults(op, custom_call, ctx);
  }

  return success();
}

LogicalResult ExportXlaOp(InfeedOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp token;
  if (failed(GetXlaOp(op.getToken(), value_map, &token, op))) return failure();

  // mhlo.infeed produces multiple results. The shape argument expected by the
  // xla client API is a tuple type with two element-types:
  // data_type : A tuple containing all the mhlo.infeedOp result types except
  //             the token type.
  // token_type : The last result type of mhlo.infeedOp.
  auto result_types = op.getResultTypes();
  auto num_results = op.getNumResults();

  xla::Shape token_shape = xla::TypeToShape(result_types[num_results - 1]);
  std::vector<xla::Shape> subshapes;
  for (const auto& item : llvm::enumerate(result_types)) {
    if (item.index() == num_results - 1) break;
    subshapes.push_back(xla::TypeToShape(item.value()));
  }

  xla::Shape data_shape = xla::ShapeUtil::MakeTupleShape(subshapes);
  auto xla_result = xla::InfeedWithToken(token, data_shape,
                                         std::string(op.getInfeedConfig()));
  ctx.builder->ClearSharding();

  if (!subshapes.empty()) {
    auto data_tuple_element = xla::GetTupleElement(xla_result, 0);
    for (const auto& item : llvm::enumerate(op.getResults())) {
      if (item.index() == num_results - 1) break;
      value_map[item.value()] =
          xla::GetTupleElement(data_tuple_element, item.index());
    }
  }

  value_map[op.getResult(num_results - 1)] =
      xla::GetTupleElement(xla_result, 1);

  return success();
}

LogicalResult ExportXlaOp(IotaOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  value_map[op] = xla::Iota(ctx.builder, xla::TypeToShape(op.getType()),
                            op.getIotaDimension());
  return success();
}

LogicalResult ExportXlaOp(MapOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaComputation computation;
  if (failed(ctx.converter->LowerRegionAsComputation(&op.getComputation(),
                                                     &computation))) {
    return failure();
  }
  llvm::SmallVector<xla::XlaOp> operands;
  if (failed(GetTuple(op, op.getInputs(), ctx, operands))) return failure();
  value_map[op] = xla::Map(ctx.builder, operands, computation,
                           Convert_dimensions(op.getDimensions()));
  return success();
}

LogicalResult ExportXlaOp(OutfeedOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;

  llvm::SmallVector<xla::XlaOp> operands;
  if (failed(GetTuple(op, op.getInputs(), ctx, operands))) return failure();

  const auto sharding = ctx.builder->sharding();
  xla::XlaOp operand;

  if (sharding.has_value() &&
      sharding->tuple_shardings_size() != operands.size()) {
    xla::XlaScopedShardingAssignment scoped_sharding(ctx.builder, std::nullopt);
    operand = Tuple(ctx.builder, operands);
  } else {
    operand = Tuple(ctx.builder, operands);
  }
  std::vector<xla::Shape> subshapes;
  for (auto operand : op.getInputs())
    subshapes.push_back(xla::TypeToShape(operand.getType()));

  xla::Shape shape_with_layout = xla::ShapeUtil::MakeTupleShape(subshapes);

  xla::XlaOp token;
  if (failed(GetXlaOp(op.getToken(), value_map, &token, op))) return failure();

  value_map[op] = xla::OutfeedWithToken(operand, token, shape_with_layout,
                                        std::string(op.getOutfeedConfig()));
  return success();
}

LogicalResult ExportXlaOp(PartitionIdOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::Shape shape = xla::TypeToShape(op.getResult().getType());
  value_map[op] =
      xla::internal::XlaBuilderFriend::BuildPartitionId(ctx.builder, shape);
  return success();
}

LogicalResult ExportXlaOp(PadOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::PaddingConfig padding_config;
  auto edge_padding_low = ConvertDenseIntAttr(op.getEdgePaddingLow());
  auto edge_padding_high = ConvertDenseIntAttr(op.getEdgePaddingHigh());
  auto interior_padding = ConvertDenseIntAttr(op.getInteriorPadding());
  for (int64_t i = 0, end = edge_padding_low.size(); i < end; ++i) {
    auto* dims = padding_config.add_dimensions();
    dims->set_edge_padding_low(edge_padding_low[i]);
    dims->set_edge_padding_high(edge_padding_high[i]);
    dims->set_interior_padding(interior_padding[i]);
  }
  xla::XlaOp operand, padding_value;
  if (failed(GetXlaOp(op.getOperand(), value_map, &operand, op)))
    return failure();
  if (failed(GetXlaOp(op.getPaddingValue(), value_map, &padding_value, op)))
    return failure();

  value_map[op] = xla::Pad(operand, padding_value, padding_config);
  return success();
}

LogicalResult ExportXlaOp(RecvOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;

  xla::XlaOp token;
  if (failed(GetXlaOp(op.getToken(), value_map, &token, op))) return failure();

  // mhlo.recvOp produces multiple results. The shape argument expected by the
  // xla client API is a tuple type with two element-types:
  // data_type : A tuple containing all the mhlo.RecvOp result types except
  //             the token type.
  // token_type : The last result type of mhlo.recvOp.
  auto result_types = op.getResultTypes();
  auto num_results = op.getNumResults();

  xla::Shape token_shape = xla::TypeToShape(result_types[num_results - 1]);
  std::vector<xla::Shape> subshapes;
  for (const auto& item : llvm::enumerate(result_types)) {
    if (item.index() == num_results - 1) break;
    subshapes.push_back(xla::TypeToShape(item.value()));
  }

  xla::Shape data_shape;
  if (subshapes.size() == 1)
    data_shape = subshapes[0];
  else
    data_shape = xla::ShapeUtil::MakeTupleShape(subshapes);

  token = xla::internal::XlaBuilderFriend::BuildRecv(
      ctx.builder, token, data_shape,
      Convert_channel_handle(op.getChannelHandle()), op.getIsHostTransfer());
  xla::XlaOp xla_result = xla::internal::XlaBuilderFriend::BuildRecvDone(
      ctx.builder, token, data_shape,
      Convert_channel_handle(op.getChannelHandle()), op.getIsHostTransfer());

  auto data_tuple_element = xla::GetTupleElement(xla_result, 0);
  if (subshapes.size() == 1) {
    value_map[op.getResult(0)] = data_tuple_element;
  } else {
    for (const auto& item : llvm::enumerate(op.getResults())) {
      if (item.index() == num_results - 1) break;
      value_map[item.value()] =
          xla::GetTupleElement(data_tuple_element, item.index());
    }
  }

  value_map[op.getResult(num_results - 1)] =
      xla::GetTupleElement(xla_result, 1);

  return success();
}

LogicalResult ExportXlaOp(ReduceOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaComputation body;
  if (failed(ctx.converter->LowerRegionAsComputation(&op.getBody(), &body))) {
    return failure();
  }
  llvm::SmallVector<xla::XlaOp> operands, init_values;
  if (failed(GetTuple(op, op.getInputs(), ctx, operands)) ||
      failed(GetTuple(op, op.getInitValues(), ctx, init_values))) {
    return failure();
  }
  xla::XlaOp result =
      xla::Reduce(ctx.builder, operands, init_values, body,
                  Convert_broadcast_dimensions(op.getDimensions()));
  if (op.getNumResults() == 1) {
    value_map[op.getResult(0)] = result;
  } else {
    BuildGetTupleElementsForTupleResults(op, result, ctx);
  }
  return success();
}

LogicalResult ExportXlaOp(ReduceWindowOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaComputation body;
  if (failed(ctx.converter->LowerRegionAsComputation(&op.getBody(), &body))) {
    return failure();
  }
  llvm::SmallVector<xla::XlaOp> operands, init_values;
  if (failed(GetTuple(op, op.getInputs(), ctx, operands)) ||
      failed(GetTuple(op, op.getInitValues(), ctx, init_values))) {
    return failure();
  }

  xla::XlaOp result = xla::ReduceWindowWithGeneralPadding(
      operands, init_values, body,
      ConvertDenseIntAttr(op.getWindowDimensions()),
      ConvertDenseIntAttr(op.getWindowStrides()),
      ConvertDenseIntAttr(op.getBaseDilations()),
      ConvertDenseIntAttr(op.getWindowDilations()),
      Convert_padding(op.getPadding()));

  if (op.getNumResults() == 1) {
    value_map[op.getResult(0)] = result;
  } else {
    BuildGetTupleElementsForTupleResults(op, result, ctx);
  }
  return success();
}

LogicalResult ExportXlaOp(ReshapeOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp operand;
  if (failed(GetXlaOp(op.getOperand(), value_map, &operand, op)))
    return failure();

  value_map[op] =
      xla::Reshape(operand, xla::TypeToShape(op.getType()).dimensions());
  return success();
}

LogicalResult ExportXlaOp(ReturnOp op, OpLoweringContext ctx) {
  // Failure on purpose because `mhlo::ReturnOp` will be handled by
  // special purpose logic in `ConvertToHloModule::Lower`.
  return failure();
}

LogicalResult ExportXlaOp(RngBitGeneratorOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto results = op.getResults();
  auto xla_arg_1 = value_map[*op.getODSOperands(0).begin()];
  auto xla_result = xla::RngBitGenerator(
      static_cast<xla::RandomAlgorithm>(op.getRngAlgorithm()),
      Unwrap(xla_arg_1), xla::TypeToShape(results[1].getType()));

  BuildGetTupleElementsForTupleResults(op, xla_result, ctx);
  return mlir::success();
}

LogicalResult ExportXlaOp(XlaRngGetAndUpdateStateOp op, OpLoweringContext ctx) {
  // This op does not exist in the XLA builder interface.
  (*ctx.values)[op.getResult()] =
      xla::internal::XlaBuilderFriend::BuildRngGetAndUpdateState(
          ctx.builder, static_cast<int64_t>(op.getDelta()),
          xla::TypeToShape(op.getType()));
  return mlir::success();
}

LogicalResult ExportXlaOp(BatchNormGradOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;

  xla::XlaOp operand, scale, mean, variance, grad_output;
  if (failed(GetXlaOp(op.getOperand(), value_map, &operand, op)))
    return failure();
  if (failed(GetXlaOp(op.getScale(), value_map, &scale, op))) return failure();
  if (failed(GetXlaOp(op.getMean(), value_map, &mean, op))) return failure();
  if (failed(GetXlaOp(op.getVariance(), value_map, &variance, op)))
    return failure();
  if (failed(GetXlaOp(op.getGradOutput(), value_map, &grad_output, op)))
    return failure();

  auto xla_result =
      xla::BatchNormGrad(operand, scale, mean, variance, grad_output,
                         ConvertAPFloat(op.getEpsilon()), op.getFeatureIndex());

  BuildGetTupleElementsForTupleResults(op, xla_result, ctx);

  return mlir::success();
}

LogicalResult ExportXlaOp(BatchNormTrainingOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;

  xla::XlaOp operand, scale, offset;
  if (failed(GetXlaOp(op.getOperand(), value_map, &operand, op)))
    return failure();
  if (failed(GetXlaOp(op.getScale(), value_map, &scale, op))) return failure();
  if (failed(GetXlaOp(op.getOffset(), value_map, &offset, op)))
    return failure();

  auto xla_result = xla::BatchNormTraining(operand, scale, offset,
                                           ConvertAPFloat(op.getEpsilon()),
                                           op.getFeatureIndex());

  BuildGetTupleElementsForTupleResults(op, xla_result, ctx);

  return mlir::success();
}

LogicalResult ExportXlaOp(RngOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp a, b;
  if (failed(GetXlaOp(op.getA(), value_map, &a, op))) return failure();
  if (failed(GetXlaOp(op.getB(), value_map, &b, op))) return failure();

  if (op.getRngDistribution() == RngDistribution::UNIFORM) {
    value_map[op] = xla::RngUniform(a, b, xla::TypeToShape(op.getType()));
    return success();
  } else if (op.getRngDistribution() == RngDistribution::NORMAL) {
    value_map[op] = xla::RngNormal(a, b, xla::TypeToShape(op.getType()));
    return success();
  }
  return failure();
}

LogicalResult ExportXlaOp(ScatterOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaComputation update_computation;
  if (failed(ctx.converter->LowerRegionAsComputation(&op.getUpdateComputation(),
                                                     &update_computation))) {
    return failure();
  }
  xla::ScatterDimensionNumbers dimension_numbers =
      Convert_scatter_dimension_numbers(op.getScatterDimensionNumbers());

  llvm::SmallVector<xla::XlaOp> operands;
  llvm::SmallVector<xla::XlaOp> updates;
  if (failed(GetTuple(op, op.getInputs(), ctx, operands))) return failure();
  if (failed(GetTuple(op, op.getUpdates(), ctx, updates))) return failure();

  xla::XlaOp scatter_indices;
  if (failed(GetXlaOp(op.getScatterIndices(), value_map, &scatter_indices, op)))
    return failure();

  auto scatter_op = xla::Scatter(
      operands, scatter_indices, updates, update_computation, dimension_numbers,
      op.getIndicesAreSorted(), op.getUniqueIndices());
  if (op->getNumResults() == 1) {
    value_map[op.getResult(0)] = scatter_op;
    return success();
  }

  // mhlo.ScatterOp supports multiple returns, untuple all the results of XLA's.
  BuildGetTupleElementsForTupleResults(op, scatter_op, ctx);

  return success();
}

LogicalResult ExportXlaOp(SelectAndScatterOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaComputation select;
  xla::XlaComputation scatter;
  if (failed(
          ctx.converter->LowerRegionAsComputation(&op.getSelect(), &select)) ||
      failed(ctx.converter->LowerRegionAsComputation(&op.getScatter(),
                                                     &scatter))) {
    return failure();
  }
  xla::XlaOp operand, source, init_value;
  if (failed(GetXlaOp(op.getOperand(), value_map, &operand, op)))
    return failure();
  if (failed(GetXlaOp(op.getSource(), value_map, &source, op)))
    return failure();
  if (failed(GetXlaOp(op.getInitValue(), value_map, &init_value, op)))
    return failure();

  value_map[op] = xla::SelectAndScatterWithGeneralPadding(
      operand, select, ConvertDenseIntAttr(op.getWindowDimensions()),
      ConvertDenseIntAttr(op.getWindowStrides()),
      Convert_padding(op.getPadding()), source, init_value, scatter);
  return success();
}

LogicalResult ExportXlaOp(SendOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;

  llvm::SmallVector<xla::XlaOp> operands;
  if (failed(GetTuple(op, op.getInputs(), ctx, operands))) return failure();

  xla::XlaOp operand;
  if (operands.size() == 1)
    operand = operands[0];
  else
    operand = Tuple(ctx.builder, operands);

  xla::XlaOp token;
  if (failed(GetXlaOp(op.getToken(), value_map, &token, op))) return failure();

  token = xla::internal::XlaBuilderFriend::BuildSend(
      ctx.builder, operand, token,
      Convert_channel_handle(op.getChannelHandle()), op.getIsHostTransfer());
  value_map[op] = xla::internal::XlaBuilderFriend::BuildSendDone(
      ctx.builder, token, Convert_channel_handle(op.getChannelHandle()),
      op.getIsHostTransfer());
  return success();
}

// TODO(b/298671312): The semantics of xla::SetDimensionSize have changed so
// that it always returns a dynamic shape.  The old semantics are still
// available through xla::RemoveDynamicDimension, so to avoid changing MHLO
// semantics we explicitly check for that case here.  However, we should
// consider adding a RemoveDynamicDimensionOp to HLO and MHLO.
mlir::LogicalResult ExportXlaOp(mlir::mhlo::SetDimensionSizeOp op,
                                OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp array;
  if (failed(GetXlaOp(op.getOperand(), value_map, &array, op)))
    return mlir::failure();
  auto dimension = Convertuint64_t(op.getDimension());
  auto shape_or = ctx.builder->GetShapePtr(array);
  if (!shape_or.ok()) {
    return op.emitError(shape_or.status().ToString());
  }
  xla::XlaOp xla_result;
  if (auto constant = llvm::dyn_cast_or_null<mlir::mhlo::ConstantOp>(
          op.getSize().getDefiningOp());
      constant != nullptr) {
    auto value = constant.getValue();
    auto values = value.getValues<mlir::IntegerAttr>();
    if ((*values.begin()).getValue().getSExtValue() ==
        shape_or.value()->dimensions(dimension)) {
      xla_result = xla::RemoveDynamicDimension(array, dimension);
    }
  }
  if (!xla_result.valid()) {
    xla::XlaOp dynamic_size;
    if (failed(GetXlaOp(op.getSize(), value_map, &dynamic_size, op)))
      return mlir::failure();
    xla_result = xla::SetDimensionSize(array, dynamic_size, dimension);
  }
  value_map[result] = xla_result;
  return mlir::success();
}

mlir::LogicalResult ExportXlaOp(mlir::mhlo::SineOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp arg;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &arg, op)))
    return mlir::failure();
  auto xla_result = xla::Sin(Unwrap(arg));
  value_map[result] = xla_result;
  return mlir::success();
}

LogicalResult ExportXlaOp(SortOp op, OpLoweringContext ctx) {
  xla::XlaComputation comparator;
  if (failed(ctx.converter->LowerRegionAsComputation(&op.getComparator(),
                                                     &comparator)))
    return failure();

  llvm::SmallVector<xla::XlaOp> operands;
  if (failed(GetTuple(op, op.getInputs(), ctx, operands))) return failure();
  auto sorted =
      xla::Sort(operands, comparator, op.getDimension(), op.getIsStable());

  auto& value_map = *ctx.values;
  auto shape_or = sorted.builder()->GetShape(sorted);
  if (!shape_or.ok()) {
    return op.emitError(shape_or.status().ToString());
  }

  xla::Shape& shape = shape_or.value();
  if (!shape.IsTuple()) {
    value_map[op.getResult(0)] = sorted;
    return success();
  }

  // MLIR's sort supports multiple returns, untuple all the results of XLA's.
  BuildGetTupleElementsForTupleResults(op, sorted, ctx);
  return success();
}

LogicalResult ExportXlaOp(SubtractOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  auto result = op.getResult();
  xla::XlaOp lhs;
  if (failed(GetXlaOp(*op.getODSOperands(0).begin(), value_map, &lhs, op)))
    return mlir::failure();
  xla::XlaOp rhs;
  if (failed(GetXlaOp(*op.getODSOperands(1).begin(), value_map, &rhs, op)))
    return mlir::failure();
  auto xla_result = xla::Sub(Unwrap(lhs), Unwrap(rhs));
  value_map[result] = xla_result;
  return mlir::success();
}

LogicalResult ExportXlaOp(TraceOp op, OpLoweringContext ctx) {
  // TODO(atondwal): remove mhlo.trace
  return success();
}

LogicalResult ExportXlaOp(UnaryEinsumOp op, OpLoweringContext ctx) {
  // Intentional as UnaryEinsumOp is always lowered to the EinsumOp with two
  // operands.
  return failure();
}

LogicalResult ExportXlaOp(WhileOp op, OpLoweringContext ctx) {
  xla::XlaComputation condition;
  xla::XlaComputation body;
  // If the results of the while op have a sharding, we use those shardings for
  // the corresponding arguments and return shardings in the body and condition.
  llvm::SmallVector<std::optional<xla::OpSharding>> res_shardings =
      GetResultShardings(ctx.builder->sharding());
  assert(res_shardings.empty() || res_shardings.size() == op->getNumResults());
  if (failed(ctx.converter->LowerRegionAsComputation(
          &op.getBody(), &body, std::nullopt, /*ensure_single_arg*/ true,
          /*arg_shardings=*/res_shardings, /*ret_shardings=*/res_shardings)) ||
      failed(ctx.converter->LowerRegionAsComputation(
          &op.getCond(), &condition, std::nullopt,
          /*ensure_single_arg*/ true, /*arg_shardings=*/res_shardings))) {
    return failure();
  }

  // In case MHLO's whileOp has multiple operands, create xla::Tuple, using
  // those operands, to be used as sole operand of xla::While.
  llvm::SmallVector<xla::XlaOp> operands;
  if (failed(GetTuple(op, op.getOperands(), ctx, operands))) return failure();

  xla::XlaOp operand = operands[0];
  if (operands.size() > 1) operand = Tuple(ctx.builder, operands);

  auto whileop = xla::While(condition, body, operand);

  auto& value_map = *ctx.values;
  auto shape_or = whileop.builder()->GetShape(whileop);
  if (!shape_or.ok()) {
    return op.emitError(shape_or.status().ToString());
  }

  xla::Shape& shape = shape_or.value();
  if (!shape.IsTuple()) {
    value_map[op.getResult(0)] = whileop;
    return success();
  }

  // mhlo.WhileOp supports multiple returns, untuple all the results of XLA's.
  BuildGetTupleElementsForTupleResults(op, whileop, ctx);

  return success();
}

LogicalResult ExportXlaOp(OptimizationBarrierOp op, OpLoweringContext ctx) {
  // In case MHLO's OptimizationBarrierOp has multiple operands,
  // create xla::Tuple, using those operands, to be used as
  // sole operand of xla::OptimizationBarrier.
  llvm::SmallVector<xla::XlaOp> operands;
  if (failed(GetTuple(op, op.getOperands(), ctx, operands))) return failure();
  if (operands.empty()) return success();

  auto& value_map = *ctx.values;
  if (operands.size() == 1) {
    value_map[op.getOperation()->getResult(0)] =
        xla::OptimizationBarrier(operands[0]);
  } else {
    auto result = xla::OptimizationBarrier(Tuple(ctx.builder, operands));
    BuildGetTupleElementsForTupleResults(op, result, ctx);
  }

  return success();
}

LogicalResult ExportXlaOp(FusionOp op, OpLoweringContext ctx) {
  if (!op.getFusionKind()) {
    op.emitOpError() << "requires fusion kind for HLO translation";
    return failure();
  }

  xla::XlaComputation fused_computation;
  if (failed(ctx.converter->LowerRegionAsComputation(&op.getFusedComputation(),
                                                     &fused_computation)))
    return failure();

  auto& values = *ctx.values;
  auto aliasInfo =
      xla::ConvertOutputOperandAliasing(op.getOutputOperandAliases());
  auto output_operand_aliasing = absl::MakeSpan(*aliasInfo);
  llvm::SmallVector<xla::XlaOp, 4> operands;
  for (auto operand : op.getInputs()) operands.push_back(values[operand]);

  auto fusion_kind_string =
      mlir::mhlo::stringifyFusionKind(op.getFusionKind().value());
  xla::XlaOp fusion = xla::internal::XlaBuilderFriend::BuildFusion(
      ctx.builder, operands,
      absl::string_view(fusion_kind_string.data(), fusion_kind_string.size()),
      fused_computation, output_operand_aliasing);
  if (op.getNumResults() == 1) {
    values[op.getResult(0)] = fusion;
  } else {
    BuildGetTupleElementsForTupleResults(op, fusion, ctx);
  }
  return success();
}

LogicalResult ExportXlaOp(BitcastOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp operand;
  if (failed(GetXlaOp(op.getOperand(), value_map, &operand, op)))
    return failure();
  xla::XlaOp bitcast = xla::internal::XlaBuilderFriend::BuildBitcast(
      ctx.builder, operand, xla::TypeToShape(op.getType()));
  value_map[op] = bitcast;
  if (ctx.converter->GetOptions().propagate_bitcast_layouts_to_backend_config) {
    // Encode the source and result layout of the bitcast into the XLA HLO
    // backend config as a protobuf. Note that this is a temporary solution
    // which will go away once XLA:GPU stops falling back to XLA HLO Elemental
    // IR emitters.
    xla::HloInstructionProto* bitcast_proto =
        xla::internal::XlaBuilderFriend::GetInstruction(bitcast);
    xla::HloInstructionProto* operand_proto =
        xla::internal::XlaBuilderFriend::GetInstruction(operand);
    xla::LayoutProto result_layout =
        ExtractLayout(op, bitcast_proto->shape().dimensions_size(),
                      "result_layout")
            .ToProto();
    xla::LayoutProto source_layout =
        ExtractLayout(op, operand_proto->shape().dimensions_size(),
                      "source_layout")
            .ToProto();
    xla::gpu::BitcastBackendConfig bitcast_config;
    *bitcast_config.mutable_source_layout() = source_layout;
    *bitcast_config.mutable_result_layout() = result_layout;
    *bitcast_proto->mutable_backend_config() =
        bitcast_config.SerializeAsString();
  }
  return success();
}

LogicalResult ExportXlaOp(UniformQuantizeOp op, OpLoweringContext ctx) {
  // Currently, it doesn't have an XLA builder equivalent.
  // TODO(b/230671877): Implement XLA import/export for quantized MHLO ops.
  return failure();
}

LogicalResult ExportXlaOp(UniformDequantizeOp op, OpLoweringContext ctx) {
  // Currently, it doesn't have an XLA builder equivalent.
  // TODO(b/230671877): Implement XLA import/export for quantized MHLO ops.
  return failure();
}

LogicalResult ExportXlaOp(TopKOp op, OpLoweringContext ctx) {
  auto& value_map = *ctx.values;
  xla::XlaOp operand;
  if (failed(GetXlaOp(op.getOperand(), value_map, &operand, op)))
    return failure();
  auto topk = xla::TopK(operand, op.getK(), op.getLargest());

  // Untuple the two results of XLA's topk.
  BuildGetTupleElementsForTupleResults(op, topk, ctx);
  return success();
}

}  // namespace
}  // namespace mhlo
}  // namespace mlir

#include "xla/translate/mhlo_to_hlo/operator_writers.inc"

namespace mlir {
namespace {

LogicalResult ConvertLayout(mlir::Operation* op, const mlir::ArrayAttr& layout,
                            xla::ShapeProto* shape) {
  // In the case of tuples, ShapeProtos can be nested, and so can the mlir
  // attribute describing the layout. So recurse into the subshapes in both data
  // structures in parallel.
  if (shape->element_type() == xla::TUPLE) {
    auto subshapes = shape->mutable_tuple_shapes();

    // 'layout' does not take the token attribute into account, so skip the
    // corresponding entry from xla shape proto.
    size_t subshapes_data_size = subshapes->size();
    if (!subshapes->empty() &&
        subshapes->Mutable(subshapes->size() - 1)->element_type() == xla::TOKEN)
      subshapes_data_size = subshapes->size() - 1;

    if (layout.size() != subshapes_data_size) {
      op->emitOpError() << "Expected layout of size " << layout.size()
                        << ", but found " << subshapes->size();
      return failure();
    }
    for (int i = 0; i < subshapes_data_size; i++) {
      mlir::Attribute child = layout[i];
      if (child.isa<mlir::UnitAttr>()) {
        // ignore unit attributes, they are used only for tokens.
        continue;
      }
      mlir::ArrayAttr c = child.dyn_cast<mlir::ArrayAttr>();
      if (!c) {
        op->emitOpError() << "Type Error: Expected layout array attribute";
        return failure();
      }
      if (failed(ConvertLayout(op, c, subshapes->Mutable(i)))) {
        return failure();
      }
    }
  } else {
    int rank = shape->dimensions().size();
    if (rank) {
      if (layout.size() != rank) {
        return failure();  // pass error down
      }
      std::vector<int64_t> array(rank);
      for (int i = 0; i < rank; i++) {
        mlir::IntegerAttr attr = layout[i].dyn_cast<mlir::IntegerAttr>();
        if (!attr) {
          op->emitOpError() << "Type Error: Expected layout integer attribute";
          return failure();
        }
        array[i] = attr.getInt();
      }
      *shape->mutable_layout() = xla::LayoutUtil::MakeLayout(array).ToProto();
    }
  }
  return success();
}

// Assigns layouts from 'layout' to shape.
// The function accepts any of the following shapes
//   one or more array-shape(s) of infeed data
//   Tuple(Tuple(zero or more array-shape w.r.t data), token_type)
//
// 'layout' of the mhlo.InfedOp 'op' is
//    [zero or more layout for each array-shape w.r.t data]
// 'layout_index' indexes into 'layout' accessing a layout corresponding to a
// shape.
LogicalResult ConvertInfeedtLayout(mlir::Operation* op,
                                   const mlir::ArrayAttr& layout,
                                   xla::ShapeProto* shape,
                                   int64_t layout_index = 0) {
  if (shape->element_type() != xla::TUPLE) {
    // Handles following shape:
    //   single array-shape of infeed data
    mlir::ArrayAttr child_layout =
        layout[layout_index].dyn_cast<mlir::ArrayAttr>();
    if (!child_layout) {
      op->emitOpError() << "Type Error: Expected layout array attribute";
      return failure();
    }

    int rank = shape->dimensions().size();
    if (rank) {
      if (child_layout.size() != rank) {
        return failure();  // pass error down
      }
      std::vector<int64_t> array(rank);
      for (int i = 0; i < rank; i++) {
        mlir::IntegerAttr attr = child_layout[i].dyn_cast<mlir::IntegerAttr>();
        if (!attr) {
          op->emitOpError() << "Type Error: Expected layout integer attribute";
          return failure();
        }
        array[i] = attr.getInt();
      }
      *shape->mutable_layout() = xla::LayoutUtil::MakeLayout(array).ToProto();
    }

    return success();
  }

  auto subshapes = shape->mutable_tuple_shapes();
  auto datashape = subshapes->Mutable(0);

  if (datashape->element_type() == xla::TUPLE) {
    //   Handles following shapes:
    //     (Tuple(zero or more array-shape w.r.t data), token_type)
    auto data_subshapes = datashape->mutable_tuple_shapes();
    if (layout.size() != data_subshapes->size()) {
      op->emitOpError() << "Expected " << data_subshapes->size()
                        << " layout attribute(s) for infeed data, but found "
                        << layout.size();
      return failure();
    }

    for (int i = 0; i < data_subshapes->size(); i++) {
      if (failed(
              ConvertInfeedtLayout(op, layout, data_subshapes->Mutable(i), i)))
        return failure();
    }
  } else {
    //   Handles following shapes:
    //     array-shapes of two or more infeed data
    if (layout.size() != subshapes->size()) {
      op->emitOpError() << "Expected " << subshapes->size()
                        << " layout attribute(s) for infeed data, but found "
                        << layout.size();
      return failure();
    }

    for (int i = 0; i < subshapes->size(); i++) {
      if (failed(ConvertInfeedtLayout(op, layout, subshapes->Mutable(i), i)))
        return failure();
    }
  }

  return success();
}

// MHLO and XLA HLO disagree on the meaning of addition of `pred` / `i1`, so
// there has to be a special case somewhere to account for the difference.  To
// get the expected behavior of an `AddOp` on `i1`, we have to use `xor`.  Since
// the majority of the conversion is generated code, we just sidestep it here
// for this single case, and inline the code to emit an `xor`.
LogicalResult ExportXlaOperatorWrapped(mlir::Operation* inst,
                                       OpLoweringContext ctx) {
  auto op = dyn_cast<mlir::mhlo::AddOp>(inst);
  if (op && op.getResult()
                .getType()
                .cast<mlir::TensorType>()
                .getElementType()
                .isSignlessInteger(1)) {
    auto& value_map = *ctx.values;
    auto result = op.getResult();
    xla::XlaOp xla_arg_0;
    if (failed(GetXlaOp(op.getLhs(), value_map, &xla_arg_0, op)))
      return mlir::failure();
    xla::XlaOp xla_arg_1;
    if (failed(GetXlaOp(op.getRhs(), value_map, &xla_arg_1, op)))
      return mlir::failure();
    auto xla_result = xla::Xor(Unwrap(xla_arg_0), Unwrap(xla_arg_1));
    value_map[result] = xla_result;
    return mlir::success();
  }

  return ExportXlaOperator(inst, ctx);
}

LogicalResult ConvertToHloModule::Lower(
    mlir::Operation* inst, bool is_entry_function,
    llvm::ArrayRef<std::optional<xla::OpSharding>> ret_shardings,
    xla::XlaBuilder* builder,
    ConvertToHloModule::ValueLoweringMap* value_lowering,
    xla::XlaOp* return_value) {
  // Explicitly fail for ops that are not supported for export.
  if (inst->getDialect() !=
          inst->getContext()->getLoadedDialect<mlir::mhlo::MhloDialect>() &&
      !mlir::isa<mlir::func::ConstantOp, mlir::arith::ConstantOp,
                 mlir::func::CallOp, mlir::tensor::CastOp,
                 mlir::func::ReturnOp>(inst)) {
    inst->emitOpError("unsupported op for export to XLA");
    return failure();
  }

  *return_value = xla::XlaOp();

  // See MlirToHloConversionOptions for more about layouts.
  auto propagate_layouts = [this](mlir::Operation* inst,
                                  xla::XlaOp xla_op) -> mlir::LogicalResult {
    if (options_.propagate_layouts) {
      auto* shape = xla::internal::XlaBuilderFriend::GetInstruction(xla_op)
                        ->mutable_shape();
      // TODO(kramm): merge this with ConvertLayout.
      mlir::FailureOr<xla::Shape> mlir_shape_or = ExtractXlaShape(inst);
      if (failed(mlir_shape_or)) return failure();
      *shape = mlir_shape_or->ToProto();
    }

    return success();
  };

  if (succeeded(ExportXlaOperatorWrapped(
          inst,
          {value_lowering, this, builder, &stack_frame_indexes_builder_}))) {
    if (inst->getNumResults() == 1) {
      auto iter = value_lowering->find(inst->getResult(0));
      if (iter == value_lowering->end()) {
        inst->emitOpError(
            "inst has a result, but it's not found in value_lowering");
        return failure();
      }
      if (failed(propagate_layouts(inst, iter->second))) {
        return failure();
      }
    }
    // For infeed ops stemming back to InfeedDequeueTuple, respect the
    // layout attribute, and create the corresponding layout in hlo.
    if (isa<mhlo::InfeedOp>(inst)) {
      mlir::ArrayAttr layout =
          inst->getAttrOfType<mlir::ArrayAttr>(kLayoutAttr);

      if (layout) {
        // We propagate layout to the following three ops:
        // L1: For each data-result of mhlo.InfeedOp, we find the exported
        // xla::kGetTupleElement and propagate the layout.
        //
        // L2: For the token-result of mhlo.InfeedOp (result at last index),
        // we extract the xla::kInfeed op using the corresponding
        // xla::kGetTupleElement and propagate the layout to it.
        //
        // L3: In case there are non-zero data-results, there exists an
        // additional xla::kGetTupleElement accessing a tuple of the
        // data-results. We need to propagate the layout to that
        // xla::kGetTupleElement as well.
        auto num_results = inst->getNumResults();
        bool propagate_layout_to_data_tuple = true;
        for (unsigned i = 0; i < num_results; i++) {
          auto iter = value_lowering->find(inst->getResult(i));
          if (iter == value_lowering->end()) {
            inst->emitOpError() << "inst's result value at index " << i
                                << " has no match in value_lowering";
            return failure();
          }
          auto xla_gte_op = iter->second;
          xla::HloInstructionProto* get_tuple_element_proto =
              xla::internal::XlaBuilderFriend::GetInstruction(xla_gte_op);

          assert(xla::StringToHloOpcode(get_tuple_element_proto->opcode())
                         .value() == xla::HloOpcode::kGetTupleElement &&
                 "The token-result of mhlo.InfeedOp should be mapped to a "
                 "xla::HloOpcode::kGetTupleElement");

          if (i == num_results - 1) {
            // L2
            xla::HloInstructionProto* xla_infeed_op_proto =
                xla::internal::XlaBuilderFriend::GetInstructionByHandle(
                    xla_gte_op.builder(),
                    get_tuple_element_proto->operand_ids(0));

            assert(
                xla::StringToHloOpcode(xla_infeed_op_proto->opcode()).value() ==
                    xla::HloOpcode::kInfeed &&
                "Expected xla::HloOpcode::kInfeed op");

            auto* shape = xla_infeed_op_proto->mutable_shape();
            if (failed(ConvertInfeedtLayout(inst, layout, shape)))
              return failure();

          } else {
            // L1
            auto* shape = get_tuple_element_proto->mutable_shape();
            if (failed(ConvertInfeedtLayout(inst, layout, shape, i)))
              return failure();

            // L3
            if (propagate_layout_to_data_tuple) {
              xla::HloInstructionProto* data_tuple_proto =
                  xla::internal::XlaBuilderFriend::GetInstructionByHandle(
                      xla_gte_op.builder(),
                      get_tuple_element_proto->operand_ids(0));
              auto* data_tuple_shape = data_tuple_proto->mutable_shape();

              assert(
                  xla::StringToHloOpcode(data_tuple_proto->opcode()).value() ==
                      xla::HloOpcode::kGetTupleElement &&
                  "Expected a xla:tupleOp for all the data results.");
              if (failed(ConvertInfeedtLayout(inst, layout, data_tuple_shape)))
                return failure();
            }
            propagate_layout_to_data_tuple = false;
          }
        }
      }
    }
    return success();
  }

  auto& value_map = *value_lowering;
  ElementsAttr const_attr;

  if (auto call_op = dyn_cast<mlir::func::CallOp>(inst)) {
    return LowerFunctionCall(call_op, builder, &value_map);
  }

  if (auto op = dyn_cast<mlir::tensor::CastOp>(inst)) {
    Value operand = op.getOperand();
    auto ty = operand.getType().dyn_cast<ShapedType>();
    // If this was a cast from a static or bounded tensors, then it is a noop
    // for export to HLO and we can use the operand.
    if (!ty || !IsBoundedOrStatic(ty)) {
      inst->emitOpError()
          << "requires static or bounded operand for HLO translation";
      return failure();
    }

    xla::XlaOp xla_operand;
    if (failed(GetXlaOp(operand, value_map, &xla_operand, op)))
      return failure();
    value_map[op.getResult()] = xla_operand;
    if (failed(propagate_layouts(inst, xla_operand))) {
      return failure();
    }
    return success();
  }

  if (matchPattern(inst, m_Constant(&const_attr))) {
    if (!inst->getResult(0).getType().isa<ShapedType>()) {
      return inst->emitError(
          "expected shaped type during constant mhlo -> hlo translation");
    }

    mlir::FailureOr<xla::Shape> shape_or = ExtractXlaShape(inst);
    if (failed(shape_or)) return failure();
    auto literal_or =
        CreateArrayLiteralFromAttr(const_attr, shape_or->layout());
    if (!literal_or.ok())
      return inst->emitError(literal_or.status().ToString());
    auto constant = xla::ConstantLiteral(builder, literal_or.value());
    value_map[inst->getResult(0)] = constant;

    return success();
  }

  if (isa<mhlo::ReturnOp, mlir::func::ReturnOp>(inst)) {
    // Construct the return value for the function. If there is a single value
    // returned, then return it directly, else create a tuple and return.
    unsigned num_return_values = inst->getNumOperands();
    std::optional<xla::OpSharding> ret_tuple_sharding =
        CreateTupleSharding(ret_shardings);
    if ((return_tuple_ && is_entry_function) || num_return_values != 1) {
      std::vector<xla::XlaOp> returns(num_return_values);
      for (OpOperand& ret : inst->getOpOperands()) {
        unsigned index = ret.getOperandNumber();
        xla::XlaOp operand;
        if (failed(GetXlaOp(ret.get(), value_map, &operand, inst)))
          return failure();

        returns[index] = operand;
        if (!is_entry_function || !ret_tuple_sharding) continue;

        xla::Shape return_shape = xla::TypeToShape(ret.get().getType());
        absl::StatusOr<xla::XlaOp> reshape =
            ReshapeWithCorrectRepresentationAndSharding(
                builder, returns[index], return_shape,
                options_.layout_preference_fn, options_.shape_representation_fn,
                ret_shardings[index], /*fast_mem=*/false);
        if (!reshape.ok())
          return inst->emitError() << reshape.status().message();

        returns[index] = reshape.value();
      }

      xla::XlaScopedShardingAssignment scoped_sharding(builder,
                                                       ret_tuple_sharding);
      *return_value = xla::Tuple(builder, returns);
    } else if (num_return_values == 1) {
      xla::XlaOp operand;
      if (failed(GetXlaOp(inst->getOperand(0), value_map, &operand, inst)))
        return failure();

      if (ret_tuple_sharding) {
        auto tuple = Tuple(builder, {operand});
        builder->SetSharding(*ret_shardings[0]);
        *return_value = GetTupleElement(tuple, 0);
        builder->ClearSharding();
      } else {
        *return_value = operand;
      }
    }

    return success();
  }

  inst->emitOpError() << "can't be translated to XLA HLO";
  return failure();
}

LogicalResult ConvertToHloModule::LowerFunctionCall(
    mlir::func::CallOp call_op, xla::XlaBuilder* builder,
    ConvertToHloModule::ValueLoweringMap* value_lowering) {
  auto& value_map = *value_lowering;
  mlir::func::FuncOp callee =
      module_.lookupSymbol<mlir::func::FuncOp>(call_op.getCallee());
  if (failed(RunOnFunction(callee))) return failure();
  std::vector<xla::XlaOp> operands;
  for (auto operand : call_op.getOperands()) {
    xla::XlaOp xla_operand;
    if (failed(GetXlaOp(operand, value_map, &xla_operand, call_op)))
      return failure();
    operands.push_back(xla_operand);
  }
  // Each call to xla::Call would insert a copy of the computation to
  // the HLO. Thus each callsite would have a unique callee in the
  // exported HLO. HLO syntactically does not require all calls to have unique
  // callees, but eventually before lowering call graph is "flattened" to
  // make that true. This is done before lowering because buffer assignment
  // needs this invariant.
  xla::FrontendAttributes fe_attrs = CreateXlaFrontendAttributesFromOp(call_op);
  xla::XlaScopedFrontendAttributesAssignment assignment(builder, fe_attrs);
  xla::XlaOp call_result =
      xla::Call(builder, lowered_computation_[callee], operands);
  // Use GetTupleElement for multiple outputs
  unsigned num_results = call_op.getNumResults();
  if (num_results > 1) {
    for (unsigned i = 0; i != num_results; ++i) {
      value_map[call_op.getResult(i)] = xla::GetTupleElement(call_result, i);
    }
  } else if (num_results == 1) {
    value_map[call_op.getResult(0)] = call_result;
  }
  return success();
}

LogicalResult ConvertToHloModule::RunOnFunction(mlir::func::FuncOp f) {
  if (lowered_computation_.count(f)) return success();
  if (!llvm::hasSingleElement(f)) {
    return f.emitError("only single block Function supported");
  }

  // Create a sub-builder if this is not the main function.
  std::unique_ptr<xla::XlaBuilder> builder_up;
  bool entry_function = f.getName() == "main";
  if (!entry_function)
    builder_up = module_builder_.CreateSubBuilder(f.getName().str());
  auto& builder = entry_function ? module_builder_ : *builder_up;

  xla::XlaComputation computation;
  std::vector<bool> entry_args_same_across_replicas;
  llvm::SmallVector<std::optional<xla::OpSharding>, 4> arg_shardings;
  llvm::SmallVector<std::optional<xla::OpSharding>, 4> ret_shardings;
  llvm::SmallVector<std::optional<xla::FrontendAttributes>, 4> arg_fe_attrs;
  if (entry_function) {
    bool any_arg_replicated = false;
    entry_args_same_across_replicas.reserve(f.getNumArguments());
    for (int64_t i = 0; i < f.getNumArguments(); ++i) {
      auto attr = f.getArgAttrOfType<mlir::BoolAttr>(i, kReplicationAttr);
      entry_args_same_across_replicas.push_back(attr != nullptr &&
                                                attr.getValue());
      any_arg_replicated |= entry_args_same_across_replicas.back();
      // Pass the alias info to the builder so that it will build the alias info
      // into the resulting HloModule.
      auto buffer_donor =
          f.getArgAttrOfType<mlir::BoolAttr>(i, "jax.buffer_donor");
      if (buffer_donor) {
        if (use_tuple_args_) {
          builder.AddBufferDonor(/*param_number=*/0, /*param_index=*/{i});
        } else {
          builder.AddBufferDonor(/*param_number=*/i, /*param_index=*/{});
        }
      }
      auto aliasing_output =
          f.getArgAttrOfType<mlir::IntegerAttr>(i, "tf.aliasing_output");
      if (!aliasing_output) continue;
      xla::ShapeIndex output_index;
      if ((return_tuple_ && entry_function) || f.getNumResults() != 1) {
        output_index = {aliasing_output.getInt()};
      } else {
        if (aliasing_output.getInt() != 0) {
          return f.emitError(
              "Aliasing output must be 0 if only one output exists");
        }
        output_index = {};
      }
      if (use_tuple_args_) {
        builder.SetUpAlias(output_index, /*param_number=*/0,
                           /*param_index=*/{i});
      } else {
        builder.SetUpAlias(output_index, /*param_number=*/i,
                           /*param_index=*/{});
      }
    }
    // Do not populate this field when nothing is replicated, since empty field
    // means no replication. This avoids the need for unrelated tests to handle
    // this field.
    if (!any_arg_replicated) entry_args_same_across_replicas.clear();

    ExtractShardingsFromFunction(f, &arg_shardings, &ret_shardings);
    ExtractFrontendAttributesFromFunction(f, &arg_fe_attrs);
  }
  if (failed(LowerBasicBlockAsFunction(&f.front(), &builder, entry_function,
                                       false, entry_args_same_across_replicas,
                                       arg_shardings, ret_shardings,
                                       arg_fe_attrs, &computation))) {
    return failure();
  }
  if (auto execution_thread =
          f->getAttrOfType<mlir::StringAttr>("execution_thread")) {
    computation.mutable_proto()->mutable_computations(0)->set_execution_thread(
        execution_thread.str());
  }
  for (int i = 0; i < f.getNumArguments(); ++i) {
    if (auto pr =
            f.getArgAttrOfType<mlir::ArrayAttr>(i, kParameterReplicationAttr)) {
      for (auto b : pr.getValue())
        for (auto& instr : *computation.mutable_proto()
                                ->mutable_computations(0)
                                ->mutable_instructions())
          if (instr.parameter_number() == i)
            instr.mutable_parameter_replication()
                ->add_replicated_at_leaf_buffers(
                    b.cast<mlir::BoolAttr>().getValue());
    }
  }
  lowered_computation_[f] = std::move(computation);
  return success();
}

LogicalResult ConvertToHloModule::SetEntryTupleShapesAndLeafReplication(
    Block* block, const std::vector<bool>& entry_args_same_across_replicas,
    llvm::SmallVectorImpl<xla::Shape>* arg_shapes,
    std::vector<bool>* leaf_replication) {
  arg_shapes->reserve(block->getNumArguments());
  leaf_replication->reserve(block->getNumArguments());
  for (BlockArgument& arg : block->getArguments()) {
    arg_shapes->push_back(xla::TypeToShape(arg.getType()));
    xla::Shape& arg_shape = arg_shapes->back();
    auto layout_preference_status =
        options_.layout_preference_fn ? options_.layout_preference_fn(arg_shape)
                                      : XlaLayoutPreference::kNoPreference;
    if (!layout_preference_status.ok())
      return block->getParentOp()->emitError()
             << layout_preference_status.status().message();

    auto arg_shape_status = options_.shape_representation_fn
                                ? options_.shape_representation_fn(
                                      arg_shape, /*use_fast_memory=*/false,
                                      layout_preference_status.value())
                                : arg_shape;
    if (!arg_shape_status.ok())
      return block->getParentOp()->emitError()
             << arg_shape_status.status().message();

    arg_shape = std::move(arg_shape_status.value());

    if (entry_args_same_across_replicas.empty()) continue;
    for (int i = 0, e = xla::ShapeUtil::GetLeafCount(arg_shape); i < e; ++i)
      leaf_replication->push_back(
          entry_args_same_across_replicas[arg.getArgNumber()]);
  }

  return success();
}

LogicalResult ConvertToHloModule::SetEntryTupleShardings(
    Block* block, xla::XlaBuilder* builder,
    llvm::ArrayRef<std::optional<xla::OpSharding>> arg_shardings,
    llvm::SmallVectorImpl<xla::Shape>* arg_shapes) {
  if (!arg_shardings.empty() && SomeOptionalShardingsAreSet(arg_shardings)) {
    xla::OpSharding sharding;
    sharding.set_type(xla::OpSharding::TUPLE);
    for (const auto& arg_sharding : llvm::enumerate(arg_shardings)) {
      if (arg_sharding.value().has_value()) {
        auto hlo_sharding = xla::HloSharding::FromProto(*arg_sharding.value());
        if (!hlo_sharding.ok())
          return block->getParentOp()->emitError()
                 << hlo_sharding.status().message();

        auto status = RewriteLayoutWithShardedShape(
            hlo_sharding.value(), /*use_fast_memory=*/false,
            options_.layout_preference_fn, options_.shape_representation_fn,
            &(*arg_shapes)[arg_sharding.index()]);
        if (!status.ok())
          return block->getParentOp()->emitError() << status.message();

        *sharding.add_tuple_shardings() = *arg_sharding.value();
      } else {
        xla::OpSharding fallback_sharding;
        fallback_sharding.set_type(xla::OpSharding::REPLICATED);
        *sharding.add_tuple_shardings() = fallback_sharding;
      }
    }

    builder->SetSharding(sharding);
  }

  return success();
}

LogicalResult ConvertToHloModule::LowerBasicBlockAsFunction(
    Block* block, xla::XlaBuilder* builder, bool is_entry_function,
    bool ensure_single_arg,
    const std::vector<bool>& entry_args_same_across_replicas,
    llvm::ArrayRef<std::optional<xla::OpSharding>> arg_shardings,
    llvm::ArrayRef<std::optional<xla::OpSharding>> ret_shardings,
    llvm::ArrayRef<std::optional<xla::FrontendAttributes>> fe_attrs,
    xla::XlaComputation* result,
    std::optional<llvm::ArrayRef<mlir::Value>> implicit_operands) {
  // Mapping from the Value to lowered XlaOp.
  ValueLoweringMap lowering;

  // If using tuples as input, then there is only one input parameter that is a
  // tuple.
  if (is_entry_function && use_tuple_args_) {
    llvm::SmallVector<xla::Shape, 4> arg_shapes;
    std::vector<bool> leaf_replication;
    if (failed(SetEntryTupleShapesAndLeafReplication(
            block, entry_args_same_across_replicas, &arg_shapes,
            &leaf_replication)))
      return failure();

    if (failed(
            SetEntryTupleShardings(block, builder, arg_shardings, &arg_shapes)))
      return failure();

    xla::Shape input_shape = xla::ShapeUtil::MakeTupleShape(arg_shapes);
    auto tuple =
        xla::Parameter(builder, 0, input_shape, "arg_tuple", leaf_replication);
    builder->ClearSharding();

    for (BlockArgument& arg : block->getArguments()) {
      xla::XlaScopedShardingAssignment scoped_sharding(
          builder, arg_shardings.empty() ? std::nullopt
                                         : arg_shardings[arg.getArgNumber()]);
      lowering[arg] = xla::GetTupleElement(tuple, arg.getArgNumber());
    }
  } else {
    if (ensure_single_arg) {
      // Applicable for mhlo.IfOp or mhlo.CaseOp or mhlo.WhileOp.
      llvm::SmallVector<xla::Shape, 4> arg_shapes;

      // The arguments of `block` are ignored if `implicit_operands` is set,
      // therefore `arg_shardings` should be empty in that case.
      assert(arg_shardings.empty() || !implicit_operands);

      auto args_size = block->getNumArguments();
      if (implicit_operands) args_size = implicit_operands->size();

      arg_shapes.reserve(args_size);
      if (implicit_operands) {
        for (auto implicit_operand : *implicit_operands)
          arg_shapes.push_back(xla::TypeToShape(implicit_operand.getType()));
      } else {
        for (BlockArgument& arg : block->getArguments())
          arg_shapes.push_back(xla::TypeToShape(arg.getType()));
      }

      if (args_size > 1) {
        xla::XlaScopedShardingAssignment scoped_sharding(
            builder, arg_shardings.empty()
                         ? std::nullopt
                         : CreateTupleSharding(arg_shardings));
        auto tuple = xla::Parameter(builder, 0,
                                    xla::ShapeUtil::MakeTupleShape(arg_shapes),
                                    "arg_tuple");

        if (implicit_operands) {
          int arg_index = 0;
          for (auto implicit_operand : *implicit_operands)
            lowering[implicit_operand] =
                xla::GetTupleElement(tuple, arg_index++);
        } else {
          for (BlockArgument& arg : block->getArguments()) {
            auto num = arg.getArgNumber();
            xla::XlaScopedShardingAssignment scoped_sharding(
                builder,
                arg_shardings.empty() ? std::nullopt : arg_shardings[num]);
            lowering[arg] = xla::GetTupleElement(tuple, num);
          }
        }
      } else if (args_size == 1) {
        if (implicit_operands) {
          lowering[(*implicit_operands)[0]] =
              xla::Parameter(builder, 0, arg_shapes[0], "Arg_");
        } else {
          xla::XlaScopedShardingAssignment scoped_sharding(
              builder,
              arg_shardings.empty() ? std::nullopt : arg_shardings.front());
          lowering[block->getArgument(0)] =
              xla::Parameter(builder, 0, arg_shapes[0], "Arg_");
        }
      } else {
        // Applicable only for IfOp or CaseOp. No implicit operands implies no
        // xla parameters. In this case, we create an empty tuple as the
        // block-parameter.
        xla::Parameter(builder, 0, xla::ShapeUtil::MakeTupleShape(arg_shapes),
                       "arg_empty_tuple");
      }
    } else {
      for (BlockArgument& arg : block->getArguments()) {
        auto num = arg.getArgNumber();
        xla::Shape shape = xla::TypeToShape(arg.getType());
        xla::XlaScopedShardingAssignment scoped_sharding(
            builder, arg_shardings.empty() ? std::nullopt : arg_shardings[num]);
        if (!fe_attrs.empty() && fe_attrs[num]) {
          // Populates frontend attributes for parameters only for the entry
          // functions with no tuple args.
          builder->SetFrontendAttributes(*fe_attrs[num]);
        }
        if (entry_args_same_across_replicas.empty()) {
          lowering[arg] =
              xla::Parameter(builder, num, shape, absl::StrCat("Arg_", num));
        } else {
          lowering[arg] = xla::Parameter(
              builder, num, shape, absl::StrCat("Arg_", num),
              std::vector<bool>(entry_args_same_across_replicas[num],
                                xla::ShapeUtil::GetLeafCount(shape)));
        }
        builder->ClearFrontendAttributes();
      }
    }
  }

  xla::XlaOp return_value;
  for (auto& inst : *block)
    if (failed(Lower(&inst, is_entry_function, ret_shardings, builder,
                     &lowering, &return_value)))
      return failure();

  // Build the XlaComputation and check for failures.
  auto computation_or =
      return_value.valid() ? builder->Build(return_value) : builder->Build();
  if (!computation_or.ok()) {
    block->back().emitError() << computation_or.status().message();
    return failure();
  }
  *result = std::move(computation_or.value());
  return success();
}

LogicalResult ConvertToHloModule::LowerRegionAsComputation(
    mlir::Region* region, xla::XlaComputation* func,
    std::optional<llvm::ArrayRef<mlir::Value>> implicit_operands,
    bool ensure_single_arg,
    llvm::ArrayRef<std::optional<xla::OpSharding>> arg_shardings,
    llvm::ArrayRef<std::optional<xla::OpSharding>> ret_shardings) {
  std::unique_ptr<xla::XlaBuilder> builder =
      module_builder_.CreateSubBuilder(absl::StrCat("region_", region_id_++));
  return LowerBasicBlockAsFunction(&region->front(), builder.get(),
                                   /*is_entry_function=*/false,
                                   /*ensure_single_arg*/ ensure_single_arg,
                                   /*entry_args_same_across_replicas=*/{},
                                   arg_shardings, ret_shardings,
                                   /*fe_attrs=*/{}, func, implicit_operands);
}

// Runs the PrepareForExport pass on the ModuleOp.
xla::Status PrepareForExport(mlir::ModuleOp module) {
  bool hasShapeOps = false;
  module.walk([&](Operation* op) {
    hasShapeOps |= isa<shape::ShapeDialect>(op->getDialect());
    hasShapeOps |= isa<mhlo::ComputeReshapeShapeOp, mhlo::CstrReshapableOp>(op);
    return hasShapeOps ? WalkResult::interrupt() : WalkResult::advance();
  });
  mlir::PassManager pm(module.getContext());
  pm.addNestedPass<mlir::func::FuncOp>(mhlo::createPrepareForExportPass());
  if (hasShapeOps) {
    // Experimental support for exporting dynamic MHLO programs to HLO.
    // Only bounded dynamism is planned to be supported; unbounded dynamism
    // is out of scope for now.
    pm.addNestedPass<mlir::func::FuncOp>(
        mhlo::createSymbolicShapeOptimizationPass());
    pm.addNestedPass<mlir::func::FuncOp>(mhlo::createShapeLegalizeToHloPass());
  }
  if (failed(pm.run(module)))
    return tsl::errors::Internal("Unable to prepare for XLA export");
  return absl::OkStatus();
}

}  // namespace

xla::Status ConvertMlirHloToHlo(mlir::ModuleOp module, xla::HloProto* hlo_proto,
                                bool use_tuple_args, bool return_tuple,
                                MlirToHloConversionOptions options) {
  // To support the ongoing migration of XLA's compiler interface from MHLO
  // to StableHLO, we've inserted this fallback to provide support for backends
  // which are converting incoming ModuleOps directly to HLO.
  // xla::MlirToXlaComputation is a better API for this purpose because it
  // supports not just MHLO, but also CHLO and StableHLO, but we will
  // temporarily support StableHLO to MHLO lowering here as well to ensure
  // a smooth migration.
  // TODO(b/263811577): Remove this functionality once we have reasonable
  // confidence that everyone has migrated from calling ConvertMlirHloToHlo
  // directly.
  bool hasStablehloOps = false;
  module.walk([&](Operation* op) {
    hasStablehloOps |= isa<stablehlo::StablehloDialect>(op->getDialect());
    return hasStablehloOps ? WalkResult::interrupt() : WalkResult::advance();
  });
  if (hasStablehloOps) {
    mlir::PassManager pm(module->getContext());
    pm.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());
    if (failed(pm.run(module)))
      return tsl::errors::Internal("Unable to convert StableHLO to MHLO");
  }

  TF_RETURN_IF_ERROR(PrepareForExport(module));
  mlir::BaseScopedDiagnosticHandler diag_handler(module.getContext());
  xla::XlaBuilder module_builder("main");
  ConvertToHloModule converter(module, module_builder, use_tuple_args,
                               return_tuple, options);
  if (failed(converter.Run())) return diag_handler.ConsumeStatus();
  auto hlo_module = converter.ConsumeMainProto();
  StringRef module_name = module.getName() ? *module.getName() : "main";
  hlo_module.set_name(module_name.str());
  if (auto cross_program_prefetches = module->getAttrOfType<mlir::ArrayAttr>(
          "mhlo.cross_program_prefetches")) {
    for (const auto& prefetch :
         Convert_cross_program_prefetches(cross_program_prefetches)) {
      *hlo_module.add_cross_program_prefetches() = std::move(prefetch);
    }
  }
  if (auto is_dynamic =
          module->getAttrOfType<mlir::BoolAttr>("mhlo.is_dynamic")) {
    hlo_module.set_is_dynamic(is_dynamic.getValue());
  }
  if (auto frontend_attributes =
          module->getAttrOfType<DictionaryAttr>(kFrontendAttributesAttr)) {
    ConstructFrontendAttributesFromAttribute(
        frontend_attributes, *hlo_module.mutable_frontend_attributes());
  }
  if (auto use_auto_spmd_partitioning = module->getAttrOfType<mlir::BoolAttr>(
          "mhlo.use_auto_spmd_partitioning")) {
    hlo_module.set_use_auto_spmd_partitioning(
        use_auto_spmd_partitioning.getValue());
  }
  if (auto spmd_output_sharding = module->getAttrOfType<mlir::StringAttr>(
          "mhlo.spmd_output_sharding")) {
    *hlo_module.mutable_spmd_output_sharding() =
        *xla::ConvertSharding(spmd_output_sharding.getValue());
  }
  if (auto spmd_parameters_sharding = module->getAttrOfType<mlir::ArrayAttr>(
          "mhlo.spmd_parameters_shardings")) {
    for (const auto& sharding : spmd_parameters_sharding.getValue()) {
      *hlo_module.add_spmd_parameters_shardings() =
          *xla::ConvertSharding(sharding.cast<mlir::StringAttr>().getValue());
    }
  }

  xla::StackFrameIndexProto stack_frame_index =
      converter.BuildStackFramesIndexProto();
  hlo_module.mutable_stack_frame_index()->Swap(&stack_frame_index);
  hlo_proto->mutable_hlo_module()->Swap(&hlo_module);
  return absl::OkStatus();
}

xla::Status BuildHloFromMlirHlo(mlir::Block& block, xla::XlaBuilder& builder,
                                llvm::ArrayRef<xla::XlaOp> xla_params,
                                std::vector<xla::XlaOp>& returns,
                                MlirToHloConversionOptions options) {
  auto module = block.getParentOp()->getParentOfType<mlir::ModuleOp>();
  TF_RETURN_IF_ERROR(PrepareForExport(module));
  ConvertToHloModule converter(module, builder,
                               /*use_tuple_args=*/false, /*return_tuple=*/false,
                               options);

  ConvertToHloModule::ValueLoweringMap lowering;
  // xla_params should only include non-constant parameters the block arguments
  // correspond to.
  if (xla_params.size() != block.getArguments().size())
    return tsl::errors::Internal("xla_params size (", xla_params.size(),
                                 ") != block arguments size (",
                                 block.getArguments().size(), ")");
  for (BlockArgument& arg : block.getArguments()) {
    auto num = arg.getArgNumber();
    lowering[arg] = xla_params[num];
  }

  mlir::BaseScopedDiagnosticHandler diag_handler(module.getContext());
  for (auto& inst : block) {
    if (isa<mhlo::ReturnOp, mlir::func::ReturnOp>(inst)) {
      returns.resize(inst.getNumOperands());
      for (OpOperand& ret : inst.getOpOperands()) {
        unsigned index = ret.getOperandNumber();
        xla::XlaOp operand;
        if (failed(GetXlaOp(ret.get(), lowering, &operand, &inst)))
          return diag_handler.ConsumeStatus();
        returns[index] = operand;
      }
    } else {
      xla::XlaOp return_value;
      if (failed(converter.Lower(&inst, /*is_entry_function=*/true,
                                 /*ret_shardings=*/{}, &builder, &lowering,
                                 &return_value)))
        return diag_handler.ConsumeStatus();
    }
  }

  return absl::OkStatus();
}

}  // namespace mlir
