/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/layout_canonicalizer.h"

#include <cstdint>
#include <iterator>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/permutation_util.h"
#include "xla/service/layout_assignment.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {
namespace {

std::vector<int64_t> CanonicalizeInstructionLayout(HloInstruction* instr,
                                                   bool is_entry_root);

bool IsLayoutDescending(const Shape& shape) {
  return absl::c_is_sorted(shape.layout().minor_to_major(),
                           [](int64_t a, int64_t b) { return a > b; });
}

// Given an instruction (with non-tuple output shape), this function updates the
// output shape such that the layout is descending. It returns the
// major-to-minor layout ordering which will be used when instr is used as an
// operand.
std::vector<int64_t> HandleOutput(HloInstruction* instr) {
  CHECK(!instr->shape().IsTuple());
  if (IsLayoutDescending(instr->shape())) {
    return {};
  }
  // Create the major-to-minor ordering to construct the new logical dimensions
  std::vector<int64_t> major_to_minor;
  absl::c_reverse_copy(instr->shape().layout().minor_to_major(),
                       std::back_inserter(major_to_minor));

  // Compose shape's dimensions with the major-to-minor layout
  std::vector<int64_t> input_new_logical_dims =
      ComposePermutations(instr->shape().dimensions(), major_to_minor);

  // Update the shape
  *instr->mutable_shape() = ShapeUtil::MakeShapeWithDescendingLayout(
      instr->shape().element_type(), input_new_logical_dims);
  return major_to_minor;
}

std::vector<int64_t> HandleBroadcast(HloInstruction* broadcast,
                                     bool is_entry_root) {
  VLOG(3) << "HandleBroadcast: " << broadcast->name();
  // Handle broadcast input
  HloInstruction* operand = broadcast->mutable_operand(0);
  std::vector<int64_t> operand_map =
      CanonicalizeInstructionLayout(operand, false);
  VLOG(3) << "operand_map = " << absl::StrJoin(operand_map, ",");

  // Handle output
  std::vector<int64_t> output_map;
  if (!is_entry_root) {
    output_map = HandleOutput(broadcast);
  }
  VLOG(3) << "output_map = " << absl::StrJoin(output_map, ",");

  // Compose dimension map with the inverse of the output map.
  if (!output_map.empty()) {
    std::vector<int64_t> inverse_output_map = InversePermutation(output_map);
    VLOG(3) << "inverse_output_map = "
            << absl::StrJoin(inverse_output_map, ",");
    std::vector<int64_t> new_broadcast_dimensions;
    new_broadcast_dimensions.reserve(broadcast->dimensions().size());
    for (int64_t dim : broadcast->dimensions()) {
      new_broadcast_dimensions.push_back(inverse_output_map[dim]);
    }
    VLOG(3) << "dimensions after applying output_map = "
            << absl::StrJoin(new_broadcast_dimensions, ",");
    *broadcast->mutable_dimensions() = new_broadcast_dimensions;
  }

  // Compose dimension map with the operand map.
  if (!operand_map.empty()) {
    std::vector<int64_t> new_broadcast_dimensions =
        ComposePermutations(broadcast->dimensions(), operand_map);
    VLOG(3) << "dimensions after applying operand_map = "
            << absl::StrJoin(new_broadcast_dimensions, ",");
    *broadcast->mutable_dimensions() = new_broadcast_dimensions;
  }
  VLOG(3) << "Broadcast after: " << broadcast->ToString();
  return output_map;
}

std::vector<int64_t> HandleTranspose(HloInstruction* transpose,
                                     bool is_entry_root) {
  VLOG(3) << "HandleTranspose: " << transpose->name();
  // Handle broadcast input
  HloInstruction* operand = transpose->mutable_operand(0);
  std::vector<int64_t> operand_map =
      CanonicalizeInstructionLayout(operand, false);
  VLOG(3) << "operand_map = " << absl::StrJoin(operand_map, ",");

  // Handle output
  std::vector<int64_t> output_map;
  if (!is_entry_root) {
    output_map = HandleOutput(transpose);
  }
  VLOG(3) << "output_map = " << absl::StrJoin(output_map, ",");

  // Compose dimension map with the output map.
  if (!output_map.empty()) {
    std::vector<int64_t> new_transpose_dimensions =
        ComposePermutations(output_map, transpose->dimensions());
    VLOG(3) << "dimensions after applying output_map = "
            << absl::StrJoin(new_transpose_dimensions, ",");
    *transpose->mutable_dimensions() = new_transpose_dimensions;
  }

  // Compose dimension map with the inverse of the operand map.
  if (!operand_map.empty()) {
    std::vector<int64_t> inverse_operand_map = InversePermutation(operand_map);
    VLOG(3) << "inverse_operand_map = "
            << absl::StrJoin(inverse_operand_map, ",");
    std::vector<int64_t> new_transpose_dimensions =
        ComposePermutations(inverse_operand_map, transpose->dimensions());
    VLOG(3) << "dimensions after applying operand_map = "
            << absl::StrJoin(new_transpose_dimensions, ",");
    *transpose->mutable_dimensions() = new_transpose_dimensions;
  }
  VLOG(3) << "Transpose after: " << transpose->ToString();
  return output_map;
}

std::vector<int64_t> CanonicalizeInstructionLayout(HloInstruction* instr,
                                                   bool is_entry_root) {
  if (!LayoutAssignment::InstructionCanChangeLayout(instr)) {
    return {};
  }
  // For now, we only handle broadcast and transpose. I will add other ops
  // gradually.
  switch (instr->opcode()) {
    case HloOpcode::kBroadcast:
      return HandleBroadcast(instr, is_entry_root);
    case HloOpcode::kTranspose:
      return HandleTranspose(instr, is_entry_root);
    default:
      break;
  }
  return {};
}
};  // namespace

absl::StatusOr<bool> LayoutCanonicalizer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(3) << "LayoutCanonicalizer::Run: \n" << module->ToString();
  for (auto* comp : module->MakeNonfusionComputations(execution_threads)) {
    // We only canonicalize the entry computation for now.
    if (comp->IsEntryComputation()) {
      CanonicalizeInstructionLayout(comp->root_instruction(), true);
    }
  }
  return true;
}

}  // namespace xla
