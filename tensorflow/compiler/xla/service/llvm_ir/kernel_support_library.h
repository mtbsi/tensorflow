/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_KERNEL_SUPPORT_LIBRARY_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_KERNEL_SUPPORT_LIBRARY_H_

#include <string>

#include "absl/strings/string_view.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_loop.h"
#include "tensorflow/compiler/xla/service/llvm_ir/llvm_util.h"

namespace xla {
// A thin wrapper around llvm_loop.h to make code generating structured control
// flow more readable.
class KernelSupportLibrary {
 public:
  // `b` is the llvm::IRBuilder instance used to generate LLVM IR.
  // `unroll_mode` specifies the desired LLVM unrolling behavior for every loop
  // generated by this instance of KernelSupportLibrary.
  explicit KernelSupportLibrary(
      llvm::IRBuilder<>* b,
      llvm_ir::UnrollMode unroll_mode = llvm_ir::UnrollMode::kNoUnroll,
      bool prevent_vectorization = true)
      : b_(b),
        unroll_mode_(unroll_mode),
        prevent_vectorization_(prevent_vectorization) {}

  // Generates the following control flow structure:
  //
  //   if (`start` < `end`) {
  //     `for_body_generator(/*ind_var=*/start, /*is_first_iteration=*/true)`;
  //     for (i64 i = `start` + `step`; i s< `end`; i += `step`)
  //       `for_body_generator(/*ind_var=*/,i, /*is_first_iteration=*/false)`;
  //   }
  Status ForWithStatus(
      absl::string_view name, llvm::Value* start, llvm::Value* end,
      llvm::Value* step,
      const std::function<Status(llvm::Value* ind_var,
                                 bool is_first_iteration)>& for_body_generator);

  void For(
      absl::string_view name, llvm::Value* start, llvm::Value* end,
      llvm::Value* step,
      const std::function<void(llvm::Value* ind_var, bool is_first_iteration)>&
          for_body_generator) {
    CHECK_EQ(OkStatus(),
             ForWithStatus(
                 name, start, end, step,
                 [&](llvm::Value* ind_var, bool is_first_iteration) -> Status {
                   for_body_generator(ind_var, is_first_iteration);
                   return OkStatus();
                 }));
  }

  Status ForWithStatus(
      absl::string_view name, int64_t start, int64_t end, int64_t step,
      const std::function<Status(
          llvm::Value* ind_var, bool is_first_iteration)>& for_body_generator) {
    return ForWithStatus(name, /*start=*/b_->getInt64(start),
                         /*end=*/b_->getInt64(end),
                         /*step=*/b_->getInt64(step), for_body_generator);
  }

  void For(
      absl::string_view name, int64_t start, int64_t end, int64_t step,
      const std::function<void(llvm::Value* ind_var, bool is_first_iteration)>&
          for_body_generator) {
    For(name, /*start=*/b_->getInt64(start),
        /*end=*/b_->getInt64(end),
        /*step=*/b_->getInt64(step), for_body_generator);
  }

  // Generates the following control flow structure if `peel_first_iteration` is
  // true:
  //
  //   if (`start` < `end`) {
  //     `for_body_generator(/*ind_var=*/start, /*is_first_iteration=*/,true)`;
  //     for (i64 i = `start` + `step`; i s< `end`; i += `step`)
  //       `for_body_generator(/*ind_var=*/,i, /*is_first_iteration=*/,false)`;
  //   }
  //
  // and the following if `peel_first_iteration` is false:
  //
  //   for (i64 i = `start`; i s< `end`; i += `step`)
  //     `for_body_generator(/*ind_var=*/,i,
  //                         /*is_first_iteration=*/,(i != `start`))`;
  Status ForWithStatus(
      absl::string_view name, llvm::Value* start, llvm::Value* end,
      llvm::Value* step, bool peel_first_iteration,
      const std::function<Status(llvm::Value* ind_var,
                                 llvm::Value* is_first_iteration)>&
          for_body_generator);

  void For(absl::string_view name, llvm::Value* start, llvm::Value* end,
           llvm::Value* step, bool peel_first_iteration,
           const std::function<void(llvm::Value* ind_var,
                                    llvm::Value* is_first_iteration)>&
               for_body_generator) {
    TF_CHECK_OK(ForWithStatus(
        name, start, end, step, peel_first_iteration,
        [&](llvm::Value* ind_var, llvm::Value* is_first_iteration) -> Status {
          for_body_generator(ind_var, is_first_iteration);
          return OkStatus();
        }));
  }

  Status ForWithStatus(
      absl::string_view name, llvm::Value* start, llvm::Value* end,
      int64_t step, bool peel_first_iteration,
      const std::function<Status(llvm::Value* ind_var,
                                 llvm::Value* is_first_iteration)>&
          for_body_generator) {
    return ForWithStatus(
        name, /*start=*/start, /*end=*/end,
        /*step=*/llvm::ConstantInt::get(start->getType(), step),
        peel_first_iteration, for_body_generator);
  }

  void For(absl::string_view name, llvm::Value* start, llvm::Value* end,
           int64_t step, bool peel_first_iteration,
           const std::function<void(llvm::Value* ind_var,
                                    llvm::Value* is_first_iteration)>&
               for_body_generator) {
    For(name, /*start=*/start, /*end=*/end,
        /*step=*/llvm::ConstantInt::get(start->getType(), step),
        peel_first_iteration, for_body_generator);
  }

  Status ForWithStatus(
      absl::string_view name, llvm::Value* start, llvm::Value* end,
      llvm::Value* step,
      const std::function<Status(llvm::Value* ind_var)>& for_body_generator) {
    return ForWithStatus(name, start, end, step,
                         /*peel_first_iteration=*/false,
                         [&](llvm::Value* indvar, llvm::Value*) -> Status {
                           return for_body_generator(indvar);
                         });
  }

  void For(
      absl::string_view name, llvm::Value* start, llvm::Value* end,
      llvm::Value* step,
      const std::function<void(llvm::Value* ind_var)>& for_body_generator) {
    For(name, start, end, step,
        /*peel_first_iteration=*/false, [&](llvm::Value* indvar, llvm::Value*) {
          return for_body_generator(indvar);
        });
  }

  Status ForWithStatus(
      absl::string_view name, llvm::Value* start, llvm::Value* end,
      int64_t step,
      const std::function<Status(llvm::Value* ind_var)>& for_body_generator) {
    return ForWithStatus(name, start, end,
                         llvm::ConstantInt::get(start->getType(), step),
                         /*peel_first_iteration=*/false,
                         [&](llvm::Value* indvar, llvm::Value*) -> Status {
                           return for_body_generator(indvar);
                         });
  }

  void For(
      absl::string_view name, llvm::Value* start, llvm::Value* end,
      int64_t step,
      const std::function<void(llvm::Value* ind_var)>& for_body_generator) {
    For(name, start, end, llvm::ConstantInt::get(start->getType(), step),
        for_body_generator);
  }

  Status ForWithStatus(
      absl::string_view name, int64_t start, int64_t end, int64_t step,
      const std::function<Status(llvm::Value* ind_var)>& for_body_generator) {
    return ForWithStatus(name, /*start=*/b_->getInt64(start),
                         /*end=*/b_->getInt64(end),
                         /*step=*/b_->getInt64(step), for_body_generator);
  }

  void For(
      absl::string_view name, int64_t start, int64_t end, int64_t step,
      const std::function<void(llvm::Value* ind_var)>& for_body_generator) {
    For(name, /*start=*/b_->getInt64(start),
        /*end=*/b_->getInt64(end),
        /*step=*/b_->getInt64(step), for_body_generator);
  }

  // Generates the following control flow structure:
  //
  //   if (`condition`)
  //     `true_block_generator()`;
  //   else
  //      `false_block_generator()`;
  // The else is skipped if false_block_generator is null.
  Status IfWithStatus(
      absl::string_view name, llvm::Value* condition,
      const std::function<Status()>& true_block_generator,
      const std::function<Status()>& false_block_generator = nullptr);

  Status IfWithStatus(
      llvm::Value* condition,
      const std::function<Status()>& true_block_generator,
      const std::function<Status()>& false_block_generator = []() -> Status {
        return OkStatus();
      }) {
    return IfWithStatus("", condition, true_block_generator,
                        false_block_generator);
  }

  void If(llvm::Value* condition,
          const std::function<void()>& true_block_generator,
          const std::function<void()>& false_block_generator = nullptr) {
    If("", condition, true_block_generator, false_block_generator);
  }

  void If(absl::string_view name, llvm::Value* condition,
          const std::function<void()>& true_block_generator,
          const std::function<void()>& false_block_generator = nullptr) {
    if (false_block_generator != nullptr) {
      TF_CHECK_OK(IfWithStatus(
          name, condition,
          [&]() {
            true_block_generator();
            return OkStatus();
          },
          [&]() {
            false_block_generator();
            return OkStatus();
          }));
    } else {
      TF_CHECK_OK(IfWithStatus(name, condition, [&]() {
        true_block_generator();
        return OkStatus();
      }));
    }
  }

  using ArgumentVector = absl::Span<llvm::Value* const>;

  // Generates the following control flow structure:
  //
  //  define @`kernel_name`(arg0, arg1, ... arg`arguments.size()`) {
  //    kernel_body_generator({arg0, arg1, ... arg`arguments.size()`});
  //  }
  //
  //  ...
  //  call @`kernel_name`(arguments[0], arguments[1] ...)
  //  ...
  //
  // If a function called `kernel_name` is already present in the module then
  // that function is re-used.  In that sense we're using the llvm::Module as a
  // cache of outlined kernels, keyed by function name.
  //
  // If any of the values in `arguments` is nullptr (i.e. a nullptr
  // llvm::Value*) then we ignore it when generating LLVM IR, and instead pass
  // in a nullptr llvm::Value* in its position to `kernel_body_generator`.
  // Currently we only support at most one nullptr value in `arguments`.
  static void EmitAndCallOutlinedKernel(
      const HloModuleConfig& module_config, llvm::IRBuilder<>* b,
      absl::string_view kernel_name, ArgumentVector arguments,
      const std::function<void(ArgumentVector)>& kernel_body_generator);

  // Thin wrappers around the more general EmitAndCallOutlinedKernel above.
  static void EmitAndCallOutlinedKernel(
      const HloModuleConfig& module_config, llvm::IRBuilder<>* b,
      absl::string_view kernel_name, llvm::Value* arg0, llvm::Value* arg1,
      llvm::Value* arg2,
      const std::function<void(llvm::Value*, llvm::Value*, llvm::Value*)>&
          kernel_body_generator) {
    EmitAndCallOutlinedKernel(module_config, b, kernel_name, {arg0, arg1, arg2},
                              [&](ArgumentVector args) {
                                kernel_body_generator(args[0], args[1],
                                                      args[2]);
                              });
  }

  static void EmitAndCallOutlinedKernel(
      const HloModuleConfig& module_config, llvm::IRBuilder<>* b,
      absl::string_view kernel_name, llvm::Value* arg0, llvm::Value* arg1,
      llvm::Value* arg2, llvm::Value* arg3,
      const std::function<void(llvm::Value*, llvm::Value*, llvm::Value*,
                               llvm::Value*)>& kernel_body_generator) {
    EmitAndCallOutlinedKernel(
        module_config, b, kernel_name, {arg0, arg1, arg2, arg3},
        [&](ArgumentVector args) {
          kernel_body_generator(args[0], args[1], args[2], args[3]);
        });
  }

 private:
  llvm::IRBuilder<>* b_;
  llvm_ir::UnrollMode unroll_mode_;
  bool prevent_vectorization_;
};
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_LLVM_IR_KERNEL_SUPPORT_LIBRARY_H_