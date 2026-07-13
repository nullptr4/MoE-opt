#include "operator_runtime/ops/vector_add.h"

#include "operator_runtime/descriptor.h"
#include "operator_runtime/detail/elementwise.h"
#include "operator_runtime/detail/tensor_checks.h"
#include "operator_runtime/detail/cuda_helpers.h"
#include "ops/vector_add/nvidia/kernel.cuh"

#include <cuda_fp16.h>

namespace {

struct VectorAddDescriptor final : oprt_operator_descriptor {
    oprt_tensor_view_t out_view;
    oprt_tensor_view_t a_view;
    oprt_tensor_view_t b_view;
    int64_t elements = 0;

    const char *op_name() const override {
        return "vector_add";
    }
};

template <typename T>
oprt_status_t launch_vector_add(const VectorAddDescriptor *desc, void *out, const void *a, const void *b, oprt_stream_t stream) {
    constexpr int threads = 256;
    int blocks = oprt::blocks_for(desc->elements, threads);
    cudaStream_t s = oprt::as_cuda_stream(stream);
    oprt::vector_add::nvidia::vector_add_contiguous_kernel<T><<<blocks, threads, 0, s>>>(
        static_cast<T *>(out), static_cast<const T *>(a), static_cast<const T *>(b), desc->elements);
    OPRT_CUDA_RETURN_IF_ERROR(cudaGetLastError());
    return OPRT_SUCCESS;
}

} // namespace

extern "C" OPRT_EXPORT oprt_status_t oprt_create_vector_add_descriptor(
    oprt_operator_descriptor_t *desc,
    const oprt_tensor_view_t *out,
    const oprt_tensor_view_t *a,
    const oprt_tensor_view_t *b) {
    if (desc == nullptr) {
        return OPRT_ERR_INVALID_ARG;
    }
    *desc = nullptr;
    auto status = oprt::check_tensor(out);
    if (status != OPRT_SUCCESS) {
        return status;
    }
    status = oprt::check_tensor(a);
    if (status != OPRT_SUCCESS) {
        return status;
    }
    status = oprt::check_tensor(b);
    if (status != OPRT_SUCCESS) {
        return status;
    }
    status = oprt::check_same_dtype(*out, *a);
    if (status != OPRT_SUCCESS) {
        return status;
    }
    status = oprt::check_same_dtype(*out, *b);
    if (status != OPRT_SUCCESS) {
        return status;
    }
    if (!oprt::elementwise_fast_path(*out, *a, *b)) {
        return OPRT_ERR_NOT_SUPPORTED;
    }

    auto *typed = new VectorAddDescriptor();
    typed->out_view = *out;
    typed->a_view = *a;
    typed->b_view = *b;
    typed->elements = oprt::numel(*out);
    typed->workspace_size = 0;
    *desc = typed;
    return OPRT_SUCCESS;
}

extern "C" OPRT_EXPORT oprt_status_t oprt_get_vector_add_workspace_size(
    oprt_operator_descriptor_t desc,
    size_t *size) {
    if (desc == nullptr || size == nullptr) {
        return OPRT_ERR_INVALID_ARG;
    }
    *size = desc->workspace_size;
    return OPRT_SUCCESS;
}

extern "C" OPRT_EXPORT oprt_status_t oprt_execute_vector_add(
    oprt_operator_descriptor_t desc,
    void *,
    size_t workspace_size,
    void *out,
    const void *a,
    const void *b,
    oprt_stream_t stream) {
    if (desc == nullptr || out == nullptr || a == nullptr || b == nullptr) {
        return OPRT_ERR_INVALID_ARG;
    }
    if (workspace_size < desc->workspace_size) {
        return OPRT_ERR_INSUFFICIENT_WORKSPACE;
    }
    auto *typed = static_cast<const VectorAddDescriptor *>(desc);
    switch (typed->out_view.dtype) {
    case OPRT_DTYPE_F16:
        return launch_vector_add<half>(typed, out, a, b, stream);
    case OPRT_DTYPE_F32:
        return launch_vector_add<float>(typed, out, a, b, stream);
    default:
        return OPRT_ERR_UNSUPPORTED_DTYPE;
    }
}

extern "C" OPRT_EXPORT oprt_status_t oprt_destroy_vector_add_descriptor(
    oprt_operator_descriptor_t desc) {
    delete desc;
    return OPRT_SUCCESS;
}
