#include "operator_runtime/ops/softmax.h"

#include "operator_runtime/descriptor.h"
#include "operator_runtime/detail/tensor_checks.h"
#include "operator_runtime/detail/cuda_helpers.h"
#include "ops/softmax/nvidia/kernel.cuh"

#include <cuda_runtime.h>

namespace {

struct SoftmaxDescriptor final : oprt_operator_descriptor {
    oprt_tensor_view_t out_view;
    oprt_tensor_view_t in_view;
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t axis = 1;

    const char *op_name() const override {
        return "softmax";
    }
};

bool is_rowwise_case(const oprt_tensor_view_t &out, const oprt_tensor_view_t &in, int64_t axis) {
    return in.dtype == OPRT_DTYPE_F32 &&
           out.dtype == OPRT_DTYPE_F32 &&
           in.ndim == 2 &&
           out.ndim == 2 &&
           axis == 1 &&
           oprt::same_shape(out, in) &&
           oprt::is_contiguous(in) &&
           oprt::is_contiguous(out);
}

} // namespace

extern "C" OPRT_EXPORT oprt_status_t oprt_create_softmax_descriptor(
    oprt_operator_descriptor_t *desc,
    const oprt_tensor_view_t *out,
    const oprt_tensor_view_t *in,
    int64_t axis) {
    if (desc == nullptr) {
        return OPRT_ERR_INVALID_ARG;
    }
    *desc = nullptr;
    auto status = oprt::check_tensor(out);
    if (status != OPRT_SUCCESS) {
        return status;
    }
    status = oprt::check_tensor(in);
    if (status != OPRT_SUCCESS) {
        return status;
    }
    if (!is_rowwise_case(*out, *in, axis)) {
        return OPRT_ERR_NOT_SUPPORTED;
    }

    auto *typed = new SoftmaxDescriptor();
    typed->out_view = *out;
    typed->in_view = *in;
    typed->rows = in->shape[0];
    typed->cols = in->shape[1];
    typed->axis = axis;
    typed->workspace_size = 0;
    *desc = typed;
    return OPRT_SUCCESS;
}

extern "C" OPRT_EXPORT oprt_status_t oprt_get_softmax_workspace_size(
    oprt_operator_descriptor_t desc,
    size_t *size) {
    if (desc == nullptr || size == nullptr) {
        return OPRT_ERR_INVALID_ARG;
    }
    *size = desc->workspace_size;
    return OPRT_SUCCESS;
}

extern "C" OPRT_EXPORT oprt_status_t oprt_execute_softmax(
    oprt_operator_descriptor_t desc,
    void *,
    size_t workspace_size,
    void *out,
    const void *in,
    oprt_stream_t stream) {
    if (desc == nullptr || out == nullptr || in == nullptr) {
        return OPRT_ERR_INVALID_ARG;
    }
    if (workspace_size < desc->workspace_size) {
        return OPRT_ERR_INSUFFICIENT_WORKSPACE;
    }
    auto *typed = static_cast<const SoftmaxDescriptor *>(desc);
    cudaStream_t s = oprt::as_cuda_stream(stream);
    constexpr int threads = 256;
    oprt::softmax::nvidia::softmax_rowwise_kernel<<<typed->rows, threads, threads * sizeof(float), s>>>(
        static_cast<float *>(out), static_cast<const float *>(in), typed->rows, typed->cols);
    OPRT_CUDA_RETURN_IF_ERROR(cudaGetLastError());
    return OPRT_SUCCESS;
}

extern "C" OPRT_EXPORT oprt_status_t oprt_destroy_softmax_descriptor(
    oprt_operator_descriptor_t desc) {
    delete desc;
    return OPRT_SUCCESS;
}
