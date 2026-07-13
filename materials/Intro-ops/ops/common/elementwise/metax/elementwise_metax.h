#pragma once

#include "operator_runtime/detail/cuda_helpers.h"
#include "operator_runtime/detail/elementwise.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <utility>

namespace oprt::elementwise::metax {

template <typename T>
__device__ inline T cast_scalar(float value) {
    return static_cast<T>(value);
}

template <>
__device__ inline half cast_scalar<half>(float value) {
    return __float2half(value);
}

__device__ inline int64_t index_to_offset(int64_t linear,
                                          int32_t ndim,
                                          const int64_t *shape,
                                          const int64_t *strides) {
    int64_t offset = 0;
    for (int32_t dim = ndim - 1; dim >= 0; --dim) {
        int64_t coord = linear % shape[dim];
        linear /= shape[dim];
        offset += coord * strides[dim];
    }
    return offset;
}

template <size_t Input>
__device__ inline int64_t input_offset(
    int64_t linear,
    const oprt::ElementwiseInfo::DeviceMeta *__restrict__ meta,
    const oprt::ElementwiseInfo::DeviceInputMeta *__restrict__ input_meta) {
    const auto &m = input_meta[Input];
    return m.contiguous != 0
             ? linear
             : index_to_offset(linear, meta->ndim, m.shape, m.strides);
}

template <typename T, size_t Input>
__device__ inline T load_input(
    int64_t linear,
    const void *const *__restrict__ inputs,
    const oprt::ElementwiseInfo::DeviceMeta *__restrict__ meta,
    const oprt::ElementwiseInfo::DeviceInputMeta *__restrict__ input_meta) {
    const T *input = static_cast<const T *>(inputs[Input]);
    return input[input_offset<Input>(linear, meta, input_meta)];
}

template <typename T, typename Op, size_t... Is, typename... Args>
__device__ inline T apply_at_index_impl(
    int64_t linear,
    const void *const *__restrict__ inputs,
    const oprt::ElementwiseInfo::DeviceMeta *__restrict__ meta,
    const oprt::ElementwiseInfo::DeviceInputMeta *__restrict__ input_meta,
    Op op,
    std::index_sequence<Is...>,
    Args... args) {
    static_assert(sizeof...(Is) > 0, "elementwise launch requires at least one input");
    return op(load_input<T, Is>(linear, inputs, meta, input_meta)..., args...);
}

template <typename T, size_t N, typename Op, typename... Args>
__device__ inline T apply_at_index(
    int64_t linear,
    const void *const *__restrict__ inputs,
    const oprt::ElementwiseInfo::DeviceMeta *__restrict__ meta,
    const oprt::ElementwiseInfo::DeviceInputMeta *__restrict__ input_meta,
    Op op,
    Args... args) {
    return apply_at_index_impl<T>(
        linear,
        inputs,
        meta,
        input_meta,
        op,
        std::make_index_sequence<N>{},
        args...);
}

template <typename T, size_t N, typename Op, typename... Args>
__global__ void elementwise_kernel(
    T *__restrict__ out,
    const void *const *__restrict__ inputs,
    const oprt::ElementwiseInfo::DeviceMeta *__restrict__ meta,
    const oprt::ElementwiseInfo::DeviceInputMeta *__restrict__ input_meta,
    Op op,
    Args... args) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
    const bool output_contiguous = meta->output_contiguous != 0;

    for (int64_t i = idx; i < meta->elements; i += stride) {
        int64_t out_offset = output_contiguous
                               ? i
                               : index_to_offset(i, meta->ndim, meta->output_shape, meta->output_strides);
        out[out_offset] = apply_at_index<T, N>(i, inputs, meta, input_meta, op, args...);
    }
}

template <typename T, size_t N, typename Op, typename... Args>
oprt_status_t launch(
    const oprt::ElementwiseInfo &info,
    void *workspace,
    void *out,
    const std::array<const void *, N> &inputs,
    oprt_stream_t stream,
    Op op,
    Args... args) {
    if (info.input_count != N || (info.workspace_bytes() != 0 && workspace == nullptr)) {
        return OPRT_ERR_INVALID_ARG;
    }
    if (info.elements == 0) {
        return OPRT_SUCCESS;
    }

    auto *workspace_bytes = static_cast<uint8_t *>(workspace);
    auto **device_inputs = reinterpret_cast<const void **>(workspace_bytes);
    auto *device_meta = reinterpret_cast<oprt::ElementwiseInfo::DeviceMeta *>(
        workspace_bytes + N * sizeof(const void *));
    auto *device_input_meta = reinterpret_cast<oprt::ElementwiseInfo::DeviceInputMeta *>(device_meta + 1);

    oprt::ElementwiseInfo::DeviceMeta host_meta{};
    host_meta.ndim = info.ndim;
    host_meta.input_count = static_cast<uint32_t>(N);
    host_meta.elements = info.elements;
    host_meta.output_contiguous = info.output_contiguous ? 1 : 0;
    for (int32_t i = 0; i < info.ndim; ++i) {
        host_meta.output_shape[i] = info.output_shape[i];
        host_meta.output_strides[i] = info.output_strides[i];
    }

    std::array<oprt::ElementwiseInfo::DeviceInputMeta, N> host_input_meta{};
    for (size_t input = 0; input < N; ++input) {
        const auto &src = info.inputs[input];
        auto &dst = host_input_meta[input];
        dst.contiguous = src.contiguous ? 1 : 0;
        dst.broadcasted = src.broadcasted ? 1 : 0;
        for (int32_t dim = 0; dim < info.ndim; ++dim) {
            dst.shape[dim] = src.shape[dim];
            dst.strides[dim] = src.strides[dim];
        }
    }

    cudaStream_t cuda_stream = oprt::as_cuda_stream(stream);
    OPRT_CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(device_inputs,
                                              inputs.data(),
                                              N * sizeof(const void *),
                                              cudaMemcpyHostToDevice,
                                              cuda_stream));
    OPRT_CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(device_meta,
                                              &host_meta,
                                              sizeof(host_meta),
                                              cudaMemcpyHostToDevice,
                                              cuda_stream));
    OPRT_CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(device_input_meta,
                                              host_input_meta.data(),
                                              N * sizeof(oprt::ElementwiseInfo::DeviceInputMeta),
                                              cudaMemcpyHostToDevice,
                                              cuda_stream));

    constexpr int threads = 256;
    int blocks = oprt::blocks_for(info.elements, threads);
    elementwise_kernel<T, N, Op, Args...><<<blocks, threads, 0, cuda_stream>>>(
        static_cast<T *>(out),
        device_inputs,
        device_meta,
        device_input_meta,
        op,
        args...);
    OPRT_CUDA_RETURN_IF_ERROR(cudaGetLastError());
    return OPRT_SUCCESS;
}

} // namespace oprt::elementwise::metax
