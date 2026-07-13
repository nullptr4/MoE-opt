#pragma once

#include <cuda_fp16.h>
#include <stdint.h>

namespace oprt::vector_add::nvidia {

template <typename T>
__device__ T add_values(T a, T b) {
    // TODO: return the elementwise sum for generic types.
    return T{};
}

template <>
__device__ inline half add_values<half>(half a, half b) {
    // TODO: return the half-precision elementwise sum.
    return half{};
}

template <typename T>
__global__ void vector_add_contiguous_kernel(T * __restrict__ out,
                                             const T * __restrict__ a,
                                             const T * __restrict__ b,
                                             int64_t n) {
    // TODO: implement a grid-stride loop vector add kernel.
    //
    // Suggested steps:
    // 1. Compute the global thread index.
    // 2. Compute the grid-wide stride.
    // 3. Loop over i = idx; i < n; i += stride.
    // 4. Write out[i] = add_values(a[i], b[i]).
}

} // namespace oprt::vector_add::nvidia
