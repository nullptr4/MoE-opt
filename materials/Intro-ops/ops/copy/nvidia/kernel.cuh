#pragma once

#include <cuda_fp16.h>
#include <stdint.h>

namespace oprt::copy::nvidia {

template <typename T>
__global__ void copy_contiguous_kernel(T *dst, const T *src, int64_t n) {
    // TODO: implement a grid-stride loop copy kernel.
    //
    // Suggested steps:
    // 1. Compute the global thread index.
    // 2. Compute the grid-wide stride.
    // 3. Loop over i = idx; i < n; i += stride.
    // 4. Copy src[i] to dst[i].
}

} // namespace oprt::copy::nvidia
