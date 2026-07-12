#pragma once

#include "operator_runtime/api.h"

#ifdef __CUDACC__
#include <cuda_runtime.h>

#define OPRT_CUDA_RETURN_IF_ERROR(expr) \
    do { \
        cudaError_t err__ = (expr); \
        if (err__ != cudaSuccess) { \
            return OPRT_ERR_RUNTIME; \
        } \
    } while (0)

namespace oprt {

inline cudaStream_t as_cuda_stream(oprt_stream_t stream) {
    return reinterpret_cast<cudaStream_t>(stream);
}

inline int blocks_for(int64_t n, int threads) {
    int64_t blocks = (n + threads - 1) / threads;
    return static_cast<int>(blocks > 0 ? blocks : 1);
}

} // namespace oprt

#endif
