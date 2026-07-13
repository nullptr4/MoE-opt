#pragma once

#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

namespace oprt::softmax::nvidia {

__global__ void softmax_rowwise_kernel(float *out, const float *in, int64_t rows, int64_t cols) {
    // TODO: implement a numerically stable row-wise softmax kernel.
    //
    // Suggested steps:
    // 1. Use one block per row.
    // 2. Reduce to find the row maximum.
    // 3. Compute exp(x - row_max), write the temporary values to out,
    //    and accumulate their sum.
    // 4. Reduce to get the row sum.
    // 5. Normalize each output element by row_sum.
}

} // namespace oprt::softmax::nvidia
