#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace oprt::reduce_sum::nvidia {

__global__ void reduce_sum_rowwise_kernel(float *out, const float *in, int64_t rows, int64_t cols) {
    // TODO: implement a row-wise reduce_sum kernel with shared memory.
    //
    // Suggested steps:
    // 1. Use one block per row.
    // 2. Let each thread accumulate a partial sum over the row.
    // 3. Store the partial sums in shared memory.
    // 4. Reduce shared memory with a tree reduction.
    // 5. Let thread 0 write the final row sum to out[row].
}

} // namespace oprt::reduce_sum::nvidia
