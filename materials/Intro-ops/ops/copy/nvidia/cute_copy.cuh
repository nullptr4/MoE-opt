#pragma once

#include <cute/tensor.hpp>
#include <cuda_runtime.h>

#include <cstdint>

namespace oprt::copy::nvidia {

template <typename T, int Threads, int ElementsPerAccess>
__global__ void copy_contiguous_cute_l40_kernel(T *dst, const T *src, int64_t n) {
    using namespace cute;

    constexpr int elements_per_block = Threads * ElementsPerAccess;

    Tensor src_tensor = make_tensor(make_gmem_ptr(src), make_layout(make_shape(n)));
    Tensor dst_tensor = make_tensor(make_gmem_ptr(dst), make_layout(make_shape(n)));

    auto block_shape = make_shape(Int<elements_per_block>{});
    auto block_coord = make_coord(blockIdx.x);

    Tensor coords = make_identity_tensor(shape(src_tensor));
    Tensor predicates = cute::lazy::transform(coords, [&](auto coord) {
        return elem_less(coord, shape(src_tensor));
    });

    Tensor src_tile = local_tile(src_tensor, block_shape, block_coord);
    Tensor dst_tile = local_tile(dst_tensor, block_shape, block_coord);
    Tensor pred_tile = local_tile(predicates, block_shape, block_coord);

    Layout thread_layout = make_layout(make_shape(Int<Threads>{}));
    Layout value_layout = make_layout(make_shape(Int<ElementsPerAccess>{}));

    using AccessType = uint_byte_t<sizeof(T) * ElementsPerAccess>;
    using CopyOp = UniversalCopy<AccessType>;
    using Atom = Copy_Atom<CopyOp, T>;

    TiledCopy tiled_copy = make_tiled_copy(Atom{}, thread_layout, value_layout);
    ThrCopy thread_copy = tiled_copy.get_thread_slice(threadIdx.x);

    Tensor thread_src = thread_copy.partition_S(src_tile);
    Tensor thread_dst = thread_copy.partition_D(dst_tile);
    Tensor thread_pred = thread_copy.partition_S(pred_tile);
    Tensor fragment = make_fragment_like(thread_src);

    copy_if(tiled_copy, thread_pred, thread_src, fragment);
    copy_if(tiled_copy, thread_pred, fragment, thread_dst);
}

template <typename T>
inline void launch_copy_contiguous_cute_l40(T *dst, const T *src, int64_t n, cudaStream_t stream) {
    constexpr int threads = 256;
    constexpr int bytes_per_access = 16;
    constexpr int elements_per_access = bytes_per_access / static_cast<int>(sizeof(T));
    static_assert(elements_per_access > 0, "copy element type is larger than the vector access");

    if (n <= 0) {
        return;
    }

    auto src_addr = reinterpret_cast<std::uintptr_t>(src);
    auto dst_addr = reinterpret_cast<std::uintptr_t>(dst);
    bool aligned = (src_addr % bytes_per_access == 0) && (dst_addr % bytes_per_access == 0);

    if (aligned) {
        constexpr int elements_per_block = threads * elements_per_access;
        int64_t blocks64 = (n + elements_per_block - 1) / elements_per_block;
        int blocks = static_cast<int>(blocks64 > 0 ? blocks64 : 1);
        copy_contiguous_cute_l40_kernel<T, threads, elements_per_access><<<blocks, threads, 0, stream>>>(
            dst, src, n);
    } else {
        int64_t blocks64 = (n + threads - 1) / threads;
        int blocks = static_cast<int>(blocks64 > 0 ? blocks64 : 1);
        copy_contiguous_cute_l40_kernel<T, threads, 1><<<blocks, threads, 0, stream>>>(
            dst, src, n);
    }
}

} // namespace oprt::copy::nvidia
