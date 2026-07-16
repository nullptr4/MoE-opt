#include <tl_templates/maca/gemm.h>
#include <tl_templates/maca/copy.h>
#include <tl_templates/maca/reduce.h>
#include <tl_templates/maca/intrin.h>
#include <tl_templates/maca/atomic.h>
#include <tl_templates/maca/threadblock_swizzle.h>
#include <tl_templates/maca/debug.h>

extern "C" __global__ void kernel_kernel(const int* __restrict__ group_idx_for_bx, const int* __restrict__ group_offsets, const int* __restrict__ group_padded_offsets, const int* __restrict__ group_sizes, const half_t* __restrict__ input, const half_t* __restrict__ routed_expert_gate, const half_t* __restrict__ routed_expert_up, half_t* __restrict__ up_logits);
extern "C" __global__ void kernel_kernel_1(const int* __restrict__ group_idx_for_bx, const int* __restrict__ group_offsets, const int* __restrict__ group_padded_offsets, const int* __restrict__ group_sizes, half_t* __restrict__ output, const half_t* __restrict__ routed_expert_down, const half_t* __restrict__ routed_expert_weights, const half_t* __restrict__ up_logits);
extern "C" __global__ void __launch_bounds__(256, 1) kernel_kernel(const int* __restrict__ group_idx_for_bx, const int* __restrict__ group_offsets, const int* __restrict__ group_padded_offsets, const int* __restrict__ group_sizes, const half_t* __restrict__ input, const half_t* __restrict__ routed_expert_gate, const half_t* __restrict__ routed_expert_up, half_t* __restrict__ up_logits) {
  float gate_logits_local[64];
  float up_logits_local[64];
  half_t input_shared[64];
  extern __shared__ __align__(1024) half_t routed_expert_up_shared[];
  half_t up_logits_local_cast[4];
  const dim3 blockIdx = tl::rasterization2DRow<8>();
  int cur_group_idx = group_idx_for_bx[((int)blockIdx.x)];
  int condval;
  if (((0 <= cur_group_idx) && (cur_group_idx < 8))) {
    condval = group_sizes[((int64_t)cur_group_idx)];
  } else {
    condval = 0;
  }
  int cur_group_size = condval;
  int condval_1;
  if (((0 <= cur_group_idx) && (cur_group_idx < 8))) {
    condval_1 = group_offsets[((int64_t)cur_group_idx)];
  } else {
    condval_1 = 0;
  }
  int condval_2;
  if (((0 <= cur_group_idx) && (cur_group_idx < 8))) {
    condval_2 = group_padded_offsets[((int64_t)cur_group_idx)];
  } else {
    condval_2 = 0;
  }
  int m_start = (((((int)blockIdx.x) * 128) + condval_1) - condval_2);
  int condval_3;
  if (((0 <= cur_group_idx) && (cur_group_idx < 8))) {
    condval_3 = group_padded_offsets[((int64_t)cur_group_idx)];
  } else {
    condval_3 = 0;
  }
  int actual_rows = max(0, min(128, ((cur_group_size + condval_3) - (((int)blockIdx.x) * 128))));
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(gate_logits_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 16; ++i_1) {
    float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(up_logits_local + (i_1 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
  }
  for (int k = 0; k < 56; ++k) {
    #pragma unroll
    for (int i_2 = 0; i_2 < 16; ++i_2) {
      half_t broadcast_var_2 = half_t(0x0p+0f/*0.000000e+00*/);
      uint2 condval_4;
      if ((((((((((int)threadIdx.x) >> 6) * 32) + ((i_2 & 1) * 16)) + m_start) + (((int)threadIdx.x) & 15)) < 131072) && (0 <= (((((((int)threadIdx.x) >> 6) * 32) + ((i_2 & 1) * 16)) + m_start) + (((int)threadIdx.x) & 15))))) {
        condval_4 = *(uint2*)(input + ((((((((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)229376) + ((((int64_t)i_2) & (int64_t)1) * (int64_t)114688)) + (((int64_t)m_start) * (int64_t)7168)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)7168)) + (((int64_t)k) * (int64_t)128)) + ((((int64_t)i_2) >> (int64_t)1) * (int64_t)16)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)4) * (int64_t)4)));
      } else {
        condval_4 = make_uint2(__pack_half2(broadcast_var_2, broadcast_var_2), __pack_half2(broadcast_var_2, broadcast_var_2));
      }
      *(uint2*)(input_shared + (i_2 * 4)) = condval_4;
    }
    __syncthreads();
    #pragma unroll
    for (int i_3 = 0; i_3 < 8; ++i_3) {
      half_t broadcast_var_3 = half_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval_5;
      if (((0 <= cur_group_idx) && (cur_group_idx < 8))) {
        condval_5 = *(uint4*)(routed_expert_gate + ((((((((int64_t)cur_group_idx) * (int64_t)14680064) + (((int64_t)((int)blockIdx.y)) * (int64_t)917504)) + (((int64_t)i_3) * (int64_t)114688)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)7168)) + (((int64_t)k) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
      } else {
        condval_5 = make_uint4(__pack_half2(broadcast_var_3, broadcast_var_3), __pack_half2(broadcast_var_3, broadcast_var_3), __pack_half2(broadcast_var_3, broadcast_var_3), __pack_half2(broadcast_var_3, broadcast_var_3));
      }
      *(uint4*)(routed_expert_up_shared + ((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_3 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_5;
    }
    half_t B_local[32];
    __syncthreads();
    for (int ki = 0; ki < 8; ++ki) {
      for (int j = 0; j < 8; ++j) {
        *(uint2*)(B_local + (j * 4)) = *(uint2*)(routed_expert_up_shared + ((((((((ki >> 2) * 8192) + (j * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)));
      }
      for (int i_4 = 0; i_4 < 2; ++i_4) {
        for (int j_1 = 0; j_1 < 8; ++j_1) {
          {
      *(((float32x4*)gate_logits_local) + ((i_4 * 8) + j_1)) = __builtin_mxc_mma_16x16x16f16(*(((float16x4*)B_local) + j_1),
                    *(((float16x4*)input_shared) + ((ki * 2) + i_4)),
                    *(((float32x4*)gate_logits_local) + ((i_4 * 8) + j_1)));
    };
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_5 = 0; i_5 < 8; ++i_5) {
      half_t broadcast_var_4 = half_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval_6;
      if (((0 <= cur_group_idx) && (cur_group_idx < 8))) {
        condval_6 = *(uint4*)(routed_expert_up + ((((((((int64_t)cur_group_idx) * (int64_t)14680064) + (((int64_t)((int)blockIdx.y)) * (int64_t)917504)) + (((int64_t)i_5) * (int64_t)114688)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)4) * (int64_t)7168)) + (((int64_t)k) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
      } else {
        condval_6 = make_uint4(__pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4));
      }
      *(uint4*)(routed_expert_up_shared + ((((((((((int)threadIdx.x) & 15) >> 3) * 8192) + (i_5 * 1024)) + ((((int)threadIdx.x) >> 4) * 64)) + (((((((int)threadIdx.x) & 127) >> 6) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_6;
    }
    half_t B_local_1[32];
    __syncthreads();
    for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
      for (int j_2 = 0; j_2 < 8; ++j_2) {
        *(uint2*)(B_local_1 + (j_2 * 4)) = *(uint2*)(routed_expert_up_shared + ((((((((ki_1 >> 2) * 8192) + (j_2 * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + ((ki_1 & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)));
      }
      for (int i_6 = 0; i_6 < 2; ++i_6) {
        for (int j_3 = 0; j_3 < 8; ++j_3) {
          {
      *(((float32x4*)up_logits_local) + ((i_6 * 8) + j_3)) = __builtin_mxc_mma_16x16x16f16(*(((float16x4*)B_local_1) + j_3),
                    *(((float16x4*)input_shared) + ((ki_1 * 2) + i_6)),
                    *(((float32x4*)up_logits_local) + ((i_6 * 8) + j_3)));
    };
        }
      }
    }
  }
  #pragma unroll
  for (int i_7 = 0; i_7 < 64; ++i_7) {
    gate_logits_local[i_7] = (gate_logits_local[i_7] * (0x1p+0f/*1.000000e+00*/ / (0x1p+0f/*1.000000e+00*/ + exp2f(((gate_logits_local[i_7] * -0x1p+0f/*-1.000000e+00*/) * 0x1.7154764ee6c2fp+0f/*1.442695e+00*/)))));
    up_logits_local[i_7] = (up_logits_local[i_7] * gate_logits_local[i_7]);
  }
  #pragma unroll
  for (int i_8 = 0; i_8 < 16; ++i_8) {
    if (((((((int)threadIdx.x) >> 6) * 32) + ((i_8 >> 3) * 16)) + (((int)threadIdx.x) & 15)) < actual_rows) {
      uint2 __1;
      float4 v_ = *(float4*)(up_logits_local + (i_8 * 4));
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
      ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&v_))[1]);
      *(uint2*)(up_logits_local_cast + 0) = __1;
      if (0 <= (((((((int)threadIdx.x) >> 6) * 32) + ((i_8 >> 3) * 16)) + m_start) + (((int)threadIdx.x) & 15))) {
        if ((((((((int)threadIdx.x) >> 6) * 32) + ((i_8 >> 3) * 16)) + m_start) + (((int)threadIdx.x) & 15)) < 131072) {
          *(uint2*)(up_logits + ((((((((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)65536) + ((((int64_t)i_8) >> (int64_t)3) * (int64_t)32768)) + (((int64_t)m_start) * (int64_t)2048)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)2048)) + (((int64_t)((int)blockIdx.y)) * (int64_t)128)) + ((((int64_t)i_8) & (int64_t)7) * (int64_t)16)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)4) * (int64_t)4))) = *(uint2*)(up_logits_local_cast + 0);
        }
      }
    }
  }
}

extern "C" __global__ void __launch_bounds__(256, 1) kernel_kernel_1(const int* __restrict__ group_idx_for_bx, const int* __restrict__ group_offsets, const int* __restrict__ group_padded_offsets, const int* __restrict__ group_sizes, half_t* __restrict__ output, const half_t* __restrict__ routed_expert_down, const half_t* __restrict__ routed_expert_weights, const half_t* __restrict__ up_logits) {
  float output_local[64];
  extern __shared__ __align__(1024) half_t routed_expert_down_shared[];
  half_t up_logits_shared[32];
  half_t routed_expert_weights_local_cast_2[4];
  half_t output_local_cast_1[4];
  const dim3 blockIdx = tl::rasterization2DRow<16>();
  int cur_group_idx = group_idx_for_bx[((int)blockIdx.x)];
  int condval;
  if (((0 <= cur_group_idx) && (cur_group_idx < 8))) {
    condval = group_sizes[((int64_t)cur_group_idx)];
  } else {
    condval = 0;
  }
  int cur_group_size = condval;
  int condval_1;
  if (((0 <= cur_group_idx) && (cur_group_idx < 8))) {
    condval_1 = group_offsets[((int64_t)cur_group_idx)];
  } else {
    condval_1 = 0;
  }
  int condval_2;
  if (((0 <= cur_group_idx) && (cur_group_idx < 8))) {
    condval_2 = group_padded_offsets[((int64_t)cur_group_idx)];
  } else {
    condval_2 = 0;
  }
  int m_start = (((((int)blockIdx.x) * 128) + condval_1) - condval_2);
  int condval_3;
  if (((0 <= cur_group_idx) && (cur_group_idx < 8))) {
    condval_3 = group_padded_offsets[((int64_t)cur_group_idx)];
  } else {
    condval_3 = 0;
  }
  int actual_rows = max(0, min(128, ((cur_group_size + condval_3) - (((int)blockIdx.x) * 128))));
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(output_local + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 4; ++i_1) {
    half_t broadcast_var_1 = half_t(0x0p+0f/*0.000000e+00*/);
    uint4 condval_4;
    if (((0 <= cur_group_idx) && (cur_group_idx < 8))) {
      condval_4 = *(uint4*)(routed_expert_down + (((((((int64_t)cur_group_idx) * (int64_t)14680064) + (((int64_t)((int)blockIdx.y)) * (int64_t)262144)) + (((int64_t)i_1) * (int64_t)65536)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)2048)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)));
    } else {
      condval_4 = make_uint4(__pack_half2(broadcast_var_1, broadcast_var_1), __pack_half2(broadcast_var_1, broadcast_var_1), __pack_half2(broadcast_var_1, broadcast_var_1), __pack_half2(broadcast_var_1, broadcast_var_1));
    }
    *(uint4*)(routed_expert_down_shared + (((((i_1 * 2048) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_4;
  }
  for (int k = 0; k < 31; ++k) {
    __syncthreads();
    #pragma unroll
    for (int i_2 = 0; i_2 < 4; ++i_2) {
      half_t broadcast_var_2 = half_t(0x0p+0f/*0.000000e+00*/);
      uint4 condval_5;
      if (((0 <= cur_group_idx) && (cur_group_idx < 8))) {
        condval_5 = *(uint4*)(routed_expert_down + (((((((((int64_t)cur_group_idx) * (int64_t)14680064) + (((int64_t)((int)blockIdx.y)) * (int64_t)262144)) + (((int64_t)i_2) * (int64_t)65536)) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)3) * (int64_t)2048)) + (((int64_t)k) * (int64_t)64)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)7) * (int64_t)8)) + (int64_t)64));
      } else {
        condval_5 = make_uint4(__pack_half2(broadcast_var_2, broadcast_var_2), __pack_half2(broadcast_var_2, broadcast_var_2), __pack_half2(broadcast_var_2, broadcast_var_2), __pack_half2(broadcast_var_2, broadcast_var_2));
      }
      *(uint4*)(routed_expert_down_shared + ((((((((k + 1) & 1) * 8192) + (i_2 * 2048)) + ((((int)threadIdx.x) >> 3) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 8))) = condval_5;
    }
    #pragma unroll
    for (int i_3 = 0; i_3 < 8; ++i_3) {
      half_t broadcast_var_3 = half_t(0x0p+0f/*0.000000e+00*/);
      uint2 condval_6;
      if ((((((((((int)threadIdx.x) >> 6) * 32) + ((i_3 & 1) * 16)) + m_start) + (((int)threadIdx.x) & 15)) < 131072) && (0 <= (((((((int)threadIdx.x) >> 6) * 32) + ((i_3 & 1) * 16)) + m_start) + (((int)threadIdx.x) & 15))))) {
        condval_6 = *(uint2*)(up_logits + ((((((((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)65536) + ((((int64_t)i_3) & (int64_t)1) * (int64_t)32768)) + (((int64_t)m_start) * (int64_t)2048)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)2048)) + (((int64_t)k) * (int64_t)64)) + ((((int64_t)i_3) >> (int64_t)1) * (int64_t)16)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)4) * (int64_t)4)));
      } else {
        condval_6 = make_uint2(__pack_half2(broadcast_var_3, broadcast_var_3), __pack_half2(broadcast_var_3, broadcast_var_3));
      }
      *(uint2*)(up_logits_shared + (i_3 * 4)) = condval_6;
    }
    half_t B_local[32];
    __syncthreads();
    for (int ki = 0; ki < 4; ++ki) {
      for (int j = 0; j < 8; ++j) {
        *(uint2*)(B_local + (j * 4)) = *(uint2*)(routed_expert_down_shared + ((((((((k & 1) * 8192) + (j * 1024)) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)));
      }
      for (int i_4 = 0; i_4 < 2; ++i_4) {
        for (int j_1 = 0; j_1 < 8; ++j_1) {
          {
      *(((float32x4*)output_local) + ((i_4 * 8) + j_1)) = __builtin_mxc_mma_16x16x16f16(*(((float16x4*)B_local) + j_1),
                    *(((float16x4*)up_logits_shared) + ((ki * 2) + i_4)),
                    *(((float32x4*)output_local) + ((i_4 * 8) + j_1)));
    };
        }
      }
    }
  }
  #pragma unroll
  for (int i_5 = 0; i_5 < 8; ++i_5) {
    half_t broadcast_var_4 = half_t(0x0p+0f/*0.000000e+00*/);
    uint2 condval_7;
    if ((((((((((int)threadIdx.x) >> 6) * 32) + ((i_5 & 1) * 16)) + m_start) + (((int)threadIdx.x) & 15)) < 131072) && (0 <= (((((((int)threadIdx.x) >> 6) * 32) + ((i_5 & 1) * 16)) + m_start) + (((int)threadIdx.x) & 15))))) {
      condval_7 = *(uint2*)(up_logits + ((((((((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)65536) + ((((int64_t)i_5) & (int64_t)1) * (int64_t)32768)) + (((int64_t)m_start) * (int64_t)2048)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)2048)) + ((((int64_t)i_5) >> (int64_t)1) * (int64_t)16)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)4) * (int64_t)4)) + (int64_t)1984));
    } else {
      condval_7 = make_uint2(__pack_half2(broadcast_var_4, broadcast_var_4), __pack_half2(broadcast_var_4, broadcast_var_4));
    }
    *(uint2*)(up_logits_shared + (i_5 * 4)) = condval_7;
  }
  half_t B_local_1[32];
  for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
    for (int j_2 = 0; j_2 < 8; ++j_2) {
      *(uint2*)(B_local_1 + (j_2 * 4)) = *(uint2*)(routed_expert_down_shared + (((((((j_2 * 1024) + ((((int)threadIdx.x) & 15) * 64)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 63) >> 5) + (((int)threadIdx.x) & 1)) & 1) * 8)) + (((((int)threadIdx.x) & 31) >> 4) * 4)) + 8192));
    }
    for (int i_6 = 0; i_6 < 2; ++i_6) {
      for (int j_3 = 0; j_3 < 8; ++j_3) {
        {
      *(((float32x4*)output_local) + ((i_6 * 8) + j_3)) = __builtin_mxc_mma_16x16x16f16(*(((float16x4*)B_local_1) + j_3),
                    *(((float16x4*)up_logits_shared) + ((ki_1 * 2) + i_6)),
                    *(((float32x4*)output_local) + ((i_6 * 8) + j_3)));
    };
      }
    }
  }
  #pragma unroll
  for (int i_7 = 0; i_7 < 16; ++i_7) {
    if (((((((int)threadIdx.x) >> 6) * 32) + ((i_7 >> 3) * 16)) + (((int)threadIdx.x) & 15)) < actual_rows) {
      half_t condval_8;
      if (((0 <= (((((((int)threadIdx.x) >> 6) * 32) + ((i_7 >> 3) * 16)) + m_start) + (((int)threadIdx.x) & 15))) && ((((((((int)threadIdx.x) >> 6) * 32) + ((i_7 >> 3) * 16)) + m_start) + (((int)threadIdx.x) & 15)) < 131072))) {
        condval_8 = routed_expert_weights[(((((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)32) + ((((int64_t)i_7) >> (int64_t)3) * (int64_t)16)) + ((int64_t)m_start)) + (((int64_t)((int)threadIdx.x)) & (int64_t)15))];
      } else {
        condval_8 = half_t(0x0p+0f/*0.000000e+00*/);
      }
      *(uint2*)(routed_expert_weights_local_cast_2 + 0) = make_uint2(__pack_half2(condval_8, condval_8), __pack_half2(condval_8, condval_8));
      uint2 __1;
      float4 __2;
        float4 v_ = *(float4*)(output_local + (i_7 * 4));
        float4 __3;
        uint2 v__1 = *(uint2*)(routed_expert_weights_local_cast_2 + 0);
        ((float2*)(&__3))[0] = __half22float2(((half2*)(&v__1))[0]);
        ((float2*)(&__3))[1] = __half22float2(((half2*)(&v__1))[1]);
        __2.x = (v_.x*__3.x);
        __2.y = (v_.y*__3.y);
        __2.z = (v_.z*__3.z);
        __2.w = (v_.w*__3.w);
      ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&__2))[0]);
      ((half2*)(&__1))[1] = __float22half2_rn(((float2*)(&__2))[1]);
      *(uint2*)(output_local_cast_1 + 0) = __1;
      if (0 <= (((((((int)threadIdx.x) >> 6) * 32) + ((i_7 >> 3) * 16)) + m_start) + (((int)threadIdx.x) & 15))) {
        if ((((((((int)threadIdx.x) >> 6) * 32) + ((i_7 >> 3) * 16)) + m_start) + (((int)threadIdx.x) & 15)) < 131072) {
          *(uint2*)(output + ((((((((((int64_t)((int)threadIdx.x)) >> (int64_t)6) * (int64_t)229376) + ((((int64_t)i_7) >> (int64_t)3) * (int64_t)114688)) + (((int64_t)m_start) * (int64_t)7168)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)7168)) + (((int64_t)((int)blockIdx.y)) * (int64_t)128)) + ((((int64_t)i_7) & (int64_t)7) * (int64_t)16)) + (((((int64_t)((int)threadIdx.x)) & (int64_t)63) >> (int64_t)4) * (int64_t)4))) = *(uint2*)(output_local_cast_1 + 0);
        }
      }
    }
  }
}

