#pragma once

#include "operator_runtime/descriptor.h"
#include "operator_runtime/tensor_view.h"
#include "operator_runtime/detail/tensor_checks.h"

#ifdef __cplusplus

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace oprt {

struct ElementwiseInfo {
    struct InputInfo {
        std::array<int64_t, OPRT_MAX_DIMS> shape{};
        std::array<int64_t, OPRT_MAX_DIMS> strides{};
        bool contiguous = false;
        bool broadcasted = false;
    };

    oprt_dtype_t dtype = OPRT_DTYPE_F32;
    int32_t ndim = 0;
    int64_t elements = 0;
    size_t input_count = 0;
    std::array<int64_t, OPRT_MAX_DIMS> output_shape{};
    std::array<int64_t, OPRT_MAX_DIMS> output_strides{};
    bool output_contiguous = false;
    std::vector<InputInfo> inputs;

    size_t meta_bytes() const {
        return sizeof(DeviceMeta) + input_count * sizeof(DeviceInputMeta);
    }

    size_t workspace_bytes() const {
        return input_count * sizeof(const void *) + meta_bytes();
    }

    struct DeviceInputMeta {
        int64_t shape[OPRT_MAX_DIMS];
        int64_t strides[OPRT_MAX_DIMS];
        uint8_t contiguous;
        uint8_t broadcasted;
    };

    struct DeviceMeta {
        int32_t ndim;
        uint32_t input_count;
        int64_t elements;
        int64_t output_shape[OPRT_MAX_DIMS];
        int64_t output_strides[OPRT_MAX_DIMS];
        uint8_t output_contiguous;
    };
};

inline bool elementwise_fast_path(const oprt_tensor_view_t &out,
                                  const oprt_tensor_view_t &a) {
    return same_shape(out, a) && is_contiguous(out) && is_contiguous(a);
}

inline bool elementwise_fast_path(const oprt_tensor_view_t &out,
                                  const oprt_tensor_view_t &a,
                                  const oprt_tensor_view_t &b) {
    return same_shape(out, a) && same_shape(out, b) &&
           is_contiguous(out) && is_contiguous(a) && is_contiguous(b);
}

inline bool elementwise_input_broadcastable_to(const oprt_tensor_view_t &out,
                                               const oprt_tensor_view_t &in) {
    if (out.ndim < 0 || out.ndim > OPRT_MAX_DIMS || in.ndim < 0 || in.ndim > OPRT_MAX_DIMS) {
        return false;
    }
    if (in.ndim > out.ndim) {
        return false;
    }
    int32_t dim_offset = out.ndim - in.ndim;
    for (int32_t out_dim = 0; out_dim < out.ndim; ++out_dim) {
        int32_t in_dim = out_dim - dim_offset;
        if (in_dim < 0) {
            continue;
        }
        int64_t in_extent = in.shape[in_dim];
        int64_t out_extent = out.shape[out_dim];
        if (in_extent != out_extent && in_extent != 1) {
            return false;
        }
    }
    return true;
}

inline bool elementwise_has_broadcast(const oprt_tensor_view_t &out,
                                      const oprt_tensor_view_t &in) {
    if (has_broadcast_dim(in)) {
        return true;
    }
    if (in.ndim != out.ndim) {
        return true;
    }
    for (int32_t i = 0; i < out.ndim; ++i) {
        if (in.shape[i] != out.shape[i]) {
            return true;
        }
    }
    return false;
}

inline bool elementwise_same_shape_inputs(const oprt_tensor_view_t &out,
                                          const std::vector<const oprt_tensor_view_t *> &inputs) {
    for (const auto *input : inputs) {
        if (input == nullptr || !same_shape(out, *input)) {
            return false;
        }
    }
    return true;
}

inline oprt_status_t create_elementwise_info(const oprt_tensor_view_t *out,
                                             const std::vector<const oprt_tensor_view_t *> &inputs,
                                             ElementwiseInfo *info) {
    if (info == nullptr || out == nullptr || inputs.empty()) {
        return OPRT_ERR_INVALID_ARG;
    }

    auto status = check_tensor(out);
    if (status != OPRT_SUCCESS) {
        return status;
    }
    if (has_broadcast_dim(*out)) {
        return OPRT_ERR_INVALID_ARG;
    }

    ElementwiseInfo next;
    next.dtype = out->dtype;
    next.ndim = out->ndim;
    next.elements = numel(*out);
    next.input_count = inputs.size();
    next.output_contiguous = is_contiguous(*out);

    for (int32_t i = 0; i < out->ndim; ++i) {
        next.output_shape[i] = out->shape[i];
        next.output_strides[i] = out->strides[i];
    }

    next.inputs.reserve(inputs.size());
    for (const auto *input : inputs) {
        status = check_tensor(input);
        if (status != OPRT_SUCCESS) {
            return status;
        }
        if (input->dtype != out->dtype) {
            return OPRT_ERR_UNSUPPORTED_DTYPE;
        }
        if (!elementwise_input_broadcastable_to(*out, *input)) {
            return OPRT_ERR_INVALID_ARG;
        }

        ElementwiseInfo::InputInfo input_info;
        input_info.contiguous = same_shape(*out, *input) && is_contiguous(*input);
        input_info.broadcasted = elementwise_has_broadcast(*out, *input);

        int32_t dim_offset = out->ndim - input->ndim;
        for (int32_t out_dim = 0; out_dim < out->ndim; ++out_dim) {
            int32_t in_dim = out_dim - dim_offset;
            if (in_dim < 0) {
                input_info.shape[out_dim] = out->shape[out_dim];
                input_info.strides[out_dim] = 0;
                input_info.broadcasted = true;
                continue;
            }
            input_info.shape[out_dim] = out->shape[out_dim];
            input_info.strides[out_dim] = input->shape[in_dim] == 1 && out->shape[out_dim] != 1
                                             ? 0
                                             : input->strides[in_dim];
        }
        next.inputs.push_back(input_info);
    }

    *info = std::move(next);
    return OPRT_SUCCESS;
}

struct ElementwiseDescriptorBase : oprt_operator_descriptor {
    oprt_tensor_view_t out_view{};
    std::vector<oprt_tensor_view_t> input_views;
    oprt::ElementwiseInfo info;
};

inline oprt_status_t init_elementwise_descriptor(ElementwiseDescriptorBase *desc,
                                                 const oprt_tensor_view_t *out,
                                                 const std::vector<const oprt_tensor_view_t *> &inputs) {
    if (desc == nullptr || out == nullptr || inputs.empty()) {
        return OPRT_ERR_INVALID_ARG;
    }

    oprt::ElementwiseInfo info;
    auto status = oprt::create_elementwise_info(out, inputs, &info);
    if (status != OPRT_SUCCESS) {
        return status;
    }

    desc->out_view = *out;
    desc->input_views.clear();
    desc->input_views.reserve(inputs.size());
    for (const auto *input : inputs) {
        desc->input_views.push_back(*input);
    }
    desc->info = std::move(info);
    desc->workspace_size = desc->info.workspace_bytes();
    return OPRT_SUCCESS;
}

inline oprt_status_t get_elementwise_workspace_size(oprt_operator_descriptor_t desc,
                                                    size_t *size) {
    if (desc == nullptr || size == nullptr) {
        return OPRT_ERR_INVALID_ARG;
    }
    *size = desc->workspace_size;
    return OPRT_SUCCESS;
}

inline oprt_status_t validate_elementwise_execute_args(oprt_operator_descriptor_t desc,
                                                       size_t workspace_size,
                                                       void *out,
                                                       const std::vector<const void *> &inputs) {
    if (desc == nullptr || out == nullptr || inputs.empty()) {
        return OPRT_ERR_INVALID_ARG;
    }
    for (const auto *input : inputs) {
        if (input == nullptr) {
            return OPRT_ERR_INVALID_ARG;
        }
    }
    if (workspace_size < desc->workspace_size) {
        return OPRT_ERR_INSUFFICIENT_WORKSPACE;
    }
    return OPRT_SUCCESS;
}

inline oprt_status_t destroy_elementwise_descriptor(oprt_operator_descriptor_t desc) {
    delete desc;
    return OPRT_SUCCESS;
}

} // namespace oprt

#endif
