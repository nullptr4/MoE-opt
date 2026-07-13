#pragma once

#include "operator_runtime/descriptor.h"
#include "operator_runtime/detail/elementwise.h"

namespace oprt::elementwise {

struct ReluDescriptor final : oprt::ElementwiseDescriptorBase {
    float negative_slope = 0.0f;

    const char *op_name() const override {
        return "relu";
    }
};

inline oprt_status_t create_relu_descriptor_common(
    oprt_operator_descriptor_t *desc,
    const oprt_tensor_view_t *out,
    const oprt_tensor_view_t *in,
    float negative_slope) {
    if (desc == nullptr) {
        return OPRT_ERR_INVALID_ARG;
    }
    *desc = nullptr;

    if (out == nullptr || in == nullptr) {
        return OPRT_ERR_INVALID_ARG;
    }

    auto *typed = new ReluDescriptor();
    auto status = oprt::init_elementwise_descriptor(typed, out, {in});
    if (status != OPRT_SUCCESS) {
        delete typed;
        return status;
    }
    typed->negative_slope = negative_slope;
    *desc = typed;
    return OPRT_SUCCESS;
}

inline oprt_status_t get_relu_workspace_size_common(oprt_operator_descriptor_t desc,
                                                    size_t *size) {
    return oprt::get_elementwise_workspace_size(desc, size);
}

inline oprt_status_t validate_relu_execute_args(oprt_operator_descriptor_t desc,
                                                size_t workspace_size,
                                                void *out,
                                                const void *in) {
    if (desc == nullptr || out == nullptr || in == nullptr) {
        return OPRT_ERR_INVALID_ARG;
    }
    return oprt::validate_elementwise_execute_args(desc, workspace_size, out, {in});
}

inline oprt_status_t destroy_relu_descriptor_common(oprt_operator_descriptor_t desc) {
    return oprt::destroy_elementwise_descriptor(desc);
}

} // namespace oprt::elementwise
