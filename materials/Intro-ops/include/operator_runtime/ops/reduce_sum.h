#pragma once

#include "operator_runtime/api.h"

#ifdef __cplusplus
extern "C" {
#endif

OPRT_EXPORT oprt_status_t oprt_create_reduce_sum_descriptor(
    oprt_operator_descriptor_t *desc,
    const oprt_tensor_view_t *out,
    const oprt_tensor_view_t *in,
    int64_t axis);

OPRT_EXPORT oprt_status_t oprt_get_reduce_sum_workspace_size(
    oprt_operator_descriptor_t desc,
    size_t *size);

OPRT_EXPORT oprt_status_t oprt_execute_reduce_sum(
    oprt_operator_descriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *out,
    const void *in,
    oprt_stream_t stream);

OPRT_EXPORT oprt_status_t oprt_destroy_reduce_sum_descriptor(
    oprt_operator_descriptor_t desc);

#ifdef __cplusplus
}
#endif
