#pragma once

#include "operator_runtime/api.h"

#ifdef __cplusplus
extern "C" {
#endif

OPRT_EXPORT oprt_status_t oprt_create_copy_descriptor(
    oprt_operator_descriptor_t *desc,
    const oprt_tensor_view_t *dst,
    const oprt_tensor_view_t *src);

OPRT_EXPORT oprt_status_t oprt_get_copy_workspace_size(
    oprt_operator_descriptor_t desc,
    size_t *size);

OPRT_EXPORT oprt_status_t oprt_execute_copy(
    oprt_operator_descriptor_t desc,
    void *workspace,
    size_t workspace_size,
    void *dst,
    const void *src,
    oprt_stream_t stream);

OPRT_EXPORT oprt_status_t oprt_destroy_copy_descriptor(
    oprt_operator_descriptor_t desc);

#ifdef __cplusplus
}
#endif
