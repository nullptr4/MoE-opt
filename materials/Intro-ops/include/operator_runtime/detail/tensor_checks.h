#pragma once

#include "operator_runtime/tensor_view.h"

#ifdef __cplusplus

namespace oprt {

inline oprt_status_t check_tensor(const oprt_tensor_view_t *view) {
    if (view == nullptr || view->data == nullptr) {
        return OPRT_ERR_INVALID_ARG;
    }
    if (view->ndim < 0 || view->ndim > OPRT_MAX_DIMS) {
        return OPRT_ERR_INVALID_ARG;
    }
    for (int32_t i = 0; i < view->ndim; ++i) {
        if (view->shape[i] < 0) {
            return OPRT_ERR_INVALID_ARG;
        }
    }
    if (view->dtype != OPRT_DTYPE_F16 && view->dtype != OPRT_DTYPE_F32) {
        return OPRT_ERR_UNSUPPORTED_DTYPE;
    }
    return OPRT_SUCCESS;
}

inline oprt_status_t check_same_dtype(const oprt_tensor_view_t &a,
                                      const oprt_tensor_view_t &b) {
    return a.dtype == b.dtype ? OPRT_SUCCESS : OPRT_ERR_UNSUPPORTED_DTYPE;
}

inline oprt_status_t check_same_shape(const oprt_tensor_view_t &a,
                                      const oprt_tensor_view_t &b) {
    return same_shape(a, b) ? OPRT_SUCCESS : OPRT_ERR_INVALID_ARG;
}

} // namespace oprt

#endif
