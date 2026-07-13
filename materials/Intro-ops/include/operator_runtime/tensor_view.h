#pragma once

#include "operator_runtime/api.h"

#ifdef __cplusplus

#include <algorithm>
#include <cstdint>

namespace oprt {

inline int64_t numel(const oprt_tensor_view_t &view) {
    if (view.ndim < 0 || view.ndim > OPRT_MAX_DIMS) {
        return 0;
    }
    int64_t total = 1;
    for (int32_t i = 0; i < view.ndim; ++i) {
        total *= view.shape[i];
    }
    return total;
}

inline bool is_contiguous(const oprt_tensor_view_t &view) {
    if (view.ndim < 0 || view.ndim > OPRT_MAX_DIMS) {
        return false;
    }
    int64_t expected = 1;
    for (int32_t i = view.ndim - 1; i >= 0; --i) {
        if (view.shape[i] == 1) {
            continue;
        }
        if (view.strides[i] != expected) {
            return false;
        }
        expected *= view.shape[i];
    }
    return true;
}

inline bool same_shape(const oprt_tensor_view_t &a, const oprt_tensor_view_t &b) {
    if (a.ndim != b.ndim) {
        return false;
    }
    for (int32_t i = 0; i < a.ndim; ++i) {
        if (a.shape[i] != b.shape[i]) {
            return false;
        }
    }
    return true;
}

inline bool has_broadcast_dim(const oprt_tensor_view_t &view) {
    if (view.ndim < 0 || view.ndim > OPRT_MAX_DIMS) {
        return false;
    }
    for (int32_t i = 0; i < view.ndim; ++i) {
        if (view.shape[i] > 1 && view.strides[i] == 0) {
            return true;
        }
    }
    return false;
}

} // namespace oprt

#endif
