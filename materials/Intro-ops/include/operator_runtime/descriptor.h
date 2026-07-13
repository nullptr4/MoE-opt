#pragma once

#include "operator_runtime/api.h"

#ifdef __cplusplus

#include <string>

struct oprt_operator_descriptor {
    virtual ~oprt_operator_descriptor() = default;
    virtual const char *op_name() const = 0;
    size_t workspace_size = 0;
    oprt_backend_t backend = OPRT_BACKEND_AUTO;
};

#endif
