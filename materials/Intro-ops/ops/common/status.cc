#include "operator_runtime/api.h"

namespace {
#ifdef CAMP_ENABLE_NVIDIA
oprt_backend_t g_current_backend = OPRT_BACKEND_NVIDIA;
#elif defined(CAMP_ENABLE_METAX)
oprt_backend_t g_current_backend = OPRT_BACKEND_METAX;
#else
oprt_backend_t g_current_backend = OPRT_BACKEND_AUTO;
#endif
}

extern "C" OPRT_EXPORT const char *oprt_status_string(oprt_status_t status) {
    switch (status) {
    case OPRT_SUCCESS:
        return "success";
    case OPRT_ERR_INVALID_ARG:
        return "invalid argument";
    case OPRT_ERR_UNSUPPORTED_DTYPE:
        return "unsupported dtype";
    case OPRT_ERR_RUNTIME:
        return "runtime error";
    case OPRT_ERR_INSUFFICIENT_WORKSPACE:
        return "insufficient workspace";
    case OPRT_ERR_NOT_SUPPORTED:
        return "not supported";
    default:
        return "unknown status";
    }
}

extern "C" OPRT_EXPORT const char *oprt_backend_string(oprt_backend_t backend) {
    switch (backend) {
    case OPRT_BACKEND_AUTO:
        return "auto";
    case OPRT_BACKEND_NVIDIA:
        return "nvidia";
    case OPRT_BACKEND_METAX:
        return "metax";
    default:
        return "unknown";
    }
}

extern "C" OPRT_EXPORT oprt_status_t oprt_set_backend(oprt_backend_t backend) {
    switch (backend) {
    case OPRT_BACKEND_AUTO:
#ifdef CAMP_ENABLE_NVIDIA
        g_current_backend = OPRT_BACKEND_NVIDIA;
        return OPRT_SUCCESS;
#elif defined(CAMP_ENABLE_METAX)
        g_current_backend = OPRT_BACKEND_METAX;
        return OPRT_SUCCESS;
#else
        return OPRT_ERR_NOT_SUPPORTED;
#endif
    case OPRT_BACKEND_NVIDIA:
#ifdef CAMP_ENABLE_NVIDIA
        g_current_backend = backend;
        return OPRT_SUCCESS;
#else
        return OPRT_ERR_NOT_SUPPORTED;
#endif
    case OPRT_BACKEND_METAX:
#ifdef CAMP_ENABLE_METAX
        g_current_backend = backend;
        return OPRT_SUCCESS;
#else
        return OPRT_ERR_NOT_SUPPORTED;
#endif
    default:
        return OPRT_ERR_INVALID_ARG;
    }
}

extern "C" OPRT_EXPORT oprt_status_t oprt_get_backend(oprt_backend_t *backend) {
    if (backend == nullptr) {
        return OPRT_ERR_INVALID_ARG;
    }
    *backend = g_current_backend;
    return OPRT_SUCCESS;
}
