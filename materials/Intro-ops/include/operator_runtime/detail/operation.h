#pragma once

#include <string>

namespace oprt {

struct OperationSpec {
    std::string name;
    std::string backend;
    std::string kind;
};

} // namespace oprt
