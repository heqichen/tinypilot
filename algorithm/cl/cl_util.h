
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <cassert>
#include <fstream>
#include <sstream>
#include <string>

namespace cooboc {
namespace algorithm {
namespace cl {

#define CL_CHECK(_expr)                \
    do {                               \
        assert(CL_SUCCESS == (_expr)); \
    } while (0)

#define CL_CHECK_ERR(_expr)               \
    ({                                    \
        cl_int err = CL_INVALID_VALUE;    \
        __typeof__(_expr) _ret = _expr;   \
        assert(_ret &&err == CL_SUCCESS); \
        _ret;                             \
    })

std::string readFile(const std::string &filepath) {
    std::ifstream file(filepath, std::ios::binary | std::ios::in);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

}    // namespace cl
}    // namespace algorithm
}    // namespace cooboc