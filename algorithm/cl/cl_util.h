
#ifndef __ALGORITHM_CL_CL_UTIL_H__
#define __ALGORITHM_CL_CL_UTIL_H__

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <cassert>
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

std::string readFile(const std::string &filepath);

}    // namespace cl
}    // namespace algorithm
}    // namespace cooboc

#endif