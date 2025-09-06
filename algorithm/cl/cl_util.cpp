#include "algorithm/cl/cl_util.h"
#include <fstream>
#include <sstream>
#include <string>

namespace cooboc {
namespace algorithm {
namespace cl {

std::string readFile(const std::string &filepath) {
    std::ifstream file(filepath, std::ios::binary | std::ios::in);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

}    // namespace cl
}    // namespace algorithm
}    // namespace cooboc
