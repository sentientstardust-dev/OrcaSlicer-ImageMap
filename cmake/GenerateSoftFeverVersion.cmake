if(NOT DEFINED INPUT)
    message(FATAL_ERROR "INPUT is required")
endif()

if(NOT DEFINED OUTPUT)
    message(FATAL_ERROR "OUTPUT is required")
endif()

file(STRINGS "${INPUT}" SOFTFEVER_VERSION LIMIT_COUNT 1)
string(STRIP "${SOFTFEVER_VERSION}" SOFTFEVER_VERSION)

set(CONTENT [=[
#include "libslic3r_version.h"

namespace Slic3r {

const char* softfever_version()
{
    return "@SOFTFEVER_VERSION@";
}

} // namespace Slic3r
]=])
string(CONFIGURE "${CONTENT}" CONTENT @ONLY)
file(WRITE "${OUTPUT}" "${CONTENT}")
