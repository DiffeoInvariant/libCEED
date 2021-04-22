#ifndef CEED_JAX_TYPESHEADER
#define CEED_JAX_TYPES_HEADER
#include <ceed/backend.h>
#include <exception>

#define CeedJaxFromChk(ierr)   \
  do {                         \
     if (ierr) {               \
      return NULL;             \
    }                          \
  } while (0)                  \


#define CeedHandleXLAException(exc)     \
  do {                                  \
    std::string error = exc.toString();				 \
    return CeedError(ceed, CEED_ERROR_BACKEND, error.c_str());	 \
  } while (0)

namespace ceed {
  namespace xla {
    typedef int (*ceedFunction)();
  }
}


#endif // CEED_JAX_TYPES_HEADER
