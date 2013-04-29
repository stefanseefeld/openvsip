#include "f2c.h"
#include "cblas.h"

/* This is the part the brings in the functions if we define the NO_INLINE_WRAP
   macro. If that is the case, we define INLINE to nothing and just grab the
   functions from cblaswr.h. If NO_INLINE_WRAP is not defined, then blaswrap.h
   defines INLINE to static inline and includes cblaswr.h.
*/

#ifdef NO_INLINE_WRAP

#define INLINE
#include "cblaswr.h"

#endif

