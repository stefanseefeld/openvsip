#include "f2c.h"

#ifdef KR_headers
double sqrt();
double d_sqrt(x) doublereal *x;
#else
#undef abs
#include "math.h"
#undef complex
double d_sqrt(doublereal *x)
#endif
{
return( sqrt(*x) );
}
