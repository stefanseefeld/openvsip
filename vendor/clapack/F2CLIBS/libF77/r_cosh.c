#include "f2c.h"

#ifdef KR_headers
double cosh();
double r_cosh(x) real *x;
#else
#undef abs
#include "math.h"
#undef complex
double r_cosh(real *x)
#endif
{
return( cosh(*x) );
}
