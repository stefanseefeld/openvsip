#include "f2c.h"

#ifdef KR_headers
double sqrt();
double r_sqrt(x) real *x;
#else
#undef abs
#include "math.h"
#undef complex
double r_sqrt(real *x)
#endif
{
return( sqrt(*x) );
}
