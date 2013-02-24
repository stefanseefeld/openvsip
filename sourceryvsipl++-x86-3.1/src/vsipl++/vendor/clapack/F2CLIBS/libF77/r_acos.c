#include "f2c.h"

#ifdef KR_headers
double acos();
double r_acos(x) real *x;
#else
#undef abs
#include "math.h"
#undef complex
double r_acos(real *x)
#endif
{
return( acos(*x) );
}
