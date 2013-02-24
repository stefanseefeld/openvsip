#include "f2c.h"

#ifdef KR_headers
double acos();
double d_acos(x) doublereal *x;
#else
#undef abs
#include "math.h"
#undef complex
double d_acos(doublereal *x)
#endif
{
return( acos(*x) );
}
