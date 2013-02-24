#include "f2c.h"

#define log10e 0.43429448190325182765

#ifdef KR_headers
double log();
double r_lg10(x) real *x;
#else
#undef abs
#include "math.h"
#undef complex
double r_lg10(real *x)
#endif
{
return( log10e * log(*x) );
}
