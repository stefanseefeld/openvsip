/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#ifndef VSIP_OPT_CBE_LOGMA_PARAMS_H
#define VSIP_OPT_CBE_LOGMA_PARAMS_H

#include <lwp_params.h>

#ifdef _cplusplus
namespace vsip
{
namespace impl
{
namespace cbe
{
#endif

// R = log10(A) * b + c, where A and R are vectors and b and c
// are scalars.

// Operations for float, complex interleaved and complex split
// op types (CLMA and ZLMA) would be defined here.
enum logma_op { LMA };

typedef struct
{
  int                cmd;
  unsigned int       length;
  unsigned long long a_ptr;
  unsigned long long r_ptr;
  double             b_value;
  double             c_value;
} Logma_params;

#define VSIP_IMPL_LOGMA_STACK_SIZE 4096
#define VSIP_IMPL_LOGMA_BUFFER_SIZE 65536
#define VSIP_IMPL_LOGMA_DTL_SIZE 128

#ifdef _cplusplus
} // namespace vsip::impl::cbe
} // namespace vsip::impl
} // namespace vsip
#endif

#endif
