/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#ifndef VSIP_OPT_CBE_REDUCTION_PARAMS_H
#define VSIP_OPT_CBE_REDUCTION_PARAMS_H

#include <lwp_params.h>

#ifdef _cplusplus
namespace vsip
{
namespace impl
{
namespace cbe
{
#endif

// Structures used in DMAs should be sized in multiples of 128-bits
enum reduction_op { SUM, SUMSQ, CSUM, CSUMSQ, ZSUM, ZSUMSQ};

typedef struct
{
  int                cmd;
  unsigned int       length;
  unsigned long long r_ptr;

  unsigned long long a_ptr;
} Reduction_params;

typedef struct
{
  // The first three members are the same
  // as in Reduction_params.
  int                cmd;
  unsigned int       length;
  unsigned long long r_ptr;

  unsigned long long re_ptr;
  unsigned long long im_ptr;
} Reduction_split_params;

#define VSIP_IMPL_REDUCTION_STACK_SIZE 4096
#define VSIP_IMPL_REDUCTION_BUFFER_SIZE 65536
#define VSIP_IMPL_REDUCTION_DTL_SIZE 128

#ifdef _cplusplus
} // namespace vsip::impl::cbe
} // namespace vsip::impl
} // namespace vsip
#endif

#endif
