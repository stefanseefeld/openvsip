/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#ifndef VSIP_OPT_CBE_TERNARY_PARAMS_H
#define VSIP_OPT_CBE_TERNARY_PARAMS_H

#include <lwp_params.h>

#ifdef _cplusplus
namespace vsip
{
namespace impl
{
namespace cbe
{
#endif

enum ternary_op { AM, MA, CAM, CMA, ZAM, ZMA};

typedef struct
{
  int                cmd;
  unsigned int       length;
  unsigned long long a_ptr;
  unsigned long long b_ptr;
  unsigned long long c_ptr;
  unsigned long long r_ptr;
} Ternary_params;

typedef struct
{
  int                cmd;
  unsigned int       length;
  unsigned long long a_im_ptr;
  unsigned long long a_re_ptr;
  unsigned long long b_im_ptr;
  unsigned long long b_re_ptr;
  unsigned long long c_im_ptr;
  unsigned long long c_re_ptr;
  unsigned long long r_im_ptr;
  unsigned long long r_re_ptr;
} Ternary_split_params;

#define VSIP_IMPL_TERNARY_STACK_SIZE 4096
#define VSIP_IMPL_TERNARY_BUFFER_SIZE 65536
#define VSIP_IMPL_TERNARY_DTL_SIZE 128

#ifdef _cplusplus
} // namespace vsip::impl::cbe
} // namespace vsip::impl
} // namespace vsip
#endif

#endif
