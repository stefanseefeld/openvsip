/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#ifndef VSIP_OPT_CBE_DOT_PARAMS_H
#define VSIP_OPT_CBE_DOT_PARAMS_H

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

typedef struct
{
  int                conj;
  unsigned int       length;
  unsigned long long a_ptr;
  unsigned long long b_ptr;
  unsigned long long r_ptr;
} Dot_params;

typedef struct
{
  int                conj;
  unsigned int       length;
  unsigned long long a_im_ptr;
  unsigned long long a_re_ptr;
  unsigned long long b_im_ptr;
  unsigned long long b_re_ptr;
  unsigned long long r_ptr;
} Dot_split_params;

#define VSIP_IMPL_DOT_STACK_SIZE 4096
#define VSIP_IMPL_DOT_BUFFER_SIZE 65536
#define VSIP_IMPL_DOT_DTL_SIZE 128

#ifdef _cplusplus
} // namespace vsip::impl::cbe
} // namespace vsip::impl
} // namespace vsip
#endif

#endif
