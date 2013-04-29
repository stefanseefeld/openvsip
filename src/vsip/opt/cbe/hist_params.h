/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#ifndef VSIP_OPT_CBE_HIST_PARAMS_H
#define VSIP_OPT_CBE_HIST_PARAMS_H

#include <lwp_params.h>

#ifdef _cplusplus
namespace vsip
{
namespace impl
{
namespace cbe
{
#endif

enum hist_op
{
  HIST,
  IHIST,
};

// Structures used in DMAs should be sized in multiples of 128-bits

typedef struct
{
  int                cmd;	// Chosen from hist_op above.
  unsigned int       length;	// Size of value vector.
  unsigned int       num_bin;	// Size of bin vector.
  float              min;
  float              max;
  unsigned long long bin_ptr;	// Pointer to bin vector.
  unsigned long long val_ptr;
} Hist_params;

#define VSIP_IMPL_HIST_STACK_SIZE 4096
#define VSIP_IMPL_HIST_BUFFER_SIZE 65536
#define VSIP_IMPL_HIST_DTL_SIZE 128
#define VSIP_IMPL_HIST_MAX_BINS 128

#ifdef _cplusplus
} // namespace vsip::impl::cbe
} // namespace vsip::impl
} // namespace vsip
#endif

#endif
