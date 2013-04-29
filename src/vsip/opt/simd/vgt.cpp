/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#include <vsip/opt/simd/vgt.hpp>

namespace vsip
{
namespace impl
{
namespace simd
{

#if !VSIP_IMPL_INLINE_LIBSIMD

template <typename T>
void
vgt(T const* op1, T const* op2, bool *res, length_type size)
{
  static bool const Is_vectorized = is_algorithm_supported<T, false, Alg_vgt>
                                      ::value;
  Vgt<T, Is_vectorized>::exec(op1, op2, res, size);
}

template void vgt(float const*, float const*, bool*, length_type);



#endif

} // namespace vsip::impl::simd
} // namespace vsip::impl
} // namespace vsip
