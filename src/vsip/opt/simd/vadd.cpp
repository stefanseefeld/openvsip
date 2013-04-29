/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#include <vsip/opt/simd/vadd.hpp>

namespace vsip
{
namespace impl
{
namespace simd
{

#if !VSIP_IMPL_INLINE_LIBSIMD

template <typename T>
void
vadd(T const *op1, T const *op2, T *res, length_type size)
{
  static bool const Is_vectorized = is_algorithm_supported<T, false, Alg_vadd>
                                      ::value;
  Vadd<T, Is_vectorized>::exec(op1, op2, res, size);
}

template void vadd(short const *, short const *, short*, length_type);
template void vadd(float const *, float const *, float*, length_type);
template void vadd(double const *, double const *, double*, length_type);
template void vadd(std::complex<float> const *, std::complex<float> const *,
		   std::complex<float>*, length_type);
template void vadd(std::complex<double> const *, std::complex<double> const *,
		   std::complex<double>*, length_type);

template <typename T>
void
vadd(std::pair<T const *,T const *> op1, std::pair<T const *,T const *> op2, std::pair<T*,T*> res,
     length_type size)
{
  static bool const Is_vectorized = is_algorithm_supported<T, true, Alg_vadd>
                                      ::value;
  Vadd<std::pair<T,T>, Is_vectorized>::exec(op1, op2, res, size);
}

template void vadd(std::pair<float const *,float const *>,
		   std::pair<float const *,float const *>,
		   std::pair<float*,float*>, length_type);
template void vadd(std::pair<double const *,double const *>,
		   std::pair<double const *,double const *>,
		   std::pair<double*,double*>, length_type);

#endif

} // namespace vsip::impl::simd
} // namespace vsip::impl
} // namespace vsip
