/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#include <vsip/opt/simd/vmul.hpp>

namespace vsip
{
namespace impl
{
namespace simd
{

#if !VSIP_IMPL_INLINE_LIBSIMD

template <typename T>
void
vmul(T const *op1, T const *op2, T *res, length_type size)
{
  static bool const Is_vectorized = is_algorithm_supported<T, false, Alg_vmul>
                                      ::value;
  Vmul<T, Is_vectorized>::exec(op1, op2, res, size);
}

template void vmul(short const *, short const *, short*, length_type);
template void vmul(float const *, float const *, float*, length_type);
template void vmul(double const *, double const *, double*, length_type);
template void vmul(std::complex<float> const *, std::complex<float> const *,
		   std::complex<float>*, length_type);
template void vmul(std::complex<double> const *, std::complex<double> const *,
		   std::complex<double>*, length_type);

template <typename T>
void
vmul(std::pair<T const *,T const *>  op1,
     std::pair<T const *,T const *>  op2,
     std::pair<T*,T*>  res,
     length_type size)
{
  static bool const Is_vectorized = is_algorithm_supported<T, true, Alg_vmul>
                                      ::value;
  Vmul<std::pair<T,T>, Is_vectorized>::exec(op1, op2, res, size);
}

template void vmul(std::pair<float const *,float const *>,
		   std::pair<float const *,float const *>,
		   std::pair<float*,float*>, length_type);
template void vmul(std::pair<double const *,double const *>,
		   std::pair<double const *,double const *>,
		   std::pair<double*,double*>, length_type);

#endif

} // namespace vsip::impl::simd
} // namespace vsip::impl
} // namespace vsip
