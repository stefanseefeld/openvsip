/* Copyright (c) 2006 by CodeSourcery.  All rights reserved. */

#include <vsip/opt/simd/vma_ip_csc.hpp>

namespace vsip
{
namespace impl
{
namespace simd
{

#if !VSIP_IMPL_INLINE_LIBSIMD

template <typename T>
void
vma_ip_cSC(std::complex<T> const& a,
	   T const*               B,
	   std::complex<T>*       R,
	   length_type n)
{
  static bool const Is_vectorized =
    is_algorithm_supported<std::complex<T>, false, Alg_vma_ip_cSC>::value;
  Vma_ip_cSC<std::complex<T>, Is_vectorized>::exec(a, B, R, n);
}

template void vma_ip_cSC(std::complex<float> const&, float const*,
			 std::complex<float>*, length_type);
template void vma_ip_cSC(std::complex<double> const&, double const*,
			 std::complex<double>*, length_type);

#endif

} // namespace vsip::impl::simd
} // namespace vsip::impl
} // namespace vsip
