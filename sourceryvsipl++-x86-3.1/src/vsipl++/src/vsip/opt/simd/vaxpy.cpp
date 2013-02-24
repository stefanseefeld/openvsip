/* Copyright (c) 2006 by CodeSourcery.  All rights reserved. */

#include <vsip/opt/simd/vaxpy.hpp>

namespace vsip
{
namespace impl
{
namespace simd
{

#if !VSIP_IMPL_INLINE_LIBSIMD

template <typename T>
void
vma_cSC(std::complex<T> const& a,
	T const*               B,
	std::complex<T> const* C,
	std::complex<T>*       R,
	length_type n)
{
  static bool const Is_vectorized =
    is_algorithm_supported<std::complex<T>, false, Alg_vma_cSC>::value;
  Vma_cSC<std::complex<T>, Is_vectorized>::exec(a, B, C, R, n);
}

template void vma_cSC(std::complex<float> const&, float const*,
		      std::complex<float> const*, std::complex<float>*, length_type);
template void vma_cSC(std::complex<double> const&, double const*,
		      std::complex<double> const*, std::complex<double>*, length_type);

#endif

} // namespace vsip::impl::simd
} // namespace vsip::impl
} // namespace vsip
