/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#ifndef VSIP_IMPL_IPP_FFT_HPP
#define VSIP_IMPL_IPP_FFT_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/core/fft/util.hpp>
#include <vsip/opt/dispatch.hpp>
#include <memory>

namespace vsip
{
namespace impl
{
namespace ipp
{

/// These are the entry points into the IPP FFT bridge.
template <typename I, dimension_type D, typename S>
std::auto_ptr<I>
create(Domain<D> const &dom, S scale, bool fast);

} // namespace vsip::impl::ipp
} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{

template <dimension_type D,
	  typename I,
	  typename O,
	  int S,
	  return_mechanism_type R,
	  unsigned N>
struct Evaluator<op::fft<D, I, O, S, R, N>, be::intel_ipp,
  std::auto_ptr<impl::fft::Fft_backend<D, I, O, S> >
  (Domain<D> const &, typename impl::Scalar_of<I>::type)>
{
  typedef typename impl::Scalar_of<I>::type scalar_type;
  static bool const ct_valid =
    // IPP supports float and double real and complex 1D FFTs,...
    (D == 1 && (impl::Type_equal<scalar_type, float>::value ||
		impl::Type_equal<scalar_type, double>::value)) ||
    // ...and complex float 2D FFTs
    (D == 2 && impl::Type_equal<scalar_type, float>::value &&
     impl::Type_equal<I, O>::value);
  static bool rt_valid(Domain<D> const &, scalar_type) { return true;}
  static std::auto_ptr<impl::fft::Fft_backend<D, I, O, S> >
  exec(Domain<D> const &dom, scalar_type scale)
  {
    bool fast = impl::fft::is_power_of_two(dom);
    return impl::ipp::create<impl::fft::Fft_backend<D, I, O, S> >(dom, scale, fast);
  }
};

template <typename I,
	  typename O,
	  int A,
	  int D,
	  return_mechanism_type R,
	  unsigned N>
struct Evaluator<op::fftm<I, O, A, D, R, N>, be::intel_ipp,
  std::auto_ptr<impl::fft::Fftm_backend<I, O, A, D> > 
  (Domain<2> const &, typename impl::Scalar_of<I>::type)>
{
  typedef typename impl::Scalar_of<I>::type scalar_type;
  static bool const ct_valid = 
    // IPP supports float and double real and complex FFTM
    impl::Type_equal<scalar_type, float>::value ||
    impl::Type_equal<scalar_type, double>::value;
  static bool rt_valid(Domain<2> const &, scalar_type) { return true;}
  static std::auto_ptr<impl::fft::Fftm_backend<I, O, A, D> > 
  exec(Domain<2> const &dom, scalar_type scale)
  {
    bool fast = impl::fft::is_power_of_two(dom);
    return impl::ipp::create<impl::fft::Fftm_backend<I, O, A, D> >(dom, scale, fast);
  }
};
} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif

