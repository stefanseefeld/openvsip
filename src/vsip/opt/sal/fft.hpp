/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/sal/fft.hpp
    @author  Stefan Seefeld
    @date    2006-02-02
    @brief   VSIPL++ Library: FFT wrappers and traits to bridge with 
             Mercury's SAL.
*/

#ifndef VSIP_IMPL_SAL_FFT_HPP
#define VSIP_IMPL_SAL_FFT_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/opt/dispatch.hpp>
#include <vsip/core/fft/util.hpp>
#include <memory>

namespace vsip
{
namespace impl
{
namespace sal
{

template <typename I, dimension_type D, typename S>
std::auto_ptr<I>
create(Domain<D> const &dom, S scale);

// Traits class to indicate whether SAL FFTM supports a given type for
// input and output.

template <typename T> struct Is_fft_avail
{ static bool const value = false; };

#if VSIP_IMPL_HAVE_SAL_FLOAT
template <> struct Is_fft_avail<float>
{ static bool const value = true; };

template <> struct Is_fft_avail<complex<float> >
{ static bool const value = true; };
#endif

#if VSIP_IMPL_HAVE_SAL_DOUBLE
template <> struct Is_fft_avail<double>
{ static bool const value = true; };

template <> struct Is_fft_avail<complex<double> >
{ static bool const value = true; };
#endif

} // namespace vsip::impl::sal
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
struct Evaluator<op::fft<D, I, O, S, R, N>, be::mercury_sal,
  std::auto_ptr<impl::fft::Fft_backend<D, I, O, S> >
  (Domain<D> const &, typename impl::scalar_of<I>::type)>
{
  typedef typename impl::scalar_of<I>::type scalar_type;
  static bool const ct_valid = 
    (D == 1 || D == 2) &&
    impl::sal::Is_fft_avail<I>::value &&
    impl::sal::Is_fft_avail<O>::value;
  static bool rt_valid(Domain<D> const &dom, scalar_type)
  {
    for (dimension_type d = 0; d != D; ++d)
      // SAL can only deal with powers of 2.
      if (dom[d].size() & (dom[d].size() - 1) ||
	  // SAL requires a minimum block size.
	  dom[d].size() < 8) return false;

    return true;
  }
  static std::auto_ptr<impl::fft::Fft_backend<D, I, O, S> >
  exec(Domain<D> const &dom, scalar_type scale)
  {
    return impl::sal::create<impl::fft::Fft_backend<D, I, O, S> >(dom, scale);
  }
};

template <typename I,
	  typename O,
	  int A,
	  int D,
	  return_mechanism_type R,
	  unsigned N>
struct Evaluator<op::fftm<I, O, A, D, R, N>, be::mercury_sal,
  std::auto_ptr<impl::fft::Fftm_backend<I, O, A, D> > 
  (Domain<2> const &, typename impl::scalar_of<I>::type)>
{
  typedef typename impl::scalar_of<I>::type scalar_type;
  static bool const ct_valid = impl::sal::Is_fft_avail<I>::value &&
                               impl::sal::Is_fft_avail<O>::value;
  static bool rt_valid(Domain<2> const &dom, scalar_type)
  {
    static int const axis = A == vsip::row ? 1 : 0;
    // SAL can only deal with powers of 2.
    if (dom[axis].size() & (dom[axis].size() - 1)) return false;
    // SAL requires a minimum block size.
    if (dom[axis].size() < 8) return false;
    else return true;
  }
  static std::auto_ptr<impl::fft::Fftm_backend<I, O, A, D> > 
  exec(Domain<2> const &dom, scalar_type scale)
  {
    return impl::sal::create<impl::fft::Fftm_backend<I, O, A, D> >(dom, scale);
  }
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif
