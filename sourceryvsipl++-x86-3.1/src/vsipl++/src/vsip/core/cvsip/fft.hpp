/* Copyright (c) 2006 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/cvsip/fft.hpp
    @author  Stefan Seefeld
    @date    2006-10-16
    @brief   VSIPL++ Library: FFT wrappers and traits to bridge with 
             C-VSIPL.
*/

#ifndef VSIP_CORE_CVSIP_FFT_HPP
#define VSIP_CORE_CVSIP_FFT_HPP

#include <memory>
#include <vsip/core/config.hpp>
#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/opt/dispatch.hpp>
#include <vsip/core/fft/util.hpp>

namespace vsip
{
namespace impl
{
namespace cvsip
{

template <typename I, dimension_type D, typename S>
std::auto_ptr<I>
create(Domain<D> const &dom, S scale, unsigned int);

} // namespace vsip::impl::cvsip
} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{
template <typename I,
	  typename O,
	  int S,
	  return_mechanism_type R,
	  unsigned N>
struct Evaluator<op::fft<1, I, O, S, R, N>, be::cvsip,
  std::auto_ptr<impl::fft::Fft_backend<1, I, O, S> >
  (Domain<1> const &, typename impl::scalar_of<I>::type)>
{
  typedef typename impl::scalar_of<I>::type scalar_type;
  static bool const has_float =
#if VSIP_IMPL_CVSIP_HAVE_FLOAT
    true
#else
    false
#endif
    ;
  static bool const has_double =
#if VSIP_IMPL_CVSIP_HAVE_DOUBLE
    true
#else
    false
#endif
    ;
  static bool const ct_valid = (has_float && 
                                is_same<scalar_type, float>::value) ||
                               (has_double && 
                                is_same<scalar_type, double>::value);
  static bool rt_valid(Domain<1> const &, scalar_type) { return true;}
  static std::auto_ptr<impl::fft::Fft_backend<1, I, O, S> >
  exec(Domain<1> const &dom, scalar_type scale)
  {
    return impl::cvsip::create<impl::fft::Fft_backend<1, I, O, S> >(dom, scale, N);
  }
};

template <typename I,
	  typename O,
	  int A,
	  int D,
	  return_mechanism_type R,
	  unsigned N>
struct Evaluator<op::fftm<I, O, A, D, R, N>, be::cvsip,
  std::auto_ptr<impl::fft::Fftm_backend<I, O, A, D> > 
  (Domain<2> const &, typename impl::scalar_of<I>::type)>
{
  typedef typename impl::scalar_of<I>::type scalar_type;
  static bool const ct_valid = !is_same<scalar_type, long double>::value;
  static bool rt_valid(Domain<2> const &, scalar_type) { return true;}
  static std::auto_ptr<impl::fft::Fftm_backend<I, O, A, D> > 
  exec(Domain<2> const &dom, scalar_type scale)
  {
    return impl::cvsip::create<impl::fft::Fftm_backend<I, O, A, D> >(dom, scale, N);
  }
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif
