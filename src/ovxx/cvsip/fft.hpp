//
// Copyright (c) 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_cvsip_fft_hpp_
#define ovxx_cvsip_fft_hpp_

#include <ovxx/config.hpp>
#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <ovxx/dispatch.hpp>
#include <ovxx/signal/fft/util.hpp>
#include <memory>

namespace ovxx
{
namespace cvsip
{

template <typename I, dimension_type D, typename S>
std::unique_ptr<I>
create(Domain<D> const &dom, S scale, unsigned int);

} // namespace ovxx::cvsip

namespace dispatcher
{
template <typename I,
	  typename O,
	  int S,
	  return_mechanism_type R,
	  unsigned N>
struct Evaluator<op::fft<1, I, O, S, R, N>, be::cvsip,
  std::unique_ptr<signal::fft::fft_backend<1, I, O, S> >
  (Domain<1> const &, typename scalar_of<I>::type)>
{
  typedef typename scalar_of<I>::type scalar_type;
  static bool const has_float =
#if OVXX_CVSIP_HAVE_FLOAT
    true
#else
    false
#endif
    ;
  static bool const has_double =
#if OVXX_CVSIP_HAVE_DOUBLE
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
  static std::unique_ptr<signal::fft::fft_backend<1, I, O, S> >
  exec(Domain<1> const &dom, scalar_type scale)
  {
    return cvsip::create<signal::fft::fft_backend<1, I, O, S> >(dom, scale, N);
  }
};

template <typename I,
	  typename O,
	  int A,
	  int D,
	  return_mechanism_type R,
	  unsigned N>
struct Evaluator<op::fftm<I, O, A, D, R, N>, be::cvsip,
  std::unique_ptr<signal::fft::fftm_backend<I, O, A, D> >
  (Domain<2> const &, typename scalar_of<I>::type)>
{
  typedef typename scalar_of<I>::type scalar_type;
  static bool const ct_valid = !is_same<scalar_type, long double>::value;
  static bool rt_valid(Domain<2> const &, scalar_type) { return true;}
  static std::unique_ptr<signal::fft::fftm_backend<I, O, A, D> >
  exec(Domain<2> const &dom, scalar_type scale)
  {
    return cvsip::create<signal::fft::fftm_backend<I, O, A, D> >(dom, scale, N);
  }
};

} // namespace ovxx::dispatcher
} // namespace ovxx

#endif
