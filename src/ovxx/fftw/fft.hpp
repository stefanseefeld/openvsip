//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_fftw_fft_hpp_
#define ovxx_fftw_fft_hpp_

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <ovxx/dispatch.hpp>
#include <ovxx/signal/fft/util.hpp>
#include <memory>

namespace ovxx
{
namespace fftw
{
using vsip::complex;

template <typename I, dimension_type D>
std::unique_ptr<I>
create(vsip::Domain<D> const &dom, unsigned);

#define OVXX_FFTW_FFT_DECL(D,I,O,S)		      \
template <>                                           \
std::unique_ptr<signal::fft::fft_backend<D,I,O,S> >    \
create(vsip::Domain<D> const &, unsigned);

#define OVXX_FFTW_FFT_DECL_T(T)			      \
OVXX_FFTW_FFT_DECL(1, T, complex<T>, 0)	              \
OVXX_FFTW_FFT_DECL(1, complex<T>, T, 0)		      \
OVXX_FFTW_FFT_DECL(1, complex<T>, complex<T>, fft_fwd)\
OVXX_FFTW_FFT_DECL(1, complex<T>, complex<T>, fft_inv)\
OVXX_FFTW_FFT_DECL(2, T, complex<T>, 0)               \
OVXX_FFTW_FFT_DECL(2, T, complex<T>, 1)               \
OVXX_FFTW_FFT_DECL(2, complex<T>, T, 0)               \
OVXX_FFTW_FFT_DECL(2, complex<T>, T, 1)               \
OVXX_FFTW_FFT_DECL(2, complex<T>, complex<T>, fft_fwd)\
OVXX_FFTW_FFT_DECL(2, complex<T>, complex<T>, fft_inv)\
OVXX_FFTW_FFT_DECL(3, T, complex<T>, 0)               \
OVXX_FFTW_FFT_DECL(3, T, complex<T>, 1)               \
OVXX_FFTW_FFT_DECL(3, T, complex<T>, 2)               \
OVXX_FFTW_FFT_DECL(3, complex<T>, T, 0)               \
OVXX_FFTW_FFT_DECL(3, complex<T>, T, 1)               \
OVXX_FFTW_FFT_DECL(3, complex<T>, T, 2)               \
OVXX_FFTW_FFT_DECL(3, complex<T>, complex<T>, fft_fwd)\
OVXX_FFTW_FFT_DECL(3, complex<T>, complex<T>, fft_inv)

#ifdef OVXX_FFTW_HAVE_FLOAT
OVXX_FFTW_FFT_DECL_T(float)
#endif
#ifdef OVXX_FFTW_HAVE_DOUBLE
OVXX_FFTW_FFT_DECL_T(double)
#endif

#undef OVXX_FFT_DECL_T
#undef OVXX_FFT_DECL

#define OVXX_FFTW_FFTM_DECL(I,O,A,D)		      \
template <>					      \
std::unique_ptr<signal::fft::fftm_backend<I,O,A,D> >    \
create(vsip::Domain<2> const &, unsigned);

#define OVXX_FFTW_FFTM_DECL_T(T)		      \
OVXX_FFTW_FFTM_DECL(T, complex<T>, 0, fft_fwd)	      \
OVXX_FFTW_FFTM_DECL(T, complex<T>, 1, fft_fwd)	      \
OVXX_FFTW_FFTM_DECL(complex<T>, T, 0, fft_inv)	      \
OVXX_FFTW_FFTM_DECL(complex<T>, T, 1, fft_inv)	      \
OVXX_FFTW_FFTM_DECL(complex<T>, complex<T>, 0, fft_fwd)\
OVXX_FFTW_FFTM_DECL(complex<T>, complex<T>, 1, fft_fwd)\
OVXX_FFTW_FFTM_DECL(complex<T>, complex<T>, 0, fft_inv)\
OVXX_FFTW_FFTM_DECL(complex<T>, complex<T>, 1, fft_inv)

#ifdef OVXX_FFTW_HAVE_FLOAT
OVXX_FFTW_FFTM_DECL_T(float)
#endif
#ifdef OVXX_FFTW_HAVE_DOUBLE
OVXX_FFTW_FFTM_DECL_T(double)
#endif

#undef OVXX_FFTW_FFTM_DECL_T
#undef OVXX_FFTW_FFTM_DECL

} // namespace ovxx::fftw

namespace dispatcher
{
template <dimension_type D,
	  typename I,
	  typename O,
	  int S,
	  vsip::return_mechanism_type R,
	  unsigned N>
struct Evaluator<op::fft<D, I, O, S, R, N>, be::fftw,
		 std::unique_ptr<signal::fft::fft_backend<D, I, O, S> >
		 (vsip::Domain<D> const &, typename scalar_of<I>::type)>
{
  typedef typename scalar_of<I>::type scalar_type;
  static bool const ct_valid =
#ifdef OVXX_FFTW_HAVE_FLOAT
    is_same<scalar_type, float>::value ||
#endif
#ifdef OVXX_FFTW_HAVE_DOUBLE
    is_same<scalar_type, double>::value ||
#endif
    false;

  static bool rt_valid(vsip::Domain<D> const &, scalar_type) { return true;}
  static std::unique_ptr<signal::fft::fft_backend<D, I, O, S> >
  exec(vsip::Domain<D> const &dom, scalar_type)
  {
    return fftw::create<signal::fft::fft_backend<D, I, O, S> >(dom, N);
  }
};

template <typename I,
	  typename O,
	  int A,
	  int D,
	  vsip::return_mechanism_type R,
	  unsigned N>
struct Evaluator<op::fftm<I, O, A, D, R, N>, be::fftw,
		 std::unique_ptr<signal::fft::fftm_backend<I, O, A, D> >
		 (vsip::Domain<2> const &, typename scalar_of<I>::type)>
{
  typedef typename scalar_of<I>::type scalar_type;
  static bool const ct_valid =
#ifdef OVXX_FFTW_HAVE_FLOAT
    is_same<scalar_type, float>::value ||
#endif
#ifdef OVXX_FFTW_HAVE_DOUBLE
    is_same<scalar_type, double>::value ||
#endif
    false;

  static bool rt_valid(vsip::Domain<2> const &, scalar_type)
  { return true;}
  static std::unique_ptr<signal::fft::fftm_backend<I, O, A, D> >
  exec(vsip::Domain<2> const &dom, scalar_type)
  {
    return fftw::create<signal::fft::fftm_backend<I, O, A, D> >(dom, N);
  }
};

} // namespace ovxx::dispatcher
} // namespace ovxx

#endif
