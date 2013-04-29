/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/fftw3/fft.hpp
    @author  Stefan Seefeld
    @date    2006-03-06
    @brief   VSIPL++ Library: FFT wrappers and traits to bridge with FFTW3.
*/

#ifndef VSIP_OPT_FFTW3_FFT_HPP
#define VSIP_OPT_FFTW3_FFT_HPP

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
namespace fftw3
{

template <typename I, dimension_type D>
std::auto_ptr<I>
create(Domain<D> const &dom, unsigned);

#define FFT_DECL(D,I,O,S)                             \
template <>                                           \
std::auto_ptr<fft::Fft_backend<D,I,O,S> >             \
create(Domain<D> const &, unsigned);

#define FFT_DECL_T(T)	                              \
FFT_DECL(1, T, std::complex<T>, 0)                    \
FFT_DECL(1, std::complex<T>, T, 0)                    \
FFT_DECL(1, std::complex<T>, std::complex<T>, fft_fwd)\
FFT_DECL(1, std::complex<T>, std::complex<T>, fft_inv)\
FFT_DECL(2, T, std::complex<T>, 0)                    \
FFT_DECL(2, T, std::complex<T>, 1)                    \
FFT_DECL(2, std::complex<T>, T, 0)                    \
FFT_DECL(2, std::complex<T>, T, 1)                    \
FFT_DECL(2, std::complex<T>, std::complex<T>, fft_fwd)\
FFT_DECL(2, std::complex<T>, std::complex<T>, fft_inv)\
FFT_DECL(3, T, std::complex<T>, 0)                    \
FFT_DECL(3, T, std::complex<T>, 1)                    \
FFT_DECL(3, T, std::complex<T>, 2)                    \
FFT_DECL(3, std::complex<T>, T, 0)                    \
FFT_DECL(3, std::complex<T>, T, 1)                    \
FFT_DECL(3, std::complex<T>, T, 2)                    \
FFT_DECL(3, std::complex<T>, std::complex<T>, fft_fwd)\
FFT_DECL(3, std::complex<T>, std::complex<T>, fft_inv)

#ifdef VSIP_IMPL_FFTW3_HAVE_FLOAT
FFT_DECL_T(float)
#endif
#ifdef VSIP_IMPL_FFTW3_HAVE_DOUBLE
FFT_DECL_T(double)
#endif
#ifdef VSIP_IMPL_FFTW3_HAVE_LONG_DOUBLE
FFT_DECL_T(long double)
#endif

#undef FFT_DECL_T
#undef FFT_DECL

#define FFTM_DECL(I,O,A,D)                             \
template <>					       \
std::auto_ptr<fft::Fftm_backend<I,O,A,D> >             \
create(Domain<2> const &, unsigned);

#define FFTM_DECL_T(T)				       \
FFTM_DECL(T, std::complex<T>, 0, fft_fwd)              \
FFTM_DECL(T, std::complex<T>, 1, fft_fwd)              \
FFTM_DECL(std::complex<T>, T, 0, fft_inv)              \
FFTM_DECL(std::complex<T>, T, 1, fft_inv)              \
FFTM_DECL(std::complex<T>, std::complex<T>, 0, fft_fwd)\
FFTM_DECL(std::complex<T>, std::complex<T>, 1, fft_fwd)\
FFTM_DECL(std::complex<T>, std::complex<T>, 0, fft_inv)\
FFTM_DECL(std::complex<T>, std::complex<T>, 1, fft_inv)

#ifdef VSIP_IMPL_FFTW3_HAVE_FLOAT
FFTM_DECL_T(float)
#endif
#ifdef VSIP_IMPL_FFTW3_HAVE_DOUBLE
FFTM_DECL_T(double)
#endif
#ifdef VSIP_IMPL_FFTW3_HAVE_LONG_DOUBLE
FFTM_DECL_T(long double)
#endif

#undef FFTM_DECL_T
#undef FFTM_DECL

} // namespace vsip::impl::fftw3
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
struct Evaluator<op::fft<D, I, O, S, R, N>, be::fftw3,
  std::auto_ptr<impl::fft::Fft_backend<D, I, O, S> >
  (Domain<D> const &, typename impl::scalar_of<I>::type)>
{
  typedef typename impl::scalar_of<I>::type scalar_type;
  static int const A = impl::fft::axis<I, O, S>::value;
  static int const E = impl::fft::exponent<I, O, S>::value;
  static bool const ct_valid =
#ifdef VSIP_IMPL_FFTW3_HAVE_FLOAT
    is_same<scalar_type, float>::value ||
#endif
#ifdef VSIP_IMPL_FFTW3_HAVE_DOUBLE
    is_same<scalar_type, double>::value ||
#endif
#ifdef VSIP_IMPL_FFTW3_HAVE_LONG_DOUBLE
    is_same<scalar_type, long double>::value ||
#endif
    false;

  static bool rt_valid(Domain<D> const &, scalar_type) { return true;}
  static std::auto_ptr<impl::fft::Fft_backend<D, I, O, S> >
  exec(Domain<D> const &dom, scalar_type)
  {
    return impl::fftw3::create<impl::fft::Fft_backend<D, I, O, S> >(dom, N);
  }
};

template <typename I,
	  typename O,
	  int A,
	  int D,
	  return_mechanism_type R,
	  unsigned N>
struct Evaluator<op::fftm<I, O, A, D, R, N>, be::fftw3,
  std::auto_ptr<impl::fft::Fftm_backend<I, O, A, D> > 
  (Domain<2> const &, typename impl::scalar_of<I>::type)>
{
  typedef typename impl::scalar_of<I>::type scalar_type;
  static bool const ct_valid =
#ifdef VSIP_IMPL_FFTW3_HAVE_FLOAT
    is_same<scalar_type, float>::value ||
#endif
#ifdef VSIP_IMPL_FFTW3_HAVE_DOUBLE
    is_same<scalar_type, double>::value ||
#endif
#ifdef VSIP_IMPL_FFTW3_HAVE_LONG_DOUBLE
    is_same<scalar_type, long double>::value ||
#endif
    false;

  static bool rt_valid(Domain<2> const &, scalar_type)
  { return true;}
  static std::auto_ptr<impl::fft::Fftm_backend<I, O, A, D> > 
  exec(Domain<2> const &dom, scalar_type)
  {
    return impl::fftw3::create<impl::fft::Fftm_backend<I, O, A, D> >(dom, N);
  }
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif

