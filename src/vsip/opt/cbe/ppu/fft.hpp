/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cbe/ppu/fft.hpp
    @author  Stefan Seefeld
    @date    2007-01-31
    @brief   VSIPL++ Library: FFT wrappers and traits to bridge with the CBE SDK.
*/

#ifndef VSIP_OPT_CBE_PPU_FFT_HPP
#define VSIP_OPT_CBE_PPU_FFT_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/core/fft/util.hpp>
#include <vsip/opt/dispatch.hpp>
#include <vsip/opt/cbe/fft_params.h>
#include <vsip/opt/cbe/ppu/task_manager.hpp>

namespace vsip
{
namespace impl
{
namespace cbe
{

template <typename I, dimension_type D, typename S>
std::auto_ptr<I>
create(Domain<D> const &dom, S scale);

#define FFT_DECL(D,I,O,S)                              \
template <>		      		               \
std::auto_ptr<fft::Fft_backend<D,I,O,S> >              \
create(Domain<D> const &, scalar_of<I>::type);

#define FFT_DECL_T(T)				       \
FFT_DECL(1, T, std::complex<T>, 0)                     \
FFT_DECL(1, std::complex<T>, T, 0)		       \
FFT_DECL(1, std::complex<T>, std::complex<T>, fft_fwd) \
FFT_DECL(1, std::complex<T>, std::complex<T>, fft_inv)

FFT_DECL_T(float)

#undef FFT_DECL_T
#undef FFT_DECL

#define FFTM_DECL(I,O,A,D)                             \
template <>                                            \
std::auto_ptr<fft::Fftm_backend<I,O,A,D> >             \
create(Domain<2> const &, scalar_of<I>::type);

#define FFTM_DECL_T(T)				       \
FFTM_DECL(T, std::complex<T>, 0, fft_fwd)              \
FFTM_DECL(T, std::complex<T>, 1, fft_fwd)	       \
FFTM_DECL(std::complex<T>, T, 0, fft_inv)              \
FFTM_DECL(std::complex<T>, T, 1, fft_inv)              \
FFTM_DECL(std::complex<T>, std::complex<T>, 0, fft_fwd)\
FFTM_DECL(std::complex<T>, std::complex<T>, 1, fft_fwd)\
FFTM_DECL(std::complex<T>, std::complex<T>, 0, fft_inv)\
FFTM_DECL(std::complex<T>, std::complex<T>, 1, fft_inv)

FFTM_DECL_T(float)

#undef FFTM_DECL_T
#undef FFTM_DECL

} // namespace vsip::impl::cbe
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
struct Evaluator<op::fft<1, I, O, S, R, N>, be::cbe_sdk,
  std::auto_ptr<impl::fft::Fft_backend<1, I, O, S> >
  (Domain<1> const &, float)>
{
  static bool const ct_valid = 
    is_same<I, complex<float> >::value &&
    is_same<I, O>::value;
  static bool rt_valid(Domain<1> const &dom, float) 
  { 
    return
      (dom.size() >= MIN_FFT_1D_SIZE) &&
      (dom.size() <= MAX_FFT_1D_SIZE) &&
      (impl::fft::is_power_of_two(dom)) &&
      impl::cbe::Task_manager::instance()->num_spes() > 0;
  }
  static std::auto_ptr<impl::fft::Fft_backend<1, I, O, S> >
  exec(Domain<1> const &dom, float scale)
  {
    return impl::cbe::create<impl::fft::Fft_backend<1, I, O, S> >(dom, scale);
  }
};

template <typename I,
	  typename O,
	  int A,
	  int D,
	  return_mechanism_type R,
	  unsigned N>
struct Evaluator<op::fftm<I, O, A, D, R, N>, be::cbe_sdk,
  std::auto_ptr<impl::fft::Fftm_backend<I, O, A, D> > 
  (Domain<2> const &, typename impl::scalar_of<I>::type)>
{
  static bool const ct_valid = 
    is_same<I, complex<float> >::value &&
    is_same<I, O>::value;
  static bool rt_valid(Domain<2> const &dom, float)
  { 
    length_type size = A == vsip::row ? dom[1].size() : dom[0].size();
    return
      (size >= MIN_FFT_1D_SIZE) &&
      (size <= MAX_FFT_1D_SIZE) &&
      (impl::fft::is_power_of_two(size)) &&
      impl::cbe::Task_manager::instance()->num_spes() > 0;
  }
  static std::auto_ptr<impl::fft::Fftm_backend<I, O, A, D> > 
  exec(Domain<2> const &dom, float scale)
  {
    return impl::cbe::create<impl::fft::Fftm_backend<I, O, A, D> >(dom, scale);
  }
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif

