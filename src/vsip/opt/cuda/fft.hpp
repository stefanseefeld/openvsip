/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cuda/fft.hpp
    @author  Don McCoy
    @date    2009-02-26
    @brief   VSIPL++ Library: FFT wrappers and traits to bridge with 
             NVidia's CUDA FFT library.
*/

#ifndef VSIP_IMPL_CUDA_FFT_HPP
#define VSIP_IMPL_CUDA_FFT_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/core/fft/util.hpp>
#include <vsip/opt/dispatch.hpp>
#include <memory>

static size_t const min_limit_elements = 2;
static size_t const max_limit_bytes = 536870912;

namespace vsip
{
namespace impl
{
namespace cuda
{

/// These are the entry points into the CUDA FFT bridge.
template <typename I, dimension_type D, typename S>
std::auto_ptr<I>
create(Domain<D> const &dom, S scale);

} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{

template <typename I, typename O, int S, return_mechanism_type R, unsigned N>
struct Evaluator<op::fft<1, I, O, S, R, N>, be::cuda,
  std::auto_ptr<impl::fft::Fft_backend<1, I, O, S> >
  (Domain<1> const &, float)>
{
  static bool const ct_valid = is_same<typename impl::scalar_of<I>::type, float>::value;
  static bool rt_valid(Domain<1> const &dom, float)
  {
    return (dom.size() >= min_limit_elements);
  }

  static std::auto_ptr<impl::fft::Fft_backend<1, I, O, S> >
  exec(Domain<1> const &dom, float scale)
  {
    return impl::cuda::create<impl::fft::Fft_backend<1, I, O, S> >(dom, scale);
  }
};

template <typename I, typename O,
	  int A, int D, return_mechanism_type R, unsigned N>
struct Evaluator<op::fftm<I, O, A, D, R, N>, be::cuda,
  std::auto_ptr<impl::fft::Fftm_backend<I, O, A, D> >
  (Domain<2> const &, float)>
{
  typedef typename impl::scalar_of<I>::type input_scalar_type;

  static bool const ct_valid = is_same<input_scalar_type, float>::value;
  static bool rt_valid(Domain<2> const &dom, float)
  {
    bool is_size_good_multiple;
    size_t batch_size, total_elements_per_batch;

    // As of cuFFT 3.1 the input data for any given FFT call must be aligned
    //  to a 256 byte boundary.  If the total size of the FFT is greater than
    //  the prescribed threshold it will need to be computed in multiple
    //  batches.  In this implementation this is accomplished via manually
    //  offsetting the pointer by an integer number of rows (columns) so
    //  verifying that the individual batch size in bytes is a multiple of
    //  256 will guarantee that the data offset will still be valid.
    if (!impl::is_complex<O>::value) // C2R
    {
      batch_size = max_limit_bytes / ((dom[1 - A].size() / 2 + 1) * sizeof(std::complex<input_scalar_type>));
      total_elements_per_batch = batch_size * (dom[1 - A].size() / 2 + 1);
      is_size_good_multiple = ((total_elements_per_batch % 32) == 0);
    }
    else if (!impl::is_complex<I>::value) // R2C
    {
      batch_size = max_limit_bytes / (dom[1 - A].size() * sizeof(input_scalar_type));
      total_elements_per_batch = batch_size * dom[1 - A].size();
      is_size_good_multiple = ((total_elements_per_batch % 64) == 0);
    }
    else // C2C
    {
      batch_size = max_limit_bytes / (dom[1 - A].size() * sizeof(std::complex<input_scalar_type>));
      total_elements_per_batch = batch_size * dom[1 - A].size();
      is_size_good_multiple = ((total_elements_per_batch % 32) == 0);
    }

    // The size of an individual FFT (row or column) must be 
    // greater than the minumum, however the total size  (rows * columns)
    // must still be under the maximum limit or the batch size must be an
    // appropriate multiple.
    return (dom[1 - A].size() >= min_limit_elements &&
               (dom.size() * sizeof(I) < max_limit_bytes || is_size_good_multiple));
  }

  static std::auto_ptr<impl::fft::Fftm_backend<I, O, A, D> > 
  exec(Domain<2> const &dom, float scale)
  {
    return impl::cuda::create<impl::fft::Fftm_backend<I, O, A, D> >(dom, scale);
  }
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif

