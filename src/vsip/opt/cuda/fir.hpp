/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#ifndef VSIP_OPT_CUDA_FIR_HPP
#define VSIP_OPT_CUDA_FIR_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/core/signal/fir_backend.hpp>
#include <vsip/opt/dispatch.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{
namespace impl
{
namespace cuda
{

extern void
fir_no_decimation(
  float const*     input,
  float*           output,
  size_t           out_length,
  float const*     kernel,
  size_t           kernel_length,
  float*           saved_state);

extern void
fir_no_decimation(
  std::complex<float> const*     input,
  std::complex<float>*           output,
  size_t                         out_len,
  std::complex<float> const*     kernel,
  size_t                         kernel_length,
  std::complex<float>*           saved_state);

extern void
assign_scalar(float value, float *out, length_type length);

extern void
assign_scalar(std::complex<float> const &value, std::complex<float> *out, length_type length);

extern void
copy(float const *in, float *out, length_type length);

extern void
copy(std::complex<float> const *in, std::complex<float> *out, length_type length);

template <typename T, symmetry_type S, obj_state C> 
class Fir_impl : public Fir_backend<T, S, C>
{
  typedef Fir_backend<T, S, C> base;
  typedef Dense<1, T> block_type;
  typedef typename get_block_layout<block_type>::type layout_type;

public:
  Fir_impl(aligned_array<T> kernel, length_type k, length_type i, length_type d)
    : base(i, k, d),
      dev_kernel_(impl::Applied_layout<layout_type>(k)),
      state_saved_(k - 1),
      dev_state_(impl::Applied_layout<layout_type>(state_saved_))
  {
    dev_kernel_.from_host(kernel.get());// Copy the kernel data to the device
    assign_scalar(T(), this->dev_state_.ptr(), this->state_saved_);// Zero the initial state
  }

  Fir_impl(Fir_impl const &fir)
    : base(fir),
      dev_kernel_(impl::Applied_layout<layout_type>(fir.input_size())),
      state_saved_(fir.state_saved_),
      dev_state_(impl::Applied_layout<layout_type>(state_saved_))
  {
    copy(fir.dev_kernel_.ptr(), this->dev_kernel_.ptr(), this->kernel_size());
    copy(fir.dev_state_.ptr(), this->dev_state_.ptr(), this->state_saved_);
  }
  virtual Fir_impl *clone() { return new Fir_impl(*this);}

  length_type apply(T const *in, stride_type in_stride, length_type in_length,
                    T *out, stride_type out_stride, length_type out_length)
  {
    T const *kernel = dev_kernel_.ptr();
    T *state = dev_state_.ptr();

    assert(in_stride == 1);
    assert(out_stride == 1);

    fir_no_decimation(in, out, this->input_size(), kernel, this->filter_order(), state);

    if (C != state_save) this->reset();

    return this->input_size();
  }

  virtual void reset() VSIP_NOTHROW
  {
    assign_scalar(T(), this->dev_state_.ptr(), state_saved_);// Zero the state vector
  }

  virtual char const* name() { return "fir-cuda"; }
  virtual bool supports_cuda_memory() { return true;}

private:         
  Device_storage<T, layout_type>      dev_kernel_;// Device storage for the coefficients
  length_type                         state_saved_;// Length of the saved state vector   
  Device_storage<T, layout_type>      dev_state_; // Device storage for the internal state vector
};
} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{
template <typename T, symmetry_type S, obj_state C> 
struct Evaluator<op::fir, be::cuda,
                 impl::Ref_counted_ptr<impl::Fir_backend<T, S, C> >
                 (impl::aligned_array<T>, 
                  length_type, length_type, length_type,
                  unsigned, alg_hint_type)>
{
  static bool const ct_valid = ((is_same<T, float>::value ||
                                 is_same<T, std::complex<float> >::value) && S == nonsym);

  typedef impl::Ref_counted_ptr<impl::Fir_backend<T, S, C> > return_type;

  static bool rt_valid(impl::aligned_array<T> const &, length_type k,
                       length_type i, length_type d,
                       unsigned, alg_hint_type)
  {
    length_type output_size = (i + d - 1) / d;
    // Verify that the decimation == 1 and that the output size > the kernel size
    return (d == 1 && output_size > k);
  }
  static return_type exec(impl::aligned_array<T> k, length_type ks,
                          length_type is, length_type d,
                          unsigned int, alg_hint_type)
  { return return_type(new impl::cuda::Fir_impl<T, S, C>(k, ks, is, d), impl::noincrement);}
};
} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif
