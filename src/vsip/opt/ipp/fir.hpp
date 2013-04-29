/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/ipp/fir.hpp
    @author  Stefan Seefeld
    @date    2006-11-02
    @brief   VSIPL++ Library: FIR IPP backend.
*/

#ifndef VSIP_OPT_IPP_FIR_HPP
#define VSIP_OPT_IPP_FIR_HPP

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
namespace ipp
{

template <typename T>
void 
fir_call(T const *kernel, length_type kernel_size,
         T const *in, T *out, length_type out_size,
         T *state, length_type *state_size,
         length_type decimation);

template <>
void 
fir_call(float const *kernel, length_type kernel_size,
         float const *in, float *out, length_type out_size,
         float *state, length_type *state_size,
         length_type decimation);

template <>
void 
fir_call(std::complex<float> const *kernel, length_type kernel_size,
         std::complex<float> const *in,
         std::complex<float> *out, length_type out_size,
         std::complex<float> *state, length_type *state_size,
         length_type decimation);

template <>
void 
fir_call(double const *kernel, length_type kernel_size,
         double const *in, double *out, length_type out_size,
         double *state, length_type *state_size,
         length_type decimation);

template <>
void 
fir_call(std::complex<double> const *kernel, length_type kernel_size,
         std::complex<double> const *in,
         std::complex<double> *out, length_type out_size,
         std::complex<double> *state, length_type *state_size,
         length_type decimation);



template <typename T>
void
copy(T const*    src,
     vsip::stride_type src_stride,
     T*          dst,
     vsip::stride_type dst_stride,
     vsip::length_type len)
{
  for (vsip::index_type i=0; i<len; ++i)
    dst[i*dst_stride] = src[i*src_stride];
}



template <typename T, symmetry_type S, obj_state C> 
class Fir_impl : public Fir_backend<T, S, C>
{
  typedef Fir_backend<T, S, C> base;
  typedef Dense<1, T> block_type;
public:
  Fir_impl(aligned_array<T> kernel, length_type k, length_type i, length_type d)
    : base(i, k, d),
      skip_(0),
      kernel_(this->kernel_size()),
      state_(2 * this->kernel_size()),
      state_saved_(0),
      temp_in_(this->input_size()),
      temp_out_(this->input_size())
  {
    // spec says a nonsym kernel size has to be >1, but symmetric can be ==1:
    assert(k > (S == nonsym));

    vcopy(kernel.get(), kernel_.get(), k);
    if (S != nonsym)
      copy(kernel.get(), 1, kernel_.get() + this->order_, -1, k);

    vzero(state_.get(), 2*this->kernel_size());
  }

  Fir_impl(Fir_impl const &fir)
    : base(fir),
      skip_(fir.skip_),
      kernel_(this->kernel_size()),
      state_ (2 * this->kernel_size()),
      state_saved_(fir.state_saved_),
      temp_in_(this->input_size()),  // allocate
      temp_out_(this->input_size())  // allocate
  {
    vcopy(fir.kernel_.get(), this->kernel_.get(), this->kernel_size());
    vcopy(fir.state_.get(), this->state_.get(), 2*this->kernel_size());
  }
  virtual Fir_impl *clone() { return new Fir_impl(*this);}

  length_type apply(T const *in, stride_type in_stride, length_type in_length,
                    T *out, stride_type out_stride, length_type out_length)
  {
    length_type const d = this->decimation();
    length_type const m = this->filter_order() - 1;
    length_type o = (this->input_size() - this->skip_ + d - 1) / d;

    if (in_stride == 1 && out_stride == 1)
    {
      ipp::fir_call(this->kernel_.get(), m + 1,
		    in, out, o,
		    this->state_.get(), &this->state_saved_, d);
    }
    else
    {
      if (in_stride != 1)
      {
	copy(in, in_stride, this->temp_in_.get(), 1, in_length);
	in = this->temp_in_.get();
      }
      if (out_stride == 1)
      {
	ipp::fir_call(this->kernel_.get(), m + 1,
		      in, out, o,
		      this->state_.get(), &this->state_saved_, d);
      }
      else
      {
	ipp::fir_call(this->kernel_.get(), m + 1,
		      in, this->temp_out_.get(), o,
		      this->state_.get(), &this->state_saved_, d);
	copy(this->temp_out_.get(), 1, out, out_stride, out_length);
      }
    }

    if (C != state_save) this->reset();

    return o;
  }

  virtual void reset() VSIP_NOTHROW
  {
    state_saved_ = skip_ = 0;
    vzero(state_.get(), 2*this->kernel_size());
  }

  virtual char const* name() { return "fir-ipp"; }

private:
  length_type           skip_;          // how much of next input to skip
  aligned_array<T>      kernel_;
  aligned_array<T>      state_;
  length_type           state_saved_;   // number of elements saved
  aligned_array<T>      temp_in_;
  aligned_array<T>      temp_out_;
};
} // namespace vsip::impl::ipp
} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{
template <typename T, symmetry_type S, obj_state C> 
struct Evaluator<op::fir, be::intel_ipp,
                 impl::Ref_counted_ptr<impl::Fir_backend<T, S, C> >
                 (impl::aligned_array<T>, 
                  length_type, length_type, length_type,
                  unsigned, alg_hint_type)>
{
  static bool const ct_valid = 
    is_same<T, float>::value ||
    is_same<T, std::complex<float> >::value ||
    is_same<T, double>::value ||
    is_same<T, std::complex<double> >::value;
  typedef impl::Ref_counted_ptr<impl::Fir_backend<T, S, C> > return_type;
  // rt_valid takes the first argument by reference to avoid taking
  // ownership.
  static bool rt_valid(impl::aligned_array<T> const &, length_type k,
                       length_type i, length_type d,
                       unsigned, alg_hint_type)
  {
    length_type o = k * (1 + (S != nonsym)) - (S == sym_even_len_odd) - 1;
    assert(i > 0); // input size
    assert(d > 0); // decimation
    assert(o + 1 > d); // M >= decimation
    assert(i >= o);    // input_size >= M 

    length_type output_size = (i + d - 1) / d;
    return i == output_size * d;
  }
  static return_type exec(impl::aligned_array<T> k, length_type ks,
                          length_type is, length_type d,
                          unsigned int, alg_hint_type)
  { return return_type(new impl::ipp::Fir_impl<T, S, C>(k, ks, is, d), impl::noincrement);}
};
} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif
