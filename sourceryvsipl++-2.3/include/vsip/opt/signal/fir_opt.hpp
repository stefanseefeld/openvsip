/* Copyright (c) 2006, 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/signal/fir_opt.hpp
    @author  Stefan Seefeld
    @date    2006-11-02
    @brief   VSIPL++ Library: FIR implementation.
*/

#ifndef VSIP_OPT_SIGNAL_FIR_OPT_HPP
#define VSIP_OPT_SIGNAL_FIR_OPT_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/core/signal/fir_backend.hpp>
#include <vsip/core/signal/conv.hpp>
#include <vsip/opt/dispatch.hpp>
#include <vsip/vector.hpp>
#include <vsip/domain.hpp>
#include <vsip/math.hpp>
#include <vsip/core/view_traits.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{
namespace impl
{

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
      state_(this->kernel_size(), T(0)),
      state_saved_(0)
  {
    // spec says a nonsym kernel size has to be >1, but symmetric can be ==1:
    assert(k > (S == nonsym));
    // mirror the kernel
    Dense<1, T> kernel_block(k, kernel.get());
    kernel_block.admit();
    Vector<T> tmp(kernel_block);
    this->kernel_(Domain<1>(this->order_, -1, k)) = tmp;
    // and maybe unmirror a copy, too
    if (S != nonsym) this->kernel_(Domain<1>(k)) = tmp;
    kernel_block.release(false);
  }

  Fir_impl(Fir_impl const &fir)
    : base(fir),
      skip_(fir.skip_),
      kernel_(fir.kernel_),
      state_(fir.state_.get(Domain<1>(fir.state_.size()))), // deep copy
      state_saved_(fir.state_saved_)
  {}
  virtual Fir_impl *clone() { return new Fir_impl(*this);}

  length_type apply(T *in, stride_type in_stride, length_type in_length,
                    T *out, stride_type out_stride, length_type out_length)
  {
    typedef impl::Subset_block<Dense<1, T> > block_type;
    typedef Vector<T, block_type> view_type;

    length_type in_extent = abs(in_stride) * (in_length - 1) + 1;
    Dense<1, T> in_block(in_extent, in_stride > 0 ? in : in - in_extent + 1);
    block_type sub_in_block(Domain<1>(in_stride > 0 ? 0 : in_extent - 1,
                                      in_stride, in_length), in_block);
    view_type input(sub_in_block);

    length_type out_extent = abs(out_stride) * (out_length - 1) + 1;
    Dense<1, T> out_block(out_extent, out_stride > 0 ? out : out - out_extent + 1);
    block_type sub_out_block(Domain<1>(out_stride > 0 ? 0 : out_extent - 1,
                                       out_stride, out_length), out_block);
    view_type output(sub_out_block);

    length_type const dec = this->decimation();
    length_type const m = this->order_;
    length_type const skip = this->skip_;
    length_type const saved = this->state_saved_;
    length_type oix = 0;
    length_type i = 0;

    in_block.admit(true);
    out_block.admit(false);

    for (; i < m - skip; ++oix, i += dec)
    {
      // Conceptually this comes second, but it's more convenient
      // to put it here.  So, read the second statement below first.
      T sum = dot(this->kernel_(Domain<1>(m - skip - i, 1, i + skip + 1)),
                  input(Domain<1>(i + skip + 1)));
      
      if (C == state_save && i < saved)
        sum += dot(this->kernel_(Domain<1>(saved - i)),
                   this->state_(Domain<1>(i, 1, saved - i)));
      output.put(oix, sum);
    }
	  
    length_type start = i - (m - skip);
    for ( ; start + m < this->input_size_; ++oix, start += dec)
    {
      T sum = dot(this->kernel_, input(Domain<1>(start, 1, m + 1)));
      output.put(oix, sum);
    }

    if (C == state_save)
    {
      this->skip_ = start + m - this->input_size_;
      // Invariant: (this->input_size % dec == 0) => (this->skip_ == 0)
      assert(this->input_size_ % dec != 0 || this->skip_ == 0);
      length_type const new_save = this->input_size_ - start;
      this->state_saved_ = new_save;
      this->state_(Domain<1>(new_save)) = input(Domain<1>(start, 1, new_save));
    }

    in_block.release(false);
    out_block.release(true);

    return oix;
  }

  virtual void reset() VSIP_NOTHROW
  {
    state_saved_ = skip_ = 0;
    state_ = T(0);
  }

  virtual char const* name() { return "fir-opt"; }

private:
  length_type skip_;          // how much of next input to skip
  Vector<T, block_type> kernel_; 
  Vector<T, block_type> state_;
  length_type state_saved_;   // number of elements saved
};



/// Create FIR kernel from coefficients
///
/// Requires:
///   :p_coeff: to be a pointer to FIR coefficients.
///   :n_coeff: to be the number of FIR coefficients.
///   :n_kernel: to be the kernel size.
///
/// Returns a vector of kernel coefficients, mirrored and with
///   necessary symmetry applied
template <typename      T,
	  typename      BlockT,
	  symmetry_type S>
Vector<T, BlockT>
create_kernel(
  T*          p_coeff,
  length_type n_coeff,
  length_type n_kernel)
{
  Vector<T, BlockT> kernel(n_kernel);

  // mirror the kernel
  Dense<1, T> coeff_block(n_coeff, p_coeff);
  coeff_block.admit();
  Vector<T> coeff(coeff_block);
  kernel(Domain<1>(n_kernel-1, -1, n_coeff)) = coeff;
  // and maybe unmirror a copy, too
  if (S != nonsym) kernel(Domain<1>(n_coeff)) = coeff;
  coeff_block.release(false);

  return kernel;
}



// Fir implementation, based on convolution

template <typename T, symmetry_type S, obj_state C> 
class Fir_conv_impl : public Fir_backend<T, S, C>
{
  typedef Fir_backend<T, S, C> base;
  typedef Dense<1, T> block_type;

  static length_type conv_size(length_type input_size, length_type order, length_type dec)
  {
    return input_size - (order%dec ? (dec-order%dec) : 0);
  }

public:
  Fir_conv_impl(
    aligned_array<T> kernel,
    length_type      k,
    length_type      i,
    length_type      d)
  : base        (i, k, d),
    skip_       (0),
    kernel_     (create_kernel<T, block_type, S>(kernel.get(), k, this->kernel_size())),
    conv_kernel_(kernel_(Domain<1>(this->kernel_size()-1, -1,
				   this->kernel_size()))),
    state_      (this->kernel_size(), T(0)),
    state_saved_(0),
    conv_       (conv_kernel_, conv_size(i, this->order_, d), d)
  {
    assert(i % d == 0);

    // spec says a nonsym kernel size has to be >1, but symmetric can be ==1:
    assert(k > (S == nonsym));
  }

  Fir_conv_impl(Fir_conv_impl const &fir)
    : base(fir),
      skip_(fir.skip_),
      kernel_(fir.kernel_),
      conv_kernel_(fir.conv_kernel_),
      state_(fir.state_.get(Domain<1>(fir.state_.size()))), // deep copy
      state_saved_(fir.state_saved_),
      conv_       (conv_kernel_, this->input_size_-(this->order_%this->decimation_ ? (this->decimation_-this->order_%this->decimation_) : 0), this->decimation_)
  {}
  virtual Fir_conv_impl *clone() { return new Fir_conv_impl(*this);}

  length_type apply(T *in, stride_type in_stride, length_type in_length,
                    T *out, stride_type out_stride, length_type out_length)
  {
    typedef impl::Subset_block<Dense<1, T> > block_type;
    typedef Vector<T, block_type> view_type;

    Dense<1, T> in_block(in_stride * (in_length - 1) + 1, in);
    block_type sub_in_block(Domain<1>(0, in_stride, in_length), in_block);
    view_type input(sub_in_block);

    Dense<1, T> out_block(out_stride * (out_length - 1) + 1, out);
    block_type sub_out_block(Domain<1>(0, out_stride, out_length), out_block);
    view_type output(sub_out_block);

    length_type const dec = this->decimation();
    length_type const m = this->order_;
    length_type const skip = this->skip_;
    length_type const saved = this->state_saved_;
    length_type oix = 0;
    length_type i = 0;

    in_block.admit(true);
    out_block.admit(false);

    for (; i < m - skip; ++oix, i += dec)
    {
      // Conceptually this comes second, but it's more convenient
      // to put it here.  So, read the second statement below first.
      T sum = dot(this->kernel_(Domain<1>(m - skip - i, 1, i + skip + 1)),
                  input(Domain<1>(i + skip + 1)));
      
      if (C == state_save && i < saved)
        sum += dot(this->kernel_(Domain<1>(saved - i)),
                   this->state_(Domain<1>(i, 1, saved - i)));
      output.put(oix, sum);
    }
	  
    length_type start = i - (m - skip);
    assert(start == (((m-skip) % dec) ? (dec-(m-skip)%dec) : 0));
    if (start + m < this->input_size_)
    {
      length_type out_size =   (this->input_size_ - (start + m)) / dec
	                   + (((this->input_size_ - (start + m)) % dec) ? 1:0);
      conv_(input(Domain<1>(start, 1, this->input_size_-start)),
	   output(Domain<1>(oix, 1, out_size)));

      length_type x = this->input_size_-(start+m);
      start += x + (x % dec ? (dec-x%dec) : 0);
      oix += out_size;
    }

    if (C == state_save)
    {
      this->skip_ = start + m - this->input_size_;
      // Invariant: (this->input_size % dec == 0) => (this->skip_ == 0)
      assert(this->input_size_ % dec != 0 || this->skip_ == 0);
      length_type const new_save = this->input_size_ - start;
      this->state_saved_ = new_save;
      this->state_(Domain<1>(new_save)) = input(Domain<1>(start, 1, new_save));
    }

    in_block.release(false);
    out_block.release(true);

    return oix;
  }

  virtual void reset() VSIP_NOTHROW
  {
    state_saved_ = skip_ = 0;
    state_ = T(0);
  }

  virtual char const* name() { return "fir-conv-opt"; }

private:
  length_type skip_;          // how much of next input to skip
  Vector<T, block_type> kernel_; 
  Vector<T, block_type> conv_kernel_; 
  Vector<T, block_type> state_;
  length_type state_saved_;   // number of elements saved
  vsip::Convolution<const_Vector, nonsym, support_min, T> conv_;
};

} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{
template <typename T, symmetry_type S, obj_state C> 
struct Evaluator<op::fir, be::opt,
                 impl::Ref_counted_ptr<impl::Fir_backend<T, S, C> >
                 (impl::aligned_array<T>,
                  length_type, length_type, length_type,
                  unsigned, alg_hint_type)>
{
  static bool const ct_valid = true;
  typedef impl::Ref_counted_ptr<impl::Fir_backend<T, S, C> > return_type;
  // We pass a reference for the first argument 
  // to not lose ownership of the data.
  static bool rt_valid(impl::aligned_array<T> const &,
                       length_type, length_type, length_type,
                       unsigned, alg_hint_type)
  { return true;}
  static return_type exec(impl::aligned_array<T> k, length_type ks,
                          length_type is, length_type d,
                          unsigned, alg_hint_type)
  {
    length_type order = impl::Fir_backend<T, S, C>::order(ks);
    length_type start = order%d ? (d-order%d) : 0;
    
    // Requirements for convolution-based Fir
    //  - input size must be a multiple of the decimation,
    //    otherwise the convolution size changes from frame to frame.
    //  - input size must be greater than the "start" position + order,
    //    otherwise the FIR is too small to perform convolution on a frame.
    return (is%d == 0 && is > start + order)
      ? return_type(new impl::Fir_conv_impl<T, S, C>(k, ks, is, d), impl::noincrement)
      : return_type(new impl::Fir_impl<T, S, C>(k, ks, is, d), impl::noincrement);
  }
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif
