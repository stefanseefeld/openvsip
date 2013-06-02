//
// Copyright (c) 2006 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_signal_fir_hpp_
#define ovxx_signal_fir_hpp_

#include <vsip/support.hpp>
#include <vsip/impl/signal/types.hpp>
#include <vsip/vector.hpp>
#include <vsip/domain.hpp>
#include <ovxx/dispatch.hpp>

namespace ovxx
{
namespace signal
{
template <typename T, symmetry_type S, obj_state C> 
class Fir_backend
{
public:
  static length_type order(length_type k)
  { return k * (1 + (S != nonsym)) - (S == sym_even_len_odd);}

  Fir_backend(length_type i, length_type k, length_type d)
    : input_size_(i), order_(order(k) - 1), decimation_(d)
  {
    OVXX_PRECONDITION(input_size_ > 0);
    OVXX_PRECONDITION(decimation_ > 0);
    OVXX_PRECONDITION(order_ + 1 > decimation_); // M >= decimation
    OVXX_PRECONDITION(input_size_ >= order_);    // input_size >= M 
    // must be after asserts because of division
    output_size_ = (input_size_ + decimation_ - 1) / decimation_;
  }
  Fir_backend(Fir_backend const &fir)
    :  input_size_(fir.input_size_),
       output_size_(fir.output_size_),
       order_(fir.order_),
       decimation_(fir.decimation_)
  {}

  virtual ~Fir_backend() {}
  virtual Fir_backend *clone() = 0;

  length_type kernel_size() const VSIP_NOTHROW { return order_ + 1;}
  length_type filter_order() const VSIP_NOTHROW { return order_ + 1;}
  length_type input_size() const VSIP_NOTHROW { return input_size_;}
  length_type output_size() const VSIP_NOTHROW { return output_size_;}
  vsip::length_type decimation() const VSIP_NOTHROW { return decimation_;}

  virtual length_type apply(T const *in, stride_type in_stride, length_type in_length,
                            T *out, stride_type out_stride, length_type out_length) = 0;
  virtual void reset() VSIP_NOTHROW = 0;
  virtual char const* name() { return "fir-backend-base";}
  virtual bool supports_cuda_memory() { return false;}

protected:
  length_type input_size_;
  length_type output_size_; 
  length_type order_;         // M in the spec
  length_type decimation_;  
};

template <typename T, symmetry_type S, obj_state C> 
class Fir : public Fir_backend<T, S, C>
{
  typedef Fir_backend<T, S, C> base;
  typedef Dense<1, T> block_type;
public:
  Fir(aligned_array<T> kernel, length_type k, length_type i, length_type d)
    : base(i, k, d),
      skip_(0),
      kernel_(this->kernel_size()),
      state_(this->kernel_size(), T(0)),
      state_saved_(0)
  {
    OVXX_PRECONDITION(k > (S == nonsym));
    // unpack the kernel
    Dense<1, T> kernel_block(k, kernel.get());
    kernel_block.admit();
    Vector<T> tmp(kernel_block);
    this->kernel_(Domain<1>(this->order_, -1, k)) = tmp;
    if (S != nonsym) this->kernel_(Domain<1>(k)) = tmp;
    kernel_block.release(false);
  }

  Fir(Fir const &fir)
    : base(fir),
      skip_(fir.skip_),
      kernel_(fir.kernel_),
      state_(fir.state_.get(Domain<1>(fir.state_.size()))),
      state_saved_(fir.state_saved_)
  {}
  virtual Fir *clone() { return new Fir(*this);}

  length_type apply(T const *in, stride_type in_stride, length_type in_length,
                    T *out, stride_type out_stride, length_type out_length)
  {
    typedef expr::Subset<Dense<1, T> > block_type;
    typedef Vector<T, block_type> view_type;

    length_type in_extent = abs(in_stride) * (in_length - 1) + 1;
    T *nc_in = const_cast<T*>(in);
    Dense<1, T> in_block(in_extent, in_stride > 0 ? nc_in : nc_in - in_extent + 1);
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
      OVXX_PRECONDITION(this->input_size_ % dec != 0 || this->skip_ == 0);
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

private:
  length_type skip_;
  Vector<T, block_type> kernel_; 
  Vector<T, block_type> state_;
  length_type state_saved_;
};

} // namespace ovxx::signal

namespace dispatcher
{
template <typename T, symmetry_type S, obj_state C> 
struct Evaluator<op::fir, be::generic,
                 shared_ptr<signal::Fir_backend<T, S, C> >
                 (aligned_array<T>,
                  length_type, length_type, length_type,
                  unsigned, alg_hint_type)>
{
  static bool const ct_valid = true;
  typedef ovxx::shared_ptr<signal::Fir_backend<T, S, C> > return_type;
  static bool rt_valid(aligned_array<T> const &,
                       length_type, length_type, length_type,
                       unsigned, alg_hint_type)
  { return true;}
  static return_type exec(aligned_array<T> k, length_type ks,
                          length_type is, length_type d,
                          unsigned, alg_hint_type)
  {
    length_type order = signal::Fir_backend<T, S, C>::order(ks);
    length_type start = order%d ? (d-order%d) : 0;
    return return_type(new signal::Fir<T, S, C>(k, ks, is, d));
  }
};

} // namespace ovxx::dispatcher
} // namespace ovxx

#endif
