/* Copyright (c) 2006 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/signal/fir_backend.hpp
    @author  Stefan Seefeld
    @date    2006-11-02
    @brief   VSIPL++ Library: FIR backend interface.
*/

#ifndef VSIP_CORE_SIGNAL_FIR_BACKEND_HPP
#define VSIP_CORE_SIGNAL_FIR_BACKEND_HPP

#include <vsip/support.hpp>
#include <vsip/core/signal/types.hpp>
#include <vsip/vector.hpp>
#include <vsip/domain.hpp>
#include <vsip/core/refcount.hpp>
#include <vsip/core/dispatch_tags.hpp>

namespace vsip
{
namespace impl
{

// base class to be used by all Fir_impl<> specializations.
template <typename T, symmetry_type S, obj_state C> 
class Fir_backend : public Ref_count<Fir_backend<T, S, C> >
{
public:
  static length_type order(length_type k)
  { return k * (1 + (S != nonsym)) - (S == sym_even_len_odd);}

  Fir_backend(length_type i, length_type k, length_type d)
    : input_size_(i), order_(order(k) - 1), decimation_(d)
  {
    assert(input_size_ > 0);
    assert(decimation_ > 0);
    assert(order_ + 1 > decimation_); // M >= decimation
    assert(input_size_ >= order_);    // input_size >= M 
    // must be after asserts because of division
    output_size_ = (input_size_ + decimation_ - 1) / decimation_;
  }
  Fir_backend(Fir_backend const &fir)
    :  Ref_count<Fir_backend>(),  // copy is unique, count starts at 1.
       input_size_(fir.input_size_),
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
  virtual char const* name() { return "fir-backend-base"; }
  virtual bool supports_cuda_memory() { return false;}

protected:
  length_type input_size_;
  length_type output_size_; 
  length_type order_;         // M in the spec
  length_type decimation_;  
};

} // namespace vsip::impl
} // namespace vsip


#endif
