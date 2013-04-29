/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cbe/cml/fir.hpp
    @author  Don McCoy
    @date    2008-06-05
    @brief   VSIPL++ Library: FIR CML backend.
*/

#ifndef VSIP_OPT_CBE_CML_FIR_HPP
#define VSIP_OPT_CBE_CML_FIR_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/dda.hpp>
#include <vsip/core/signal/fir_backend.hpp>
#include <vsip/opt/dispatch.hpp>

#include <cml.h>
#include <cml_core.h>


/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{
namespace impl
{
namespace cml
{


// CML wrappers for T == float

inline int
fir_create(
  cml_fir_f**         fir_obj_handle,
  float*              K,
  ptrdiff_t           K_stride,
  size_t              d,
  cml_filter_state    state,
  size_t              nk,
  size_t              n)
{
  return
    cml_fir_create_f(
      fir_obj_handle,
      K, K_stride,
      d, state,
      nk, n);
}

inline void
fir_copy_state(
  cml_fir_f const* src_fir_obj_handle,
  cml_fir_f*       dst_fir_obj_handle)
{
  cml_impl_fir_copy_state_f(src_fir_obj_handle, dst_fir_obj_handle);
}

inline void
fir_apply(
  cml_fir_f*            fir_obj_ptr,
  float const*          A,
  ptrdiff_t             A_stride,
  float*                Z,
  ptrdiff_t             Z_stride)
{
  cml_fir_apply_f(
    fir_obj_ptr,
    A, A_stride,
    Z, Z_stride);
}

inline void
fir_destroy(cml_fir_f* fir_obj_ptr)
{
  cml_fir_destroy_f(fir_obj_ptr);
}



// Implementation

template <typename T, symmetry_type S, obj_state C> 
class Fir_impl : public Fir_backend<T, S, C>
{
  typedef Fir_backend<T, S, C> base;
  typedef Dense<1, T> block_type;

public:
  Fir_impl(aligned_array<T> kernel, length_type k, length_type i, length_type d)
    : base(i, k, d),
      fir_obj_ptr_(NULL),
      filter_state_(C == state_save ? SAVE_STATE : DONT_SAVE_STATE)
  {
    // spec says a nonsym kernel size has to be >1, but symmetric can be ==1:
    assert(k > (S == nonsym));

    // copy the kernel
    Dense<1, T> kernel_block(k, kernel.get());
    kernel_block.admit();
    Vector<T> tmp(kernel_block);
    Vector<T> coeffs(this->kernel_size());
    coeffs(Domain<1>(k)) = tmp;    

    // and expand the second half if symmetric
    if (S != nonsym) coeffs(Domain<1>(this->order_, -1, k)) = tmp;
    kernel_block.release(false);

    dda::Data<block_type, dda::out> coeffs_data(coeffs.block());

    fir_create(&fir_obj_ptr_,
	       coeffs_data.ptr(),
	       1, // kernel stride
	       this->decimation(),
	       this->filter_state_,
	       this->kernel_size(),
	       this->input_size());
  }


  Fir_impl(Fir_impl const &fir)
    : base(fir),
      fir_obj_ptr_(0),
      filter_state_(fir.filter_state_)
  {
    fir_create(&fir_obj_ptr_,
	       fir.fir_obj_ptr_->K,
	       1, // kernel stride
	       this->decimation(),
	       this->filter_state_,
	       this->kernel_size(),
	       this->input_size());
    fir_copy_state(fir.fir_obj_ptr_, fir_obj_ptr_);
  }

  ~Fir_impl()
  {
    fir_destroy(this->fir_obj_ptr_);
  }

  virtual Fir_impl *clone() { return new Fir_impl(*this); }

  length_type apply(T const *in, stride_type in_stride, length_type in_length,
                    T *out, stride_type out_stride, length_type out_length)
  {
    assert(in_length == this->input_size());
    assert(out_length == this->output_size());

    fir_apply(this->fir_obj_ptr_,
	      in, in_stride,
	      out, out_stride);

    return this->output_size();
  }

  virtual void reset() VSIP_NOTHROW
  {
    cml_fir_reset_f(this->fir_obj_ptr_);
  }

private:
  cml_fir_f*            fir_obj_ptr_;
  cml_filter_state      filter_state_;
};

} // namespace vsip::impl::cml
} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{
template <typename T, symmetry_type S, obj_state C> 
struct Evaluator<op::fir, be::cml,
                 impl::Ref_counted_ptr<impl::Fir_backend<T, S, C> >
                 (impl::aligned_array<T>, 
                  length_type, length_type, length_type,
                  unsigned, alg_hint_type)>
{
  static bool const ct_valid = is_same<T, float>::value;

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

    // CML FIR objects have fixed output size, whereas VSIPL++ FIR objects
    // have fixed input size.  If input size is not a multiple of the
    // decimation, output size will vary from frame to frame.  The 
    // following check ensures that the CML backend is not used in
    // those cases.
    return
      (i % d) == 0
      ;
  }
  static return_type exec(impl::aligned_array<T> k, length_type ks,
                          length_type is, length_type d,
                          unsigned int, alg_hint_type)
  {
    return return_type(new impl::cml::Fir_impl<T, S, C>(k, ks, is, d),
		       impl::noincrement);
  }
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif
