/* Copyright (c) 2006 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/cvsip/fir.hpp
    @author  Stefan Seefeld
    @date    2006-11-10
    @brief   VSIPL++ Library: FIR  C-VSIPL backend.
*/

#ifndef VSIP_CORE_CVSIP_FIR_HPP
#define VSIP_CORE_CVSIP_FIR_HPP

#include <vsip/support.hpp>
#include <vsip/core/signal/fir_backend.hpp>
#include <vsip/core/cvsip/view.hpp>
#ifndef VSIP_IMPL_REF_IMPL
# include <vsip/opt/dispatch.hpp>
#endif
extern "C" {
#include <vsip.h>
}

namespace vsip
{
namespace impl
{
namespace cvsip
{

#if HAVE_VSIP_FIR_ATTR // This is a TVCPP bug

typedef vsip_fir_attr vsip_fir_attr_f;
typedef vsip_fir_attr vsip_fir_attr_d;
typedef vsip_cfir_attr vsip_cfir_attr_f;
typedef vsip_cfir_attr vsip_cfir_attr_d;

#endif

template <typename T> struct Fir_traits;

template <> struct Fir_traits<float>
{
  typedef vsip_vview_f view_type;
  typedef vsip_fir_f fir_type;
 
  static fir_type *create(view_type *k,
                          vsip_symmetry s, vsip_length i, vsip_length d,
                          vsip_obj_state c, unsigned int n, vsip_alg_hint h)
   {
    fir_type *fir = vsip_fir_create_f(k, s, i, d, c, n, h);
    if (!fir) VSIP_IMPL_THROW((std::bad_alloc()));
    return fir;
  }
  static fir_type *clone(fir_type *fir, view_type *new_kernel,
                         unsigned int n, vsip_alg_hint h)
  {
    vsip_fir_attr_f attrs;
    vsip_fir_getattr_f(fir, &attrs);
    return create(new_kernel, attrs.symm, attrs.in_len, attrs.decimation,
                  attrs.state, n, h);
  }
  static void destroy(fir_type *fir)
  {
    int status = vsip_fir_destroy_f(fir);
    assert(status == 0);
  }
  static int call(fir_type *fir, view_type const *in, view_type const *out)
  { return vsip_firflt_f(fir, in, out);}
  static void reset(fir_type *fir) { vsip_fir_reset_f(fir);}
};

template <> struct Fir_traits<std::complex<float> >
{
  typedef vsip_cvview_f view_type;
  typedef vsip_cfir_f fir_type;
 
  static fir_type *create(view_type *k,
                          vsip_symmetry s, vsip_length i, vsip_length d,
                          vsip_obj_state c, unsigned int n, vsip_alg_hint h)
  {
    fir_type *fir = vsip_cfir_create_f(k, s, i, d, c, n, h);
    if (!fir) VSIP_IMPL_THROW((std::bad_alloc()));
    return fir;
  }
  static fir_type *clone(fir_type *fir, view_type *new_kernel,
                         unsigned int n, vsip_alg_hint h)
  {
    vsip_cfir_attr_f attrs;
    vsip_cfir_getattr_f(fir, &attrs);
    return create(new_kernel, attrs.symm, attrs.in_len, attrs.decimation,
                  attrs.state, n, h);
  }
  static void destroy(fir_type *fir)
  {
    int status = vsip_cfir_destroy_f(fir);
    assert(status == 0);
  }
  static int call(fir_type *fir, view_type const *in, view_type const *out)
  { return vsip_cfirflt_f(fir, in, out);}
  static void reset(fir_type *fir) { vsip_cfir_reset_f(fir);}
};

template <> struct Fir_traits<double>
{
  typedef vsip_vview_d view_type;
  typedef vsip_fir_d fir_type;
 
  static fir_type *create(view_type *k,
                          vsip_symmetry s, vsip_length i, vsip_length d,
                          vsip_obj_state c, unsigned int n, vsip_alg_hint h)
  {
    fir_type *fir = vsip_fir_create_d(k, s, i, d, c, n, h);
    if (!fir) VSIP_IMPL_THROW((std::bad_alloc()));
    return fir;
  }
  static fir_type *clone(fir_type *fir, view_type *new_kernel,
                         unsigned int n, vsip_alg_hint h)
  {
    vsip_fir_attr_d attrs;
    vsip_fir_getattr_d(fir, &attrs);
    return create(new_kernel, attrs.symm, attrs.in_len, attrs.decimation,
                  attrs.state, n, h);
  }
  static void destroy(fir_type *fir)
  {
    int status = vsip_fir_destroy_d(fir);
    assert(status == 0);
  }
  static int call(fir_type *fir, view_type const *in, view_type const *out)
  { return vsip_firflt_d(fir, in, out);}
  static void reset(fir_type *fir) { vsip_fir_reset_d(fir);}
};

template <> struct Fir_traits<std::complex<double> >
{
  typedef vsip_cvview_d view_type;
  typedef vsip_cfir_d fir_type;
  
  static fir_type *create(view_type *k,
                          vsip_symmetry s, vsip_length i, vsip_length d,
                          vsip_obj_state c, unsigned int n, vsip_alg_hint h)
  {
    fir_type *fir = vsip_cfir_create_d(k, s, i, d, c, n, h);
    if (!fir) VSIP_IMPL_THROW((std::bad_alloc()));
    return fir;
  }
  static fir_type *clone(fir_type *fir, view_type *new_kernel,
                         unsigned int n, vsip_alg_hint h)
  {
    vsip_cfir_attr_d attrs;
    vsip_cfir_getattr_d(fir, &attrs);
    return create(new_kernel, attrs.symm, attrs.in_len, attrs.decimation,
                  attrs.state, n, h);
  }
  static void destroy(fir_type *fir)
  {
    int status = vsip_cfir_destroy_d(fir);
    assert(status == 0);
  }
  static int call(fir_type *fir, view_type const *in, view_type const *out)
  { return vsip_cfirflt_d(fir, in, out);}
  static void reset(fir_type *fir) { vsip_cfir_reset_d(fir);}
};

template <typename T, symmetry_type S, obj_state C> 
class Fir_impl : public Fir_backend<T, S, C>
{
  typedef Fir_backend<T, S, C> base;
  typedef Dense<1, T> block_type;
  typedef Fir_traits<T> traits;

public:
  Fir_impl(aligned_array<T> kernel, length_type k, length_type i, length_type d,
      unsigned n, alg_hint_type h)
    : base(i, k, d),
      kernel_data_(kernel),
      kernel_(kernel_data_.get(), 0, 1, k),
      n_(n),
      h_(h),
      fir_(traits::create(kernel_.ptr(), symmetry(S), i, d, save(C), n, hint(h)))
  {
    // spec says a nonsym kernel size has to be >1, but symmetric can be ==1:
    assert(k > (S == nonsym));
  }

  Fir_impl(Fir_impl const &fir)
    : base(fir),
      kernel_data_(VSIP_IMPL_ALLOC_ALIGNMENT, fir.kernel_.size(),
                   fir.kernel_data_.get()),
      kernel_(kernel_data_.get(), 0, 1, fir.kernel_.size()),
      n_(fir.n_),
      h_(fir.h_),
      fir_(traits::clone(fir.fir_, kernel_.ptr(), n_, hint(h_)))
  {
    // The C-VSIPL API seems to be missing a way to access
    // (or set / clone) a fir object's current state.
    // Thus, this copy-constructor only creates a true copy if
    // the original's 'reset()' method is called.
    VSIP_IMPL_THROW(vsip::impl::unimplemented
                    ("Backend does not allow copy-construction."));
  }
  virtual Fir_impl *clone() { return new Fir_impl(*this);}

  length_type apply(T const *in, stride_type in_stride, length_type in_length,
                    T *out, stride_type out_stride, length_type out_length)
  {
    length_type in_offset = in_stride > 0 ? 0 : -in_stride * (in_length - 1) + 1;
    const_View<1, T> input(in, in_offset, in_stride, in_length);
    length_type out_offset = out_stride > 0 ? 0 : -out_stride * (out_length - 1) + 1;
    View<1, T> output(out, out_offset, out_stride, out_length);
    return traits::call(fir_, input.ptr(), output.ptr());
  }

  virtual void reset() VSIP_NOTHROW { traits::reset(fir_);}
private:
  aligned_array<T> kernel_data_;
  View<1, T> kernel_;
  unsigned n_;
  alg_hint_type h_;
  typename traits::fir_type *fir_;
};
} // namespace vsip::impl::cvsip
} // namespace vsip::impl
} // namespace vsip

#ifndef VSIP_IMPL_REF_IMPL

namespace vsip_csl
{
namespace dispatcher
{
template <typename T, symmetry_type S, obj_state C> 
struct Evaluator<op::fir, be::cvsip,
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
  // We pass a reference for the first argument 
  // to not lose ownership of the data.
  static bool rt_valid(impl::aligned_array<T> const &, length_type,
                       length_type, length_type,
                       unsigned, alg_hint_type)
  { return true;}
  static return_type exec(impl::aligned_array<T> k, length_type ks,
                          length_type is, length_type d,
                          unsigned n, alg_hint_type h)
  {
    return return_type(new impl::cvsip::Fir_impl<T, S, C>(k, ks, is, d, n, h), 
                       impl::noincrement);
  }
};
} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif // VSIP_IMPL_REF_IMPL

#endif
