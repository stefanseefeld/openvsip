//
// Copyright (c) 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_cvsip_fir_hpp_
#define ovxx_cvsip_fir_hpp_

#include <vsip/support.hpp>
#include <ovxx/signal/fir.hpp>
#include <ovxx/dispatch.hpp>
#include <ovxx/cvsip/view.hpp>

namespace ovxx
{
namespace cvsip
{

#if HAVE_VSIP_FIR_ATTR // This is a TVCPP bug

typedef vsip_fir_attr vsip_fir_attr_f;
typedef vsip_fir_attr vsip_fir_attr_d;
typedef vsip_cfir_attr vsip_cfir_attr_f;
typedef vsip_cfir_attr vsip_cfir_attr_d;

#endif

template <typename T> struct fir_traits;

template <> struct fir_traits<float>
{
  typedef vsip_vview_f view_type;
  typedef vsip_fir_f fir_type;
 
  static fir_type *create(view_type *k,
                          vsip_symmetry s, vsip_length i, vsip_length d,
                          vsip_obj_state c, unsigned int n, vsip_alg_hint h)
   {
    fir_type *fir = vsip_fir_create_f(k, s, i, d, c, n, h);
    if (!fir) OVXX_DO_THROW((std::bad_alloc()));
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

template <> struct fir_traits<std::complex<float> >
{
  typedef vsip_cvview_f view_type;
  typedef vsip_cfir_f fir_type;
 
  static fir_type *create(view_type *k,
                          vsip_symmetry s, vsip_length i, vsip_length d,
                          vsip_obj_state c, unsigned int n, vsip_alg_hint h)
  {
    fir_type *fir = vsip_cfir_create_f(k, s, i, d, c, n, h);
    if (!fir) OVXX_DO_THROW((std::bad_alloc()));
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

template <> struct fir_traits<double>
{
  typedef vsip_vview_d view_type;
  typedef vsip_fir_d fir_type;
 
  static fir_type *create(view_type *k,
                          vsip_symmetry s, vsip_length i, vsip_length d,
                          vsip_obj_state c, unsigned int n, vsip_alg_hint h)
  {
    fir_type *fir = vsip_fir_create_d(k, s, i, d, c, n, h);
    if (!fir) OVXX_DO_THROW((std::bad_alloc()));
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

template <> struct fir_traits<std::complex<double> >
{
  typedef vsip_cvview_d view_type;
  typedef vsip_cfir_d fir_type;
  
  static fir_type *create(view_type *k,
                          vsip_symmetry s, vsip_length i, vsip_length d,
                          vsip_obj_state c, unsigned int n, vsip_alg_hint h)
  {
    fir_type *fir = vsip_cfir_create_d(k, s, i, d, c, n, h);
    if (!fir) OVXX_DO_THROW((std::bad_alloc()));
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
class Fir : public signal::Fir_backend<T, S, C>
{
  typedef signal::Fir_backend<T, S, C> base;
  typedef Dense<1, T> block_type;
  typedef fir_traits<T> traits;

public:
  Fir(aligned_array<T> kernel, length_type k, length_type i, length_type d,
      unsigned n, alg_hint_type h)
    : base(i, k, d),
      kernel_data_(kernel),
      kernel_(kernel_data_.get(), 0, 1, k),
      n_(n),
      h_(h),
      fir_(traits::create(kernel_.ptr(), symmetry(S), i, d, save(C), n, hint(h)))
  {
    // spec says a nonsym kernel size has to be >1, but symmetric can be ==1:
    OVXX_PRECONDITION(k > (S == nonsym));
  }

  Fir(Fir const &fir)
    : base(fir),
      kernel_data_(OVXX_ALLOC_ALIGNMENT, fir.kernel_.size(),
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
    OVXX_DO_THROW(unimplemented
		  ("Backend does not allow copy-construction."));
  }
  virtual Fir *clone() { return new Fir(*this);}

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
} // namespace ovxx::cvsip

namespace dispatcher
{
template <typename T, symmetry_type S, obj_state C> 
struct Evaluator<op::fir, be::cvsip,
                 std::shared_ptr<signal::Fir_backend<T, S, C> >
                 (aligned_array<T>, 
                  length_type, length_type, length_type,
                  unsigned, alg_hint_type)>
{
  static bool const ct_valid = 
    is_same<T, float>::value ||
    is_same<T, complex<float> >::value ||
    is_same<T, double>::value ||
    is_same<T, complex<double> >::value;
  typedef std::shared_ptr<signal::Fir_backend<T, S, C> > return_type;
  // We pass a reference for the first argument 
  // to not lose ownership of the data.
  static bool rt_valid(aligned_array<T> const &, length_type,
                       length_type, length_type,
                       unsigned, alg_hint_type)
  { return true;}
  static return_type exec(aligned_array<T> k, length_type ks,
                          length_type is, length_type d,
                          unsigned n, alg_hint_type h)
  {
    return return_type(new cvsip::Fir<T, S, C>(k, ks, is, d, n, h));
  }
};
} // namespace ovxx::dispatcher
} // namespace ovxx

#endif
