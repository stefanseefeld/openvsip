//
// Copyright (c) 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_CORE_CVSIP_CORR_HPP
#define VSIP_CORE_CVSIP_CORR_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/config.hpp>
#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/core/domain_utils.hpp>
#include <vsip/core/signal/corr_common.hpp>
#include <vsip/core/cvsip/block.hpp>
#include <vsip/core/cvsip/view.hpp>
#include <vsip/core/cvsip/common.hpp>
extern "C" 
{
#include <vsip.h>
}

/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{
namespace impl
{
namespace cvsip
{

template <dimension_type D, typename T> struct Corr_traits;

#if HAVE_VSIP_CORR1D_CREATE_F == 1
template <>
struct Corr_traits<1, float>
{
  typedef vsip_corr1d_f corr_type;
  typedef vsip_vview_f view_type;

  static corr_type *create(length_type r, length_type i,
                           support_region_type s, unsigned n, alg_hint_type a)
  {
    corr_type *c = vsip_corr1d_create_f(r, i, support(s), n, hint(a));
    if (!c) VSIP_IMPL_THROW(std::bad_alloc());
    return c;
  }
  static void destroy(corr_type *c) 
  {
    int status = vsip_corr1d_destroy_f(c);
    assert(status == 0);
  }
  static void call(corr_type *c, vsip_bias b,
                   view_type *ref, view_type *in, view_type *out)
  { vsip_correlate1d_f(c, b, ref, in, out);}
};

template <>
struct Corr_traits<1, std::complex<float> >
{
  typedef vsip_ccorr1d_f corr_type;
  typedef vsip_cvview_f view_type;

  static corr_type *create(length_type r, length_type i,
                           support_region_type s, unsigned n, alg_hint_type a)
  {
    corr_type *c = vsip_ccorr1d_create_f(r, i, support(s), n, hint(a));
    if (!c) VSIP_IMPL_THROW(std::bad_alloc());
    return c;
  }
  static void destroy(corr_type *c) 
  {
    int status = vsip_ccorr1d_destroy_f(c);
    assert(status == 0);
  }
  static void call(corr_type *c, vsip_bias b,
                   view_type *ref, view_type *in, view_type *out)
  { vsip_ccorrelate1d_f(c, b, ref, in, out);
  }
};
#endif
#if HAVE_VSIP_CORR2D_CREATE_F == 1
template <>
struct Corr_traits<2, float>
{
  typedef vsip_corr2d_f corr_type;
  typedef vsip_mview_f view_type;

  static corr_type *create(length_type a, length_type b,
                           length_type p, length_type q,
                           support_region_type r, unsigned n, alg_hint_type a)
  {
    corr_type *c = vsip_corr2d_create_f(a, b, p, q, support(r), n, hint(a));
    if (!c) VSIP_IMPL_THROW(std::bad_alloc());
    return c;
  }
  static void destroy(corr_type *c) 
  {
    int status = vsip_corr2d_destroy_f(c);
    assert(status == 0);
  }
  static void call(corr_type *c, vsip_bias b,
                   view_type *ref, view_type *in, view_type *out)
  { vsip_correlate2d_f(c, b, ref, in, out);}
};

template <>
struct Corr_traits<2, std::complex<float> >
{
  typedef vsip_ccorr2d_f corr_type;
  typedef vsip_cmview_f view_type;

  static corr_type *create(length_type a, length_type b,
                           length_type p, length_type q,
                           support_region_type r, unsigned n, alg_hint_type a)
  {
    corr_type *c = vsip_corr2d_create_f(a, b, p, q, support(r), n, hint(a));
    if (!c) VSIP_IMPL_THROW(std::bad_alloc());
    return c;
  }
  static void destroy(corr_type *c) 
  {
    int status = vsip_corr2d_destroy_f(c);
    assert(status == 0);
  }
  static void call(corr_type *c, vsip_bias b,
                   view_type *ref, view_type *in, view_type *out)
  { vsip_ccorrelate2d_f(c, b, ref, in, out);}
};
#endif
#if HAVE_VSIP_CORR1D_CREATE_D == 1
template <>
struct Corr_traits<1, double>
{
  typedef vsip_corr1d_d corr_type;
  typedef vsip_vview_d view_type;

  static corr_type *create(length_type r, length_type i,
                           support_region_type s, unsigned n, alg_hint_type a)
  {
    corr_type *c = vsip_corr1d_create_d(r, i, support(s), n, hint(a));
    if (!c) VSIP_IMPL_THROW(std::bad_alloc());
    return c;
  }
  static void destroy(corr_type *c) 
  {
    int status = vsip_corr1d_destroy_d(c);
    assert(status == 0);
  }
  static void call(corr_type *c, vsip_bias b,
                   view_type *ref, view_type *in, view_type *out)
  { vsip_correlate1d_d(c, b, ref, in, out);}
};

template <>
struct Corr_traits<1, std::complex<double> >
{
  typedef vsip_ccorr1d_d corr_type;
  typedef vsip_cvview_d view_type;

  static corr_type *create(length_type r, length_type i,
                           support_region_type s, unsigned n, alg_hint_type a)
  {
    corr_type *c = vsip_ccorr1d_create_d(r, i, support(s), n, hint(a));
    if (!c) VSIP_IMPL_THROW(std::bad_alloc());
    return c;
  }
  static void destroy(corr_type *c) 
  {
    int status = vsip_ccorr1d_destroy_d(c);
    assert(status == 0);
  }
  static void call(corr_type *c, vsip_bias b,
                   view_type *ref, view_type *in, view_type *out)
  { vsip_ccorrelate1d_d(c, b, ref, in, out);}
};
#endif
#if HAVE_VSIP_CORR2D_CREATE_D == 1
template <>
struct Corr_traits<2, double>
{
  typedef vsip_corr2d_d corr_type;
  typedef vsip_mview_d view_type;

  static corr_type *create(length_type a, length_type b,
                           length_type p, length_type q,
                           support_region_type r, unsigned n, alg_hint_type a)
  {
    corr_type *c = vsip_corr2d_create_f(a, b, p, q, support(r), n, hint(a));
    if (!c) VSIP_IMPL_THROW(std::bad_alloc());
    return c;
  }
  static void destroy(corr_type *c) 
  {
    int status = vsip_corr2d_destroy_d(c);
    assert(status == 0);
  }
  static void call(corr_type *c, vsip_bias b,
                   view_type *ref, view_type *in, view_type *out)
  { vsip_correlate2d_d(c, b, ref, in, out);}
};

template <>
struct Corr_traits<2, std::complex<double> >
{
  typedef vsip_ccorr2d_d corr_type;
  typedef vsip_cmview_d view_type;

  static corr_type *create(length_type a, length_type b,
                           length_type p, length_type q,
                           support_region_type r, unsigned n, alg_hint_type a)
  {
    corr_type *c = vsip_corr2d_create_f(a, b, p, q, support(r), n, hint(a));
    if (!c) VSIP_IMPL_THROW(std::bad_alloc());
    return c;
  }
  static void destroy(corr_type *c) 
  {
    int status = vsip_corr2d_destroy_d(c);
    assert(status == 0);
  }
  static void call(corr_type *c, vsip_bias b,
                   view_type *ref, view_type *in, view_type *out)
  { vsip_ccorrelate2d_d(c, b, ref, in, out);}
};
#endif

template <dimension_type D,
          typename T,
          support_region_type S>
class Correlation_base
{
public:
  static dimension_type const dim = D;
  static support_region_type const supprt  = S;

  Correlation_base(Domain<D> const &ref_size, Domain<D> const &input_size)
  : ref_size_(normalize(ref_size)),
    input_size_(normalize(input_size)),
    output_size_(conv_output_size(S, ref_size_, input_size_, 1))
  {}

  Domain<D> const &reference_size() const VSIP_NOTHROW { return ref_size_;}
  Domain<D> const &input_size() const VSIP_NOTHROW { return input_size_;}
  Domain<D> const &output_size() const VSIP_NOTHROW { return output_size_;}

protected:
  Domain<D> ref_size_;
  Domain<D> input_size_;
  Domain<D> output_size_;
};

template <dimension_type      D,
	  support_region_type R,
	  typename            T,
	  unsigned            N,
          alg_hint_type       H>
class Correlation;

template <support_region_type R,
	  typename            T,
	  unsigned            N,
          alg_hint_type       H>
class Correlation<1, R, T, N, H> : public Correlation_base<1, T, R>
{
  typedef impl::cvsip::Corr_traits<1, T> traits;

public:
  Correlation(Domain<1> const& ref_size, Domain<1> const& input_size)
    : Correlation_base<1, T, R>(ref_size, input_size),
      impl_(traits::create(ref_size.size(), input_size.size(), R, N, H))
  {}
  ~Correlation() VSIP_NOTHROW {traits::destroy(impl_);}

  template <typename Block0, typename Block1, typename Block2>
  void impl_correlate(bias_type b,
                      const_Vector<T, Block0> ref,
                      const_Vector<T, Block1> in,
                      Vector<T, Block2> out) VSIP_NOTHROW
  {
    dda::Data<Block0, dda::in> data_ref(ref.block());
    dda::Data<Block1, dda::in> data_in(in.block());
    dda::Data<Block2, dda::out> data_out(out.block());
    const_View<1, T> rview(data_ref.ptr(), 0,
			   data_ref.stride(0), data_ref.size(0));
    const_View<1, T> iview(data_in.ptr(), 0,
			   data_in.stride(0), data_in.size(0));
    View<1, T> oview(data_out.ptr(), 0,
		     data_out.stride(0), data_out.size(0));
    traits::call(impl_, impl::cvsip::bias(b),
		 rview.ptr(), iview.ptr(), oview.ptr());
  }

private:
  typename traits::corr_type *impl_;
};

template <support_region_type R,
	  typename            T,
	  unsigned            N,
          alg_hint_type       H>
class Correlation<2, R, T, N, H> : public Correlation_base<2, T, R>
{
  typedef impl::cvsip::Corr_traits<2, T> traits;
public:

  Correlation(Domain<2> const& ref_size, Domain<2> const& input_size)
    : Correlation_base<2, T, R>(ref_size, input_size),
      impl_(traits::create(ref_size[0].size(), ref_size[1].size(),
                           input_size[0].size(), input_size[1].size(),
                           R, N, H))
  {}
  ~Correlation() VSIP_NOTHROW {traits::destroy(impl_);}

  template <typename Block0, typename Block1, typename Block2>
  void impl_correlate(bias_type b,
                      const_Matrix<T, Block0> ref,
                      const_Matrix<T, Block1> in,
                      Matrix<T, Block2> out) VSIP_NOTHROW
  {
    dda::Data<Block0, dda::in> ref_data(ref.block());
    dda::Data<Block1, dda::in> in_data(in.block());
    dda::Data<Block2, dda::out> out_data(out.block());
    const_View<1, T> rview(ref_data.ptr(), 0,
			   ref_data.stride(0), ref_data.size(0),
			   ref_data.stride(1), ref_data.size(1));
    const_View<1, T> iview(in_data.ptr(), 0,
			   in_data.stride(0), in_data.size(0),
			   in_data.stride(1), in_data.size(1));
    View<1, T> oview(out_data.ptr(), 0,
		     out_data.stride(0), out_data.size(0),
		     out_data.stride(1), out_data.size(1));
    traits::call(impl_, impl::cvsip::bias(b),
		 rview.ptr(), iview.ptr(), oview.ptr());
  }

private:
  typename traits::corr_type *impl_;
};

} // namespace vsip::impl::cvsip
} // namespace vsip::impl
} // namespace vsip

#if !VSIP_IMPL_REF_IMPL

namespace vsip_csl
{
namespace dispatcher
{
# if HAVE_VSIP_CORR1D_CREATE_F == 1
template <support_region_type R,
	  unsigned            N,
          alg_hint_type       H>
struct Evaluator<op::corr<1, R, float, N, H>, be::cvsip>
{
  static bool const ct_valid = true;
  typedef vsip::impl::cvsip::Correlation<1, R, float, N, H> backend_type;
};
template <support_region_type R,
	  unsigned            N,
          alg_hint_type       H>
struct Evaluator<op::corr<1, R, std::complex<float>, N, H>, be::cvsip>
{
  static bool const ct_valid = true;
  typedef vsip::impl::cvsip::Correlation<1, R, std::complex<float>, N, H> backend_type;
};
# endif
# if HAVE_VSIP_CORR2D_CREATE_F == 1
template <support_region_type R,
	  unsigned            N,
          alg_hint_type       H>
struct Evaluator<op::corr<2, R, float, N, H>, be::cvsip>
{
  static bool const ct_valid = true;
  typedef vsip::impl::cvsip::Correlation<2, R, float, N, H> backend_type;
};
template <support_region_type R,
	  unsigned            N,
          alg_hint_type       H>
struct Evaluator<op::corr<2, R, std::complex<float>, N, H>, be::cvsip>
{
  static bool const ct_valid = true;
  typedef vsip::impl::cvsip::Correlation<2, R, std::complex<float>, N, H> backend_type;
};
# endif
# if HAVE_VSIP_CORR1D_CREATE_D == 1
template <support_region_type R,
	  unsigned            N,
          alg_hint_type       H>
struct Evaluator<op::corr<1, R, double, N, H>, be::cvsip>
{
  static bool const ct_valid = true;
  typedef vsip::impl::cvsip::Correlation<1, R, double, N, H> backend_type;
};
template <support_region_type R,
	  unsigned            N,
          alg_hint_type       H>
struct Evaluator<op::corr<1, R, std::complex<double>, N, H>, be::cvsip>
{
  static bool const ct_valid = true;
  typedef vsip::impl::cvsip::Correlation<1, R, std::complex<double>, N, H> backend_type;
};
# endif
# if HAVE_VSIP_CORR2D_CREATE_D == 1
template <support_region_type R,
	  unsigned            N,
          alg_hint_type       H>
struct Evaluator<op::corr<2, R, double, N, H>, be::cvsip>
{
  static bool const ct_valid = true;
  typedef vsip::impl::cvsip::Correlation<2, R, double, N, H> backend_type;
};
template <support_region_type R,
	  unsigned            N,
          alg_hint_type       H>
struct Evaluator<op::corr<2, R, std::complex<double>, N, H>, be::cvsip>
{
  static bool const ct_valid = true;
  typedef vsip::impl::cvsip::Correlation<2, R, std::complex<double>, N, H> backend_type;
};
# endif

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif // !VSIP_IMPL_REF_IMPL

#endif // VSIP_CORE_CVSIP_CORR_HPP
