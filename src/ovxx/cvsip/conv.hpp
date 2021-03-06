//
// Copyright (c) 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_cvsip_conv_hpp_
#define ovxx_cvsip_conv_hpp_

#include <ovxx/config.hpp>
#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <ovxx/domain_utils.hpp>
#include <ovxx/signal/conv.hpp>
#include <ovxx/cvsip/block.hpp>
#include <ovxx/cvsip/view.hpp>
#include <ovxx/cvsip/common.hpp>

// Define this to 1 to fix the convolution results returned from
// C-VSIP.  This works around incorrect results produced by the
// C-VSIP ref-impl for a subset of the symmetry/support-region
// combinations.  This may not be necessary for other C-VSIP
// implementations.
//
// Defining this to 1 will produce results consistent with the
// other Convolution implementations (Ext, IPP, and SAL).

#define OVXX_FIX_CVSIP_CONV 1

namespace ovxx
{
namespace cvsip
{

template <dimension_type D, typename T> struct conv_traits
{ static bool const valid = false; };

#if HAVE_VSIP_CONV1D_CREATE_F == 1
template <>
struct conv_traits<1, float>
{
  static bool const valid = true;

  typedef vsip_conv1d_f conv_type;
  typedef vsip_vview_f view_type;

  static conv_type *create(view_type *h, symmetry_type s, length_type l,
                           length_type d, support_region_type r, unsigned n,
                           alg_hint_type a)
  {
    conv_type *c = vsip_conv1d_create_f(h, symmetry(s), l, d,
                                        support(r), n, hint(a));
    if (!c) OVXX_DO_THROW(std::bad_alloc());
    return c;
  }
  static void destroy(conv_type *c) 
  {
    int status = vsip_conv1d_destroy_f(c);
    assert(status == 0);
  }
  static void call(conv_type *c, view_type *in, view_type *out)
  { vsip_convolve1d_f(c, in, out);}
};
#endif
#if HAVE_VSIP_CONV2D_CREATE_F == 1
template <>
struct conv_traits<2, float>
{
  static bool const valid = true;

  typedef vsip_conv2d_f conv_type;
  typedef vsip_mview_f view_type;

  static conv_type *create(view_type *h, symmetry_type s,
                           length_type p, length_type q,
                           length_type d, support_region_type r, unsigned n,
                           alg_hint_type a)
  {
    conv_type *c = vsip_conv2d_create_f(h, symmetry(s), p, q, d, support(r), n, a);
    if (!c) OVXX_DO_THROW(std::bad_alloc());
    return c;
  }
  static void destroy(conv_type *c) 
  {
    int status = vsip_conv2d_destroy_f(c);
    assert(status == 0);
  }
  static void call(conv_type *c, view_type *in, view_type *out)
  { vsip_convolve2d_f(c, in, out);}
};
#endif
#if HAVE_VSIP_CONV1D_CREATE_D == 1
template <>
struct conv_traits<1, double>
{
  static bool const valid = true;

  typedef vsip_conv1d_d conv_type;
  typedef vsip_vview_d view_type;

  static conv_type *create(view_type *h, symmetry_type s, length_type l,
                           length_type d, support_region_type r, unsigned n,
                           alg_hint_type a)
  {
    conv_type *c = vsip_conv1d_create_d(h, symmetry(s), l, d,
                                        support(r), n, hint(a));
    if (!c) OVXX_DO_THROW(std::bad_alloc());
    return c;
  }
  static void destroy(conv_type *c) 
  {
    int status = vsip_conv1d_destroy_d(c);
    assert(status == 0);
  }
  static void call(conv_type *c, view_type *in, view_type *out)
  { vsip_convolve1d_d(c, in, out);}
};
#endif
#if HAVE_VSIP_CONV2D_CREATE_D == 1
template <>
struct conv_traits<2, double>
{
  static bool const valid = true;

  typedef vsip_conv2d_d conv_type;
  typedef vsip_mview_d view_type;

  static conv_type *create(view_type *h, symmetry_type s,
                           length_type p, length_type q,
                           length_type d, support_region_type r, unsigned n,
                           alg_hint_type a)
  {
    conv_type *c = vsip_conv2d_create_d(h, symmetry(s), p, q, d, support(r), n, a);
    if (!c) OVXX_DO_THROW(std::bad_alloc());
    return c;
  }
  static void destroy(conv_type *c) 
  {
    int status = vsip_conv2d_destroy_d(c);
    assert(status == 0);
  }
  static void call(conv_type *c, view_type *in, view_type *out)
  { vsip_convolve2d_d(c, in, out);}
};
#endif

template <dimension_type D,
          typename T,
          symmetry_type S,
	  support_region_type R>
class Convolution_base
{
public:
  static dimension_type const dim = D;
  static symmetry_type const symmtry = S;
  static support_region_type const supprt  = R;

  Convolution_base(Domain<D> const &c, Domain<D> const &i, length_type d)
    : kernel_size_(signal::conv_kernel_size<S>(c)),
      input_size_(i),
      output_size_(signal::conv_output_size(R, kernel_size_, input_size_, d)),
      decimation_(d)
  {}

  Domain<D> const& kernel_size() const VSIP_NOTHROW { return kernel_size_;}
  Domain<D> const& filter_order() const VSIP_NOTHROW { return kernel_size_;}
  Domain<D> const& input_size() const VSIP_NOTHROW { return input_size_;}
  Domain<D> const& output_size() const VSIP_NOTHROW { return output_size_;}
  symmetry_type symmetry() const VSIP_NOTHROW { return S;}
  support_region_type support() const VSIP_NOTHROW { return R;}
  length_type decimation() const VSIP_NOTHROW { return decimation_;}

  float impl_performance(char const* what) const { return 0.f;}

protected:
  Domain<D> kernel_size_;
  Domain<D> input_size_;
  Domain<D> output_size_;
  length_type decimation_;
};

// Bogus type name to encapsulate error message.

template <typename T>
struct conv_cvsip_backend_does_not_support_type;

template <dimension_type      D,
          symmetry_type       S,
	  support_region_type R,
	  typename            T,
	  unsigned            N,
          alg_hint_type       H>
class Convolution;

template <symmetry_type       S,
	  support_region_type R,
	  typename            T,
	  unsigned            N,
          alg_hint_type       H>
class Convolution<1, S, R, T, N, H>
  : public Convolution_base<1, T, S, R>,
    ct_assert_msg<conv_traits<1, T>::valid,
		  conv_cvsip_backend_does_not_support_type<T> >
{
  typedef conv_traits<1, T> traits;

public:
  template <typename Block>
  Convolution(const_Vector<T, Block> coeffs, Domain<1> const& input_size,
              length_type decimation)
    : Convolution_base<1, T, S, R>(view_domain(coeffs), input_size, decimation),
      coeffs_(coeffs.size()),
      impl_(0)
  {
    dda::Data<Block, dda::in> c_data(coeffs.block());
    const_View<1, T> tmp(c_data.ptr(), 0, c_data.stride(0), c_data.size(0));
    coeffs_ = tmp;
    impl_ = traits::create(coeffs_.ptr(), S, input_size.size(), decimation, R, N, H);
  }
  ~Convolution() {traits::destroy(impl_);}

protected:
  template <typename Block0, typename Block1>
  void convolve(const_Vector<T, Block0> in, Vector<T, Block1> out)
  {
    {
      dda::Data<Block0, dda::in> in_data(in.block());
      dda::Data<Block1, dda::out> out_data(out.block());
      const_View<1, T> iview(in_data.ptr(), 0,
			     in_data.stride(0), in_data.size(0));
      View<1, T> oview(out_data.ptr(), 0,
		       out_data.stride(0), out_data.size(0));
      traits::call(impl_, iview.ptr(), oview.ptr());
    }


#if OVXX_FIX_CVSIP_CONV
    // Fixup C-VSIP results.
    if (S == vsip::sym_even_len_even && R == vsip::support_same)
    {
      typedef view_traits<1, T> traits;

      T sum = T();
      length_type coeff_size = traits::length(coeffs_.ptr());
      index_type D = this->decimation();
      
      index_type n = out.size()-1;
      for (index_type k=0; k<2*coeff_size; ++k)
      {
	if (n*D + (coeff_size)   >= k &&
	    n*D + (coeff_size)-k <  in.size())
	{
	  if (k<coeff_size)
	    sum += traits::get(coeffs_.ptr(), k)
	         * in.get((n*D+coeff_size-k));
	  else
	    sum += traits::get(coeffs_.ptr(), 2*coeff_size - k - 1)
	         * in.get((n*D+coeff_size-k));
	}
      }
      out.put(n, sum);
    }

    if (S == vsip::sym_even_len_even && R  == vsip::support_full)
    {
      typedef view_traits<1, T> traits;
      length_type coeff_size = traits::length(coeffs_.ptr());
      index_type D = this->decimation();

      for (index_type i=0; i<coeff_size; ++i)
      {
	index_type n = out.size()-1-i;

	T sum = T();
      
	for (index_type k=0; k<2*coeff_size; ++k)
	{
	  if (n*D >= k && n*D-k < in.size())
	  {
	    if (k<coeff_size)
	      sum += traits::get(coeffs_.ptr(), k)
	           * in.get((n*D-k));
	    else
	      sum += traits::get(coeffs_.ptr(), 2*coeff_size - k - 1)
	           * in.get((n*D-k));
	  }
	}
	out.put(n, sum);
      }
    }

    if (S == vsip::nonsym && R == vsip::support_same)
    {
      typedef view_traits<1, T> traits;

      T sum = T();
      length_type coeff_size = traits::length(coeffs_.ptr());
      index_type D = this->decimation();
      
      index_type n = out.size()-1;
      for (index_type k=0; k<coeff_size; ++k)
      {
	if (n*D + (coeff_size/2)   >= k &&
	    n*D + (coeff_size/2)-k <  in.size())
	{
	    sum += traits::get(coeffs_.ptr(), k)
	         * in.get((n*D+(coeff_size/2)-k));
	}
      }
      out.put(n, sum);
    }

    if (S == vsip::nonsym && R == vsip::support_full)
    {
      typedef view_traits<1, T> traits;
      length_type coeff_size = traits::length(coeffs_.ptr());
      index_type D = this->decimation();

      for (index_type i=0; i<(coeff_size+1)/2; ++i)
      {
	index_type n = out.size()-1-i;

	T sum = T();
      
	for (index_type k=0; k<coeff_size; ++k)
	{
	  if (n*D >= k && n*D-k < in.size())
	  {
	      sum += traits::get(coeffs_.ptr(), k)
	           * in.get((n*D-k));
	  }
	}
	out.put(n, sum);
      }
    }
#endif
  }

private:
  View<1, T, false> coeffs_;
  typename traits::conv_type *impl_;
};

template <symmetry_type       S,
	  support_region_type R,
	  typename            T,
	  unsigned            N,
          alg_hint_type       H>
class Convolution<2, S, R, T, N, H>
  : public Convolution_base<2, T, S, R>,
    ct_assert_msg<conv_traits<2, T>::valid,
		  conv_cvsip_backend_does_not_support_type<T> >
{
  typedef conv_traits<2, T> traits;
public:

  template <typename Block>
  Convolution(const_Vector<T, Block> coeffs,
              Domain<2> const& input_size,
              length_type decimation)
    : Convolution_base<2, T, S, R>(view_domain(coeffs), input_size, decimation),
      coeffs_(coeffs.size()),
      impl_(0)
  {
    dda::Data<Block, dda::in> c_data(coeffs.block());
    View<2, T> tmp(c_data.ptr(), 0,
                   c_data.stride(0), c_data.size(0),
                   c_data.stride(1), c_data.size(1));
    coeffs_ = tmp;
    impl_ = traits::create(coeffs_.ptr(), S, input_size.size(), c_data.size(1),
                           decimation, R, N, H);
  }
  ~Convolution() {traits::destroy(impl_);}

protected:
  template <typename Block0, typename Block1>
  void convolve(const_Matrix<T, Block0> in, Matrix<T, Block1> out)
  {
    dda::Data<Block0, dda::in> in_data(in.block());
    dda::Data<Block1, dda::out> out_data(out.block());
    View<1, T> iview(in_data.ptr(), 0,
                     in_data.stride(0), in_data.size(0),
                     in_data.stride(1), in_data.size(1));
    View<1, T> oview(out_data.ptr(), 0,
                     out_data.stride(0), out_data.size(0),
                     out_data.stride(1), out_data.size(1));
    traits::call(impl_, iview.ptr(), oview.ptr());
  }

private:
  View<2, T, false> coeffs_;
  typename traits::conv_type *impl_;
};

} // namespace ovxx::cvsip

namespace dispatcher
{
# if HAVE_VSIP_CONV1D_CREATE_F == 1
template <symmetry_type       S,
	  support_region_type R,
	  unsigned            N,
          alg_hint_type       H>
struct Evaluator<op::conv<1, S, R, float, N, H>, be::cvsip>
{
  static bool const ct_valid = true;
  typedef cvsip::Convolution<1, S, R, float, N, H> backend_type;
};
# endif
# if HAVE_VSIP_CONV2D_CREATE_F == 1
template <symmetry_type       S,
	  support_region_type R,
	  unsigned            N,
          alg_hint_type       H>
struct Evaluator<op::conv<2, S, R, float, N, H>, be::cvsip>
{
  static bool const ct_valid = true;
  typedef cvsip::Convolution<2, float, S, R> backend_type;
};
# endif
# if HAVE_VSIP_CONV1D_CREATE_D == 1
template <symmetry_type       S,
	  support_region_type R,
	  unsigned            N,
          alg_hint_type       H>
struct Evaluator<op::conv<1, S, R, double, N, H>, be::cvsip>
{
  static bool const ct_valid = true;
  typedef cvsip::Convolution<1, S, R, double, N, H> backend_type;
};
# endif
# if HAVE_VSIP_CONV2D_CREATE_D == 1
template <symmetry_type       S,
	  support_region_type R,
	  unsigned            N,
          alg_hint_type       H>
struct Evaluator<op::conv<2, S, R, double, N, H>, be::cvsip>
{
  static bool const ct_valid = true;
  typedef cvsip::Convolution<2, S, R, double, N, H> backend_type;
};
# endif

} // namespace ovxx::dispatcher
} // namespace ovxx

#endif
