/* Copyright (c) 2006 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/cvsip/conv.hpp
    @author  Stefan Seefeld
    @date    2006-10-30
    @brief   VSIPL++ Library: Convolution class implementation using C-VSIPL.
*/

#ifndef VSIP_CORE_CVSIP_CONV_HPP
#define VSIP_CORE_CVSIP_CONV_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/config.hpp>
#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/core/domain_utils.hpp>
#include <vsip/core/profile.hpp>
#include <vsip/core/signal/conv_common.hpp>
#include <vsip/core/cvsip/block.hpp>
#include <vsip/core/cvsip/view.hpp>
#include <vsip/core/cvsip/common.hpp>
extern "C" 
{
#include <vsip.h>
}

// Define this to 1 to fix the convolution results returned from
// C-VSIP.  This works around incorrect results produced by the
// C-VSIP ref-impl for a subset of the symmetry/support-region
// combinations.  This may not be necessary for other C-VSIP
// implementations.
//
// Defining this to 1 will produce results consistent with the
// other Convolution implementations (Ext, IPP, and SAL).

#define VSIP_IMPL_FIX_CVSIP_CONV 1



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{
namespace impl
{
namespace cvsip
{

template <dimension_type D, typename T> struct Conv_traits
{ static bool const valid = false; };

#if HAVE_VSIP_CONV1D_CREATE_F == 1
template <>
struct Conv_traits<1, float>
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
    if (!c) VSIP_IMPL_THROW(std::bad_alloc());
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
struct Conv_traits<2, float>
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
    if (!c) VSIP_IMPL_THROW(std::bad_alloc());
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
struct Conv_traits<1, double>
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
    if (!c) VSIP_IMPL_THROW(std::bad_alloc());
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
struct Conv_traits<2, double>
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
    if (!c) VSIP_IMPL_THROW(std::bad_alloc());
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
    : kernel_size_(conv_kernel_size<S>(c)),
      input_size_(i),
      output_size_(conv_output_size(R, kernel_size_, input_size_, d)),
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
struct Conv_cvsip_backend_does_not_support_type;

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
    Compile_time_assert_msg<Conv_traits<1, T>::valid,
                            Conv_cvsip_backend_does_not_support_type<T> >
{
  typedef Conv_traits<1, T> traits;

public:
  template <typename Block>
  Convolution(const_Vector<T, Block> coeffs, Domain<1> const& input_size,
              length_type decimation)
    : Convolution_base<1, T, S, R>(view_domain(coeffs), input_size, decimation),
      coeffs_(coeffs.size()),
      impl_(0)
  {
    Ext_data<Block> ext_c(coeffs.block());
    View<1, T> tmp(ext_c.data(), 0, ext_c.stride(0), ext_c.size(0));
    coeffs_ = tmp;
    impl_ = traits::create(coeffs_.ptr(), S, input_size.size(), decimation, R, N, H);
  }
  ~Convolution() {traits::destroy(impl_);}

protected:
  template <typename Block0, typename Block1>
  void convolve(const_Vector<T, Block0> in, Vector<T, Block1> out)
  {
    {
    Ext_data<Block0> ext_in(const_cast<Block0&>(in.block()));
    Ext_data<Block1> ext_out(out.block());
    View<1, T> iview(ext_in.data(), 0,
                     ext_in.stride(0), ext_in.size(0));
    View<1, T> oview(ext_out.data(), 0,
                     ext_out.stride(0), ext_out.size(0));
    traits::call(impl_, iview.ptr(), oview.ptr());
    }


#if VSIP_IMPL_FIX_CVSIP_CONV
    // Fixup C-VSIP results.
    if (S == vsip::sym_even_len_even && R == vsip::support_same)
    {
      typedef View_traits<1, T> traits;

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
      typedef View_traits<1, T> traits;
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
      typedef View_traits<1, T> traits;

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
      typedef View_traits<1, T> traits;
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
    Compile_time_assert_msg<Conv_traits<2, T>::valid,
			    Conv_cvsip_backend_does_not_support_type<T> >
{
  typedef Conv_traits<2, T> traits;
public:

  template <typename Block>
  Convolution(const_Vector<T, Block> coeffs,
              Domain<2> const& input_size,
              length_type decimation)
    : Convolution_base<2, T, S, R>(view_domain(coeffs), input_size, decimation),
      coeffs_(coeffs.size()),
      impl_(0)
  {
    Ext_data<Block> ext_c(coeffs.block());
    View<2, T> tmp(ext_c.data(), 0,
                   ext_c.stride(0), ext_c.size(0),
                   ext_c.stride(1), ext_c.size(1));
    coeffs_ = tmp;
    impl_ = traits::create(coeffs_.ptr(), S, input_size.size(), ext_c.size(1),
                           decimation, R, N, H);
  }
  ~Convolution() {traits::destroy(impl_);}

protected:
  template <typename Block0, typename Block1>
  void convolve(const_Matrix<T, Block0> in, Matrix<T, Block1> out)
  {
    Ext_data<Block0> ext_in(const_cast<Block0&>(in.block()));
    Ext_data<Block1> ext_out(out.block());
    View<1, T> iview(ext_in.data(), 0,
                     ext_in.stride(0), ext_in.size(0),
                     ext_in.stride(1), ext_in.size(1));
    View<1, T> oview(ext_out.data(), 0,
                     ext_out.stride(0), ext_out.size(0),
                     ext_out.stride(1), ext_out.size(1));
    traits::call(impl_, iview.ptr(), oview.ptr());
  }

private:
  View<2, T, false> coeffs_;
  typename traits::conv_type *impl_;
};

} // namespace vsip::impl::cvsip
} // namespace vsip::impl
} // namespace vsip

#if !VSIP_IMPL_REF_IMPL

namespace vsip_csl
{
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
  typedef impl::cvsip::Convolution<1, S, R, float, N, H> backend_type;
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
  typedef impl::cvsip::Convolution<2, float, S, R> backend_type;
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
  typedef impl::cvsip::Convolution<1, S, R, double, N, H> backend_type;
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
  typedef impl::cvsip::Convolution<2, S, R, double, N, H> backend_type;
};
# endif

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif // !VSIP_IMPL_REF_IMPL

#endif // VSIP_CORE_CVSIP_CONV_HPP
