//
// Copyright (c) 2005,2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_COMPLEX_HPP
#define VSIP_COMPLEX_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/core/complex_decl.hpp>
#include <vsip/core/expr/fns_elementwise.hpp>
#include <vsip/math.hpp>



/***********************************************************************
  Definitions - Conversion [complex.convert]
***********************************************************************/

namespace vsip
{

/// Convert from rectangular to polar for single complex value.

/// recttopolar -- convert complex number in rectangular coordinates
///                (real and imaginary) into polar coordinates (magnitude
///                and phase.

template <typename T1,
	  typename T2>
inline void
recttopolar(
  complex<T1> const& rect,
  T2&                mag,
  T2&                phase)
  VSIP_NOTHROW
{
  mag   = abs(rect);
  phase = arg(rect);
}



/// Convert from rectangular to polar for view of complex values.

template <typename                            T1,
	  typename                            T2,
	  template <typename, typename> class const_View,
	  template <typename, typename> class View,
	  typename                            Block0,
	  typename                            Block1,
	  typename                            Block2>
inline void
recttopolar(const_View<complex<T1>, Block0> z,
	    View<T2, Block1> rho, View<T2, Block2> theta)
  VSIP_NOTHROW
{
  assert(z.size() == rho.size() && z.size() == theta.size());
  for (index_type i = 0; i < z.size(); i++)
  {
    // Use abs/arg instead of calling scalar recttopolar because that
    // would require lvalue_proxy for split-storage.
    complex<T1> tmp = z.get(i);
    rho.put(i,   abs(tmp));
    theta.put(i, arg(tmp));
  }
}

namespace impl
{
template <typename T>
struct realtocomplex_functor
{
  typedef complex<T> result_type;
  static char const* name() { return "realtocomplex"; }
  static result_type apply(T rho) { return complex<T>(rho);}
  result_type operator() (T rho) const { return apply(rho);}
};

template <typename T1, typename T2>
struct polartorect_functor
{
  typedef typename Promotion<complex<T1>, complex<T2> >::type result_type;
  static char const* name() { return "polartorect"; }
  static result_type apply(T1 rho, T2 theta) { return polar(rho, theta);}
  result_type operator() (T1 rho, T2 theta) const { return apply(rho, theta);}
};

template <typename T1, typename T2>
struct Dispatch_polartorect :
  conditional<Is_view_type<T1>::value || Is_view_type<T2>::value,
	      Binary_func_view<polartorect_functor, T1, T2>,
	      polartorect_functor<T1, T2> >::type
{
};

template <typename T1, typename T2>
struct cmplx_functor
{
  typedef typename Promotion<complex<T1>, complex<T2> >::type result_type;
  static char const* name() { return "cmplx"; }
  static result_type apply(T1 real, T2 imag) { return result_type(real, imag);}
  result_type operator() (T1 real, T2 imag) const { return apply(real, imag);}
};

} // namespace impl

template <typename T>
inline complex<T>
polartorect(T rho) VSIP_NOTHROW { return complex<T>(rho);}

template <typename                            T,
	  template <typename, typename> class const_View,
	  typename                            Block0>
inline const_View<complex<T>, 
                  impl::Unary_func_view<impl::realtocomplex_functor, T> >
polartorect(const_View<T, Block0> rho) VSIP_NOTHROW
{
  return impl::Unary_func_view<impl::realtocomplex_functor, T>::apply(rho);
}

template <typename T1, typename T2>
inline typename impl::Dispatch_polartorect<T1, T2>::result_type
polartorect(T1 rho, T2 theta) VSIP_NOTHROW
{
  return impl::Dispatch_polartorect<T1, T2>::apply(rho, theta);
}

template <typename T1, typename T2,
	  template <typename, typename> class Const_view,
	  typename Block1, typename Block2>
inline typename 
impl::Binary_func_view<impl::cmplx_functor, 
		       Const_view<T1, Block1>,
		       Const_view<T2, Block2> >::result_type
cmplx(Const_view<T1, Block1> real, Const_view<T2, Block2> imag) VSIP_NOTHROW
{
  return impl::Binary_func_view<impl::cmplx_functor,
    Const_view<T1, Block1>,
    Const_view<T2, Block2> >::apply(real, imag);
}

} // namespace vsip

#endif // VSIP_COMPLEX_HPP
