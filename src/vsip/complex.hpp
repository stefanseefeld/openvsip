//
// Copyright (c) 2005,2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef vsip_complex_hpp_
#define vsip_complex_hpp_

#include <vsip/support.hpp>
#include <vsip/impl/complex_decl.hpp>
#include <ovxx/view/fns_elementwise.hpp>
#include <vsip/math.hpp>

//  Definitions - Conversion [complex.convert]

namespace vsip
{

/// Convert from rectangular to polar for single complex value.

/// recttopolar -- convert complex number in rectangular coordinates
///                (real and imaginary) into polar coordinates (magnitude
///                and phase.
template <typename T1, typename T2>
inline void
recttopolar(complex<T1> const &rect, T2 &mag, T2 &phase) VSIP_NOTHROW
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
} // namespace vsip

namespace ovxx { namespace expr { namespace op {
template <typename T>
struct R2C
{
  typedef complex<T> result_type;
  static result_type apply(T rho) { return complex<T>(rho);}
  result_type operator() (T rho) const { return apply(rho);}
};

template <typename T1, typename T2>
struct Polar
{
  typedef typename Promotion<complex<T1>, complex<T2> >::type result_type;
  static result_type apply(T1 rho, T2 theta) { return polar(rho, theta);}
  result_type operator() (T1 rho, T2 theta) const { return apply(rho, theta);}
};

template <typename T1, typename T2 = T1>
struct Complex
{
  typedef typename Promotion<complex<T1>, complex<T2> >::type result_type;
  static result_type apply(T1 real, T2 imag) { return result_type(real, imag);}
  result_type operator() (T1 real, T2 imag) const { return apply(real, imag);}
};
}}}

namespace ovxx { namespace detail {
template <typename T1, typename T2>
struct Dispatch_polar :
  conditional<is_view_type<T1>::value || is_view_type<T2>::value,
	      functors::binary_view<expr::op::Polar, T1, T2>,
	      expr::op::Polar<T1, T2> >::type
{};
}}

namespace vsip
{
template <typename T>
inline complex<T>
polartorect(T rho) VSIP_NOTHROW { return complex<T>(rho);}

template <typename                            T,
	  template <typename, typename> class const_View,
	  typename                            Block0>
inline const_View<complex<T>, 
                  ovxx::functors::unary_view<ovxx::expr::op::R2C, T> >
polartorect(const_View<T, Block0> rho) VSIP_NOTHROW
{
  return ovxx::functors::unary_view<ovxx::expr::op::R2C, T>::apply(rho);
}

template <typename T1, typename T2>
inline typename ovxx::detail::Dispatch_polar<T1, T2>::result_type
polartorect(T1 rho, T2 theta) VSIP_NOTHROW
{
  return ovxx::detail::Dispatch_polar<T1, T2>::apply(rho, theta);
}

template <typename T1, typename T2,
	  template <typename, typename> class Const_view,
	  typename B1, typename B2>
inline typename 
ovxx::functors::binary_view<ovxx::expr::op::Complex, 
		       Const_view<T1, B1>,
		       Const_view<T2, B2> >::result_type
cmplx(Const_view<T1, B1> real, Const_view<T2, B2> imag) VSIP_NOTHROW
{
  return ovxx::functors::binary_view<ovxx::expr::op::Complex,
    Const_view<T1, B1>,
    Const_view<T2, B2> >::apply(real, imag);
}

} // namespace vsip

#endif
