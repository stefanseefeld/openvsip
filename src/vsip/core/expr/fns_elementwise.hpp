//
// Copyright (c) 2005 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_CORE_EXPR_FNS_ELEMENTWISE_HPP
#define VSIP_CORE_EXPR_FNS_ELEMENTWISE_HPP

#include <vsip/core/expr/functor.hpp>
#include <vsip/core/promote.hpp>
#include <vsip/core/fns_scalar.hpp>
#include <vsip/core/metaprogramming.hpp>
#include <vsip/core/lvalue_proxy.hpp>

/// Macro to define a unary function on views in terms of
/// its homologe on scalars.
#define VSIP_IMPL_UNARY_FUNCTOR(f, F)					  \
namespace vsip_csl { namespace expr { namespace op {                      \
template <typename T>                                                     \
struct F                                                                  \
{                                                                         \
  typedef T result_type;                                                  \
  static char const *name() { return #f;}                                 \
  static result_type apply(T t) { return fn::f(t);}                       \
  result_type operator()(T t) const { return apply(t);}                   \
};                                                                        \
}}}

#define VSIP_IMPL_UNARY_FUNCTOR_RETN(f, F, retn)			  \
namespace vsip_csl { namespace expr { namespace op {                      \
template <typename T>                                                     \
struct F                                                                  \
{                                                                         \
  typedef retn result_type;                                               \
  static char const *name() { return #f;}                                 \
  static result_type apply(T t) { return fn::f(t);}                       \
  result_type operator()(T t) const { return apply(t);}                   \
};                                                                        \
}}}

#define VSIP_IMPL_UNARY_DISPATCH(f, F)	  				  \
namespace vsip { namespace impl {                                         \
template <typename T>                                                     \
struct Dispatch_##f :                                                     \
  conditional<Is_view_type<T>::value,				          \
    Unary_func_view<vsip_csl::expr::op::F, T>,	                          \
    typename conditional<Is_lvalue_proxy_type<T>::value,		  \
      expr::op::F<typename Is_lvalue_proxy_type<T>::value_type>,          \
      expr::op::F<T> >::type>::type      		                  \
{};                                                                       \
}}

#define VSIP_IMPL_UNARY_FUNCTION(f, F)		 			  \
namespace vsip {                                                          \
template <typename T>                                                     \
inline                                                                    \
typename impl::Dispatch_##f<T>::result_type	                          \
f(T const& t)							          \
{ return impl::Dispatch_##f<T>::apply(t);}				  \
}

/// This function gateway is roughly specialized to VSIPL++ view types.
/// This prevents it from competing with more general function overloads.
/// For example, cmath defines template <typename T> bool isnan(T& t)
/// which is ambiguous with the normal VSIP_IMPL_UNARY_FUNCTION.
#define VSIP_IMPL_UNARY_VIEW_FUNCTION(f)				  \
namespace vsip {                                                          \
template <template <typename, typename> class V,                          \
          typename T, typename B>                                         \
inline                                                                    \
typename impl::Dispatch_##f<V<T,B> >::result_type			  \
f(V<T,B> const &t)							  \
{ return impl::Dispatch_##f<V<T,B> >::apply(t);}			  \
}

#define VSIP_IMPL_UNARY_FUNC(f, F)					  \
VSIP_IMPL_UNARY_FUNCTOR(f, F)						  \
VSIP_IMPL_UNARY_DISPATCH(f, F)						  \
VSIP_IMPL_UNARY_FUNCTION(f, F)


#define VSIP_IMPL_UNARY_FUNC_RETN(f, F, retn)				  \
VSIP_IMPL_UNARY_FUNCTOR_RETN(f, F, retn)				  \
VSIP_IMPL_UNARY_DISPATCH(f, F)						  \
VSIP_IMPL_UNARY_FUNCTION(f, F)

#define VSIP_IMPL_UNARY_VIEW_FUNC_RETN(f, F, retn)			  \
VSIP_IMPL_UNARY_FUNCTOR_RETN(f, F, retn)				  \
VSIP_IMPL_UNARY_DISPATCH(f, F)						  \
VSIP_IMPL_UNARY_VIEW_FUNCTION(f)

/// Define a unary operator. Assume the associated Dispatch 
/// is already defined.
#define VSIP_IMPL_UNARY_OP(op, fname)                                     \
namespace vsip { namespace impl {                                         \
template <typename T>                                                     \
typename Dispatch_##fname<typename Is_view_type<T>::type>::result_type    \
operator op(T t)                                                          \
{ return Dispatch_##fname<T>::apply(t);}                                  \
}}

/// Macro to define a binary function on views in terms of
/// its homologe on scalars.
#define VSIP_IMPL_BINARY_FUNCTOR(f, F)					  \
namespace vsip_csl { namespace expr { namespace op {                      \
template <typename T1, typename T2>                                       \
struct F                                                                  \
{                                                                         \
  typedef typename Promotion<T1, T2>::type result_type;                   \
  static char const *name() { return #f;}                                 \
  static result_type apply(T1 t1, T2 t2) { return fn::f(t1, t2);}         \
  result_type operator()(T1 t1, T2 t2) const { return apply(t1, t2);}     \
};                                                                        \
}}}

#define VSIP_IMPL_BINARY_FUNCTOR_RETN(f, F, retn)			  \
namespace vsip_csl { namespace expr { namespace op {                      \
template <typename T1, typename T2>                                       \
struct F                                                                  \
{                                                                         \
  typedef retn result_type;                                               \
  static char const *name() { return #f;}                                 \
  static result_type apply(T1 t1, T2 t2) { return fn::f(t1, t2);}         \
  result_type operator()(T1 t1, T2 t2) const { return apply(t1, t2);}     \
};                                                                        \
}}}

#define VSIP_IMPL_BINARY_FUNCTOR_SCALAR_RETN(f, F)	 	 	  \
namespace vsip_csl { namespace expr { namespace op {                      \
template <typename T1, typename T2>                                       \
struct F                                                                  \
{                                                                         \
  typedef typename impl::scalar_of<typename Promotion<T1, T2>::type>::type\
                result_type;                                              \
  static char const *name() { return #f;}                                 \
  static result_type apply(T1 t1, T2 t2) { return fn::f(t1, t2);}         \
  result_type operator()(T1 t1, T2 t2) const { return apply(t1, t2);}     \
};                                                                        \
}}}

#define VSIP_IMPL_BINARY_DISPATCH(f, F)					  \
namespace vsip { namespace impl {                                         \
template <typename T1, typename T2>                                       \
struct Dispatch_##f :                                                     \
  conditional<Is_view_type<T1>::value || Is_view_type<T2>::value,         \
              Binary_func_view<expr::op::F, T1, T2>,		          \
	      expr::op::F<T1, T2> >::type			          \
{};  									  \
}}

/// Define a dispatcher that only matches if at least one of the arguments
/// is a view type.
#define VSIP_IMPL_BINARY_OP_DISPATCH(f, F)				  \
namespace vsip { namespace impl {                                         \
template <typename T1, typename T2,                                       \
          bool P = Is_view_type<T1>::value || Is_view_type<T2>::value>    \
struct Dispatch_op_##f : Binary_func_view<expr::op::F, T1, T2> {};        \
template <typename T1, typename T2>                                       \
struct Dispatch_op_##f<T1, T2, false> {};                                 \
}}

#define VSIP_IMPL_BINARY_FUNCTION(f)	  			 	  \
namespace vsip {                                                          \
template <typename T1, typename T2>                                       \
inline                                                                    \
typename impl::Dispatch_##f<T1, T2>::result_type			  \
f(T1 const &t1, T2 const &t2)					          \
{ return impl::Dispatch_##f<T1, T2>::apply(t1, t2);}                      \
}

#define VSIP_IMPL_BINARY_OPERATOR_ONE(op, fname)                          \
namespace vsip {                                                          \
template <typename T1, typename T2>                                       \
inline                                                                    \
typename impl::Dispatch_op_##fname<T1, T2>::result_type		          \
operator op(T1 const& t1, T2 const& t2)					  \
{ return impl::Dispatch_op_##fname<T1, T2>::apply(t1, t2);}		  \
}

#define VSIP_IMPL_BINARY_OPERATOR_TWO(op, fname)                          \
namespace vsip {                                                          \
template <template <typename, typename> class View,                       \
          typename T1, typename Block1, typename T2>                      \
inline                                                                    \
typename impl::Dispatch_op_##fname<View<T1, Block1>, T2>::result_type	  \
operator op(View<T1, Block1> const& t1, T2 t2)				  \
{ return impl::Dispatch_op_##fname<View<T1, Block1>, T2>::apply(t1, t2);} \
                                                                          \
template <template <typename, typename> class View,                       \
          typename T1, typename T2, typename Block2>                      \
inline                                                                    \
typename impl::Dispatch_op_##fname<T1, View<T2, Block2> >::result_type	  \
operator op(T1 t1, View<T2, Block2> const& t2)				  \
{ return impl::Dispatch_op_##fname<T1, View<T2, Block2> >::apply(t1, t2);}\
                                                                          \
template <template <typename, typename> class LView,                      \
          template <typename, typename> class RView,                      \
          typename T1, typename Block1,                                   \
          typename T2, typename Block2>                                   \
inline                                                                    \
typename impl::Dispatch_op_##fname<LView<T1, Block1>,			  \
                             RView<T2, Block2> >::result_type             \
operator op(LView<T1, Block1> const& t1, RView<T2, Block2> const& t2)	  \
{ return impl::Dispatch_op_##fname<LView<T1, Block1>,			  \
                             RView<T2, Block2> >::apply(t1, t2);}         \
}

#if (defined(__GNUC__) && __GNUC__ < 4) || defined(__ghs__) || defined(__ICL)
# define VSIP_IMPL_BINARY_OPERATOR(op, fname)                             \
VSIP_IMPL_BINARY_OPERATOR_ONE(op, fname)
#else
# define VSIP_IMPL_BINARY_OPERATOR(op, fname)                             \
VSIP_IMPL_BINARY_OPERATOR_TWO(op, fname)
#endif


#define VSIP_IMPL_BINARY_VIEW_FUNCTION(f) 	 	 	 	  \
namespace vsip {                                                          \
template <template <typename, typename> class V,                          \
          typename T, typename B>                                         \
inline                                                                    \
typename impl::Dispatch_##f<V<T,B>, V<T,B> >::result_type		  \
f(V<T,B> const &t1, V<T,B> const &t2)				          \
{ return impl::Dispatch_##f<V<T,B>, V<T,B> >::apply(t1, t2);}             \
}

#define VSIP_IMPL_BINARY_FUNC(f, F)	 			  	  \
VSIP_IMPL_BINARY_FUNCTOR(f, F)					          \
VSIP_IMPL_BINARY_DISPATCH(f, F)					          \
VSIP_IMPL_BINARY_FUNCTION(f)				                  \
VSIP_IMPL_BINARY_VIEW_FUNCTION(f)

/// Binary function that map to binary operators. For those the functor
/// already exists.
#define VSIP_IMPL_BINARY_FUNC_USEOP(f, F)			          \
VSIP_IMPL_BINARY_DISPATCH(f, F) 				          \
VSIP_IMPL_BINARY_FUNCTION(f)	    		 	                  \
VSIP_IMPL_BINARY_VIEW_FUNCTION(f)

#define VSIP_IMPL_BINARY_FUNC_RETN(f, F, retn)				  \
VSIP_IMPL_BINARY_FUNCTOR_RETN(f, F, retn)				  \
VSIP_IMPL_BINARY_DISPATCH(f, F)					          \
VSIP_IMPL_BINARY_FUNCTION(f)

#define VSIP_IMPL_BINARY_FUNC_SCALAR_RETN(f, F)				  \
VSIP_IMPL_BINARY_FUNCTOR_SCALAR_RETN(f, F)				  \
VSIP_IMPL_BINARY_DISPATCH(f, F)					          \
VSIP_IMPL_BINARY_FUNCTION(f)

#define VSIP_IMPL_BINARY_OP(op, f, F)					  \
VSIP_IMPL_BINARY_OP_DISPATCH(f, F)				          \
VSIP_IMPL_BINARY_OPERATOR(op, f)

/// Macro to define a ternary function on views in terms of
/// its homologe on scalars.
#define VSIP_IMPL_TERNARY_FUNC(f, F)	 				  \
namespace vsip_csl { namespace expr { namespace op {                      \
template <typename T1, typename T2, typename T3>                          \
struct F                                                                  \
{                                                                         \
  typedef typename Promotion<typename Promotion<T1, T2>::type,            \
                             T3>::type result_type;                       \
  static char const *name() { return #f;}                                 \
  static result_type apply(T1 t1, T2 t2, T3 t3)                           \
  { return fn::f(t1, t2, t3);}                                            \
  result_type operator()(T1 t1, T2 t2, T3 t3) const                       \
  { return apply(t1, t2, t3);}                                            \
};                                                                        \
}}}                                                                       \
namespace vsip { namespace impl {				          \
template <typename T1, typename T2, typename T3>                          \
struct Dispatch_##f :                                                     \
  conditional<Is_view_type<T1>::value ||                                  \
           Is_view_type<T2>::value ||                                     \
           Is_view_type<T3>::value,                                       \
           Ternary_func_view<expr::op::F, T1, T2, T3>,	                  \
           expr::op::F<T1, T2, T3> >::type		                  \
{};                                                                       \
} 								 	  \
template <typename T1, typename T2, typename T3>                          \
inline									  \
typename impl::Dispatch_##f<T1, T2, T3>::result_type		          \
f(T1 const &t1, T2 const &t2, T3 const &t3)				  \
{ return impl::Dispatch_##f<T1, T2, T3>::apply(t1, t2, t3);}              \
}

/***********************************************************************
  Unary Functions
***********************************************************************/

VSIP_IMPL_UNARY_FUNC(acos, Acos)

namespace vsip_csl { namespace expr { namespace op {
template <typename T> struct Arg {};

template <typename T>
struct Arg<std::complex<T> >
{
  typedef T result_type;
  static char const *name() { return "arg"; }                
  static result_type apply(std::complex<T> t) { return fn::arg(t);}
  result_type operator()(std::complex<T> t) const { return apply(t);}
};
}}}

VSIP_IMPL_UNARY_DISPATCH(arg, Arg)
VSIP_IMPL_UNARY_FUNCTION(arg, Arg)

VSIP_IMPL_UNARY_FUNC(asin, Asin)
VSIP_IMPL_UNARY_FUNC(atan, Atan)
VSIP_IMPL_UNARY_FUNC(bnot, Bnot)
VSIP_IMPL_UNARY_FUNC(ceil, Ceil)
VSIP_IMPL_UNARY_FUNC(conj, Conj)
VSIP_IMPL_UNARY_FUNC(cos, Cos)
VSIP_IMPL_UNARY_FUNC(cosh, Cosh)

namespace vsip_csl { namespace expr { namespace op {
template <typename T>
struct Euler
{
  typedef std::complex<T> result_type;
  static char const *name() { return "euler"; }                
  static result_type apply(T t) { return fn::euler(t);}
  result_type operator()(T t) const { return apply(t);}
};
}}}

VSIP_IMPL_UNARY_DISPATCH(euler, Euler)
VSIP_IMPL_UNARY_FUNCTION(euler, Euler)

VSIP_IMPL_UNARY_FUNC(exp, Exp)
VSIP_IMPL_UNARY_FUNC(exp10, Exp10)
VSIP_IMPL_UNARY_FUNC(floor, Floor)

namespace vsip_csl { namespace expr { namespace op {
template <typename T> struct Imag {};

template <typename T>
struct Imag<std::complex<T> >
{
  typedef T result_type;
  static char const *name() { return "imag"; }
  static result_type apply(std::complex<T> t) { return fn::imag(t);}
  result_type operator()(std::complex<T> t) const { return apply(t);}
};
}}}

VSIP_IMPL_UNARY_DISPATCH(imag, Imag)
VSIP_IMPL_UNARY_FUNCTION(imag, Imag)

VSIP_IMPL_UNARY_FUNC_RETN(is_finite, Is_finite, bool)
VSIP_IMPL_UNARY_FUNC_RETN(is_nan, Is_nan, bool)
VSIP_IMPL_UNARY_FUNC_RETN(is_normal, Is_normal, bool)

VSIP_IMPL_UNARY_FUNC_RETN(lnot, Lnot, bool)
VSIP_IMPL_UNARY_FUNC(log, Log)
VSIP_IMPL_UNARY_FUNC(log10, Log10)
VSIP_IMPL_UNARY_FUNC_RETN(mag, Mag, typename impl::scalar_of<T>::type)
VSIP_IMPL_UNARY_FUNC_RETN(magsq, Magsq, typename impl::scalar_of<T>::type)
VSIP_IMPL_UNARY_FUNC(neg, Neg)

namespace vsip_csl { namespace expr { namespace op {

template <typename T> struct Real {};

template <typename T>
struct Real<std::complex<T> >
{
  typedef T result_type;
  static char const *name() { return "real"; }                
  static result_type apply(std::complex<T> t) { return fn::real(t);}
  result_type operator()(std::complex<T> t) const { return apply(t);}
};
}}}

VSIP_IMPL_UNARY_DISPATCH(real, Real)
VSIP_IMPL_UNARY_FUNCTION(real, Real)

VSIP_IMPL_UNARY_FUNC(recip, Recip)
VSIP_IMPL_UNARY_FUNC(rsqrt, Rsqrt)
VSIP_IMPL_UNARY_FUNC(sin, Sin)
VSIP_IMPL_UNARY_FUNC(sinh, Sinh)
VSIP_IMPL_UNARY_FUNC(sq, Sq)
VSIP_IMPL_UNARY_FUNC(sqrt, Sqrt)
VSIP_IMPL_UNARY_FUNC(tan, Tan)
VSIP_IMPL_UNARY_FUNC(tanh, Tanh)

VSIP_IMPL_UNARY_FUNC(impl_conj, Impl_conj)

namespace vsip_csl { namespace expr { namespace op {
template <typename T>
struct Impl_real
{
  typedef typename impl::scalar_of<T>::type result_type;
  static char const *name() { return "impl_real";}
  static result_type apply(T t) { return fn::impl_real(t);}
  result_type operator()(T t) const { return apply(t);}
};
}}}

VSIP_IMPL_UNARY_DISPATCH(impl_real, Impl_real)
VSIP_IMPL_UNARY_FUNCTION(impl_real, Impl_real)

namespace vsip_csl { namespace expr { namespace op {
template <typename T>
struct Impl_imag
{
  typedef typename impl::scalar_of<T>::type result_type;
  static char const *name() { return "impl_imag";}
  static result_type apply(T t) { return fn::impl_imag(t);}
  result_type operator()(T t) const { return apply(t);}
};
}}}

VSIP_IMPL_UNARY_DISPATCH(impl_imag, Impl_imag)
VSIP_IMPL_UNARY_FUNCTION(impl_imag, Impl_imag)

namespace vsip_csl { namespace expr { namespace op {
// This unary operator gives a hint to the compiler that this block is
// unaligned
template <typename T>
struct Unaligned
{
  typedef T result_type;
  static char const *name() { return "unaligned"; }                
  static result_type apply(T t) { return t;}
  result_type operator()(T t) const { return apply(t);}
};
}}}

VSIP_IMPL_UNARY_DISPATCH(unaligned, Unaligned)
VSIP_IMPL_UNARY_FUNCTION(unaligned, Unaligned)

/***********************************************************************
  Binary Functions
***********************************************************************/

VSIP_IMPL_BINARY_FUNC_USEOP(add, Add)
VSIP_IMPL_BINARY_FUNC(atan2, Atan2)
VSIP_IMPL_BINARY_FUNC(band, Band)
VSIP_IMPL_BINARY_FUNC(bor, Bor)
VSIP_IMPL_BINARY_FUNC(bxor, Bxor)
VSIP_IMPL_BINARY_FUNC_USEOP(div, Div)
VSIP_IMPL_BINARY_FUNC_RETN(eq, Eq, bool)
VSIP_IMPL_BINARY_FUNC(fmod, Fmod)
VSIP_IMPL_BINARY_FUNC_RETN(ge, Ge, bool)
VSIP_IMPL_BINARY_FUNC_RETN(gt, Gt, bool)
VSIP_IMPL_BINARY_FUNC(hypot, Hypot)
VSIP_IMPL_BINARY_FUNC(jmul, Jmul)
VSIP_IMPL_BINARY_FUNC_RETN(land, Land, bool)
VSIP_IMPL_BINARY_FUNC_RETN(le, Le, bool)
VSIP_IMPL_BINARY_FUNC_RETN(lt, Lt, bool)
VSIP_IMPL_BINARY_FUNC_RETN(lor, Lor, bool)
VSIP_IMPL_BINARY_FUNC_RETN(lxor, Lxor, bool)
VSIP_IMPL_BINARY_FUNC_USEOP(mul, Mult)
VSIP_IMPL_BINARY_FUNC(max, Max)
VSIP_IMPL_BINARY_FUNC(maxmg, Maxmg)
VSIP_IMPL_BINARY_FUNC_SCALAR_RETN(maxmgsq, Maxmgsq)
VSIP_IMPL_BINARY_FUNC(min, Min)
VSIP_IMPL_BINARY_FUNC(minmg, Minmg)
VSIP_IMPL_BINARY_FUNC_SCALAR_RETN(minmgsq, Minmgsq)
VSIP_IMPL_BINARY_FUNC_RETN(ne, Ne, bool)
VSIP_IMPL_BINARY_FUNC(pow, Pow)
VSIP_IMPL_BINARY_FUNC_USEOP(sub, Sub)

/***********************************************************************
  Ternary Functions
***********************************************************************/

VSIP_IMPL_TERNARY_FUNC(am, Am)
VSIP_IMPL_TERNARY_FUNC(expoavg, Expoavg)
VSIP_IMPL_TERNARY_FUNC(ma, Ma)
VSIP_IMPL_TERNARY_FUNC(msb, Msb)
VSIP_IMPL_TERNARY_FUNC(sbm, Sbm)
VSIP_IMPL_TERNARY_FUNC(ite, Ite)

/***********************************************************************
  Unary Operators
***********************************************************************/

VSIP_IMPL_UNARY_OP(!, lnot)
VSIP_IMPL_UNARY_OP(~, bnot)

/***********************************************************************
  Binary Operators
***********************************************************************/

VSIP_IMPL_BINARY_OP(==, eq, Eq)
VSIP_IMPL_BINARY_OP(>=, ge, Ge)
VSIP_IMPL_BINARY_OP(>, gt, Gt)
VSIP_IMPL_BINARY_OP(<=, le, Le)
VSIP_IMPL_BINARY_OP(<, lt, Lt)
VSIP_IMPL_BINARY_OP(!=, ne, Ne)
VSIP_IMPL_BINARY_OP(&&, land, Land)
VSIP_IMPL_BINARY_OP(&, band, Band)
VSIP_IMPL_BINARY_OP(||, lor, Lor)
VSIP_IMPL_BINARY_OP(|, bor, Bor)

namespace vsip_csl { namespace expr { namespace op {
template <typename T1, typename T2>
struct Bxor_or_lxor
{
  typedef typename Promotion<T1, T2>::type result_type;
  static char const *name() { return "bxor";}
  static result_type apply(T1 t1, T2 t2) { return fn::bxor(t1, t2);}
  result_type operator()(T1 t1, T2 t2) const { return apply(t1, t2);}
};

template <>
struct Bxor_or_lxor<bool, bool>
{
  typedef bool result_type;
  static char const *name() { return "lxor";}
  static result_type apply(bool t1, bool t2) { return fn::lxor(t1, t2);}
  result_type operator()(bool t1, bool t2) const { return apply(t1, t2);}
};
}}}

VSIP_IMPL_BINARY_OP(^, bxor_or_lxor, Bxor_or_lxor)

#if 0
} // namespace vsip::impl

using impl::acos;
using impl::arg;
using impl::asin;
using impl::atan;
using impl::bnot;
using impl::ceil;
using impl::conj;
using impl::cos;
using impl::cosh;
using impl::euler;
using impl::exp;
using impl::exp10;
using impl::floor;
using impl::imag;
using impl::lnot;
using impl::log;
using impl::log10;
using impl::mag;
using impl::magsq;
using impl::neg;
using impl::real;
using impl::recip;
using impl::rsqrt;
using impl::sin;
using impl::sinh;
using impl::sq;
using impl::sqrt;
using impl::tan;
using impl::tanh;

using impl::add;
using impl::atan2;
using impl::band;
using impl::bor;
using impl::bxor;
using impl::div;
using impl::eq;
using impl::fmod;
using impl::ge;
using impl::gt;
using impl::hypot;
using impl::jmul;
using impl::land;
using impl::le;
using impl::lt;
using impl::lor;
using impl::lxor;
using impl::max;
using impl::maxmg;
using impl::maxmgsq;
using impl::min;
using impl::minmg;
using impl::minmgsq;
using impl::mul;
using impl::ne;
using impl::pow;
using impl::sub;

using impl::am;
using impl::expoavg;
using impl::ma;
using impl::msb;
using impl::sbm;

using impl::operator!;
using impl::operator~;
using impl::operator==;
using impl::operator>=;
using impl::operator>;
using impl::operator<=;
using impl::operator<;
using impl::operator!=;
using impl::operator&&;
using impl::operator&;
using impl::operator||;
using impl::operator|;

} // namespace vsip
#endif

#undef VSIP_IMPL_TERNARY_FUNC
#undef VSIP_IMPL_BINARY_FUNC_RETN
#undef VSIP_IMPL_BINARY_FUNC
#undef VSIP_IMPL_UNARY_FUNC_RETN
#undef VSIP_IMPL_UNARY_FUNC
#undef VSIP_IMPL_BINARY_FUNCTOR_RETN
#undef VSIP_IMPL_BINARY_FUNCTOR
#undef VSIP_IMPL_BINARY_DISPATCH
#undef VSIP_IMPL_BINARY_FUNCTION
#undef VSIP_IMPL_UNARY_FUNCTOR_RETN
#undef VSIP_IMPL_UNARY_FUNCTOR
#undef VSIP_IMPL_UNARY_DISPATCH
#undef VSIP_IMPL_UNARY_FUNCTION

#endif
