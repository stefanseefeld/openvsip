//
// Copyright (c) 2005 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_view_fns_lementwise_hpp_
#define ovxx_view_fns_lementwise_hpp_

#include <ovxx/expr/operations.hpp>
#include <ovxx/view/functors.hpp>
#include <ovxx/view/traits.hpp>
#include <vsip/impl/promotion.hpp>
#include <ovxx/math/scalar.hpp>
#include <ovxx/element_proxy.hpp>

/// Macro to define a unary function on views in terms of
/// its homologe on scalars.
#define OVXX_UNARY_FUNCTOR(f, F)					  \
namespace ovxx { namespace expr { namespace op {                          \
template <typename T>                                                     \
struct F                                                                  \
{                                                                         \
  typedef T result_type;                                                  \
  static result_type apply(T t) { return math::f(t);}			  \
  result_type operator()(T t) const { return apply(t);}                   \
};                                                                        \
}}}

#define OVXX_UNARY_FUNCTOR_RETN(f, F, retn)				  \
namespace ovxx { namespace expr { namespace op {                          \
template <typename T>                                                     \
struct F                                                                  \
{                                                                         \
  typedef retn result_type;                                               \
  static result_type apply(T t) { return math::f(t);}			  \
  result_type operator()(T t) const { return apply(t);}                   \
};                                                                        \
}}}

#define OVXX_UNARY_DISPATCH(f, F)	  				  \
namespace ovxx { namespace detail {                                       \
template <typename T>                                                     \
struct Dispatch_##f :                                                     \
  conditional<is_view_type<T>::value,				          \
    functors::unary_view<expr::op::F, T>,		                  \
      typename conditional<is_element_proxy<T>::value,		          \
       expr::op::F<typename is_element_proxy<T>::value_type>,             \
      expr::op::F<T> >::type>::type      		                  \
{};                                                                       \
}}

#define OVXX_UNARY_FUNCTION(f, F)		 			  \
namespace vsip {                                                          \
template <typename T>                                                     \
inline                                                                    \
typename ovxx::detail::Dispatch_##f<T>::result_type			  \
f(T const& t)							          \
{ return ovxx::detail::Dispatch_##f<T>::apply(t);}			  \
}

/// This function gateway is roughly specialized to VSIPL++ view types.
/// This prevents it from competing with more general function overloads.
/// For example, cmath defines template <typename T> bool isnan(T& t)
/// which is ambiguous with the normal OVXX_UNARY_FUNCTION.
#define OVXX_UNARY_VIEW_FUNCTION(f)					  \
namespace vsip {                                                          \
template <template <typename, typename> class V,                          \
          typename T, typename B>                                         \
inline                                                                    \
typename ovxx::detail::Dispatch_##f<V<T,B> >::result_type		  \
f(V<T,B> const &t)							  \
{ return ovxx::detail::Dispatch_##f<V<T,B> >::apply(t);}		  \
}

#define OVXX_UNARY_FUNC(f, F)						  \
OVXX_UNARY_FUNCTOR(f, F)						  \
OVXX_UNARY_DISPATCH(f, F)						  \
OVXX_UNARY_FUNCTION(f, F)


#define OVXX_UNARY_FUNC_RETN(f, F, retn)				  \
OVXX_UNARY_FUNCTOR_RETN(f, F, retn)					  \
OVXX_UNARY_DISPATCH(f, F)						  \
OVXX_UNARY_FUNCTION(f, F)

#define OVXX_UNARY_VIEW_FUNC_RETN(f, F, retn)				  \
OVXX_UNARY_FUNCTOR_RETN(f, F, retn)					  \
OVXX_UNARY_DISPATCH(f, F)						  \
OVXX_UNARY_VIEW_FUNCTION(f)

/// Define a unary operator. Assume the associated Dispatch 
/// is already defined.
#define OVXX_UNARY_OP(op, fname)					  \
namespace ovxx { namespace detail {                                       \
template <typename T>                                                     \
typename Dispatch_##fname<typename is_view_type<T>::type>::result_type    \
operator op(T t)                                                          \
{ return Dispatch_##fname<T>::apply(t);}                                  \
}}

/// Macro to define a binary function on views in terms of
/// its homologe on scalars.
#define OVXX_BINARY_FUNCTOR(f, F)					  \
namespace ovxx { namespace expr { namespace op {                          \
template <typename T1, typename T2>                                       \
struct F                                                                  \
{                                                                         \
  typedef typename vsip::Promotion<T1, T2>::type result_type;		  \
  static result_type apply(T1 t1, T2 t2) { return math::f(t1, t2);}	  \
  result_type operator()(T1 t1, T2 t2) const { return apply(t1, t2);}     \
};                                                                        \
}}}

#define OVXX_BINARY_FUNCTOR_RETN(f, F, retn)				  \
namespace ovxx { namespace expr { namespace op {                          \
template <typename T1, typename T2>                                       \
struct F                                                                  \
{                                                                         \
  typedef retn result_type;                                               \
  static result_type apply(T1 t1, T2 t2) { return math::f(t1, t2);}       \
  result_type operator()(T1 t1, T2 t2) const { return apply(t1, t2);}     \
};                                                                        \
}}}

#define OVXX_BINARY_FUNCTOR_SCALAR_RETN(f, F)				  \
namespace ovxx { namespace expr { namespace op {                          \
template <typename T1, typename T2>                                       \
struct F                                                                  \
{                                                                         \
  typedef typename scalar_of<typename Promotion<T1, T2>::type>::type	  \
                result_type;                                              \
  static result_type apply(T1 t1, T2 t2) { return math::f(t1, t2);}       \
  result_type operator()(T1 t1, T2 t2) const { return apply(t1, t2);}     \
};                                                                        \
}}}

#define OVXX_BINARY_DISPATCH(f, F)					  \
namespace ovxx { namespace detail {                                       \
template <typename T1, typename T2>                                       \
struct Dispatch_##f :                                                     \
  conditional<is_view_type<T1>::value || is_view_type<T2>::value,         \
              functors::binary_view<expr::op::F, T1, T2>,		  \
	      expr::op::F<T1, T2> >::type			          \
{};  									  \
}}

/// Define a dispatcher that only matches if at least one of the arguments
/// is a view type.
#define OVXX_BINARY_OP_DISPATCH(f, F)					  \
namespace ovxx { namespace detail {                                       \
template <typename T1, typename T2,                                       \
          bool P = is_view_type<T1>::value || is_view_type<T2>::value>    \
struct Dispatch_op_##f : functors::binary_view<expr::op::F, T1, T2> {};   \
template <typename T1, typename T2>                                       \
struct Dispatch_op_##f<T1, T2, false> {};                                 \
}}

#define OVXX_BINARY_FUNCTION(f)						  \
namespace vsip {                                                          \
template <typename T1, typename T2>                                       \
inline                                                                    \
typename ovxx::detail::Dispatch_##f<T1, T2>::result_type		  \
f(T1 const &t1, T2 const &t2)					          \
{ return ovxx::detail::Dispatch_##f<T1, T2>::apply(t1, t2);}		  \
}

#define OVXX_BINARY_OPERATOR_ONE(op, fname)				  \
namespace vsip {							  \
template <typename T1, typename T2>                                       \
inline                                                                    \
typename ovxx::detail::Dispatch_op_##fname<T1, T2>::result_type	          \
operator op(T1 const& t1, T2 const& t2)					  \
{ return ovxx::detail::Dispatch_op_##fname<T1, T2>::apply(t1, t2);}	  \
}

#define OVXX_BINARY_OPERATOR_TWO(op, fname)				  \
namespace vsip {                                                          \
template <template <typename, typename> class V,			  \
          typename T1, typename B, typename T2>                           \
inline                                                                    \
typename ovxx::detail::Dispatch_op_##fname<V<T1, B>, T2>::result_type     \
operator op(V<T1, B> const& t1, T2 t2)					  \
{ return ovxx::detail::Dispatch_op_##fname<V<T1, B>, T2>::apply(t1, t2);} \
                                                                          \
template <template <typename, typename> class V,                          \
          typename T1, typename T2, typename B>                           \
inline                                                                    \
typename ovxx::detail::Dispatch_op_##fname<T1, V<T2, B> >::result_type	  \
operator op(T1 t1, V<T2, B> const& t2)				          \
{ return ovxx::detail::Dispatch_op_##fname<T1, V<T2, B> >::apply(t1, t2);} \
                                                                          \
template <template <typename, typename> class V1,                         \
          template <typename, typename> class V2,                         \
          typename T1, typename B1,                                       \
          typename T2, typename B2>                                       \
inline                                                                    \
typename ovxx::detail::Dispatch_op_##fname<V1<T1, B1>,	  	          \
                                           V2<T2, B2> >::result_type      \
operator op(V1<T1, B1> const& t1, V2<T2, B2> const& t2)	                  \
{ return ovxx::detail::Dispatch_op_##fname<V1<T1, B1>,			  \
                                           V2<T2, B2> >::apply(t1, t2);}  \
}

#if (defined(__GNUC__) && __GNUC__ < 4) || defined(__ghs__) || defined(__ICL)
# define OVXX_BINARY_OPERATOR(op, fname) OVXX_BINARY_OPERATOR_ONE(op, fname)
#else
# define OVXX_BINARY_OPERATOR(op, fname) OVXX_BINARY_OPERATOR_TWO(op, fname)
#endif


#define OVXX_BINARY_VIEW_FUNCTION(f)					  \
namespace vsip {                                                          \
template <template <typename, typename> class V,                          \
          typename T, typename B>                                         \
inline                                                                    \
typename ovxx::detail::Dispatch_##f<V<T,B>, V<T,B> >::result_type	  \
f(V<T,B> const &t1, V<T,B> const &t2)				          \
{ return ovxx::detail::Dispatch_##f<V<T,B>, V<T,B> >::apply(t1, t2);}	  \
}

#define OVXX_BINARY_FUNC(f, F)						  \
OVXX_BINARY_FUNCTOR(f, F)					          \
OVXX_BINARY_DISPATCH(f, F)					          \
OVXX_BINARY_FUNCTION(f)						          \
OVXX_BINARY_VIEW_FUNCTION(f)

/// Binary function that map to binary operators. For those the functor
/// already exists.
#define OVXX_BINARY_FUNC_USEOP(f, F)					  \
OVXX_BINARY_DISPATCH(f, F)						  \
OVXX_BINARY_FUNCTION(f)					                  \
OVXX_BINARY_VIEW_FUNCTION(f)

#define OVXX_BINARY_FUNC_RETN(f, F, retn)				  \
OVXX_BINARY_FUNCTOR_RETN(f, F, retn)					  \
OVXX_BINARY_DISPATCH(f, F)					          \
OVXX_BINARY_FUNCTION(f)

#define OVXX_BINARY_FUNC_SCALAR_RETN(f, F)				  \
OVXX_BINARY_FUNCTOR_SCALAR_RETN(f, F)				          \
OVXX_BINARY_DISPATCH(f, F)					          \
OVXX_BINARY_FUNCTION(f)

#define OVXX_BINARY_OP(op, f, F)					  \
OVXX_BINARY_OP_DISPATCH(f, F)						  \
OVXX_BINARY_OPERATOR(op, f)

/// Macro to define a ternary function on views in terms of
/// its homologe on scalars.
#define OVXX_TERNARY_FUNC(f, F)						  \
namespace ovxx { namespace expr { namespace op {                          \
template <typename T1, typename T2, typename T3>                          \
struct F                                                                  \
{                                                                         \
  typedef typename vsip::Promotion<typename vsip::Promotion<T1, T2>::type,\
                             T3>::type result_type;                       \
  static result_type apply(T1 t1, T2 t2, T3 t3)                           \
  { return math::f(t1, t2, t3);}					  \
  result_type operator()(T1 t1, T2 t2, T3 t3) const                       \
  { return apply(t1, t2, t3);}                                            \
};                                                                        \
}}}                                                                       \
namespace ovxx { namespace detail {				          \
template <typename T1, typename T2, typename T3>                          \
struct Dispatch_##f :                                                     \
  conditional<is_view_type<T1>::value ||                                  \
           is_view_type<T2>::value ||                                     \
           is_view_type<T3>::value,                                       \
	   functors::ternary_view<expr::op::F, T1, T2, T3>,		  \
           expr::op::F<T1, T2, T3> >::type		                  \
{};                                                                       \
}}									  \
namespace vsip {							  \
template <typename T1, typename T2, typename T3>                          \
inline									  \
typename ovxx::detail::Dispatch_##f<T1, T2, T3>::result_type		  \
f(T1 const &t1, T2 const &t2, T3 const &t3)				  \
{ return ovxx::detail::Dispatch_##f<T1, T2, T3>::apply(t1, t2, t3);}	  \
}

/***********************************************************************
  Unary Functions
***********************************************************************/

OVXX_UNARY_FUNC(acos, Acos)

namespace ovxx { namespace expr { namespace op {
template <typename T> struct Arg {};

template <typename T>
struct Arg<std::complex<T> >
{
  typedef T result_type;
  static result_type apply(complex<T> t) { return math::arg(t);}
  result_type operator()(complex<T> t) const { return apply(t);}
};
}}}

OVXX_UNARY_DISPATCH(arg, Arg)
OVXX_UNARY_FUNCTION(arg, Arg)

OVXX_UNARY_FUNC(asin, Asin)
OVXX_UNARY_FUNC(atan, Atan)
OVXX_UNARY_FUNC(bnot, Bnot)
OVXX_UNARY_FUNC(ceil, Ceil)
OVXX_UNARY_FUNC(conj, Conj)
OVXX_UNARY_FUNC(cos, Cos)
OVXX_UNARY_FUNC(cosh, Cosh)

namespace ovxx { namespace expr { namespace op {
template <typename T>
struct Euler
{
  typedef complex<T> result_type;
  static result_type apply(T t) { return math::euler(t);}
  result_type operator()(T t) const { return apply(t);}
};
}}}

OVXX_UNARY_DISPATCH(euler, Euler)
OVXX_UNARY_FUNCTION(euler, Euler)

OVXX_UNARY_FUNC(exp, Exp)
OVXX_UNARY_FUNC(exp10, Exp10)
OVXX_UNARY_FUNC(floor, Floor)

namespace ovxx { namespace expr { namespace op {
template <typename T> struct Imag {};

template <typename T>
struct Imag<complex<T> >
{
  typedef T result_type;
  static result_type apply(complex<T> t) { return math::imag(t);}
  result_type operator()(complex<T> t) const { return apply(t);}
};
}}}

OVXX_UNARY_DISPATCH(imag, Imag)
OVXX_UNARY_FUNCTION(imag, Imag)

OVXX_UNARY_FUNC_RETN(is_finite, Is_finite, bool)
OVXX_UNARY_FUNC_RETN(is_nan, Is_nan, bool)
OVXX_UNARY_FUNC_RETN(is_normal, Is_normal, bool)

OVXX_UNARY_FUNC_RETN(lnot, Lnot, bool)
OVXX_UNARY_FUNC(log, Log)
OVXX_UNARY_FUNC(log10, Log10)
OVXX_UNARY_FUNC_RETN(mag, Mag, typename scalar_of<T>::type)
OVXX_UNARY_FUNC_RETN(magsq, Magsq, typename scalar_of<T>::type)
OVXX_UNARY_FUNC(neg, Neg)

namespace ovxx { namespace expr { namespace op {

template <typename T> struct Real {};

template <typename T>
struct Real<complex<T> >
{
  typedef T result_type;
  static result_type apply(complex<T> t) { return math::real(t);}
  result_type operator()(complex<T> t) const { return apply(t);}
};
}}}

OVXX_UNARY_DISPATCH(real, Real)
OVXX_UNARY_FUNCTION(real, Real)

OVXX_UNARY_FUNC(recip, Recip)
OVXX_UNARY_FUNC(rsqrt, Rsqrt)
OVXX_UNARY_FUNC(sin, Sin)
OVXX_UNARY_FUNC(sinh, Sinh)
OVXX_UNARY_FUNC(sq, Sq)
OVXX_UNARY_FUNC(sqrt, Sqrt)
OVXX_UNARY_FUNC(tan, Tan)
OVXX_UNARY_FUNC(tanh, Tanh)

OVXX_UNARY_FUNC(impl_conj, Impl_conj)

namespace ovxx { namespace expr { namespace op {
template <typename T>
struct Impl_real
{
  typedef typename scalar_of<T>::type result_type;
  static result_type apply(T t) { return math::impl_real(t);}
  result_type operator()(T t) const { return apply(t);}
};
}}}

OVXX_UNARY_DISPATCH(impl_real, Impl_real)
OVXX_UNARY_FUNCTION(impl_real, Impl_real)

namespace ovxx { namespace expr { namespace op {
template <typename T>
struct Impl_imag
{
  typedef typename scalar_of<T>::type result_type;
  static result_type apply(T t) { return math::impl_imag(t);}
  result_type operator()(T t) const { return apply(t);}
};
}}}

OVXX_UNARY_DISPATCH(impl_imag, Impl_imag)
OVXX_UNARY_FUNCTION(impl_imag, Impl_imag)

namespace ovxx { namespace expr { namespace op {
// This unary operator gives a hint to the compiler that this block is
// unaligned
template <typename T>
struct Unaligned
{
  typedef T result_type;
  static result_type apply(T t) { return t;}
  result_type operator()(T t) const { return apply(t);}
};
}}}

OVXX_UNARY_DISPATCH(unaligned, Unaligned)
OVXX_UNARY_FUNCTION(unaligned, Unaligned)

/***********************************************************************
  Binary Functions
***********************************************************************/

OVXX_BINARY_FUNC_USEOP(add, Add)
OVXX_BINARY_FUNC(atan2, Atan2)
OVXX_BINARY_FUNC(band, Band)
OVXX_BINARY_FUNC(bor, Bor)
OVXX_BINARY_FUNC(bxor, Bxor)
OVXX_BINARY_FUNC_USEOP(div, Div)
OVXX_BINARY_FUNC_RETN(eq, Eq, bool)
OVXX_BINARY_FUNC(fmod, Fmod)
OVXX_BINARY_FUNC_RETN(ge, Ge, bool)
OVXX_BINARY_FUNC_RETN(gt, Gt, bool)
OVXX_BINARY_FUNC(hypot, Hypot)
OVXX_BINARY_FUNC(jmul, Jmul)
OVXX_BINARY_FUNC_RETN(land, Land, bool)
OVXX_BINARY_FUNC_RETN(le, Le, bool)
OVXX_BINARY_FUNC_RETN(lt, Lt, bool)
OVXX_BINARY_FUNC_RETN(lor, Lor, bool)
OVXX_BINARY_FUNC_RETN(lxor, Lxor, bool)
OVXX_BINARY_FUNC_USEOP(mul, Mult)
OVXX_BINARY_FUNC(max, Max)
OVXX_BINARY_FUNC_SCALAR_RETN(maxmg, Maxmg)
OVXX_BINARY_FUNC_SCALAR_RETN(maxmgsq, Maxmgsq)
OVXX_BINARY_FUNC(min, Min)
OVXX_BINARY_FUNC(minmg, Minmg)
OVXX_BINARY_FUNC_SCALAR_RETN(minmgsq, Minmgsq)
OVXX_BINARY_FUNC_RETN(ne, Ne, bool)
OVXX_BINARY_FUNC(pow, Pow)
OVXX_BINARY_FUNC_USEOP(sub, Sub)

/***********************************************************************
  Ternary Functions
***********************************************************************/

OVXX_TERNARY_FUNC(am, Am)
OVXX_TERNARY_FUNC(expoavg, Expoavg)
OVXX_TERNARY_FUNC(ma, Ma)
OVXX_TERNARY_FUNC(msb, Msb)
OVXX_TERNARY_FUNC(sbm, Sbm)
OVXX_TERNARY_FUNC(ite, Ite)

/***********************************************************************
  Unary Operators
***********************************************************************/

OVXX_UNARY_OP(!, lnot)
OVXX_UNARY_OP(~, bnot)

/***********************************************************************
  Binary Operators
***********************************************************************/

OVXX_BINARY_OP(==, eq, Eq)
OVXX_BINARY_OP(>=, ge, Ge)
OVXX_BINARY_OP(>, gt, Gt)
OVXX_BINARY_OP(<=, le, Le)
OVXX_BINARY_OP(<, lt, Lt)
OVXX_BINARY_OP(!=, ne, Ne)
OVXX_BINARY_OP(&&, land, Land)
OVXX_BINARY_OP(&, band, Band)
OVXX_BINARY_OP(||, lor, Lor)
OVXX_BINARY_OP(|, bor, Bor)

namespace ovxx { namespace expr { namespace op {
template <typename T1, typename T2>
struct Bxor_or_lxor
{
  typedef typename Promotion<T1, T2>::type result_type;
  static char const *name() { return "bxor";}
  static result_type apply(T1 t1, T2 t2) { return math::bxor(t1, t2);}
  result_type operator()(T1 t1, T2 t2) const { return apply(t1, t2);}
};

template <>
struct Bxor_or_lxor<bool, bool>
{
  typedef bool result_type;
  static result_type apply(bool t1, bool t2) { return math::lxor(t1, t2);}
  result_type operator()(bool t1, bool t2) const { return apply(t1, t2);}
};
}}}

OVXX_BINARY_OP(^, bxor_or_lxor, Bxor_or_lxor)

#undef OVXX_TERNARY_FUNC
#undef OVXX_BINARY_FUNC_RETN
#undef OVXX_BINARY_FUNC
#undef OVXX_UNARY_FUNC_RETN
#undef OVXX_UNARY_FUNC
#undef OVXX_BINARY_FUNCTOR_RETN
#undef OVXX_BINARY_FUNCTOR
#undef OVXX_BINARY_DISPATCH
#undef OVXX_BINARY_FUNCTION
#undef OVXX_UNARY_FUNCTOR_RETN
#undef OVXX_UNARY_FUNCTOR
#undef OVXX_UNARY_DISPATCH
#undef OVXX_UNARY_FUNCTION

#endif
