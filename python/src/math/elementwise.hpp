//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef elementwise_hpp_
#define elementwise_hpp_

#include <ovxx/python/block.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>

namespace pyvsip
{
using namespace ovxx;
using namespace ovxx::python;

#define UNARY(F)					\
template <dimension_type D, typename T>		        \
inline bpl::object F(Block<D, T> const &b)		\
{							\
  typedef Block<D, T> B;				\
  typename view_of<B>::const_type v(const_cast<B&>(b)); \
  Domain<D> dom = ovxx::block_domain<D>(b);		\
  B *result = new B(dom);				\
  typename view_of<B>::type r(*result);			\
  r = vsip::F(v);					\
  return bpl::object(boost::shared_ptr<B>(result));	\
}

#define UNARY_RETN(F, R)					\
template <dimension_type D, typename T>		                \
inline bpl::object F(Block<D, T> const &b)			\
{								\
  typedef Block<D, T> B;					\
  typename view_of<B>::const_type v(const_cast<B&>(b));		\
  Domain<D> dom = ovxx::block_domain<D>(b);			\
  Block<D, R> *result = new Block<D, R>(dom);			\
  typename view_of<Block<D, R> >::type r(*result);		\
  r = ovxx::F(v);						\
  return bpl::object(boost::shared_ptr<Block<D, R> >(result));	\
}

#define BINARY(F)							\
template <dimension_type D, typename T>		                        \
inline bpl::object F(Block<D, T> const &b1, Block<D, T> const &b2)	\
{									\
  typedef Block<D, T> B;						\
  typename view_of<B>::const_type v1(const_cast<B&>(b1));		\
  typename view_of<B>::const_type v2(const_cast<B&>(b2));		\
  Domain<D> dom = ovxx::block_domain<D>(b1);				\
  B *result = new B(dom);						\
  typename view_of<B>::type r(*result);					\
  r = ovxx::F(v1, v2);							\
  return bpl::object(boost::shared_ptr<B>(result));			\
}

#define BINARY_RETN(F, R)						\
template <dimension_type D, typename T>		                        \
inline bpl::object F(Block<D, T> const &b1, Block<D, T> const &b2)	\
{									\
  typedef Block<D, T> B;						\
  typename view_of<B>::const_type v1(const_cast<B&>(b1));		\
  typename view_of<B>::const_type v2(const_cast<B&>(b2));		\
  Domain<D> dom = ovxx::block_domain<D>(b1);				\
  Block<D, R> *result = new Block<D, R>(dom);				\
  typename view_of<Block<D, R> >::type r(*result);			\
  r = ovxx::F(v1, v2);							\
  return bpl::object(boost::shared_ptr<Block<D, R> >(result));		\
}

#define TERNARY(F)							                  \
template <dimension_type D, typename T>				                          \
inline bpl::object F(Block<D, T> const &b1, Block<D, T> const &b2, Block<D, T> const &b3) \
{									                  \
  typedef Block<D, T> B;						                  \
  typename view_of<B>::const_type v1(const_cast<B&>(b1));		                  \
  typename view_of<B>::const_type v2(const_cast<B&>(b2));		                  \
  typename view_of<B>::const_type v3(const_cast<B&>(b3));		                  \
  Domain<D> dom = ovxx::block_domain<D>(b1);				                  \
  B *result = new B(dom);						                  \
  typename view_of<B>::type r(*result);					                  \
  r = ovxx::F(v1, v2, v3);							          \
  return bpl::object(boost::shared_ptr<B>(result));			                  \
}


BINARY(add)
UNARY(acos)
TERNARY(am)
UNARY_RETN(arg, typename ovxx::scalar_of<T>::type)
UNARY(asin)
UNARY(atan)
BINARY(atan2)
UNARY(ceil)
UNARY(conj)
UNARY(cos)
UNARY(cosh)
BINARY(div)
UNARY_RETN(euler, vsip::complex<T>)
BINARY_RETN(eq, bool)
UNARY(exp)
UNARY(exp10)
TERNARY(expoavg)
UNARY(floor)
BINARY(fmod)
BINARY_RETN(ge, bool)
BINARY_RETN(gt, bool)
BINARY(hypot)
UNARY_RETN(imag, typename ovxx::scalar_of<T>::type)
UNARY_RETN(is_finite, bool)
UNARY_RETN(is_nan, bool)
UNARY_RETN(is_normal, bool)
TERNARY(ite)
BINARY(jmul)
BINARY_RETN(land, bool)
BINARY_RETN(le, bool)
UNARY_RETN(lnot, bool)
UNARY(log)
UNARY(log10)
BINARY_RETN(lor, bool)
BINARY_RETN(lt, bool)
BINARY_RETN(lxor, bool)
TERNARY(ma)
UNARY_RETN(mag, typename ovxx::scalar_of<T>::type)
UNARY_RETN(magsq, typename ovxx::scalar_of<T>::type)
BINARY(max)
BINARY_RETN(maxmg, typename ovxx::scalar_of<T>::type)
BINARY_RETN(maxmgsq, typename ovxx::scalar_of<T>::type)
BINARY(min)
BINARY_RETN(minmg, typename ovxx::scalar_of<T>::type)
BINARY_RETN(minmgsq, typename ovxx::scalar_of<T>::type)
TERNARY(msb)
BINARY(mul)
BINARY_RETN(ne, bool)
UNARY(neg)
BINARY(pow)
UNARY_RETN(real, typename ovxx::scalar_of<T>::type)
UNARY(recip)
UNARY(rsqrt)
TERNARY(sbm)
UNARY(sin)
UNARY(sinh)
UNARY(sq)
UNARY(sqrt)
BINARY(sub)
UNARY(tan)
UNARY(tanh)

template <dimension_type D, typename T>
void define_elementwise()
{
  typedef Block<D, T> block_type;
  bpl::def("add", add<D, T>);
  bpl::def("acos", acos<D, T>);
  bpl::def("am", am<D, T>);
  bpl::def("asin", asin<D, T>);
  bpl::def("atan", atan<D, T>);
  bpl::def("atan2", atan2<D, T>);
  bpl::def("ceil", ceil<D, T>);
  bpl::def("cos", cos<D, T>);
  bpl::def("cosh", cosh<D, T>);
  bpl::def("div", div<D, T>);
  bpl::def("eq", eq<D, T>);
  bpl::def("exp", exp<D, T>);
  bpl::def("exp10", exp10<D, T>);
  bpl::def("expoavg", expoavg<D, T>);
  bpl::def("euler", euler<D, T>);
  bpl::def("floor", floor<D, T>);
  bpl::def("fmod", fmod<D, T>);
  bpl::def("ge", ge<D, T>);
  bpl::def("gt", gt<D, T>);
  bpl::def("hypot", hypot<D, T>);
  bpl::def("is_finite", is_finite<D, T>);
  bpl::def("is_nan", is_nan<D, T>);
  bpl::def("is_normal", is_normal<D, T>);
  bpl::def("ite", ite<D, T>);
  bpl::def("land", land<D, T>);
  bpl::def("le", le<D, T>);
  bpl::def("lnot", lnot<D, T>);
  bpl::def("log", log<D, T>);
  bpl::def("log10", log10<D, T>);
  bpl::def("lor", lor<D, T>);
  bpl::def("lt", lt<D, T>);
  bpl::def("lxor", lxor<D, T>);
  bpl::def("ma", ma<D, T>);
  bpl::def("mag", mag<D, T>);
  bpl::def("magsq", magsq<D, T>);
  bpl::def("max", max<D, T>);
  bpl::def("maxmg", maxmg<D, T>);
  bpl::def("maxmgsq", maxmgsq<D, T>);
  bpl::def("min", min<D, T>);
  bpl::def("minmg", minmg<D, T>);
  bpl::def("minmgsq", minmgsq<D, T>);
  bpl::def("msb", msb<D, T>);
  bpl::def("mul", mul<D, T>);
  bpl::def("ne", ne<D, T>);
  bpl::def("neg", neg<D, T>);
  bpl::def("pow", pow<D, T>);
  bpl::def("recip", recip<D, T>);
  bpl::def("rsqrt", rsqrt<D, T>);
  bpl::def("sbm", sbm<D, T>);
  bpl::def("sin", sin<D, T>);
  bpl::def("sinh", sinh<D, T>);
  bpl::def("sq", sq<D, T>);
  bpl::def("sqrt", sqrt<D, T>);
  bpl::def("sub", sub<D, T>);
  bpl::def("tan", tan<D, T>);
  bpl::def("tanh", tanh<D, T>);
}

template <dimension_type D, typename T>
void define_complex_elementwise()
{
  typedef Block<D, T> block_type;
  bpl::def("am", am<D, T>);
  bpl::def("add", add<D, T>);
  bpl::def("arg", arg<D, T>);
  bpl::def("conj", conj<D, T>);
  bpl::def("cos", cos<D, T>);
  bpl::def("cosh", cosh<D, T>);
  bpl::def("div", div<D, T>);
  bpl::def("eq", eq<D, T>);
  bpl::def("exp", exp<D, T>);
  bpl::def("expoavg", expoavg<D, T>);
  bpl::def("imag", imag<D, T>);
  bpl::def("jmul", jmul<D, T>);
  bpl::def("real", real<D, T>);
  bpl::def("is_finite", is_finite<D, T>);
  bpl::def("is_nan", is_nan<D, T>);
  bpl::def("is_normal", is_normal<D, T>);
  bpl::def("log", log<D, T>);
  bpl::def("log10", log10<D, T>);
  bpl::def("ma", ma<D, T>);
  bpl::def("mag", mag<D, T>);
  bpl::def("magsq", magsq<D, T>);
  bpl::def("maxmg", maxmg<D, T>);
  bpl::def("maxmgsq", maxmgsq<D, T>);
  bpl::def("msb", msb<D, T>);
  bpl::def("mul", mul<D, T>);
  bpl::def("ne", ne<D, T>);
  bpl::def("neg", neg<D, T>);
  bpl::def("pow", pow<D, T>);
  bpl::def("recip", recip<D, T>);
  bpl::def("rsqrt", rsqrt<D, T>);
  bpl::def("sbm", sbm<D, T>);
  bpl::def("sin", sin<D, T>);
  bpl::def("sinh", sinh<D, T>);
  bpl::def("sq", sq<D, T>);
  bpl::def("sqrt", sqrt<D, T>);
  bpl::def("sub", sub<D, T>);
  bpl::def("tan", tan<D, T>);
  bpl::def("tanh", tanh<D, T>);
}

} // namespace pyvsip

#endif
