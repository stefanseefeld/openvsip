/* Copyright (c) 2005 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/fns_scalar.hpp
    @author  Stefan Seefeld
    @date    2005-04-21
    @brief   VSIPL++ Library: [math.fns.scalar]

*/

#ifndef VSIP_CORE_FNS_SCALAR_HPP
#define VSIP_CORE_FNS_SCALAR_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/config.hpp>
#include <vsip/support.hpp>
#include <vsip/core/promote.hpp>
#include <vsip/core/metaprogramming.hpp>
#include <cmath>
#include <cstdlib> // ghs imports ::abs into std here.
#include <complex>

#if !HAVE_DECL_HYPOTF
# if HAVE_HYPOTF
extern "C" float hypotf(float, float);
# endif
#endif
#if !HAVE_DECL_HYPOT
# if HAVE_HYPOT
extern "C" double hypot(double, double);
# endif
#endif

namespace vsip_csl
{
namespace fn
{

/***********************************************************************
 Scalar functions
***********************************************************************/

using std::acos;
using std::arg;
using std::asin;
using std::atan;

template <typename T>
inline T 
bnot(T t) VSIP_NOTHROW { return ~t;}

using std::ceil;
using std::conj;
using std::cos;
using std::cosh;

/// Some general comments.
///
/// :Returns:
///   The complex number corresponding to the angle of a unit vector in the
///   complex plane, i.e., exp(j * x) for argument x .
template <typename T>
inline complex<T>
euler (T x) VSIP_NOTHROW { return std::polar(T (1), x);}

using std::log;
using std::log10;

using std::exp;

#if HAVE_DECL_EXP10
// If the C++ library provides ::exp10 (as an extension to ISO C++),
// we can use it.
using ::exp10;
#else // !HAVE_DECL_EXP10
// If the C++ library does not provide ::exp10 we have to provide our
// own version.
inline double 
exp10(double d) VSIP_NOTHROW {
  return exp(d * log(10.0));
}
#endif // !HAVE_DECL_EXP10

// Provide an overloaded version of exp10 that accepts a "float".
inline float
exp10(float f) VSIP_NOTHROW {
#if HAVE_DECL_EXP10F
  return exp10f (f);
#else // !HAVE_DECL_EXP10F
  return (float) exp10((double) f);
#endif // !HAVE_DECL_EXP10F
}

// Provide an overloaded version of exp10 that accepts a "long
// double".
inline long double
exp10(long double ld) VSIP_NOTHROW {
#if HAVE_DECL_EXP10L
  return exp10l (ld);
#else // !HAVE_DECL_EXP10L
  // ISO C++ requires that there an overload form of "exp" that 
  // takes a "long double" argument.  
  return exp(ld * log(10.0l));
#endif // !HAVE_DECL_EXP10L
}

template <typename T1,
	  typename T2,
	  typename T3>
inline typename Promotion<T1, typename Promotion<T2, T3>::type>::type
expoavg (T1 a,
	 T2 b,
	 T3 c) VSIP_NOTHROW
{
  return a * b + (T1(1) - a) * c;
}

using std::floor;
using std::imag;

template <typename T>
inline T 
ite(bool pred, T a, T b) VSIP_NOTHROW { return pred ? a : b;}

// isfinite, isnan, and isnormal are macros provided by C99 <math.h>
// They are not part of C++ <cmath>.
//
// GCC's cmath captures them, removing the macros, and providing
// functions std::isfinite, std::isnan, and std::isnormal.
//
// GreenHills on MCOE only provides macros.
//
// 070502: MCOE GCC 3.4.4 does not capture them.

// Pull isfinite, isnan, and isnormal into fn namespace so Fp_traits
// can see them.
#if HAVE_STD_ISFINITE
using std::isfinite;
#endif
#if HAVE_STD_ISNAN
using std::isnan;
#endif
#if HAVE_STD_ISNORMAL
using std::isnormal;
#endif

template <typename T>
struct Fp_traits
{
  static bool is_finite(T val)
  {
#ifdef  _MSC_VER
    return _isfinite(val); 
#else
    return isfinite(val); 
#endif
  }
  static bool is_nan(T val)    
  {
#ifdef  _MSC_VER
    return _isnan(val);
#else
    return isnan(val);
#endif
  }
  static bool is_normal(T val) { return isnormal(val); }
};

template <typename T>
 struct Fp_traits<complex<T> >
{
  static bool is_finite(complex<T> const& val)
  {
#ifdef  _MSC_VER
    return _isfinite(val.real()) && _isfinite(val.imag()); 
#else
    return isfinite(val.real()) && isfinite(val.imag()); 
#endif
  }

  static bool is_nan(complex<T> const& val)    
  {
#ifdef  _MSC_VER
    return _isnan(val.real()) || _isnan(val.imag()); 
#else
    return isnan(val.real()) || isnan(val.imag()); 
#endif
  }

  static bool is_normal(complex<T> const& val) 
  { return isnormal(val.real()) && isnormal(val.imag()); }
};


// is_finite -- returns nonzero/true if x is finite not plus or minus inf,
//              and not NaN.

template <typename T>
inline bool is_finite(T val)
{
  return Fp_traits<T>::is_finite(val);
}


// is_nan -- returns nonzero/true if x is NaN.

template <typename T>
inline bool is_nan(T val)
{
  return Fp_traits<T>::is_nan(val);
}


// isnormal -- returns nonzero/true if x is finite and normalized. (C99).

template <typename T>
inline bool
is_normal(T val)
{
  return Fp_traits<T>::is_normal(val);
}



template <typename T>
inline T 
lnot(T t) VSIP_NOTHROW { return !t;}

namespace abs_detail
{

// GreenHills <cmath> defines ::abs(float) and ::abs(double), but does
// not place them into the std namespace when targeting mercury (when
// _MC_EXEC is defined).

#if VSIP_IMPL_FIX_MISSING_ABS
using ::abs;
#endif
using std::abs;
} // namespace abs_detail

template <typename T>
inline T 
sq(T t) VSIP_NOTHROW { return t*t;}

template <typename T>
inline typename impl::scalar_of<T>::type 
mag(T t) VSIP_NOTHROW { return abs_detail::abs(t);}

namespace magsq_detail
{

template <typename T, typename R>
struct Magsq_impl_base
{
  static R exec(T val) { return val*val; }
};

template <typename T, typename R>
struct Magsq_impl_base<complex<T>, R>
{
  static R exec(complex<T> const& val)
  { return sq(val.real()) + sq(val.imag()); }
};

template <typename T>
struct Magsq_impl : public Magsq_impl_base<T,T> {};

template <typename T>
struct Magsq_impl<complex<T> >
{
  static T exec(complex<T> const& val)
  { return sq(val.real()) + sq(val.imag()); }
};

template <typename R>
struct Magsq_impl_helper
{
  template <typename T>
  struct Magsq_impl : public Magsq_impl_base<T,R> {};
};

} // namespace magsq_detail

template <typename T>
inline typename impl::scalar_of<T>::type 
magsq(T t) VSIP_NOTHROW 
{
  return magsq_detail::Magsq_impl<T>::exec(t);
}

template <typename T, typename ResultT>
inline ResultT
magsq(T t, ResultT) VSIP_NOTHROW 
{
  return magsq_detail::Magsq_impl_helper<ResultT>::template Magsq_impl<T>::exec(t);
}

template <typename T>
inline T
neg(T t) VSIP_NOTHROW { return -t;}

using std::real;

template <typename T>
inline T 
recip(T t) VSIP_NOTHROW { return T(1)/t;}

template <typename T>
inline T 
rsqrt(T t) VSIP_NOTHROW { return T(1)/sqrt(t);}

using std::sin;
using std::sinh;

using std::sqrt;
using std::tan;
using std::tanh;

template <typename T1, typename T2>
inline typename Promotion<T1, T2>::type
add(T1 t1, T2 t2) VSIP_NOTHROW { return t1 + t2;}

using std::atan2;

template <typename T1, typename T2>
inline typename Promotion<T1, T2>::type
band(T1 t1, T2 t2) VSIP_NOTHROW { return t1 & t2;}

template <typename T1, typename T2>
inline typename Promotion<T1, T2>::type
bxor(T1 t1, T2 t2) VSIP_NOTHROW { return t1 ^ t2;}

using std::div;
using std::fmod;

template <typename T1, typename T2>
inline bool
ge(T1 t1, T2 t2) VSIP_NOTHROW { return t1 >= t2;}

template <typename T1, typename T2>
inline bool
gt(T1 t1, T2 t2) VSIP_NOTHROW { return t1 > t2;}

inline double
hypot(double t1, double t2) VSIP_NOTHROW {
#if HAVE_HYPOT
  return ::hypot(t1, t2);
#else
  return sqrt(sq(t1) + sq(t2));
#endif
}

inline float
hypot(float t1, float t2) VSIP_NOTHROW 
{
#if HAVE_HYPOTF
  return ::hypotf(t1, t2);
#else
  return hypot((double)t1, (double)t2);
#endif
}

template <typename T1, typename T2>
inline bool
land(T1 t1, T2 t2) VSIP_NOTHROW { return t1 && t2;}

template <typename T1, typename T2>
inline bool
lor(T1 t1, T2 t2) VSIP_NOTHROW { return t1 || t2;}

template <typename T1, typename T2>
inline bool
lxor(T1 t1, T2 t2) VSIP_NOTHROW { return (t1 && !t2) || (!t1 && t2);}

template <typename T1, typename T2>
inline typename Promotion<complex<T1>, complex<T2> >::type
jmul(complex<T1> t1, complex<T2> t2) VSIP_NOTHROW 
{
  typename Promotion<complex<T1>, complex<T2> >::type retn(t1);
  t1 *= conj(t2);
  return t1;
}

template <typename T1, typename T2>
inline bool
le(T1 t1, T2 t2) VSIP_NOTHROW { return t1 <= t2;}

template <typename T1, typename T2>
inline bool
lt(T1 t1, T2 t2) VSIP_NOTHROW { return t1 < t2;}

using std::max;

template <typename T1, typename T2>
inline typename Promotion<T1, T2>::type
impl_max(T1 t1, T2 t2) VSIP_NOTHROW { return t1 > t2 ? t1 : t2; }

template <typename T1, typename T2>
inline typename impl::scalar_of<typename Promotion<T1, T2>::type>::type
maxmg(T1 t1, T2 t2) VSIP_NOTHROW { return impl_max(mag(t1), mag(t2));}

template <typename T1, typename T2>
inline typename impl::scalar_of<typename Promotion<T1, T2>::type>::type
maxmgsq(T1 t1, T2 t2) VSIP_NOTHROW { return impl_max(fn::magsq(t1), fn::magsq(t2));}

using std::min;

template <typename T1, typename T2>
inline typename Promotion<T1, T2>::type
impl_min(T1 t1, T2 t2) VSIP_NOTHROW { return t1 < t2 ? t1 : t2; }

template <typename T1, typename T2>
inline typename impl::scalar_of<typename Promotion<T1, T2>::type>::type
minmg(T1 t1, T2 t2) VSIP_NOTHROW { return impl_min(mag(t1), mag(t2));}

template <typename T1, typename T2>
inline typename impl::scalar_of<typename Promotion<T1, T2>::type>::type
minmgsq(T1 t1, T2 t2) VSIP_NOTHROW { return impl_min(fn::magsq(t1), fn::magsq(t2));}

using std::pow;

#ifdef __GXX_EXPERIMENTAL_CXX0X__
// in C++11, std::pow<float, int> returns double... (see DR 550)
// As this isn't compatible with the VSIPL++ spec, we need to provide
// our own version.
inline complex<float> pow(complex<float> const &x, int y)
{ return complex<float>(std::pow(x, y));}
#endif

template <typename T1, typename T2>
inline typename Promotion<T1, T2>::type
sub(T1 t1, T2 t2) VSIP_NOTHROW { return t1 - t2;}

template <typename T1, typename T2>
inline typename Promotion<T1, T2>::type
mul(T1 t1, T2 t2) VSIP_NOTHROW { return t1 * t2;}

template <typename T1, typename T2>
inline typename Promotion<T1, T2>::type
div(T1 t1, T2 t2) VSIP_NOTHROW { return t1 / t2;}

template <typename T1, typename T2>
inline bool 
eq(T1 t1, T2 t2) VSIP_NOTHROW { return t1 == t2;}

template <typename T1, typename T2>
inline bool 
ne(T1 t1, T2 t2) VSIP_NOTHROW { return !eq(t1, t2);}

template <typename T1, typename T2>
inline typename Promotion<T1, T2>::type
bor(T1 t1, T2 t2) VSIP_NOTHROW { return t1 | t2;}

template <typename T1, typename T2, typename T3>
inline typename Promotion<typename Promotion<T1, T2>::type, T3>::type
am(T1 t1, T2 t2, T3 t3) VSIP_NOTHROW { return (t1 + t2) * t3;}

template <typename T1, typename T2, typename T3>
inline typename Promotion<typename Promotion<T1, T2>::type, T3>::type
ma(T1 t1, T2 t2, T3 t3) VSIP_NOTHROW { return t1 * t2 + t3;}

template <typename T1, typename T2, typename T3>
inline typename Promotion<typename Promotion<T1, T2>::type, T3>::type
msb(T1 t1, T2 t2, T3 t3) VSIP_NOTHROW { return t1 * t2 - t3;}

template <typename T1, typename T2, typename T3>
inline typename Promotion<typename Promotion<T1, T2>::type, T3>::type
sbm(T1 t1, T2 t2, T3 t3) VSIP_NOTHROW { return (t1 - t2) * t3;}

template <typename T>
struct Impl_complex_class
{
  static T conj(T val) { return val; }
  static T real(T val) { return val; }
  static T imag(T val) { return T(); }
};

template <typename T>
struct Impl_complex_class<complex<T> >
{
  static complex<T> conj(complex<T> val) { return std::conj(val); }
  static T real(complex<T> val) { return val.real(); }
  static T imag(complex<T> val) { return val.imag(); }
};

template <typename T>
inline T
impl_conj(T val) { return Impl_complex_class<T>::conj(val); }

template <typename T>
inline typename impl::scalar_of<T>::type
impl_real(T val) { return Impl_complex_class<T>::real(val); }

template <typename T>
inline typename impl::scalar_of<T>::type
impl_imag(T val) { return Impl_complex_class<T>::imag(val); }

} // namespace vsip_csl::fn
} // namespace vsip_csl

namespace vsip
{
namespace impl
{
namespace fn = vsip_csl::fn;
} // namespace vsip::impl
} // namespace vsip

#endif
