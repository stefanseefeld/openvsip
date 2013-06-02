//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include <ovxx/config.hpp>
#include <vsip/selgen.hpp>
#include <vsip/impl/signal/fft.hpp>
#include <vsip/impl/signal/freqswap.hpp>
#include <ovxx/signal/window.hpp>
#include <vsip/parallel.hpp>
#include <cmath>

namespace ovxx
{
namespace signal
{
namespace
{

#if HAVE_DECL_ACOSH
// If the C++ library provides ::acosh, we'll use it.
using ::acosh;
#else // !HAVE_DECL_ACOSH
// Otherwise, we have to provide our own version.
inline double
acosh(double f)
{
  return log(f + sqrt(f+1)*sqrt(f-1));
}
#endif

template <typename T, typename B1, typename B2>
void 
acosh(const_Vector<T, B1> x, Vector<complex<T>, B2> r)
{
  r = sq(x) - 1.0;
  r = log(x + sqrt(r));
}

template <typename T>
T
bessel_I_0(T x)
{
  // If -3 <= x < 3, then use polynomial approximation to compute
  // the I_0(x) (modified bessel function of the first kind).
  //
  // This approximation is accurate to withing 1.6 * 10^-7  
  // [See Abramowitz and Stegun p378, S9.8.1 -- note that
  //  t = beta * 3 / 3.75, which accounts for the difference
  //  in the way the constants appear.]
  //
  // Otherwise, use iterative method.
  //
  const T a1 = 2.2499997;
  const T a2 = 1.2656208;
  const T a3 = 0.3163866;
  const T a4 = 0.0444479;
  const T a5 = 0.0039444;
  const T a6 = 0.0002100;
  T ans;

  if (static_cast<T>(fabs(x)) <= 3.0)
  {
    x /= 3.0;   
    x *= x; 
    ans = 1 + x * (a1 + x * (a2 + x * (a3 + x * (a4 + x * (a5 + x * a6)))));
  }
  else
  {
    T x1 = x * x * .25;
    T x0 = x1;
    T n0 = 1;
    T diff = 1;
    ans = 1 + x1;
    
    length_type n = 1;
    while (diff > .00000001)
    {
      n++;
      n0 *= static_cast<T>(n);
      x1 *= x0;
      diff = x1 / (n0 * n0);
      ans += diff;
    }
  }
  return ans;
}

} // namespace <unnamed>

// Generates Blackman window
template <typename T>
const_Vector<T>
blackman(length_type len)
{
  OVXX_PRECONDITION(len > 1);

  Vector<T> v(len);

  T temp1 = 2 * OVXX_PI / (len - 1);
  T temp2 = 2 * temp1;

  for (index_type n = 0; n < len; ++n)
    v.put(n, 0.42 - 0.5 * cos(temp1 * n) + 0.08 * cos(temp2 * n));

  return v;
}

// Generates Chebyshev window
template <typename T>
const_Vector<T>
cheby(length_type len, T ripple)
{
  OVXX_PRECONDITION(len > 1);

  T dp = pow(10.0, -ripple / 20.0);
  T df = acos(1.0 / 
    cosh(acosh((1.0 + dp) / dp) / (len - 1.0))) / OVXX_PI;
  T x0 = (3.0 - cos(2 * OVXX_PI * df)) / (1.0 + cos(2 * OVXX_PI * df));

  Vector<T> f = ramp(0.f, 1.f / len, len);

  T alpha = (x0 + 1.0) / 2.0;
  T beta  = (x0 - 1.0) / 2.0;
  Vector<T> x(len);
  x = alpha * cos(2.0 * OVXX_PI * f) + beta;

  // tmp = (mag(x) > 1);
  Vector<T> tmp(len);
  for (index_type i = 0; i < len; ++i)
    tmp.put(i, (((x.get(i) >= 0) ? x.get(i) : -x.get(i)) > 1) ? 1.0 : 0.0);

  Vector<complex<T> > wf(len, 0.0);
  typename Vector<complex<T> >::realview_type wfR(wf.real());
  Vector<complex<T> > Cfoo(len, 0.0);

  /* wf = dp*(tmp.*(cosh(((len-1.0)/2).*acosh(x)))+
     (1-tmp).*cos(((len-1.0)/2).*acos(x)));*/
  { 
    wfR = clip(x, -1.0, 1.0, -1.0, 1.0);
    wfR = (1.0 - tmp) * cos(((len - 1.0) / 2.0) * acos(wfR));
    acosh(x, Cfoo);
    Cfoo = tmp * cosh(static_cast<T>((len - 1.0) / 2.0) * Cfoo);
    wf = dp * (Cfoo + wf);
  }

  int odd = len % 2;
  if (!odd)
  {
    /*wf = real(wf).*exp(-j*pi*f);*/ 
    { 
      wf = wfR * euler(static_cast<scalar_f>(-1 * OVXX_PI) * f);
    }

    /* wf(n2+1:len) = -wf(n2+1:len); */
    { 
      for (index_type i = len / 2; i < len; ++i)
        wf.put(i, -wf.get(i));
    }
  }

  /* wt = fft(wf); */
  { 
    typedef vsip::Fft<const_Vector, complex<T>, complex<T>, 
      fft_fwd, by_reference, 1, alg_noise> f_fft_type;

    f_fft_type f_fft(Domain<1>(len), 1.0 / len);
    f_fft(wf);
    
    complex<T> scale = wf.get(0);
    scale /= magsq(scale);
    wf = scale * wf;
  }
  
  Vector<T> ret(len);
  ret = freqswap(wfR);

  return ret;
}


// Generates Hanning window
template <typename T>
const_Vector<T>
hanning(length_type len)
{
  OVXX_PRECONDITION(len > 1);

  Vector<T> v(len);

  T temp = 2 * OVXX_PI / (len + 1);

  for (unsigned int n = 0; n < len; ++n)
    v.put(n, 0.5 * (1 - static_cast<T>(cos(temp * (n + 1)))));
  return v;
}

// Generates Kaiser window
template <typename T>
const_Vector<T>
kaiser(length_type len, T beta)
{
  OVXX_PRECONDITION(len > 1);

  T Ibeta;
  T x = beta;
  T c1 = 2.0 / (len-1);

  Ibeta = bessel_I_0(x);
 
  Vector<T> v(len);
  for (length_type n = 0; n < len; ++n)
  {
    T c3 = c1 * n - 1;
    if (c3 > 1.0)
      c3 = 1.0;
    x = beta * static_cast<T>(sqrt(1 - (c3 * c3)));
    v.put(n, bessel_I_0(x) / Ibeta);
  }

  return v;
}

template
const_Vector<float>
blackman(length_type len) VSIP_THROW((std::bad_alloc));
template
const_Vector<double>
blackman(length_type len) VSIP_THROW((std::bad_alloc));

template
const_Vector<float>
cheby(length_type len, float ripple) VSIP_THROW((std::bad_alloc));
template
const_Vector<double>
cheby(length_type len, double ripple) VSIP_THROW((std::bad_alloc));

template
const_Vector<float>
hanning(length_type len) VSIP_THROW((std::bad_alloc));
template
const_Vector<double>
hanning(length_type len) VSIP_THROW((std::bad_alloc));

template
const_Vector<float>
kaiser(length_type len, float beta) VSIP_THROW((std::bad_alloc));
template
const_Vector<double>
kaiser(length_type len, double beta) VSIP_THROW((std::bad_alloc));

} // namespace ovxx::signal
} // namespace ovxx

