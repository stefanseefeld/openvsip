//
// Copyright (c) 2005 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_CORE_SIGNAL_WINDOW_HPP
#define VSIP_CORE_SIGNAL_WINDOW_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/math.hpp>

/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

// Generates Blackman window
const_Vector<scalar_f>
blackman(length_type len) VSIP_THROW((std::bad_alloc));

// Generates Chebyshev window
const_Vector<scalar_f>
cheby(length_type len, scalar_f ripple) VSIP_THROW((std::bad_alloc));

// Generates Hanning window
const_Vector<scalar_f>
hanning(length_type len) VSIP_THROW((std::bad_alloc));

// Generates Kaiser window
const_Vector<scalar_f>
kaiser( length_type len, scalar_f beta ) VSIP_THROW((std::bad_alloc));


namespace impl
{

template <typename T, typename B1, typename B2>
void 
acosh(const_Vector<T, B1> x, Vector<std::complex<T>, B2> r)
{
  r = sq(x) - 1.0;
  r = log( x + sqrt(r) );
}


template <typename T>
T
bessel_I_0( T x )
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

  if ( static_cast<T>( fabs(x) ) <= 3.0 )
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
    while ( diff > .00000001 )
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

} // namespace impl

} // namespace vsip

#endif // VSIP_CORE_SIGNAL_WINDOW_HPP
