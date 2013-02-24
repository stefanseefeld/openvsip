/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    fft.hpp
    @author  Jules Bergmann
    @date    30 Mar 2005
    @brief   VSIPL++ Library: Simple FFT function

    Crufty old FFT.
*/

#ifndef TEST_FFT_HPP
#define TEST_FFT_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/complex.hpp>

#include "fft-common.hpp"



/***********************************************************************
  Declarations
***********************************************************************/

namespace test_fft
{

inline unsigned
revbin_update(unsigned r, unsigned n)
{
  for (unsigned m=n>>1; (!((r^=m)&m)); m >>= 1)
    ;
  return r;
}



// -------------------------------------------------------------------- //
template <typename T,
	  typename Block>
void
revbin_permute(
  vsip::Vector<T, Block> A
  )
{
  int	n = A.size();
  int	r = 0;
  T	tmp;

  if (n <= 2) return;

  for (int x=1; x<n; x++)
  {
    r = revbin_update(r, n);
    if (r > x)
    {
      tmp = A.get(x);
      A.put(x, A.get(r));
      A.put(r, tmp);
    }
  }

}
   



// -------------------------------------------------------------------- //
template <typename T>
vsip::complex<T>
SinCos(double phi)
{
  return vsip::complex<T>(cos(phi), sin(phi));
}



// -------------------------------------------------------------------- //
// 1-dimensional FFT, using radix-2 decimation in frequency algorithm
// source: algorithms for programmers.

// Requires:
//   ? A to be a power-of-2 size FFT?
template <typename T,
	  typename Block>
void
fft1d(
  vsip::Vector<vsip::complex<T>, Block> A,
  int				        isign)
{
  vsip::complex<T>	e, u, v;

  int		n = A.size();
  int		ldn;
  int		i;

  /* m = (int) log2((double)n); */
  for (i=n,ldn=0; i>1; ldn++,i/=2)
    ;

  // printf("n=%d\n", n);
  // printf("ldn=%d\n", ldn);

  for (int ldm = ldn; ldm >= 1; --ldm)
  {
    int m  = (1<<ldm); // 2**ldm
    int mh = m/2;
    // printf("ldm: %d (m=%d  mh=%d)\n", ldm, m, mh);

    double phi = isign * (2*VSIP_IMPL_PI) / m;

    double s1 = 0.0;
    double c1 = 1.0;
    double a1 = sin(0.5*phi); a1 = 2.0*a1*a1;
    double b1 = sin(phi);
    double tmp;

    for (int j=0; j<mh; j++)
    {
      // printf("  j: %d\n", j);
      // e = exp(isign*2*VSIP_IMPL_PI*I*j/m);

      // e = SinCos<float>(phi * j);

      // compute sin,cos recursively (from AfP, p17)
      // printf("%d cos = %6.3f   %6.3f\n", j, c1, cos(phi*j));
      // printf("%d sin = %6.3f   %6.3f\n", j, s1, sin(phi*j));
      e = vsip::complex<T>(c1, s1);
      tmp = c1 - (a1*c1 + b1*s1);
      s1  = s1 - (a1*s1 - b1*c1);
      c1  = tmp;

      for (int r=0; r<n; r+=m)
      {
	// printf("    - u=%d v=%d\n", r+j, r+j+mh);
	u = A.get(r+j);
	v = A.get(r+j+mh);
	
	A.put(r+j,    (u + v));
	A.put(r+j+mh, (u - v) * e);
      }
    }
  }
  revbin_permute(A);
}

} // namespace test_fft




template <typename T,
	  typename Dir>
class Test_FFT
{
public:
  Test_FFT(vsip::Domain<1> size, float scale)
    : size_  (size.size()),
      scale_ (scale),
      timer_ ()
  {}

  template <typename Block>
  void operator()(vsip::Vector<T, Block> vec)
  {
    timer_.start();
    assert(vec.length() == size_);
    test_fft::fft1d(vec, Dir::isign);
    vec *= scale_;
    timer_.stop();
  }

  float mflops()
  {
    return timer_.count()*5.f*size_*logf(1.f*size_)/logf(2.f) /
           (1e6*timer_.total());
  }

  float compute_mflops()
  {
    return timer_.count()*5.f*size_*logf(1.f*size_)/logf(2.f) /
           (1e6*timer_.total());
  }

  // Member data.
private:
  vsip::length_type size_;
  float             scale_;
  AccTimer          timer_;
};

#endif // TEST_FFT_HPP
