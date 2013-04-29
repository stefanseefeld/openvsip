/* Copyright (c) 2005 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/signal/window.hpp
    @author  Don McCoy
    @date    2005-09-15
    @brief   VSIPL++ Library: Window functions [signal.windows]
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/config.hpp>
#include <cmath>

#include <vsip/selgen.hpp>
#include <vsip/core/fft.hpp>
#include <vsip/core/signal/freqswap.hpp>
#include <vsip/core/signal/window.hpp>

/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{


/// Creates Blackman window.
/// Requires: len > 1.
/// Returns: A const_Vector initialized with Blackman window weights 
/// and having length len.
/// Throws: std::bad_alloc upon memory allocation error.
const_Vector<scalar_f>
blackman(length_type len) VSIP_THROW((std::bad_alloc))
{
  assert( len > 1 );

  Vector<scalar_f> v(len);

  scalar_f temp1 = 2 * VSIP_IMPL_PI / (len - 1);
  scalar_f temp2 = 2 * temp1;

  for ( index_type n = 0; n < len; ++n )
    v.put( n, 0.42 - 0.5 * cos(temp1 * n) + 0.08 * cos(temp2 * n) );

  return v;
}

namespace impl
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

} // namespace impl

/// Creates Chebyshev window with user-specified ripple.
/// Requires: len > 1.
/// Returns: A const_Vector initialized with Dolph-Chebyshev window 
/// weights and having length len.
/// Throws: std::bad_alloc upon memory allocation error.
const_Vector<scalar_f>
cheby(length_type len, scalar_f ripple) VSIP_THROW((std::bad_alloc))
{
  assert( len > 1 );

  scalar_f dp = pow( 10.0, -ripple / 20.0 );
  scalar_f df = acos( 1.0 / 
    cosh( impl::acosh( (1.0 + dp) / dp) / (len - 1.0) ) ) / VSIP_IMPL_PI;
  scalar_f x0 = (3.0 - cos( 2 * VSIP_IMPL_PI * df )) / (1.0 + cos( 2 * VSIP_IMPL_PI * df ));

  Vector<scalar_f> f = ramp(0.f, 1.f / len, len);

  scalar_f alpha = (x0 + 1.0) / 2.0;
  scalar_f beta  = (x0 - 1.0) / 2.0;
  Vector<scalar_f> x(len);
  x = alpha * cos( 2.0 * VSIP_IMPL_PI * f ) + beta;

  // tmp = (mag(x) > 1);
  Vector<scalar_f> tmp(len);
  for ( index_type i = 0; i < len; ++i )
    tmp.put( i, (((x.get(i) >= 0) ? x.get(i) : -x.get(i)) > 1) ? 1.0 : 0.0 );


  Vector<std::complex<scalar_f> > wf(len, 0.0);
  Vector<std::complex<scalar_f> >::realview_type wfR( wf.real() );
  Vector<std::complex<scalar_f> > Cfoo(len, 0.0);

  /* wf = dp*(tmp.*(cosh(((len-1.0)/2).*acosh(x)))+
     (1-tmp).*cos(((len-1.0)/2).*acos(x)));*/
  { 
    wfR = clip( x, -1.0, 1.0, -1.0, 1.0 );

    wfR = (1.0 - tmp) * cos( ((len - 1.0) / 2.0) * acos( wfR ) );
    impl::acosh( x, Cfoo );
    Cfoo = tmp * cosh( static_cast<scalar_f>((len - 1.0) / 2.0) * Cfoo );

    wf = dp * (Cfoo + wf);
  }

  int odd = len % 2;
  if ( !odd )
  {
    /*wf = real(wf).*exp(-j*pi*f);*/ 
    { 
      wf = wfR * euler( static_cast<scalar_f>(-1 * VSIP_IMPL_PI) * f );
    }

    /* wf(n2+1:len) = -wf(n2+1:len); */
    { 
      for ( index_type i = len / 2; i < len; ++i )
        wf.put( i, -wf.get(i) );
    }
  }

  /* wt = fft(wf); */
  { 
    typedef Fft<const_Vector, std::complex<scalar_f>, std::complex<scalar_f>, 
      fft_fwd, by_reference, 1, alg_noise> f_fft_type;

    f_fft_type f_fft( Domain<1>(len), 1.0 / len );
    f_fft(wf);
    
    std::complex<scalar_f> scale = wf.get(0);
    scale /= magsq(scale);
    wf = scale * wf;
  }
  
  Vector<scalar_f> ret(len);
  ret = freqswap(wfR);

  return ret;
}


        


/// Creates Hanning window.
/// Requires: len > 1.
/// Returns: A const_Vector initialized with Hanning window weights
/// and having length len. 
/// Throws: std::bad_alloc upon memory allocation error.
const_Vector<scalar_f>
hanning(length_type len) VSIP_THROW((std::bad_alloc))
{
  assert( len > 1 );

  Vector<scalar_f> v(len);

  scalar_f temp = 2 * VSIP_IMPL_PI / (len + 1);

  for ( unsigned int n = 0; n < len; ++n )
    v.put( n, 0.5 * (1 - static_cast<scalar_f>(cos(temp * (n + 1)))) );

  return v;
}



/// Creates Kaiser window.
/// Requires: len > 1.
/// Returns: A const_Vector initialized with Kaiser window weights 
/// with transition width parameter beta and having length len.
/// Throws: std::bad_alloc upon memory allocation error.
const_Vector<scalar_f>
kaiser( length_type len, scalar_f beta ) VSIP_THROW((std::bad_alloc))
{
  assert( len > 1 );

  scalar_f Ibeta;
  scalar_f x = beta;
  scalar_f c1 = 2.0 / (len -1 );

  Ibeta = impl::bessel_I_0(x);
 
  Vector<scalar_f> v(len);
  for ( length_type n = 0; n < len; ++n )
  {
    scalar_f c3 = c1 * n - 1;
    if ( c3 > 1.0 )
      c3 = 1.0;
    x = beta * static_cast<scalar_f>( sqrt(1 - (c3 * c3)) );
    v.put( n, impl::bessel_I_0(x) / Ibeta );
  }

  return v;
}



// The GreenHills compiler on the mercury does not automatically 
// instantiate templates until link time, which is too late for
// applications linking against VSIPL++.  We use pragmas to force
// the instantiation of templates necessary for this source file.
// We need to make sure that we don't force the instantiation of
// the same templates in multiple library files because that will
// result in multiple symbol definition errors.

#if defined(__ghs__)
#pragma instantiate float vsip::impl::bessel_I_0(float)

#pragma instantiate  Vector<float, Dense<1, float, row1_type, Local_map> > vsip::freqswap<Vector, float, Dense<1, float, row1_type, Local_map> >(Vector<float, Dense<1, float, row1_type, Local_map> >)

#pragma instantiate vsip::const_Vector<float, Dense<1, float, row1_type, Local_map> > vsip::impl::freqswap<float, Dense<1, float, row1_type, Local_map> >(vsip::const_Vector<float, Dense<1, float, row1_type, Local_map> >)

#pragma instantiate vsip::const_Vector<float, const vsip::impl::Generator_expr_block<(unsigned int)1, vsip::impl::Ramp_generator<float> > > vsip::ramp<float>(float, float, unsigned int)

#pragma instantiate vsip::impl::Clip_return_type<double, float, double, Vector, Dense<1, float, row1_type, Local_map> >::type vsip::clip<double, float, double, Vector, Dense<1, float, row1_type, Local_map> >(Vector<float, Dense<1, float, row1_type, Local_map> >, double, double, double, double)

#pragma instantiate void vsip::impl::acosh<float>(vsip::Vector<float, vsip::Dense<(unsigned int)1, float, vsip::tuple<(unsigned int)0, (unsigned int)1, (unsigned int)2>, vsip::Local_map> > &, vsip::Vector<std::complex<float>, vsip::Dense<(unsigned int)1, std::complex<float>, vsip::tuple<(unsigned int)0, (unsigned int)1, (unsigned int)2>, vsip::Local_map> > &)

#pragma instantiate Vector<complex<float>, impl::Strided<1, complex<float>, Layout<1, row1_type, impl::packing::dense, interleaved_complex>, Local_map> > vsip::impl::fft::new_view<Vector<complex<float>, impl::Strided<1, complex<float>, Layout<1, row1_type, impl::packing::dense, interleaved_complex>, Local_map> > >(const Domain<1>&)

#pragma instantiate Vector<complex<float>, impl::Strided<1, complex<float>, Layout<1, row1_type, impl::packing::dense, impl::split_complex>, Local_map> > vsip::impl::fft::new_view<Vector<complex<float>, impl::Strided<1, complex<float>, Layout<1, row1_type, impl::packing::dense, impl::split_complex>, Local_map> > >(const Domain<1>&)

#pragma instantiate bool vsip::impl::data_access::is_direct_ok<Dense<1, complex<float>, row1_type, Local_map>, impl::Rt_layout<1> >(const Dense<1, complex<float>, row1_type, Local_map> &, const impl::Rt_layout<1>  &)

#pragma instantiate bool vsip::impl::data_access::is_direct_ok<impl::Strided<1, complex<float>, Layout<1, row1_type, impl::packing::dense, interleaved_complex>, Local_map>, impl::Rt_layout<1> >(const impl::Strided<1, complex<float>, Layout<1, row1_type, impl::packing::dense, interleaved_complex>, Local_map> &, const impl::Rt_layout<1>&)

#pragma instantiate bool vsip::impl::data_access::is_direct_ok<impl::Strided<1, complex<float>, Layout<1, row1_type, impl::packing::dense, impl::split_complex>, Local_map>, impl::Rt_layout<1> >(const impl::Strided<1, complex<float>, Layout<1, row1_type, impl::packing::dense, impl::split_complex>, Local_map> &, const impl::Rt_layout<1>&)
#endif

} // namespace vsip

