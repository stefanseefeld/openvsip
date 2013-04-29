//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/signal.hpp>
#include <vsip/vector.hpp>
#include <vsip_csl/test.hpp>

#if VSIP_IMPL_SAL_FFT
#  define TEST_NON_POWER_OF_2  0
#else
#  define TEST_NON_POWER_OF_2  1
#endif

using namespace vsip;
using namespace vsip_csl;


/***********************************************************************
  Definitions
***********************************************************************/

// The following test vectors were generated with C-VSIPL
// with the parameters as shown

// blackman N = 24
const scalar_f testvec_blackman[] = 
{
 -0.00000000,  0.00689491,  0.02959550,  0.07326404,
  0.14383306,  0.24489509,  0.37486633,  0.52538290,
  0.68154979,  0.82413213,  0.93320990,  0.99237636,
  0.99237636,  0.93320990,  0.82413213,  0.68154979,
  0.52538290,  0.37486633,  0.24489509,  0.14383306,
  0.07326404,  0.02959550,  0.00689491,  -0.00000000
};

// chebyshev N = 24, ripple = 60.0
const scalar_f testvec_cheby[] =
{
  0.01936452,  0.04531865,  0.09143121,  0.15967714,
  0.25135020,  0.36450197,  0.49349094,  0.62918443,
  0.75986841,  0.87276374,  0.95590014,  1.00000000,
  1.00000000,  0.95590014,  0.87276374,  0.75986841,
  0.62918443,  0.49349094,  0.36450197,  0.25135020,
  0.15967714,  0.09143121,  0.04531865,  0.01936452
};

// chebyshev N = 33, ripple = 60.0
const scalar_f testvec_cheby_odd[] =
{
  0.01891763,  0.03291801,  0.05887285,
  0.09530187,  0.14344737,  0.20388673,
  0.27633980,  0.35953825,  0.45118023,
  0.54798575,  0.64585693,  0.74013547,
  0.82593681,  0.89853081,  0.95373111,
  0.98825275,  1.00000000,  0.98825275,
  0.95373111,  0.89853081,  0.82593681,
  0.74013547,  0.64585693,  0.54798575,
  0.45118023,  0.35953825,  0.27633980,
  0.20388673,  0.14344737,  0.09530187,
  0.05887285,  0.03291801,  0.01891763
};

// hanning N = 24
const scalar_f testvec_hanning[] =
{
  0.01570842,  0.06184666,  0.13551569,  0.23208660,
  0.34549150,  0.46860474,  0.59369066,  0.71288965,
  0.81871199,  0.90450850,  0.96488824,  0.99605735,
  0.99605735,  0.96488824,  0.90450850,  0.81871199,
  0.71288965,  0.59369066,  0.46860474,  0.34549150,
  0.23208660,  0.13551569,  0.06184666,  0.01570842
};

// kaiser N = 24, beta = 3.5
const scalar_f testvec_kaiser[] =
{
  0.13553435,  0.21389064,  0.30308711,  0.40028181,
  0.50195216,  0.60407554,  0.70238150,  0.79249126,
  0.87030892,  0.93219106,  0.97518679,  0.99722036,
  0.99722036,  0.97518679,  0.93219106,  0.87030892,
  0.79249126,  0.70238150,  0.60407554,  0.50195216,
  0.40028181,  0.30308711,  0.21389064,  0.13553435
};


int
main(int argc, char** argv)
{
  vsipl init(argc, argv);


  // Blackman
  {
    const length_type N = 24;
    const_Vector<scalar_f> v = blackman(N);

    for ( unsigned int n = 0; n < N; ++n )
      test_assert( equal( v.get(n), testvec_blackman[n] ) );
  }

#if defined(VSIP_IMPL_FFT_USE_FLOAT)
#  if TEST_NON_POWER_OF_2
  // Chebyshev
  {
    const length_type N = 24;
    const scalar_f ripple = 60.0;
    const_Vector<scalar_f> v = cheby(N, ripple);

    for ( unsigned int n = 0; n < N; ++n )
      test_assert( equal( v.get(n), testvec_cheby[n] ) );
  }


  // Chebyshev odd
  {
    const length_type N = 33;
    const scalar_f ripple = 60.0;
    const_Vector<scalar_f> v = cheby(N, ripple);

    for ( unsigned int n = 0; n < N; ++n )
      test_assert( equal( v.get(n), testvec_cheby_odd[n] ) );
  }
#  endif
#endif

  // Hanning
  {
    const length_type N = 24;
    const_Vector<scalar_f> v = hanning(N);

    for ( unsigned int n = 0; n < N; ++n )
      test_assert( equal( v.get(n), testvec_hanning[n] ) );
  }

  // Kaiser
  {
    const length_type N = 24;
    const scalar_f beta = 3.5;
    const_Vector<scalar_f> v = kaiser(N, beta);

    for ( unsigned int n = 0; n < N; ++n )
      test_assert( equal( v.get(n), testvec_kaiser[n] ) );
  }

  return EXIT_SUCCESS;
}
