//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <vsip/initfin.hpp>
#include <vsip/random.hpp>
#include <vsip/support.hpp>
#include <vsip_csl/test.hpp>
#include <vsip_csl/output.hpp>

/***********************************************************************
  Function Definitions
***********************************************************************/



// C VSIPL code - the following typedefs allow this code to run
// essentially unmodified from the original source. 

// This is placed in a namespace to avoid conflicting with real
// C-VSIPL if it is used as a backend.

namespace test
{

typedef double vsip_scalar_d;
typedef struct { vsip_scalar_d  r, i; } vsip_cscalar_d;
typedef unsigned int vsip_scalar_ue32;
typedef unsigned int vsip_index;

struct vsip_randstate {
  vsip_scalar_ue32  a; /* multiplier in LCG */
  vsip_scalar_ue32  c; /* adder in LCG */
  vsip_scalar_ue32  a1;
  vsip_scalar_ue32  c1;
  vsip_scalar_ue32  X; /* Last or initial X */
  vsip_scalar_ue32  X1;
  vsip_scalar_ue32  X2;
     
  int               type;
};



#define A0 1664525
#define C0 1013904223
#define A1 69069

vsip_randstate* vsip_randcreate(
  vsip_index seed,
  vsip_index numseqs,
  vsip_index id,
  bool typegen)
{
  vsip_scalar_ue32 x0 = (vsip_scalar_ue32) seed;
  vsip_scalar_ue32 k  = (vsip_scalar_ue32) numseqs;
  vsip_randstate *state = (vsip_randstate*) malloc(sizeof(vsip_randstate));
  memset( state, 0, sizeof(vsip_randstate) );

  if(state == NULL) return (vsip_randstate*)NULL;
  state->type = (int)typegen;
  if(typegen)
  { /* create non portable generator */
    vsip_scalar_ue32 a;
    vsip_scalar_ue32 c;
    vsip_scalar_ue32 i,n,k0;
    vsip_scalar_ue32 t;
    for(i=0; i< id; i++)
      x0 = A0 * x0 + C0;
    state->X = x0; /*find the seed to start out for id */
    n = 0;
    k0 = k;
    while((k0 % 2) == 0){
      k0 = k0/2;
      n++;
    }
    i = k - 1;
    a = A0;
    while(i-- >0) a *= A0; /* find a for numseqs */
    c = 1; /* find c for numseqs */
    t = 1;
    for(i=0; i<k0; i++) t *=A0;
    while(n-- >0) {
      c *= (t + 1);
      t *= t;
    }
    t = 1;
    n = A0;
    for(i=1; i<k0; i++){
      t += n;
      n *= A0;
    }
    c *= (t * C0);
    state->a = a;
    state->c = c;
  } else { /* create portable generator */
    vsip_scalar_ue32 c[]=
      {  3,   5,   7,  11,  13,  17,  19,  23,  29,  31,
         37,  41,  43,  47,  53,  59,  61,  67,  71,  73,
         79,  83,  89,  97, 101, 103, 107, 109, 113, 127,
         131, 137, 139, 149, 151, 157, 163, 167, 173, 179,
         181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
         239, 241, 251, 257, 263, 269, 271, 277, 281, 283,
         293, 307, 311, 313, 317, 331, 337, 347, 349, 353,
         359, 367, 373, 379, 383, 389, 397, 401, 409, 419,
         421, 431, 433, 439, 443, 449, 457, 461, 463, 467,
         479, 487, 491, 499, 503, 509, 521, 523, 541, 547}; /* 100 prime numbers */
    vsip_scalar_ue32 c1 = c[id-1];
    if(id > 1) { 
      vsip_scalar_ue32 a0 = A0,
        c0 = C0;
      vsip_scalar_ue32 mask = 1;
      vsip_scalar_ue32 big  = 4294967295ul;
      vsip_scalar_ue32 skip = (big/k) * (id -1);
      int i; 
      for(i=0; i<32; i++) {
        if(mask & skip) {
          x0 = a0 * x0 + c0;
        }
        c0 = (a0+1) * c0;
        a0 = a0*a0;
        mask <<= 1;
      } 
    }
    state->X  = x0;
    state->X1 = 1;
    state->X2 = 1;
    state->a  = A0;
    state->c  = C0;
    state->a1 = A1; 
    state->c1 = c1;
  }

  return state;
}


int
vsip_randdestroy(
  vsip_randstate *state)
{
  if (state != NULL) free(state);
  return 0;
}


vsip_scalar_d vsip_randu_d(
  vsip_randstate *state)
{
  if(state->type)
  { /* nonportable generator */
    vsip_scalar_ue32 a = state->a,
      c = state->c,
      X = state->X;

    X    = a * X + c;
    state->X = X;
    return ((vsip_scalar_d)X/4294967296.0);
  } else { /* portable generator */
    vsip_scalar_ue32 itemp;
    state->X  = state->X * state->a + state->c;
    state->X1 = state->X1 * state->a1 + state->c1;
    itemp     = state->X - state->X1;
    if(state->X1 == state->X2){
      state->X1++;
      state->X2++;
    }
    return((vsip_scalar_d)itemp/4294967296.0);
  }
}


vsip_scalar_d vsip_randn_d(
  vsip_randstate *state)
{
  vsip_index i;
  vsip_scalar_d rp;
  if(state->type)
  { /* nonportable generator */
    vsip_scalar_ue32 a = state->a,
      c = state->c,
      X = state->X;
    rp = 0;
    for(i=0; i<12; i++){
      X    = a * X + c;
      rp  += (vsip_scalar_d)X/4294967296.0;
    }
    state->X = X;
    rp -= 6.0;
  } else { /* portable generator */
    vsip_scalar_ue32 itemp;
    rp = 0;
    for(i=0; i<12; i++){
      state->X  = state->X * state->a + state->c;
      state->X1 = state->X1 * state->a1 + state->c1;
      itemp     = state->X - state->X1;
      if(state->X1 == state->X2){
        state->X1++;
        state->X2++;
      }
      rp  += (vsip_scalar_d)itemp/4294967296.0;
    }
    rp = 6.0 - rp;
  }
  return rp;
}


vsip_cscalar_d (vsip_cmplx_d)(
  vsip_scalar_d r, 
  vsip_scalar_d i) 
{
  vsip_cscalar_d z; 
  z.r = r; 
  z.i = i; 
  return z; 
}


vsip_cscalar_d vsip_crandu_d(
  vsip_randstate *state)
{
  vsip_scalar_d real,imag;
  if(state->type)
  { /* nonportable generator */
    vsip_scalar_ue32 a = state->a,
      c = state->c,
      X = state->X;
    X    = a * X + c;
    real = (vsip_scalar_d)X/4294967296.0;
    X    = a * X + c;
    imag = (vsip_scalar_d)X/4294967296.0;
    state->X = X;
  } else { /* portable generator */
    vsip_scalar_ue32 itemp;
    state->X  = state->X * state->a + state->c;
    state->X1 = state->X1 * state->a1 + state->c1;
    itemp     = state->X - state->X1;
    if(state->X1 == state->X2){
      state->X1++;
      state->X2++;
    }
    real  = (vsip_scalar_d)itemp/4294967296.0;
    state->X  = state->X * state->a + state->c;
    state->X1 = state->X1 * state->a1 + state->c1;
    itemp     = state->X - state->X1;
    if(state->X1 == state->X2){
      state->X1++;
      state->X2++;
    } 
//    imag  = (vsip_scalar_d)itemp/16777216.0;  !!!! error
    imag  = (vsip_scalar_d)itemp/4294967296.0;
  }
  return vsip_cmplx_d(real,imag);
}


vsip_cscalar_d vsip_crandn_d(
  vsip_randstate *state)
{
  vsip_scalar_d real,imag;
  if(state->type)
  { /* nonportable generator */
    vsip_scalar_ue32 a = state->a,
      c = state->c,
      X = state->X;
    vsip_scalar_d t2;
    X     = a * X + c;
    real  = (vsip_scalar_d)X/4294967296.0;
    X     = a * X + c;
    real += (vsip_scalar_d)X/4294967296.0;
    X     = a * X + c;
    real += (vsip_scalar_d)X/4294967296.0;
    X     = a * X + c;
    t2    = (vsip_scalar_d)X/4294967296.0;
    X     = a * X + c;
    t2   += (vsip_scalar_d)X/4294967296.0;
    X     = a * X + c;
    t2   += (vsip_scalar_d)X/4294967296.0;
    imag = real - t2;
    real = 3 - t2 - real;
    state->X = X;
  } else { /* portable generator */
    vsip_scalar_ue32 itemp;
    vsip_scalar_d t2;
    state->X  = state->X * state->a + state->c;
    state->X1 = state->X1 * state->a1 + state->c1;
    itemp     = state->X - state->X1;
    if(state->X1 == state->X2){
      state->X1++;
      state->X2++;
    }
    real  = (vsip_scalar_d)itemp/4294967296.0;
    state->X  = state->X * state->a + state->c;
    state->X1 = state->X1 * state->a1 + state->c1;
    itemp     = state->X - state->X1;
    if(state->X1 == state->X2){
      state->X1++;
      state->X2++;
    }
    real  += (vsip_scalar_d)itemp/4294967296.0;
    state->X  = state->X * state->a + state->c;
    state->X1 = state->X1 * state->a1 + state->c1;
    itemp     = state->X - state->X1;
    if(state->X1 == state->X2){
      state->X1++;
      state->X2++;
    }
    real  += (vsip_scalar_d)itemp/4294967296.0;
    /* end t1 */
    state->X  = state->X * state->a + state->c;
    state->X1 = state->X1 * state->a1 + state->c1;
    itemp     = state->X - state->X1;
    if(state->X1 == state->X2){
      state->X1++;
      state->X2++;
    } 
    t2  = (vsip_scalar_d)itemp/4294967296.0;
    state->X  = state->X * state->a + state->c;
    state->X1 = state->X1 * state->a1 + state->c1;
    itemp     = state->X - state->X1;
    if(state->X1 == state->X2){
      state->X1++;
      state->X2++;
    } 
    t2  += (vsip_scalar_d)itemp/4294967296.0;
    state->X  = state->X * state->a + state->c;
    state->X1 = state->X1 * state->a1 + state->c1;
    itemp     = state->X - state->X1;
    if(state->X1 == state->X2){
      state->X1++;
      state->X2++;
    } 
    t2  += (vsip_scalar_d)itemp/4294967296.0;
    /* end t2 */
    imag = real - t2;
    real = 3 - t2 - real;
  }
  return vsip_cmplx_d(real,imag);
}

}; // namespace test
// end C VSIPL code



int
main(int argc, char** argv)
{
  using namespace vsip;
  using namespace vsip_csl;
  vsipl init(argc, argv);


  // Random generation tests -- Compare against C VSIPL generator.

  // scalar values, portable or not, Normal or Uniform distributions

  // Normal distribution, portable
  {
    vsip::Rand<double> rgen(0, 1);
    test::vsip_randstate *rstate;
    rstate = test::vsip_randcreate( 0, 1, 1, 0 );
    for ( int i = 0; i < 8; ++i )
    {
      double a = rgen.randn();
      double b = test::vsip_randn_d( rstate );
      test_assert( equal( a, b ) );
    }
    test::vsip_randdestroy( rstate );
  }

  // Normal distribution, non-portable
  {
    vsip::Rand<double> rgen(0, 0);
    test::vsip_randstate *rstate;
    rstate = test::vsip_randcreate( 0, 1, 1, 1 );
    for ( int i = 0; i < 8; ++i ) 
    {
      double a = rgen.randn();
      double b = test::vsip_randn_d( rstate );
      test_assert( equal( a, b ) );
    }
    test::vsip_randdestroy( rstate );
  }

  // Uniform distribution, portable
  {
    vsip::Rand<double> rgen(0, 1);
    test::vsip_randstate *rstate;
    rstate = test::vsip_randcreate( 0, 1, 1, 0 );
    for ( int i = 0; i < 8; ++i ) 
    {
      double a = rgen.randu();
      double b = test::vsip_randu_d( rstate );
      test_assert( equal( a, b ) );
    }
    test::vsip_randdestroy( rstate );
  }

  // Uniform distribution, non-portable
  {
    vsip::Rand<double> rgen(0, 0);
    test::vsip_randstate *rstate;
    rstate = test::vsip_randcreate( 0, 1, 1, 1 );
    for ( int i = 0; i < 8; ++i ) 
    {
      double a = rgen.randu();
      double b = test::vsip_randu_d( rstate );
      test_assert( equal( a, b ) );
    }
    test::vsip_randdestroy( rstate );
  }


  // Do the same for complex values

  // Normal distribution, portable
  {
    vsip::Rand<complex<double> > rgen(0, 1);
    test::vsip_randstate *rstate;
    rstate = test::vsip_randcreate( 0, 1, 1, 0 );
    for ( int i = 0; i < 8; ++i ) 
    {
      complex<double> a = rgen.randn();
      test::vsip_cscalar_d z = test::vsip_crandn_d( rstate );
      complex<double> b(z.r, z.i);
      test_assert( equal( a, b ) );
    }
    test::vsip_randdestroy( rstate );
  }

  // Normal distribution, non-portable
  {
    vsip::Rand<complex<double> > rgen(0, 0);
    test::vsip_randstate *rstate;
    rstate = test::vsip_randcreate( 0, 1, 1, 1 );
    for ( int i = 0; i < 8; ++i )
    {
      complex<double> a = rgen.randn();
      test::vsip_cscalar_d z = test::vsip_crandn_d( rstate );
      complex<double> b(z.r, z.i);
      test_assert( equal( a, b ) );
    }
    test::vsip_randdestroy( rstate );
  }

  // Uniform distribution, portable
  {
    vsip::Rand<complex<double> > rgen(0, 1);
    test::vsip_randstate *rstate;
    rstate = test::vsip_randcreate( 0, 1, 1, 0 );
    for ( int i = 0; i < 8; ++i ) 
    {
      complex<double> a = rgen.randu();
      test::vsip_cscalar_d z = test::vsip_crandu_d( rstate );
      complex<double> b(z.r, z.i);
      test_assert( equal( a, b ) );
    }
    test::vsip_randdestroy( rstate );
  }

  // Uniform distribution, non-portable
  {
    vsip::Rand<complex<double> > rgen(0, 0);
    test::vsip_randstate *rstate;
    rstate = test::vsip_randcreate( 0, 1, 1, 1 );
    for ( int i = 0; i < 8; ++i ) 
    {
      complex<double> a = rgen.randu();
      test::vsip_cscalar_d z = test::vsip_crandu_d( rstate );
      complex<double> b(z.r, z.i);
      test_assert( equal( a, b ) );
    }
    test::vsip_randdestroy( rstate );
  }



  // Vector and Matrix tests -- compare values from two different 
  // generators with the same starting seed, one filling a vector
  // and one filling a matrix.  

  // Uniform
  vsip::Rand<> vgen(0, 0);
  vsip::Rand<>::vector_type v1 = vgen.randu(5 * 7);

  vsip::Rand<> mgen(0, 0);
  vsip::Rand<>::matrix_type m1 = mgen.randu(7, 5);

  for ( index_type i = 0; i < m1.size(0); ++i )
    for ( index_type j = 0; j < m1.size(1); ++j )
      test_assert( equal( v1.get(i * m1.size(1) + j), m1.get(i, j) ) );

  // Normal
  vsip::Rand<>::vector_type v2 = vgen.randn(3 * 9);
  vsip::Rand<>::matrix_type m2 = mgen.randn(9, 3);

  for ( index_type i = 0; i < m2.size(0); ++i )
    for ( index_type j = 0; j < m2.size(1); ++j )
      test_assert( equal( v2.get(i * m2.size(1) + j), m2.get(i, j) ) );


  return EXIT_SUCCESS;
}
