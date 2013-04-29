/***********************************************************************

  File:   fft-coverage.cpp
  Author: Jules Bergmann, CodeSourcery, LLC.
  Date:   05/21/2005

  Contents: Additional tests for FFTs and FFTMs.

Copyright 2005 Georgia Tech Research Corporation, all rights reserved.

A non-exclusive, non-royalty bearing license is hereby granted to all
Persons to copy, distribute and produce derivative works for any
purpose, provided that this copyright notice and following disclaimer
appear on All copies: THIS LICENSE INCLUDES NO WARRANTIES, EXPRESSED
OR IMPLIED, WHETHER ORAL OR WRITTEN, WITH RESPECT TO THE SOFTWARE OR
OTHER MATERIAL INCLUDING, BUT NOT LIMITED TO, ANY IMPLIED WARRANTIES
OF MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE, OR ARISING
FROM A COURSE OF PERFORMANCE OR DEALING, OR FROM USAGE OR TRADE, OR OF
NON-INFRINGEMENT OF ANY PATENTS OF THIRD PARTIES. THE INFORMATION IN
THIS DOCUMENT SHOULD NOT BE CONSTRUED AS A COMMITMENT OF DEVELOPMENT
BY ANY OF THE ABOVE PARTIES.

The US Government has a license under these copyrights, and this
Material may be reproduced by or for the US Government.
***********************************************************************/

/***********************************************************************

  Included Files
***********************************************************************/

#include <stdlib.h>
#include <utility>
#include "test.hpp"
#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/signal.hpp>
#include <vsip/vector.hpp>



/***********************************************************************
  Function Definitions
***********************************************************************/

using vsip::Vector;
using vsip::Matrix;
using vsip::const_Vector;
using vsip::const_Matrix;
using vsip::Domain;
using vsip::Fft;
using vsip::Fftm;

using vsip::index_type;
using vsip::length_type;
using vsip::scalar_f;
using vsip::cscalar_f;
using vsip::alg_hint_type;
using vsip::return_mechanism_type;

using vsip::alg_time;
using vsip::alg_space;
using vsip::fft_fwd;
using vsip::fft_inv;
using vsip::by_value;
using vsip::by_reference;



// -------------------------------------------------------------------- //
// ApplyFFT -- Overloaded utility functions to apply an Fft.
//
// These enable a single test function to cover both by_reference and
// by_value.

// Fft BY-VALUE
template <template <typename, typename> class CV,
	  typename			      IT,
	  typename			      OT,
	  int				      SD,
	  unsigned 			      NOT,
	  alg_hint_type			      AH,
	  typename			      Block1,
	  typename			      Block2>
void ApplyFFT(
   Fft<CV, IT, OT, SD, by_value, NOT, AH>& fft,
   const_Vector<IT, Block1> in,
   Vector      <OT, Block2> out)
{
   out = fft(in);
}


// Fft BY-REFERENCE
template <template <typename, typename> class CV,
	  typename			      IT,
	  typename			      OT,
	  int				      SD,
	  unsigned 			      NOT,
	  alg_hint_type			      AH,
	  typename			      Block1,
	  typename			      Block2>
void ApplyFFT(
   Fft<CV, IT, OT, SD, by_reference, NOT, AH>& fft,
   const_Vector<IT, Block1> in,
   Vector      <OT, Block2> out)
{
   fft(in, out);
}


// Fft BY-REFERENCE, plus check in-place operator
template <template <typename, typename> class CV,
	  typename			      T,
	  int				      SD,
	  unsigned 			      NOT,
	  alg_hint_type			      AH,
	  typename			      Block1,
	  typename			      Block2>
void ApplyFFT(
   Fft<CV, T, T, SD, by_reference, NOT, AH>& fft,
   const_Vector<T, Block1> in,
   Vector      <T, Block2> out)
{
  fft(in, out);

  // check in-place
  Vector<T> vec = in;
  fft(vec);
  insist(equal(maxdiff(vec, out), 0.f));
}



// -------------------------------------------------------------------- //
// ApplyFFTM -- Overloaded utility functions to apply an Fftm.
//
// These enable a single test function to cover both by_reference and
// by_value.

template <typename			      IT,
	  typename			      OT,
	  int				      SD,
	  int				      Dir,
	  unsigned 			      NOT,
	  alg_hint_type			      AH,
	  typename			      Block1,
	  typename			      Block2>
void ApplyFFTM(
   Fftm<IT, OT, SD, Dir, by_value, NOT, AH>& fftm,
   const_Matrix<IT, Block1> in,
   Matrix      <OT, Block2> out)
{
   out = fftm(in);
}



template <typename			      IT,
	  typename			      OT,
	  int				      SD,
	  int				      Dir,
	  unsigned 			      NOT,
	  alg_hint_type			      AH,
	  typename			      Block1,
	  typename			      Block2>
void ApplyFFTM(
   Fftm<IT, OT, SD, Dir, by_reference, NOT, AH>& fftm,
   const_Matrix<IT, Block1> in,
   Matrix      <OT, Block2> out)
{
   fftm(in, out);
}



template <typename			      T,
	  int				      SD,
	  int				      Dir,
	  unsigned 			      NOT,
	  alg_hint_type			      AH,
	  typename			      Block1,
	  typename			      Block2>
void ApplyFFTM(
   Fftm<T, T, SD, Dir, by_reference, NOT, AH>& fftm,
   const_Matrix<T, Block1> in,
   Matrix      <T, Block2> out)
{
  fftm(in, out);

  // check in-place
  Matrix<T> mat = in;
  fftm(mat);
  insist(equal(maxdiff(mat, out), 0.f));
}



// test_1d_cc -- Test single, 1-dimensional, complex-to-complex Fft.

// Template Parameters:
//   RM is the return mechanism to test (by_value or by_reference).
//   HINT is the algorithm hint to use (alg_space or alg_noise avoid
//      triggering FFTW's planning, which can be expensive for Fft
//      objects that are used a small number of times.
//
// Requires:
//   LEN to be Fft length to test.  Can be any natural number.

template <return_mechanism_type RM,
	  alg_hint_type         hint>
void
test_1d_cc(length_type const len)
{
  Vector<cscalar_f> in   (len, cscalar_f ());
  Vector<cscalar_f> trans(len);
  Vector<cscalar_f> out  (len);

  Vector<cscalar_f> chk_trans(len, cscalar_f());

  typedef Fft<const_Vector, cscalar_f, cscalar_f, fft_fwd,
              RM, 0, hint> ForFFT;
  typedef Fft<const_Vector, cscalar_f, cscalar_f, fft_inv,
              RM, 0, hint> InvFFT;

  ForFFT for_fft ((Domain<1>(len)), 1.f);
  InvFFT inv_fft ((Domain<1>(len)), 1.f / len);

  // Test accessors.
  insist (equal (for_fft.input_size(),  Domain<1>(len)));
  insist (equal (for_fft.output_size(), Domain<1>(len)));
  insist (equal (for_fft.scale(),   1.f));
  insist (equal (for_fft.forward(), true));

  insist (equal (inv_fft.input_size(),  Domain<1>(len)));
  insist (equal (inv_fft.output_size(), Domain<1>(len)));
  insist (equal (inv_fft.scale(),   1.f / len));
  insist (equal (inv_fft.forward(), false));

  // Test the operator.
  //  - zeros in, expect zeros out.
  ApplyFFT(for_fft, in, trans);
  insist(equal(maxdiff(trans, chk_trans), 0.f));

  ApplyFFT(inv_fft, trans, out);
  insist(equal(maxdiff(in, out), 0.f));


  // Test the operator with synthesized frequencies.
  const scalar_f pi = 4.0 * vsip::atan(1.f);
  float f1 = 0.25;
  float f2 = 0.1;

  for (index_type i=0; i<len; ++i)
     in.put(i, sin(2.f * pi * f1 * i) +
	       sin(2.f * pi * f2 * i));

  ApplyFFT(for_fft, in, trans);
  ApplyFFT(inv_fft, trans, out);

  insist(equal(maxdiff(in, out), 0.f));
}



// test_1d_rc -- Test single, 1-dimensional, real-to-complex and
//               complex-to-real FFTs.

// Template Parameters:
//   RM is the return mechanism to test (by_value or by_reference).
//   HINT is the algorithm hint to use (alg_space or alg_noise avoid
//      triggering FFTW's planning, which can be expensive for Fft
//      objects that are used a small number of times.
//
// Requires:
//   LEN to be Fft length to test.  Can be any natural, even number.

template <return_mechanism_type RM,
	  alg_hint_type         hint>
void
test_1d_rc(length_type const len)
{
  Vector<scalar_f>  in(len, scalar_f());
  Vector<cscalar_f> trans(len/2+1);
  Vector<scalar_f>  out(len);

  Vector<cscalar_f> chk_trans(len/2+1, cscalar_f());


  typedef Fft<const_Vector, scalar_f, cscalar_f, 0, RM, 0, hint> ForFFT;
  typedef Fft<const_Vector, cscalar_f, scalar_f, 0, RM, 0, hint> InvFFT;

  ForFFT for_fft ((Domain<1>(len)), 1.f);
  InvFFT inv_fft ((Domain<1>(len)), 1.f / len);

  // Test accessors.
  insist(equal(for_fft.input_size(),  Domain<1>(len)));
  insist(equal(for_fft.output_size(), Domain<1>(len/2+1)));
  insist(equal(for_fft.scale(),   1.f));
  insist(equal(for_fft.forward(), true));

  insist(equal(inv_fft.input_size(),  Domain<1>(len/2+1)));
  insist(equal(inv_fft.output_size(), Domain<1>(len)));
  insist(equal(inv_fft.scale(),   1.f / len));
  insist(equal(inv_fft.forward(), false));

  // Test the operator.
  //  - zeros in, expect zeros out.
  ApplyFFT(for_fft, in, trans);
  insist(equal(maxdiff(trans, chk_trans), 0.f));

  ApplyFFT(inv_fft, trans, out);
  insist(equal(maxdiff(in, out), 0.f));

  // Check that copy-assignment works properly.
#if 0
  Vector<cscalar_f> trans2 = for_fft(in);
  Vector<scalar_f>  out2   = inv_fft(trans2);
  insist(equal(maxdiff(trans, trans2), 0.f));
  insist(equal(maxdiff(out,   out2), 0.f));
#endif

  // Test the operator with synthesized frequencies.
  const scalar_f pi = 4.0 * vsip::atan(1.f);
  float f1 = 0.25;
  float f2 = 0.1;

  for (index_type i=0; i<len; ++i)
     in.put(i, sin(2.f * pi * f1 * i) +
	       sin(2.f * pi * f2 * i));

  ApplyFFT(for_fft, in, trans);
  ApplyFFT(inv_fft, trans, out);

  insist(equal(maxdiff(in, out), 0.f));
}



// test_fftm_cc - Test Fftm complex-to-complex case.
//
// Requires:
//   RM is the return mechanism to test (by_value or by_reference).
//   SD to be the special dimension, 0 or 1.
//      SD == 0 indicates row FFTs,
//      SD == 1 indicates column FFTs.
//   HINT is the algorithm hint to use (alg_space or alg_noise avoid
//      triggering FFTW's planning, which can be expensive for Fft
//      objects that are used a small number of times.
//
// Requires:
//   LEN0, LEN1 to be dimensions Fft to test.

template <return_mechanism_type RM,
	  int                   SD,
	  alg_hint_type         hint>
void
test_fftm_cc(length_type len0, length_type len1)
{
  Matrix<cscalar_f> in   (len0, len1, cscalar_f());
  Matrix<cscalar_f> trans(len0, len1);
  Matrix<cscalar_f> out  (len0, len1);
  Domain<1>         dom0 (len0);
  Domain<1>         dom1 (len1);

  Matrix<cscalar_f> chk_trans(len0, len1, cscalar_f());

  typedef Fftm<cscalar_f, cscalar_f, SD, fft_fwd, RM, 0, hint> ForFFTM;
  typedef Fftm<cscalar_f, cscalar_f, SD, fft_inv, RM, 0, hint> InvFFTM;

  typedef Fft <const_Vector, cscalar_f, cscalar_f, fft_fwd, by_reference,
	       0, hint> ForFFT;
  typedef Fft <const_Vector, cscalar_f, cscalar_f, fft_inv, by_reference,
	       0, hint> InvFFT;

  float scale = 1.f / (SD == 0 ? len1 : len0);

  ForFFTM	for_fftm(Domain<2>(len0, len1), 1.f);
  InvFFTM	inv_fftm(Domain<2>(len0, len1), scale);
  
  // Test accessors.
  insist(equal(for_fftm.input_size(),  Domain<2>(len0, len1)));
  insist(equal(for_fftm.output_size(), Domain<2>(len0, len1)));
  insist(equal(for_fftm.scale(), 1.f));
  insist(equal(for_fftm.forward(), true));
  
  insist(equal(inv_fftm.input_size(),  Domain<2>(len0, len1)));
  insist(equal(inv_fftm.output_size(), Domain<2>(len0, len1)));
  insist(equal(inv_fftm.scale(), scale));
  insist(equal(inv_fftm.forward(), false));
  
  // Test the operator.
  //  - zeros in, expect zeros out.
  ApplyFFTM(for_fftm, in, trans);
  insist(equal(maxdiff(trans, chk_trans), 0.f));
  
  ApplyFFTM(inv_fftm, trans, out);
  insist(equal(maxdiff(in, out), 0.f));
  

  // Test the operator with synthesized frequencies.
  const scalar_f pi = 4.0 * vsip::atan(1.f);
  float f1 = 0.25;
  float f2 = 0.1;

  // Put data into rows / columns, based on special dimension.
  if (SD == 0) {
     // Place frequencies into rows.
     for (index_type i=0; i<len0; ++i)
	for (index_type j=0; j<len1; ++j)
	   in.put(i, j, sin(2.f * pi * f1 * j) +
		        sin(2.f * pi * f2 * j));
     }
  else {
     // Place frequencies into columns.
     for (index_type i=0; i<len0; ++i)
	for (index_type j=0; j<len1; ++j)
	   in.put(i, j, sin(2.f * pi * f1 * i) +
		        sin(2.f * pi * f2 * i));
     }

  ApplyFFTM(for_fftm, in,    trans);
  ApplyFFTM(inv_fftm, trans, out);

  // Check that round-trip works.
  insist(equal(maxdiff(in, out), 0.f));

  // SD == 0: Fft each row
  if (SD == 0) {
     // Check that data is consistent row-to-row, both input & output.
     for (index_type r=1; r<trans.size(0); ++r) {
	insist(equal(maxdiff(in.row(0),    in.row(r)), 0.f));
	insist(equal(maxdiff(trans.row(0), trans.row(r)), 0.f));
	}

     // Check rows, one-by-one
     ForFFT chk_fft(dom1, 1.f);
     InvFFT chk_inv(dom1, scale);
     Vector<cscalar_f> vec1(len1);
     Vector<cscalar_f> vec2(len1);
     for (index_type r=0; r<trans.size(0); ++r) {
	chk_fft(in.row(r), vec1);
	insist(equal(maxdiff(trans.row(r), vec1), 0.f));

	chk_inv(vec1, vec2);
	insist(equal(maxdiff(out.row(r), vec2), 0.f));
	}
     }
  // SD == 1: Fft each column
  else {
     // Check that data is consistent col-to-col, both input & output.
     for (index_type c=1; c<trans.size(1); ++c) {
	insist(equal(maxdiff(in.col(0), in.col(c)), 0.f));
	insist(equal(maxdiff(trans.col(0), trans.col(c)), 0.f));
	}

     // Check columns, one-by-one
     ForFFT chk_fft(dom0, 1.f);
     InvFFT chk_inv(dom0, scale);
     Vector<cscalar_f> vec1(len0);
     Vector<cscalar_f> vec2(len0);
     for (index_type c=0; c<trans.size(1); ++c) {
	chk_fft(in.col(c), vec1);
	insist(equal(maxdiff(trans.col(c), vec1), 0.f));

	chk_inv(vec1, vec2);
	insist(equal(maxdiff(out.col(c), vec2), 0.f));
	}
     }

}



// test_fftm_rc - Test Fftm real-to-complex and complex-to-real cases.
//
// Requires:
//   RM is the return mechanism to test (by_value or by_reference).
//   SD to be the special dimension, 0 or 1.
//      SD == 0 indicates row FFTs,
//      SD == 1 indicates column FFTs.
//   HINT is the algorithm hint to use (alg_space or alg_noise avoid
//      triggering FFTW's planning, which can be expensive for Fft
//      objects that are used a small number of times.
//
// Requires:
//   LEN0, LEN1 to be dimensions Fftm to test.

template <return_mechanism_type RM,
	  int                   SD,
	  alg_hint_type         hint>
void
test_fftm_rc(length_type len0, length_type len1)
{
  length_type const trans_len0 = (SD == 0) ? len0 : len0/2+1;
  length_type const trans_len1 = (SD == 1) ? len1 : len1/2+1;

  Matrix<scalar_f>  in   (len0, len1, scalar_f());
  Matrix<cscalar_f> trans(trans_len0, trans_len1);
  Matrix<scalar_f>  out  (len0, len1);

  Domain<1>         dom0 (len0);
  Domain<1>         dom1 (len1);

  Matrix<cscalar_f> chk_trans(trans_len0, trans_len1, cscalar_f());

  typedef Fftm<scalar_f, cscalar_f, SD, fft_fwd, RM, 0, hint> ForFFTM;
  typedef Fftm<cscalar_f, scalar_f, SD, fft_inv, RM, 0, hint> InvFFTM;

  typedef Fft <const_Vector, scalar_f, cscalar_f, 0, by_reference,
	       0, hint> ForFFT;
  typedef Fft <const_Vector, cscalar_f, scalar_f, 0, by_reference,
               0, hint> InvFFT;

  float scale = 1.f / (SD == 0 ? len1 : len0);

  ForFFTM	for_fftm(Domain<2>(len0, len1), 1.f);
  InvFFTM	inv_fftm(Domain<2>(len0, len1), scale);

  // Test accessors.
  insist(equal(for_fftm.input_size(),  Domain<2>(len0, len1)));
  insist(equal(for_fftm.output_size(), Domain<2>(trans_len0, trans_len1)));
  insist(equal(for_fftm.scale(), 1.f));
  insist(equal(for_fftm.forward(), true));

  insist(equal(inv_fftm.input_size(),  Domain<2>(trans_len0, trans_len1)));
  insist(equal(inv_fftm.output_size(), Domain<2>(len0, len1)));
  insist(equal(inv_fftm.scale(), scale));
  insist(equal(inv_fftm.forward(), false));

  // Test the operator.
  //  - zeros in, expect zeros out.
  ApplyFFTM(for_fftm, in, trans);
  insist(equal(maxdiff(trans, chk_trans), 0.f));

  ApplyFFTM(inv_fftm, trans, out);
  insist(equal(maxdiff(in, out), 0.f));


  // Test the operator with synthesized frequencies.
  const scalar_f pi = 4.0 * vsip::atan(1.f);
  float f1 = 0.25;
  float f2 = 0.1;

  // Put data into rows / columns, based on special dimension.
  if (SD == 0) {
     // Place frequencies into rows.
     for (index_type i=0; i<len0; ++i)
	for (index_type j=0; j<len1; ++j)
	   in.put(i, j, sin(2.f * pi * f1 * j) +
		        sin(2.f * pi * f2 * j));
     }
  else {
     // Place frequencies into columns.
     for (index_type i=0; i<len0; ++i)
	for (index_type j=0; j<len1; ++j)
	   in.put(i, j, sin(2.f * pi * f1 * i) +
		        sin(2.f * pi * f2 * i));
     }

  ApplyFFTM(for_fftm, in,    trans);
  ApplyFFTM(inv_fftm, trans, out);

  // Check that round-trip works.
  insist(equal(maxdiff(in, out), 0.f));

  // SD == 0: Fft each row
  if (SD == 0) {
     // Check that data is consistent row-to-row, both input & output.
     for (index_type r=1; r<in.size(0); ++r) {
	insist(equal(maxdiff(in.row(0),    in.row(r)), 0.f));
	insist(equal(maxdiff(trans.row(0), trans.row(r)), 0.f));
	}

     // Check rows, one-by-one
     ForFFT chk_fft(dom1, 1.f);
     InvFFT chk_inv(dom1, scale);
     Vector<cscalar_f> vec1(trans_len1);
     Vector<scalar_f>  vec2(len1);
     for (index_type r=0; r<in.size(0); ++r) {
	chk_fft(in.row(r), vec1);
	insist(equal(maxdiff(trans.row(r), vec1), 0.f));

	chk_inv(vec1, vec2);
	insist(equal(maxdiff(out.row(r), vec2), 0.f));
	}
     }
  // SD == 1: Fft each column
  else {
     // Check that data is consistent col-to-col, both input & output.
     for (index_type c=1; c<in.size(1); ++c) {
	insist(equal(maxdiff(in.col(0), in.col(c)), 0.f));
	insist(equal(maxdiff(trans.col(0), trans.col(c)), 0.f));
	}

     // Check columns, one-by-one
     ForFFT chk_fft(dom0, 1.f);
     InvFFT chk_inv(dom0, scale);
     Vector<cscalar_f> vec1(trans_len0);
     Vector<scalar_f>  vec2(len0);
     for (index_type c=0; c<in.size(1); ++c) {
	chk_fft(in.col(c), vec1);
	insist(equal(maxdiff(trans.col(c), vec1), 0.f));

	chk_inv(vec1, vec2);
	insist(equal(maxdiff(out.col(c), vec2), 0.f));
	}
     }
}



// test_suite -- run all tests.

template <alg_hint_type hint>
void
test_suite()
{
  test_1d_cc<by_value, hint>(31);
  test_1d_cc<by_value, hint>(32);
  test_1d_cc<by_value, hint>(256);
  test_1d_cc<by_value, hint>(1024);
  test_1d_cc<by_reference, hint>(31);
  test_1d_cc<by_reference, hint>(32);
  test_1d_cc<by_reference, hint>(256);
  test_1d_cc<by_reference, hint>(1024);

  test_1d_rc<by_value, hint>(30);
  test_1d_rc<by_value, hint>(32);
  test_1d_rc<by_value, hint>(256);
  test_1d_rc<by_value, hint>(1024);
  test_1d_rc<by_reference, hint>(30);
  test_1d_rc<by_reference, hint>(32);
  test_1d_rc<by_reference, hint>(256);
  test_1d_rc<by_reference, hint>(1024);

  test_fftm_cc<by_value, 0, hint>(32,   31);
  test_fftm_cc<by_value, 0, hint>(32,   32);
  test_fftm_cc<by_value, 0, hint>(32,  256);
  test_fftm_cc<by_value, 0, hint>(32, 1024);
  test_fftm_cc<by_value, 1, hint>(  31, 32);
  test_fftm_cc<by_value, 1, hint>(  32, 32);
  test_fftm_cc<by_value, 1, hint>( 256, 32);
  test_fftm_cc<by_value, 1, hint>(1024, 32);
  test_fftm_cc<by_reference, 0, hint>(32,   31);
  test_fftm_cc<by_reference, 0, hint>(32,   32);
  test_fftm_cc<by_reference, 0, hint>(32,  256);
  test_fftm_cc<by_reference, 0, hint>(32, 1024);
  test_fftm_cc<by_reference, 1, hint>(  31, 32);
  test_fftm_cc<by_reference, 1, hint>(  32, 32);
  test_fftm_cc<by_reference, 1, hint>( 256, 32);
  test_fftm_cc<by_reference, 1, hint>(1024, 32);

  test_fftm_rc<by_reference, 0, hint>(8,     8);
  test_fftm_rc<by_reference, 0, hint>(32,   30);
  test_fftm_rc<by_reference, 0, hint>(32,   32);
  test_fftm_rc<by_reference, 0, hint>(32,  256);
  test_fftm_rc<by_reference, 0, hint>(32, 1024);
  test_fftm_rc<by_reference, 1, hint>(8,     8);
  test_fftm_rc<by_reference, 1, hint>(30,   32);
  test_fftm_rc<by_reference, 1, hint>(32,   32);
  test_fftm_rc<by_reference, 1, hint>(256,  32);
  test_fftm_rc<by_reference, 1, hint>(1024, 32);

  test_fftm_rc<by_value, 0, hint>(8,     8);
  test_fftm_rc<by_value, 0, hint>(32,   30);
  test_fftm_rc<by_value, 0, hint>(32,   32);
  test_fftm_rc<by_value, 0, hint>(32,  256);
  test_fftm_rc<by_value, 0, hint>(32, 1024);
  test_fftm_rc<by_value, 1, hint>(8,     8);
  test_fftm_rc<by_value, 1, hint>(32,   32);
  test_fftm_rc<by_value, 1, hint>(30,   32);
  test_fftm_rc<by_value, 1, hint>(256,  32);
  test_fftm_rc<by_value, 1, hint>(1024, 32);
}



int
main (int argc, char** argv)
{
  vsip::vsipl	init(argc, argv);

  // For FFTW, alg_time causes each FFTW plan to MEASURE, which is
  // expensive, esp since we're only doing a small number of FFTs with
  // each object.  Setting the hint to something else (SPACE of NOISE)
  // avoids that.

  test_suite<alg_space>();	//   0.419 sec
  // test_suite<alg_time>();	// 443.    sec

  return EXIT_SUCCESS;
}
