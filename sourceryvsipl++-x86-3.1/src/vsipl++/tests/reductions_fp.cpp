/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Tests for math reductions, floating-point types.

#include "reductions.hpp"

using namespace vsip;
using vsip_csl::equal;
using vsip_csl::sumval;
using vsip_csl::sumsqval;
using vsip_csl::meansqval;
using vsip_csl::meanval;

void
simple_tests()
{
  Vector<float> vec(4);

  vec(0) = 0.;
  vec(1) = 1.;
  vec(2) = 2.;
  vec(3) = 3.;

  test_assert(equal(sumval(vec),    6.0f));
  test_assert(equal(meanval(vec),   1.5f));
  test_assert(equal(sumsqval(vec), 14.0f));
  test_assert(equal(meansqval(vec), 3.5f));

  test_assert(equal(sumval(vec+vec), 12.0f));

  Matrix<double> mat(2, 2);

  mat(0, 0) = 1.;
  mat(0, 1) = 2.;
  mat(1, 0) = 3.;
  mat(1, 1) = 4.;

  test_assert(equal(sumval(mat),   10.0));
  test_assert(equal(meanval(mat),   2.5));
  test_assert(equal(sumsqval(mat), 30.0));
  test_assert(equal(meansqval(mat), 7.5));

  Tensor<float> ten(2, 1, 2);

  ten(0, 0, 0) = 2.;
  ten(0, 0, 1) = 3.;
  ten(1, 0, 0) = 4.;
  ten(1, 0, 1) = 5.;

  test_assert(equal(sumval(ten),    14.0f));
  test_assert(equal(meanval(ten),    3.5f));
  test_assert(equal(sumsqval(ten),  54.0f));
  test_assert(equal(meansqval(ten), 13.5f));

  Vector<complex<float> > cvec(2);

  cvec(0) = complex<float>(3.f,  4.f); // -7 + 24i
  cvec(1) = complex<float>(3.f, -4.f); // -7 - 24i

  test_assert(equal(sumval(cvec),    complex<float>(6.0f, 0.0f)));
  // test_assert(equal(meanval(cvec), complex<float>(3.f, 0.f)));
  test_assert(equal(sumsqval(cvec),  complex<float>(-14.0f, 0.0f)));
  test_assert(equal(meansqval(cvec), 25.0f));


  Vector<bool> bvec(4);

  bvec(0) = true;
  bvec(1) = true;
  bvec(2) = false;
  bvec(3) = true;

  test_assert(equal(sumval(bvec), static_cast<length_type>(3)));

  // Simple test for alternate form.
  Vector<unsigned short> uvec(4);

  uvec(0) = 65535;
  uvec(1) = 1;
  uvec(2) = 2;
  uvec(3) = 3;

  typedef unsigned long W;

  test_assert(equal(sumval(uvec, W()), W(65541)));
  test_assert(equal(meanval(uvec, W()), W(65541/4)));
  uvec(0) = 256;
  test_assert(equal(sumsqval(uvec, W()), W(65550)));
  W w = meansqval(uvec, W());
  if( !equal(w, W(65550/4)) )
    std::cout << "w=" << w << ", expected=" << W(65550/4) << "\n";
  test_assert(equal(w, W(65550/4)));
}



int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  simple_tests();

  par_cover_sumval<float, Replicated_map<1> >();
  par_cover_sumval<float, Map<Block_dist> >();

  cover_sumval<float>();
  cover_sumval<complex<float> >();

  cover_sumsqval<float>();
  cover_sumsqval<complex<float> >();

  cover_meanval<float>();
  cover_meanval<complex<float> >();

  cover_meansqval<float>();
  cover_meansqval<complex<float> >();

#if VSIP_IMPL_TEST_DOUBLE
  cover_sumval<double>();
  cover_sumval<complex<double> >();

  cover_sumsqval<double>();
  cover_sumsqval<complex<double> >();

  cover_meanval<double>();

  cover_meansqval<double>();
  cover_meansqval<complex<double> >();
#endif
}
