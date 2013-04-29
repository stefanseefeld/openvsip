/* Copyright (c) 2006 by CodeSourcery, LLC.  All rights reserved. */

/** @file    tests/vsip_csl/sfilt.cpp
    @author  Jules Bergmann
    @date    2007-10-04
    @brief   VSIPL++ Library: Extra unit tests for separable filters.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#define VERBOSE 0

#if VERBOSE
#  include <iostream>
#  include <vsip/opt/diag/eval.hpp>
#endif

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/domain.hpp>
#include <vsip/random.hpp>
#include <vsip/signal.hpp>

#include <vsip_csl/img/separable_filter.hpp>

#include <vsip_csl/test.hpp>
#include <vsip_csl/error_db.hpp>
#if VERBOSE
#  include <vsip_csl/output.hpp>
#endif

using namespace vsip;
using namespace vsip_csl;


/***********************************************************************
  Definitions - Utility Functions
***********************************************************************/

template <typename Vector1T,
	  typename Vector2T,
	  typename MatrixT>
void
test_sfilt(
  Vector1T k0,
  Vector2T k1,
  MatrixT  in)

{
  using vsip_csl::img::Separable_filter;
  using vsip_csl::img::edge_zero;

  typedef typename MatrixT::value_type T;

  typedef Separable_filter<T, support_min_zeropad, edge_zero> filt_type;

  length_type nk0  = k0.size();
  length_type nk1  = k1.size();
  length_type rows = in.size(0);
  length_type cols = in.size(1);

  filt_type filt(k0, k1, Domain<2>(rows, cols));

  Matrix<T> out(rows, cols);

  for (index_type r=0; r<rows; ++r)
    in.row(r) = ramp<T>(r, 1, cols);

  out = T(-1);

  filt(in, out);


  Matrix<T> ref_k(nk0, nk1);
 
  for (index_type i=0; i<nk0; ++i)
    for (index_type j=0; j<nk1; ++j)
      ref_k(nk0-i-1, nk1-j-1) = k0(i) * k1(j);

  typedef Convolution<const_Matrix, nonsym, support_min, T>
    conv_type;

  conv_type conv(ref_k, Domain<2>(rows, cols), 1);

  Matrix<T> ref_out(rows, cols);
  ref_out = T(); // zero-pad

  conv(in, ref_out(Domain<2>(Domain<1>(nk0/2, 1, rows - nk0 + 1),
			     Domain<1>(nk1/2, 1, cols - nk1 + 1))));

  float error = error_db(out, ref_out);

#if VERBOSE
  std::cout << "error: " << error << std::endl;
  if (error >= -100)
  {
    std::cout << "BE:"
	      << vsip::impl::diag_detail::Dispatch_name<
                    typename filt_type::impl_tag>::name() << std::endl;
    std::cout << "k0:\n" << k0;
    std::cout << "k1:\n" << k1;
    std::cout << "ref_k:\n" << ref_k;
    std::cout << "out:\n" << out;
    std::cout << "ref_out:\n" << ref_out;
  }
#endif

  test_assert(error < -100);
}

template <typename T>
void
test_ident(
  length_type nk0,
  length_type nk1,
  index_type  pk0,
  index_type  pk1,
  length_type rows,
  length_type cols)
{
  Vector<T> k0(nk0);
  Vector<T> k1(nk1);

  k0 = T(); k0(pk0) = T(1);
  k1 = T(); k1(pk1) = T(1);

  Matrix<T> in (rows, cols);

  test_sfilt(k0, k1, in);
}



int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  test_ident<float>(3, 3, 0, 0, 16, 16);
  test_ident<float>(4, 4, 0, 0, 16, 16);
  test_ident<float>(5, 3, 0, 0, 16, 16);
  test_ident<float>(3, 5, 0, 0, 16, 16);
  test_ident<float>(5, 3, 1, 2, 16, 16);
  test_ident<float>(3, 5, 1, 1, 16, 16);
  test_ident<float>(5, 4, 0, 0, 16, 16);
  test_ident<float>(4, 6, 0, 0, 16, 16);

  test_ident<float>(3, 3, 0, 0, 16, 24);
  test_ident<float>(4, 4, 0, 0, 16, 24);
  test_ident<float>(5, 3, 0, 0, 16, 24);
  test_ident<float>(3, 5, 0, 0, 16, 24);
  test_ident<float>(5, 3, 1, 2, 16, 24);
  test_ident<float>(3, 5, 1, 1, 16, 24);
  test_ident<float>(5, 4, 0, 0, 16, 24);
  test_ident<float>(4, 6, 0, 0, 16, 24);

  test_ident<float>(3, 3, 0, 0, 15, 17);
  test_ident<float>(4, 4, 0, 0, 17, 15);
  test_ident<float>(5, 3, 0, 0, 15, 17);
  test_ident<float>(3, 5, 0, 0, 15, 17);
  test_ident<float>(5, 3, 1, 2, 15, 17);
  test_ident<float>(3, 5, 1, 1, 15, 17);
  test_ident<float>(5, 4, 0, 0, 15, 17);
  test_ident<float>(4, 6, 0, 0, 15, 17);
}
