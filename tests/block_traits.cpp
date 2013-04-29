/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/** @file    tests/block_traits.cpp
    @author  Jules Bergmann
    @date    2007-01-26
    @brief   VSIPL++ Library: Test block traits.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/core/us_block.hpp>

#include <vsip_csl/test.hpp>

using namespace std;
using namespace vsip;


/***********************************************************************
  Definitions
***********************************************************************/

template <typename T>
struct put_block
{
  typedef T value_type;
  value_type get() { return T(1); }
  void put(index_type, value_type) {}
};

template <typename T>
struct noput_block
{
  typedef T value_type;
  value_type get() { return T(1); }
};

struct derived_block : put_block<float>
{
};


void
test_modifiable_1()
{
  using vsip::impl::is_modifiable_block;
  using vsip::impl::Strided;
  using vsip::impl::Us_block;
  using vsip::impl::Dense_storage;
  using vsip::interleaved_complex;

  test_assert((is_modifiable_block<Dense<1, float> >::value == true));
  test_assert((is_modifiable_block<Strided<1, float> >::value == true));

  test_assert((is_modifiable_block<Us_block<1, float> >::value == true));
  test_assert((is_modifiable_block<Us_block<1, complex<float> > >::value == true));
  test_assert((is_modifiable_block<Us_block<2, float> >::value == true));
  // test_assert((is_modifiable_block<Us_block<3, complex<float> > >::value == true));

  // Has_modifiable mostly does the right thing on its own.
  test_assert((is_modifiable_block<noput_block<float> >::value == false));
  test_assert((is_modifiable_block<put_block<float> >::value == true));

  // However, it can get confused.
  test_assert((is_modifiable_block<derived_block>::value == false));
}



template <typename ViewT>
void
check_modifiable(ViewT, bool modifiable)
{
  using vsip::impl::is_modifiable_block;
  test_assert((is_modifiable_block<typename ViewT::block_type>::value ==
	  modifiable));
}



void
test_modifiable_2()
{
  Vector<float> vec(16);
  Matrix<float> mat(8, 12);
  Matrix<float> ten(3, 4, 5);

  check_modifiable(vec, true);
  check_modifiable(vec(Domain<1>(4)), true);

  check_modifiable(vec + vec, false);
  check_modifiable((vec + vec)(Domain<1>(4)), false);

  check_modifiable(mat, true);
  check_modifiable(mat.row(0), true);
  check_modifiable(mat.diag(0), true);
  check_modifiable(mat.transpose(), true);
  check_modifiable(mat(Domain<2>(4, 6)), true);

  check_modifiable(mat + 1, false);
  check_modifiable((mat + 1).col(0), false);
  check_modifiable((-mat).transpose(), false);
  check_modifiable((mat + mat)(Domain<2>(4, 6)), false);

  check_modifiable(ten, true);

  check_modifiable(ten + ten, false);


  Vector<complex<float> > cvec(16);

  check_modifiable(cvec, true);
  check_modifiable(cvec(Domain<1>(4)), true);
  check_modifiable(cvec.real(), true);
  check_modifiable(cvec.imag(), true);

  check_modifiable(cvec.real() + cvec.imag(), false);
  check_modifiable((cvec + cvec).real(), false);
}



/***********************************************************************
  Main
***********************************************************************/

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  test_modifiable_1();
  test_modifiable_2();

  return 0;
}
