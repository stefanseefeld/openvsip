/* Copyright (c) 2009 by CodeSourcery, LLC.  All rights reserved. */

/** @file    tests/plugin/vmmul.cpp
    @author  Jules Bergmann
    @date    2009-05-22
    @brief   VSIPL++ Library: Test coverage for vmmul plugin cases.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#define VERBOSE 1

#if VERBOSE
#  include <iostream>
#endif

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/domain.hpp>
#include <vsip/random.hpp>
#include <vsip/selgen.hpp>

#include <vsip_csl/test.hpp>

using namespace vsip;
using vsip_csl::equal;

/***********************************************************************
  Definitions - Utility Functions
***********************************************************************/

template <typename VT,
	  typename MT,
	  storage_format_type C,
	  int      SD>
void
t_vmmul(length_type rows, length_type cols)
{
  using namespace std;

  typedef Layout<1, row1_type, dense, C> LP1;
  typedef Layout<2, row2_type, dense, C> LP2;
  typedef impl::Strided<1, VT, LP1> block1_type;
  typedef impl::Strided<2, MT, LP2> block2_type;

  length_type v_size = SD == row ? cols : rows;

  Matrix<MT, block2_type> in (rows, cols, MT(3));
  Matrix<MT, block2_type> out(rows, cols, MT(-100));
  Vector<VT, block1_type> vec(v_size, VT(4));

  vec = ramp<VT>(VT(0), VT(1), v_size);

  out = vmmul<SD>(vec, in);

  if (SD == row)
  {
    for (index_type r=0; r<rows; ++r)
      for (index_type c=0; c<cols; ++c)
      {
#if VERBOSE
	if (!equal(out.get(r, c), vec.get(c) * in.get(r, c)))
	{
	  cout << "r, c:             = " << r << ", " << c << endl;
	  cout << "out(r, c)         = " << out(r, c) << endl;
	  cout << "vec(c) * in(r, c) = " << vec.get(c) * in.get(r, c) << endl;
	}
#endif
	test_assert(out.get(r, c) == vec.get(c) * in.get(r, c));
      }
  }
  else
  {
    for (index_type r=0; r<rows; ++r)
      for (index_type c=0; c<cols; ++c)
      {
#if VERBOSE
	if (!equal(out.get(r, c), vec.get(r) * in.get(r, c)))
	{
	  cout << "r, c:             = " << r << ", " << c << endl
	       << "out(r, c)         = " << out(r, c) << endl
	       << "vec(r) * in(r, c) = " << vec.get(r) * in.get(r, c)
	       << " = " << vec.get(r) << " * " << in.get(r, c) << endl;
	}
#endif
	test_assert(out.get(r, c) == vec.get(r) * in.get(r, c));
      }
  }
}



template <typename VT,
	  typename MT,
	  storage_format_type C,
	  int      SD>
void
test_vmmul(char const* what)
{
#if VERBOSE
  std::cout << what << std::endl;
#else
  (void)what;
#endif

  t_vmmul<VT, MT, C, SD>(32, 1024);
  t_vmmul<VT, MT, C, SD>(32, 8192);
  t_vmmul<VT, MT, C, SD>(8192, 32);
  t_vmmul<VT, MT, C, SD>(2048, 1024);
  t_vmmul<VT, MT, C, SD>(256, 8192);
}



int
main(int argc, char** argv)
{
  typedef float          F;
  typedef complex<float> CF;

  vsipl init(argc, argv);

  test_vmmul<F,  F,  split_complex, row>("rr row");
  test_vmmul<F,  F,  split_complex, row>("rr col");

  test_vmmul<CF, CF, interleaved_complex, row>("cc row"); // dispatched
  test_vmmul<CF, CF, interleaved_complex, col>("cc col");

  test_vmmul<CF, CF, split_complex, row>("zz row"); // dispatched
  test_vmmul<CF, CF, split_complex, col>("zz col"); // dispatched

  test_vmmul<F,  CF, interleaved_complex, row>("rc row");
  test_vmmul<F,  CF, interleaved_complex, col>("rc col");

  test_vmmul<F,  CF, split_complex, row>("rz row");
  test_vmmul<F,  CF, split_complex, col>("rz col"); // dispatched
}
