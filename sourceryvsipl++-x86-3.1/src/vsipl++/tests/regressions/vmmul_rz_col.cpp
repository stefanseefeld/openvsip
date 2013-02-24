/* Copyright (c) 2008 by CodeSourcery.  All rights reserved. */

/** @file    tests/regressions/vmmul_rz_col.cpp
    @author  Jules Bergmann
    @date    2008-12-30
    @brief   VSIPL++ Library: Regression test for vmmul (issue #243).
*/

/***********************************************************************
  Included Files
***********************************************************************/

#define SHOW_DIAG 0
#define VERBOSE 0

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
#include <vsip/opt/assign_diagnostics.hpp>

#include <vsip_csl/test.hpp>

using namespace vsip;


/***********************************************************************
  Definitions - Utility Functions
***********************************************************************/

template <typename VT,
	  typename MT,
	  storage_format_type C,
	  int      SD>
void
test_vmmul(length_type rows, length_type cols)
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

#if SHOW_DIAG
  vsip_csl::assign_diagnostics(out, vmmul<SD>(vec, in));
#endif
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



int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  test_vmmul<float, complex<float>, split_complex, col>(32, 1024);
}
