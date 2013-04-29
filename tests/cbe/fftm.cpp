/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/** @file    tests/plugin/fftm.cpp
    @author  Jules Bergmann
    @date    2009-06-02
    @brief   VSIPL++ Library: Test Fftm cases supported by CBE plugins.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#define VERBOSE 1

#if VERBOSE
#  include <iostream>
#endif

#include <algorithm>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/signal.hpp>
#include <vsip/map.hpp>

#include <vsip_csl/test.hpp>

using namespace std;
using namespace vsip;


/***********************************************************************
  Definitions
***********************************************************************/

// By-value Fftm, out-of-place

template <typename T,
	  typename MapT>
void
test_fftm_bv_op(
  length_type rows,
  length_type cols,
  length_type in_cols,
  length_type out_cols,
  bool        scale,
  MapT const& map = MapT())
{
  typedef Fftm<T, T, row, fft_fwd, by_value, 1> fftm_type;

  fftm_type fftm(Domain<2>(rows, cols), scale ? 1.f / cols : 1.f);

  typedef Matrix<T, Dense<2, T, row2_type, MapT> > matrix_type;
  typedef typename matrix_type::subview_type       subview_type;

  matrix_type big_in (rows, in_cols,           map);
  matrix_type big_out(rows, out_cols, T(-100), map);

  Domain<2> dom(rows, cols);

  subview_type in  = big_in(dom);
  subview_type out = big_out(dom);

  for (index_type r=0; r<rows; ++r)
    in.row(r) = T(r+1);

  out = fftm(in); 

  for (index_type r=0; r<rows; ++r)
  {
    if (!(out.get(r, 0) == T(scale ? r+1 : (r+1)*cols)))
    {
      cout << "test_fftm_bv_op: miscompare for row " << r << endl
	   << "  expected: " << T(scale ? (r+1) : (r+1)*cols) << endl
	   << "  got     : " << out.get(r, 0) << endl;
    }
    test_assert(out.get(r, 0) == T(scale ? r+1 : (r+1)*cols));
  }
}



/***********************************************************************
  Main
***********************************************************************/

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  // Plugins support sizes 4 through 8192, test across and beyond that range
  for (length_type size=4; size <= 16384; size *= 2)
  {
    test_fftm_bv_op<complex<float>, Local_map>(128, size, size, size, true);
    test_fftm_bv_op<complex<float>, Local_map>(128, size, size, size+16, true);
    test_fftm_bv_op<complex<float>, Local_map>(128, size, size+16, size, true);
    test_fftm_bv_op<complex<float>, Local_map>(128, size, size+16, size+16,
					       true);
  }

  return 0;
}
