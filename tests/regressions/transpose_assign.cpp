//
// Copyright (c) 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

//    This test triggers a bug with Intel C++ for Windows 9.1 Build 20060816Z,
//    32-bit version.  Soucery VSIPL++ works around this bug by disabling
//    some dispatch in fast-transpose.hpp.

#include <memory>
#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/domain.hpp>
#include <vsip/random.hpp>
#include <test.hpp>
#include "../test_common.hpp"

using namespace ovxx;

/***********************************************************************
  Definitions - Utility Functions
***********************************************************************/

// High-level transpose test.

template <typename T,
	  typename DstOrderT,
	  typename SrcOrderT>
void
test_hl(length_type rows, length_type cols)
{
  typedef Dense<2, T, SrcOrderT> src_block_type;
  typedef Dense<2, T, DstOrderT> dst_block_type;

  Matrix<T, src_block_type> src(rows, cols, T(-1));
  Matrix<T, dst_block_type> dst(rows, cols, T(-2));

  setup(src, 1);

  dst = src;

  check(dst, 1);
}



template <typename T,
	  typename DstOrderT,
	  typename SrcOrderT>
void
cover_hl(int verbose)
{
  if (verbose >= 1) std::cout << "cover_hl\n";
  // These tests fail for Intel C++ 9.1 for Windows prior
  // to workaround in fast-transpose.hpp:
  test_hl<T, DstOrderT, SrcOrderT>(5, 3);  // known bad case
  test_hl<T, DstOrderT, SrcOrderT>(16, 3); // known bad case
  test_hl<T, DstOrderT, SrcOrderT>(17, 3); // known bad case

  {
    length_type max_rows = 32;
    length_type max_cols = 32;
    for (index_type rows=1; rows<max_rows; ++rows)
    {
      if (verbose >= 2) std::cout << " - " << rows << " / " << max_rows << "\n";
      for (index_type cols=1; cols<max_cols; ++cols)
	test_hl<T, DstOrderT, SrcOrderT>(rows, cols);
    }
  }

  {
    length_type max_rows = 256;
    length_type max_cols = 256;
    for (index_type rows=1; rows<max_rows; rows+=3)
    {
      if (verbose >= 2) std::cout << " - " << rows << " / " << max_rows << "\n";
      for (index_type cols=1; cols<max_cols; cols+=5)
      {
	test_hl<T, DstOrderT, SrcOrderT>(rows, cols);
	test_hl<T, DstOrderT, SrcOrderT>(cols, rows);
      }
    }
  }
}



// Low-level transpose test (call transpose_unit directly).

template <typename T>
void
test_ll(length_type rows, length_type cols)
{
  std::auto_ptr<T> src(new T[rows*cols]);
  std::auto_ptr<T> dst(new T[rows*cols]);

  for (index_type r=0; r<rows; r++)
    for (index_type c=0; c<cols; c++)
    {
      src.get()[r*cols + c] = T(100*r + c);
      dst.get()[r + c*rows] = T(-100);
    }

  assignment::transpose(dst.get(), rows,
			src.get(), cols,
			rows, cols);

  for (index_type r=0; r<rows; r++)
    for (index_type c=0; c<cols; c++)
      test_assert(dst.get()[r + c*rows] == T(100*r + c));

  delete[] dst.release();
  delete[] src.release();
}


template <typename T>
void
cover_ll(int verbose)
{
  if (verbose >= 1) std::cout << "cover_ll\n";
  {
    length_type max_rows = 32;
    length_type max_cols = 32;
    for (index_type rows=1; rows<max_rows; ++rows)
    {
      if (verbose >= 2) std::cout << " - " << rows << " / " << max_rows << "\n";
      for (index_type cols=1; cols<max_cols; ++cols)
	test_ll<T>(rows, cols);
    }
  }

  {
    length_type max_rows = 256;
    length_type max_cols = 256;
    for (index_type rows=1; rows<max_rows; rows+=3)
    {
      if (verbose >= 2) std::cout << " - " << rows << " / " << max_rows << "\n";
      for (index_type cols=1; cols<max_cols; cols+=5)
      {
	test_ll<T>(rows, cols);
	test_ll<T>(cols, rows);
      }
    }
  }
}



int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  int verbose = 0;
  if (argc == 2 && argv[1][0] == '1') verbose = 1;
  if (argc == 2 && argv[1][0] == '2') verbose = 2;

  cover_hl<float, row2_type, col2_type>(verbose);
  cover_hl<complex<float>, row2_type, col2_type>(verbose);

  cover_ll<float>(verbose);
  cover_ll<complex<float> >(verbose);

  return 0;
}
