/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/fftshift.cpp
    @author  Jules Bergmann
    @date    2006-12-08
    @brief   VSIPL++ Library: Test fftshift, including distributed cases.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>
#include <algorithm>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/parallel.hpp>
#include <vsip/core/metaprogramming.hpp>

#include <vsip_csl/matlab_utils.hpp>
#include <vsip_csl/test.hpp>
#include <vsip_csl/output.hpp>

using namespace std;
using namespace vsip;
using namespace vsip_csl;

using vsip_csl::equal;
using vsip::impl::Int_type;


/***********************************************************************
  Algorithms for fftshift
***********************************************************************/

// Algorithm #1 -- use stock fftshift from vsip_csl

template <typename T1,
          typename T2,
          typename Block1,
          typename Block2>
void
fftshift(
  const_Matrix<T1, Block1> in,
  Matrix<T2, Block2>       out,
  Int_type<1>)
{
  vsip_csl::matlab::fftshift(in, out);
}



// Algorithm #2 -- process by row

template <typename T1,
          typename T2,
          typename Block1,
          typename Block2>
void
fftshift(
  const_Matrix<T1, Block1> in,
  Matrix<T2, Block2>       out,
  Int_type<2>)
{
  length_type rows = in.size(0); test_assert(rows == out.size(0));
  length_type cols = in.size(1); test_assert(cols == out.size(1));

  Domain<1> ldom(0,      1, cols/2);
  Domain<1> rdom(cols/2, 1, cols/2);

  for (index_type r=0; r<rows; ++r)
  {
    index_type xr = (r < rows/2) ? (rows/2 + r) : (r - rows/2);
    out.row(xr)(ldom) = in.row(r)(rdom);
    out.row(xr)(rdom) = in.row(r)(ldom);
  }
}



// Algorithm #3 -- process by row/col as guided by output

template <typename T1,
          typename T2,
          typename Block1,
          typename Block2>
void
fftshift(
  const_Matrix<T1, Block1> in,
  Matrix<T2, Block2>       out,
  Int_type<3>)
{
  length_type rows = in.size(0); test_assert(rows == out.size(0));
  length_type cols = in.size(1); test_assert(cols == out.size(1));

  // if distributed by row
  if (out.block().map().num_subblocks(1) == 1)
  {
    Domain<1> ldom(0,      1, cols/2);
    Domain<1> rdom(cols/2, 1, cols/2);

    for (index_type r=0; r<rows; ++r)
    {
      index_type xr = (r < rows/2) ? (rows/2 + r) : (r - rows/2);
      out.row(xr)(ldom) = in.row(r)(rdom);
      out.row(xr)(rdom) = in.row(r)(ldom);
    }
  }
  else
  {
    Domain<1> ldom(0,      1, rows/2);
    Domain<1> rdom(rows/2, 1, rows/2);

    for (index_type c=0; c<cols; ++c)
    {
      index_type xc = (c < cols/2) ? (cols/2 + c) : (c - cols/2);
      out.col(xc)(ldom) = in.col(c)(rdom);
      out.col(xc)(rdom) = in.col(c)(ldom);
    }
  }
}



// Algorithm #4 -- fft shift on root.

template <typename T1,
          typename T2,
          typename Block1,
          typename Block2>
void
fftshift(
  const_Matrix<T1, Block1> in,
  Matrix<T2, Block2>       out,
  Int_type<4>)
{
  Matrix<T2, Dense<2, T2, row2_type, Map<> > > rin (in.size(0), in.size(1));
  Matrix<T2, Dense<2, T2, row2_type, Map<> > > rout(out.size(0), out.size(1));

  rin = in;
  vsip_csl::matlab::fftshift(rin, rout);
  out = rout;
}



/***********************************************************************
  Test driver
***********************************************************************/

template <typename T,
	  int      Impl,
          typename OrderIn,
          typename OrderOut,
	  typename MapInT,
	  typename MapOutT>
void
test_fftshift(
  MapInT&     map_in,
  MapOutT&    map_out,
  length_type rows,
  length_type cols)
{
  typedef Dense<2, T, OrderIn, MapInT>    in_block_type;
  typedef Dense<2, T, OrderOut, MapOutT>  out_block_type;
  typedef Matrix<T, in_block_type>        in_view_type;
  typedef Matrix<T, out_block_type>        out_view_type;

  in_view_type  in(rows, cols,  T(-1), map_in);
  out_view_type out(rows, cols, T(-2), map_out);
   
  // setup input.
  if (subblock(in) != no_subblock)
  {
    for (index_type lr=0; lr<in.local().size(0); ++lr)
      for (index_type lc=0; lc<in.local().size(1); ++lc)
      {
	index_type gr = global_from_local_index(in, 0, lr); 
	index_type gc = global_from_local_index(in, 1, lc); 
	in.local().put(lr, lc, T(gr*cols + gc));
      }
  }

  // shift it.
  fftshift(in, out, Int_type<Impl>());

  // checkout output.
  if (subblock(out) != no_subblock)
  {
    for (index_type lr=0; lr<out.local().size(0); ++lr)
      for (index_type lc=0; lc<out.local().size(1); ++lc)
      {
	index_type gr = global_from_local_index(out, 0, lr); 
	index_type gc = global_from_local_index(out, 1, lc); 
	index_type xr = (gr < rows/2) ? (rows/2 + gr) : (gr - rows/2);
	index_type xc = (gc < cols/2) ? (cols/2 + gc) : (gc - cols/2);
	test_assert(equal(out.local().get(lr, lc),
			  T(xr*cols + xc)));
      }
  }
}



/***********************************************************************
  Main
***********************************************************************/

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

#if 0
  // Enable this section for easier debugging.
  vsip::impl::Communicator& comm = impl::default_communicator();
  pid_t pid = getpid();

  cout << "rank: "   << comm.rank()
       << "  size: " << comm.size()
       << "  pid: "  << pid
       << endl;

  // Stop each process, allow debugger to be attached.
  if (comm.rank() == 0) fgetc(stdin);
  comm.barrier();
  cout << "start\n";
#endif

  length_type np = num_processors();


#if 0
  // Simple case for debugging.
  Map<> r_map(np, 1);
  Map<> c_map(1, np);
  test_fftshift<float, 1, row2_type, row2_type>(r_map, c_map, 256, 256);
#endif

  // fftshift local views. --------------------------------------------

  Local_map l_map;

  test_fftshift<float, 1, row2_type, row2_type>(l_map, l_map, 256, 256);
  test_fftshift<float, 1, row2_type, col2_type>(l_map, l_map, 64, 48);
  test_fftshift<float, 1, col2_type, row2_type>(l_map, l_map, 128, 256);
  test_fftshift<float, 1, col2_type, col2_type>(l_map, l_map, 34, 50);
  test_fftshift<complex<float>, 1, row2_type, row2_type>(l_map, l_map, 12, 28);
  test_fftshift<complex<float>, 1, row2_type, col2_type>(l_map, l_map, 84, 18);
  test_fftshift<complex<float>, 1, col2_type, row2_type>(l_map, l_map, 32, 16);
  test_fftshift<complex<float>, 1, col2_type, col2_type>(l_map, l_map, 16, 32); 

  test_fftshift<float, 2, row2_type, row2_type>(l_map, l_map, 256, 256);
  test_fftshift<float, 3, row2_type, row2_type>(l_map, l_map, 256, 256);

  // Algorithm #4 only works on distributed views.
  // test_fftshift<float, 4, row2_type, row2_type>(l_map, l_map, 256, 256);


  // fftshift distributed views ---------------------------------------

  Map<> r_map(np, 1);
  Map<> c_map(1, np);

  test_fftshift<float, 1, row2_type, row2_type>(r_map, r_map, 256, 256);
  test_fftshift<float, 2, row2_type, row2_type>(r_map, r_map, 256, 256);
  test_fftshift<float, 3, row2_type, row2_type>(r_map, r_map, 256, 256);
  test_fftshift<float, 4, row2_type, row2_type>(r_map, r_map, 256, 256);

  test_fftshift<float, 1, row2_type, row2_type>(c_map, c_map, 256, 256);
  test_fftshift<float, 2, row2_type, row2_type>(c_map, c_map, 256, 256);
  test_fftshift<float, 3, row2_type, row2_type>(c_map, c_map, 256, 256);
  test_fftshift<float, 4, row2_type, row2_type>(c_map, c_map, 256, 256);

  test_fftshift<float, 1, row2_type, row2_type>(r_map, c_map, 256, 256);
  test_fftshift<float, 2, row2_type, row2_type>(r_map, c_map, 256, 256);
  test_fftshift<float, 3, row2_type, row2_type>(r_map, c_map, 256, 256);
  test_fftshift<float, 4, row2_type, row2_type>(r_map, c_map, 256, 256);

  test_fftshift<float, 1, row2_type, row2_type>(c_map, r_map, 256, 256);
  test_fftshift<float, 2, row2_type, row2_type>(c_map, r_map, 256, 256);
  test_fftshift<float, 3, row2_type, row2_type>(c_map, r_map, 256, 256);
  test_fftshift<float, 4, row2_type, row2_type>(c_map, r_map, 256, 256);
}
