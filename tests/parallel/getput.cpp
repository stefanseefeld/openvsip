/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/parallel/getput.cpp
    @author  Jules Bergmann
    @date    2005-12-24
    @brief   VSIPL++ Library: Unit tests for distributed blocks.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/map.hpp>
#include <vsip/tensor.hpp>
#include <vsip/parallel.hpp>

#include <vsip_csl/test.hpp>
#include <vsip_csl/output.hpp>
#include "util.hpp"
#include "util-par.hpp"

using namespace std;
using namespace vsip;
using namespace vsip_csl;


/***********************************************************************
  Definitions
***********************************************************************/

template <typename T>
void
test_getput(length_type size)
{
  typedef Map<Block_dist>                  map_type;
  typedef Dense<1, T, row1_type, map_type> block_type;
  typedef Vector<T, block_type>            view_type;

  map_type    map(num_processors());
  view_type   view(size, T(), map);

  for (index_type i=0; i<size; ++i)
  {
    view.put(i, T(i));
  }

  for (index_type i=0; i<size; ++i)
  {
    test_assert(equal(view.get(i), T(i)));
  }
}



template <typename T,
	  typename MapT>
void
test_getput(length_type rows, length_type cols)
{
  typedef Dense<2, T, row2_type, MapT> block_type;
  typedef Matrix<T, block_type>        view_type;

  MapT      map(num_processors(), 1);
  view_type view(rows, cols, T(), map);

  for (index_type r=0; r<rows; ++r)
    for (index_type c=0; c<cols; ++c)
      view.put(r, c, T(r*cols+c));

  for (index_type r=0; r<rows; ++r)
    for (index_type c=0; c<cols; ++c)
      test_assert(equal(view.get(r, c), T(r*cols+c)));
}



template <typename T,
	  typename MapT>
void
test_getput(length_type len0, length_type len1, length_type len2)
{
  typedef Dense<3, T, row3_type, MapT> block_type;
  typedef Tensor<T, block_type>        view_type;

  MapT      map(num_processors(), 1);
  view_type view(len0, len1, len2, T(), map);

  for (index_type i=0; i<len0; ++i)
    for (index_type j=0; j<len1; ++j)
      for (index_type k=0; k<len2; ++k)
	view.put(i, j, k, T(i*len1*len2 + j*len2 + k));

  for (index_type i=0; i<len0; ++i)
    for (index_type j=0; j<len1; ++j)
      for (index_type k=0; k<len2; ++k)
	test_assert(equal(view.get(i, j, k), T(i*len1*len2 + j*len2 + k)));
}



/// Test assign_local

template <typename T,
	  typename MapT>
void
test_assign_local(length_type rows, length_type cols)
{
  typedef Dense<2, T, row2_type, MapT> block_type;
  typedef Matrix<T, block_type>        view_type;

  MapT      map(num_processors(), 1);
  view_type g1_view(rows, cols, T(0), map);
  view_type g2_view(rows, cols, T(0), map);
  Matrix<T> l_view(rows, cols, T(0));

  for (index_type r=0; r<rows; ++r)
    for (index_type c=0; c<cols; ++c)
      g1_view.put(r, c, T(r*cols+c));

  assign_local(l_view,  g1_view);
  assign_local(g2_view, l_view);

  for (index_type r=0; r<rows; ++r)
    for (index_type c=0; c<cols; ++c)
      test_assert(equal(g2_view.get(r, c), T(r*cols+c)));
}




int
main(int argc, char** argv)
{
  vsipl vpp(argc, argv);

#if 0
  // Enable this section for easier debugging.
  impl::Communicator comm = impl::default_communicator();
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

  test_getput<float>(8);
  test_getput<float, Map<Block_dist, Block_dist> >(5, 7);
  test_getput<float, Map<Block_dist, Block_dist, Block_dist> >(5, 7, 3);
  test_assign_local<float, Map<Block_dist, Block_dist> >(5, 7);

  return 0;
}
