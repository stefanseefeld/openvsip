/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/corner-turn.cpp
    @author  Jules Bergmann
    @date    2006-02-14
    @brief   VSIPL++ Library: Functional test for corner-turns.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>

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
corner_turn(
  length_type rows,
  length_type cols)
{
  typedef Map<Block_dist, Block_dist>      map_type;
  typedef Dense<2, T, row2_type, map_type> block_type;

  processor_type np   = num_processors();

  map_type root_map(1, 1);
  map_type row_map (np, 1);
  map_type col_map (1, np);

  Matrix<T, block_type> src(rows, cols, root_map);
  Matrix<T, block_type> A  (rows, cols, row_map);
  Matrix<T, block_type> B  (rows, cols, col_map);
  Matrix<T, block_type> dst(rows, cols, root_map);

  if (root_map.subblock() != no_subblock)
  {
    // cout << local_processor() << "/" << np << ": initializing " << endl;
    for (index_type r=0; r<rows; ++r)
      for (index_type c=0; c<cols; ++c)
	src.local().put(r, c, T(r*cols+c));
  }

  A   = src; // scatter
  B   = A;   // corner-turn
  dst = B;   // gather

  if (root_map.subblock() != no_subblock)
  {
    // cout << local_processor() << "/" << np << ": checking " << endl;
    for (index_type r=0; r<rows; ++r)
      for (index_type c=0; c<cols; ++c)
	test_assert(equal(src.local().get(r, c),
			  dst.local().get(r, c)));
  }
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
  
  corner_turn<float>(32, 64);

  corner_turn<complex<float> >(32, 64);
  corner_turn<complex<float> >(31, 15);

  corner_turn<complex<float> >(11, 1);
  corner_turn<complex<float> >(11, 2);
  corner_turn<complex<float> >(11, 3);
  corner_turn<complex<float> >(11, 4);
  corner_turn<complex<float> >(11, 5);
  corner_turn<complex<float> >(11, 6);
  corner_turn<complex<float> >(11, 7);

  return 0;
}
