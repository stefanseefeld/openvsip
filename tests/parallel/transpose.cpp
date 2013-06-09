//
// Copyright (c) 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Unit tests for parallel matrix transpose.

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/domain.hpp>
#include <vsip/random.hpp>
#include <vsip/signal.hpp>
#include <vsip/selgen.hpp>
#include <test.hpp>
#include "util.hpp"

using namespace ovxx;

// Test, exercising subviews
template <typename MapT>
void
test_subviews(MapT &map, length_type rows, length_type cols, bool verbose)
{
  length_type row, col;

  typedef complex<float> value_type;

  typedef Dense<2, value_type, row2_type, MapT> block_type;
  typedef Matrix<value_type, block_type>        view_type;

  view_type in1(rows, cols, map);
  view_type in2(rows, cols, map);
  view_type tp1(cols, rows, map);
  view_type tp2(cols, rows, map);
  view_type tp3(cols, rows, map);
  view_type tp4(cols, rows, map);
  view_type tp5(cols, rows, map);
  view_type tp6(cols, rows, map);

  // Fill in the matrix with data
  for (row = 0; row < rows; row++)
  {
    in1.row(row).real() = +(100.0*row + ramp<float>(0, 1, cols));
    in1.row(row).imag() = -(100.0*row + ramp<float>(0, 1, cols));

    in2.row(row).real() = +(1.0*row + ramp<float>(0, 100, cols));
    in2.row(row).imag() = -(1.0*row + ramp<float>(0, 100, cols));
  }

  tp1 = in1.transpose();
  tp2 = in1.transpose() + in2.transpose();
  tp3 = tp1 + in1.transpose() + in2.transpose();
  // Not supported:
  // tp4 = in1(Domain<2>(rows, cols)).transpose();
  tp4 = in1.transpose()(Domain<2>(cols, rows));
  tp5(Domain<2>(cols, rows)) = in1.transpose()(Domain<2>(cols, rows));
  tp6(Domain<2>(cols, rows)) = in1.transpose()(Domain<2>(cols, rows)) +
    in2.transpose();

  if (verbose)
  {
    dump_view("in1", in1);
    dump_view("in2", in2);
    dump_view("tp1", tp1);
    dump_view("tp2", tp2);
  }

  for (row = 0; row < rows; row++)
    for (col = 0; col < cols; col++)
    {
      test_assert(in1.get(row, col).real() == +(100.0*row + 1.0*col));
      test_assert(in1.get(row, col).imag() == -(100.0*row + 1.0*col));
      test_assert(in2.get(row, col).real() == +(1.0*row + 100.0*col));
      test_assert(in2.get(row, col).imag() == -(1.0*row + 100.0*col));

      test_assert(tp1.get(col, row) == in1.get(row, col));
      test_assert(tp2.get(col, row) == (in1.get(row, col) + in2.get(row, col)));
      test_assert(tp3.get(col, row) == 
       	          (tp1.get(col, row) + in1.get(row, col) + in2.get(row, col)));
      test_assert(tp4.get(col, row) == in1.get(row, col));
      test_assert(tp5.get(col, row) == in1.get(row, col));
      test_assert(tp6.get(col, row) == 
                  (in1.get(row, col) + in2.get(row, col)));
    }
}



// Test, without exercising subviews

template <typename MapT>
void
test_wo_subviews(MapT &map, length_type rows, length_type cols)
{
  length_type row, col;

  typedef complex<float> value_type;

  typedef Dense<2, value_type, row2_type, MapT> block_type;
  typedef Matrix<value_type, block_type>        view_type;

  view_type in1(rows, cols, map);
  view_type in2(rows, cols, map);
  view_type tp1(cols, rows, map);
  view_type tp2(cols, rows, map);
  view_type tp3(cols, rows, map);

  // Fill in the matrix with data
  for (row = 0; row < rows; row++)
  {
    in1.row(row).real() = +(2.0*row + ramp<float>(0, 1, cols));
    in1.row(row).imag() = -(2.0*row + ramp<float>(0, 1, cols));

    in2.row(row).real() = +(1.0*row + ramp<float>(0, 2, cols));
    in2.row(row).imag() = -(1.0*row + ramp<float>(0, 2, cols));
  }

  tp1 = in1.transpose();
  tp2 = in1.transpose() + in2.transpose();
  tp3 = tp1 + in1.transpose() + in2.transpose();

  for (row = 0; row < rows; row++)
    for (col = 0; col < cols; col++)
    {
      test_assert(tp1.get(col, row) == in1.get(row, col));
      test_assert(tp2.get(col, row) == 
                  (in1.get(row, col) + in2.get(row, col)));
      test_assert(tp3.get(col, row) == 
       	          (tp1.get(col, row) + in1.get(row, col) + in2.get(row, col)));
    }
}

template <typename MapT>
void
test_x(MapT &map, length_type rows, length_type cols)
{
  length_type row, col;

  typedef complex<float> value_type;

  typedef Dense<2, value_type, row2_type, MapT> block_type;
  typedef Matrix<value_type, block_type>        view_type;

  view_type in1(rows, cols, map);
  view_type in2(rows, cols, map);
  view_type tp2(cols, rows, map);

  // Fill in the matrix with data
  for (row = 0; row < rows; row++)
  {
    in1.row(row).real() = +(2.0*row + ramp<float>(0, 1, cols));
    in1.row(row).imag() = -(2.0*row + ramp<float>(0, 1, cols));

    in2.row(row).real() = +(1.0*row + ramp<float>(0, 2, cols));
    in2.row(row).imag() = -(1.0*row + ramp<float>(0, 2, cols));
  }

  tp2 = in1.transpose() + in2.transpose();

  for (row = 0; row < rows; row++)
    for (col = 0; col < cols; col++)
    {
      test_assert(tp2.get(col, row) == 
		  (in1.get(row, col) + in2.get(row, col)));
    }
}




int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

#if 0
  // Enable this section for easier debugging.
  impl::Communicator& comm = impl::default_communicator();
  pid_t pid = getpid();

  std::cout << "rank: "   << comm.rank()
	    << "  size: " << comm.size()
	    << "  pid: "  << pid
	    << std::endl;

  // Stop each process, allow debugger to be attached.
  if (comm.rank() == 0) fgetc(stdin);
  comm.barrier();
  std::cout << "start\n";
#endif
  Map<> m;
  // Block-cylic maps ---------------------------------------------------
  {
    msg(m, "block-cyclic - 1\n");
    typedef Map<Block_dist, Whole_dist> map_type;
    map_type map = map_type(num_processors(), 1);
    test_subviews(map, 4, 8, true);
  }
  {
    msg(m, "block-cyclic - 2\n");
    typedef Map<Whole_dist, Block_dist> map_type;
    map_type map = map_type(1, num_processors());
    test_subviews(map, 4, 8, true);
  }
  {
    msg(m, "block-cyclic - 3\n");
    length_type np = num_processors();
    length_type npr, npc;
    get_np_square(np, npr, npc);
    typedef Map<Block_dist, Block_dist> map_type;
    map_type map = map_type(npr, npc);
    test_subviews(map, 4, 8, false);
  }
  {
    msg(m, "block-cyclic - 4\n");
    length_type np = num_processors();
    length_type npr, npc;
    get_np_square(np, npr, npc);
    typedef Map<Cyclic_dist, Block_dist> map_type;
    map_type map = map_type(npr, npc);
    test_wo_subviews(map, 16, 32);
  }

  // Local map ----------------------------------------------------------
  // {
  //   msg(m, "local\n");
  //   typedef Local_map map_type;
  //   map_type map;
  //   test_subviews(map, 4, 8, 0);
  // }

  // Replicated map ---------------------------------------------------------
  {
    msg(m, "replicated\n");
    typedef Replicated_map<2> map_type;
    map_type map;
    test_subviews(map, 4, 8, false);
  }
}
