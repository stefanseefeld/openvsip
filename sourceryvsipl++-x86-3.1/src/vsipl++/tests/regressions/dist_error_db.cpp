/* Copyright (c) 2007 by CodeSourcery.  All rights reserved. */

/// Description
///   Regression for distributed error_db.
///
///   This regression covers three errors.  See description of test_error_db.

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/random.hpp>

#include <vsip_csl/test.hpp>
#include <vsip_csl/error_db.hpp>

using namespace vsip;
using namespace vsip_csl;

// Test error_db between two distributed views with identical data.

// Distributed error_db was triggering three errors:
//  - compile-time error in par_reductions
//    (covered here and in test_maxval_1).
//  - run-time error with np == 2, MPI does not support reductions on
//    vectors of MPI_BOOL, causing distributed anytrue to fail.
//    (covered here and test_anytrue).
//  - run-time error with np == 2, converting a ternary_expr_block
//    containing a scalar_block from global to local was not updating
//    size stored in Scalar_block.
//    (covered here and in test_ternary_expr_size).

template <typename T,
	  typename Map1T,
	  typename Map2T>
void
test_error_db(Map1T const& map1, Map2T const& map2, length_type size)
{
  typedef Dense<1, T, row1_type, Map1T> block1_type;
  typedef Dense<1, T, row1_type, Map2T> block2_type;

  Vector<T, block1_type> A(size, map1);
  Vector<T, block2_type> B(size, map2);

  Rand<T> gen(0, 0);
  A = gen.randu(size);
  B = A;

  double error = error_db(A, B);

  test_assert(error < -100.0);
}



// Test maxval(magsq(A), idx) of view with map 'map'.

// Triggers a bug when np == 2, MPI cannot reduce a struct of MPI_BOOL.
// Fixed.

template <typename MapT>
void
test_maxval_1(MapT const& map, length_type size)
{
  dimension_type const dim = 1;
  typedef complex<float> T;
  typedef Dense<dim, T, row1_type, MapT> block_type;

  Index<dim> idx;

  Vector<T, block_type> A(size, map);

  Rand<T> gen(0, 0);
  A = gen.randu(size);

  float refmax1 = maxval(magsq(A), idx);

  float refmax2 = magsq(A.get(0));

  for (index_type i=1; i<size; ++i)
    if (magsq(A.get(i)) > refmax2)
      refmax2 = magsq(A.get(i));

  test_assert(equal(refmax1, refmax2));
}



template <typename Map1T,
	  typename Map2T>
void
test_maxval_2(
  Map1T const& map1,
  Map2T const& map2, 
  length_type  size)
{
  dimension_type const dim = 1;
  typedef complex<float> T;

  typedef Dense<dim, T, row1_type, Map1T> block1_type;
  typedef Dense<dim, T, row1_type, Map2T> block2_type;

  Index<dim> idx;

  Vector<T, block1_type> A(size, map1);
  Vector<T, block2_type> B(size, map2);

  Rand<T> gen(0, 0);
  A = gen.randu(size);
  B = gen.randu(size);

  float refmax1 = maxval(ite(magsq(A) > magsq(B), magsq(A), magsq(B)), idx);

  (void)refmax1;
}



// Test distributed anytrue(A, idx) of view with map 'map'.

// Triggers a bug when np == 2, MPI cannot reduce a struct of MPI_BOOL.
// Fixed.

template <typename MapT>
void
test_anytrue(MapT const& map, length_type size)
{
  dimension_type const dim = 1;
  typedef Dense<dim, bool, row1_type, MapT> block_type;

  Index<dim> idx;

  Vector<bool, block_type> A(size, false, map);

  bool at = anytrue(A);
  test_assert(at == false);

  A.put(0, true);

  at = anytrue(A);
  test_assert(at == true);
}



// Helper function for test_ternary_expr_size

template <typename ViewT>
void
check_block_size(length_type g_size, length_type l_size, ViewT const& view)
{
  test_assert(g_size == view.size(0));
  test_assert(g_size == view.block().size(1, 0));

  test_assert(l_size == view.local().size(0));
  test_assert(l_size == view.local().block().size(1, 0));

  test_assert(g_size == view.size());
  test_assert(g_size == view.block().size());

  test_assert(l_size == view.local().size());
  test_assert(l_size == view.local().block().size());
}



// Test is size of ternary expr block can be accessed.
//
// Triggers a bug when np == 2 and MapT == Map<>.
//
// Problem is Scalar_block gets initialized with global size, but
// global to local conversion keeps global size, causing an assertion
// failure in expr::Ternary.  Could fix Scalar_block to do the
// right thing ... but eventually we want to move the size knowledge
// out altogether.  Fix is to have expr::Ternary use
// Is_sized_block.
// 
// Fixed.

template <typename MapT>
void
test_ternary_expr_size(MapT const& map, length_type size)
{
  typedef float T;

  Rand<T> gen(0, 0);

  typedef Dense<1, T, row1_type, MapT> block_type;

  Vector<T, block_type> A(size, map);
  Vector<T, block_type> B(size, map);
  Vector<T, block_type> C(size, map);

  Index<1> idx;

  A = gen.randu(size);
  B = gen.randu(size);

  length_type l_size = A.local().size();

  check_block_size(size, l_size, ite(A > B, sq(A), sq(B)));  // OK
  check_block_size(size, l_size, ite(A > B, -100.0, sq(B))); // Error
}



int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  length_type np = num_processors();

  Local_map     l_map;
  Map<>         g_map(np);
  Map<>         r_map(1);
  Replicated_map<1> x_map;

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

  test_error_db<complex<float> >(l_map, l_map, 1024);
  test_error_db<complex<float> >(g_map, g_map, 1028);
  test_error_db<complex<float> >(r_map, r_map, 1023);
  test_error_db<complex<float> >(x_map, x_map, 1023);


  test_maxval_1(l_map, 16);
  test_maxval_1(g_map, 16);

  test_ternary_expr_size(g_map, 18);

  test_maxval_2(g_map, g_map, 16);

  test_anytrue(l_map, 16);
  test_anytrue(g_map, 18);

# if 0
  // Reductions of complex distributed expressions (expressions with
  // blocks that have different distributions) are not implemented.

  test_error_db<complex<float> >(g_map, r_map, 1028);
  test_error_db<complex<float> >(g_map, x_map, 1028);
  test_maxval_2(g_map, r_map, 16);
#endif
}
