/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Unit tests for parallel expressions

#include <iostream>
#include <cassert>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/map.hpp>
#include <vsip/math.hpp>
#include <vsip/parallel.hpp>
#include <vsip/core/length.hpp>
#include <vsip/core/domain_utils.hpp>

#include <vsip_csl/test.hpp>
#include <vsip_csl/output.hpp>
#include "util.hpp"
#include "util-par.hpp"

using namespace vsip;

using vsip::impl::Length;
using vsip::impl::valid;
using vsip::impl::extent;
using vsip::impl::next;

// Demonstrate pre-binding of an expression.

class Prebound_expr
{
private:
  class Holder_base
  {
  public:
    virtual ~Holder_base();
    virtual void exec() = 0;
  };

  template <dimension_type Dim,
	    typename       DstBlock,
	    typename       SrcBlock>
  class Holder : public Holder_base
  {
    typedef typename DstBlock::value_type value1_type;
    typedef typename SrcBlock::value_type value2_type;
  public:
    Holder(typename impl::view_of<DstBlock>::type dst,
	   typename impl::view_of<SrcBlock>::const_type src)
      : par_expr_(dst, src)
      {}

    ~Holder()
      {}

    void exec()
      { par_expr_(); }


    // Member data
  private:
    vsip::impl::Par_expr<Dim, DstBlock, SrcBlock> par_expr_;
  };

  // Constructors.
public:
   template <template <typename, typename> class View1,
	     template <typename, typename> class View2,
	     typename                            T1,
	     typename                            Block1,
	     typename                            T2,
	     typename                            Block2>
  Prebound_expr(
    View1<T1, Block1> dst,
    View2<T2, Block2> src)
    : holder_(new Holder<View1<T1, Block1>::dim, Block1, Block2>(dst, src))
  {}

  ~Prebound_expr() 
  { delete holder_; }

  void operator()()
  { holder_->exec(); }
  
// Member Data
private:
  Holder_base* holder_;

};


Prebound_expr::Holder_base::~Holder_base()
{}



// Test distributed expression with single binary-operator.

template <typename       T,
	  dimension_type Dim,
	  typename       MapRes,
	  typename       MapOp1,
	  typename       MapOp2>
void
test_distributed_expr(
  Domain<Dim> dom,
  MapRes      map_res,
  MapOp1      map_op1,
  MapOp2      map_op2,
  int         loop)

{
  typedef Map<Block_dist, Block_dist> map0_t;

  typedef typename impl::Row_major<Dim>::type order_type;

  typedef Dense<Dim, T, order_type, map0_t> dist_block0_t;
  typedef Dense<Dim, T, order_type, MapRes> dist_block_res_t;
  typedef Dense<Dim, T, order_type, MapOp1> dist_block_op1_t;
  typedef Dense<Dim, T, order_type, MapOp2> dist_block_op2_t;

  typedef typename impl::view_of<dist_block0_t>::type view0_t;
  typedef typename impl::view_of<dist_block_res_t>::type view_res_t;
  typedef typename impl::view_of<dist_block_op1_t>::type view_op1_t;
  typedef typename impl::view_of<dist_block_op2_t>::type view_op2_t;

  // map0 is not distributed (effectively).
  map0_t  map0(Block_dist(1), Block_dist(1));

  // Non-distributed view to check results.
  view0_t chk1(create_view<view0_t>(dom, map0));
  view0_t chk2(create_view<view0_t>(dom, map0));

  // Distributed views for actual parallel-expression.
  view_res_t Z1(create_view<view_res_t>(dom, T(0), map_res));
  view_res_t Z2(create_view<view_res_t>(dom, T(0), map_res));
  view_res_t Z3(create_view<view_res_t>(dom, T(0), map_res));
  view_res_t Z4(create_view<view_res_t>(dom, T(0), map_res));
  view_res_t Z5(create_view<view_res_t>(dom, T(0), map_res));
  view_res_t Z6(create_view<view_res_t>(dom, T(0), map_res));
  view_res_t Z7(create_view<view_res_t>(dom, T(0), map_res));
  view_op1_t A(create_view<view_op1_t>(dom, T(3), map_op1));
  view_op2_t B(create_view<view_op2_t>(dom, T(4), map_op2));

  impl::Communicator& comm = impl::default_communicator();

  // cout << "(" << local_processor() << "): test_distributed_view\n";

  foreach_point(A, Set_identity<Dim>(dom, 2, 1));
  foreach_point(B, Set_identity<Dim>(dom, 3, 2));

  for (int l=0; l<loop; ++l)
  {
    foreach_point(Z1, Set_identity<Dim>(dom));
    foreach_point(Z2, Set_identity<Dim>(dom));

    Z1 = A + B;
    Z2 = B - A;
    Z3 = -A;
    Z4 = -(A - B);
    Z5 = T(2) + A;
    Z6 = A + T(2);
    Z7 = T(2) * A + T(1) + B;

    // Calls:
    //    vsip::impl::par_expr(Z, A + B);
    // from dispatch_assign.hpp

    // Squirrel result away to check later.
    chk1 = Z1;
    chk2 = Z2;
  }


  // Check results.
  comm.barrier();

  Check_identity<Dim> checkerA(dom, 2, 1);
  Check_identity<Dim> checkerB(dom, 3, 2);
  foreach_point(A, checkerA);
  foreach_point(B, checkerB);

  Check_identity<Dim> checker1(dom, 5, 3);
  foreach_point(Z1, checker1);
  foreach_point(chk1, checker1);
  test_assert(checker1.good());

  Check_identity<Dim> checker2(dom, 1, 1);
  foreach_point(chk2, checker2);
  foreach_point(Z2, checker2);
  foreach_point(Z4, checker2);
  test_assert(checker2.good());

  Check_identity<Dim> checker3(dom, -2, -1);
  foreach_point(Z3, checker3);
  test_assert(checker3.good());

  Check_identity<Dim> checker4(dom, 2, 3);
  foreach_point(Z5, checker4);
  foreach_point(Z6, checker4);
  test_assert(checker4.good());

  Check_identity<Dim> checker5(dom, 2*2+3, 2*1+1+2);
  foreach_point(Z7, checker5);
  test_assert(checker5.good());

  if (map0.subblock() != no_subblock)
  {
    typename view0_t::local_type local_view = chk1.local();

    // Check that local_view is in fact the entire view.
    test_assert(extent(local_view) == extent(dom));

    // Check that each value is correct.
    bool good = true;
    Length<Dim> ext = extent(local_view);
    for (Index<Dim> idx; valid(ext,idx); next(ext, idx))
    {
      T value = T();
      for (dimension_type d=0; d<Dim; ++d)
      {
	value *= local_view.size(d);
	value += idx[d];
      }
      T expect1 = T(5)*value + T(3);
      // T expect2 = T(1)*value + T(1);

      if (get(local_view, idx) != expect1)
      {
	std::cout << "FAIL: index: " << idx
		  << "  expected " << expect1
		  << "  got "      << get(local_view, idx)
		  << std::endl;
	good = false;
      }
    }

    // cout << "CHECK: " << (good ? "good" : "BAD") << endl;
    test_assert(good);
  }
}



// Test distributed expression with two binary-operators.

template <typename       T,
	  dimension_type Dim,
	  typename       MapRes,
	  typename       MapOp1,
	  typename       MapOp2,
	  typename       MapOp3>
void
test_distributed_expr3(
  Domain<Dim> dom,
  MapRes      map_res,
  MapOp1      map_op1,
  MapOp2      map_op2,
  MapOp3      map_op3,
  int         loop)

{
  typedef Map<Block_dist, Block_dist> map0_t;

  typedef typename impl::Row_major<Dim>::type order_type;

  typedef Dense<Dim, T, order_type, map0_t> dist_block0_t;
  typedef Dense<Dim, T, order_type, MapRes> dist_block_res_t;
  typedef Dense<Dim, T, order_type, MapOp1> dist_block_op1_t;
  typedef Dense<Dim, T, order_type, MapOp2> dist_block_op2_t;
  typedef Dense<Dim, T, order_type, MapOp3> dist_block_op3_t;

  typedef typename impl::view_of<dist_block0_t>::type view0_t;
  typedef typename impl::view_of<dist_block_res_t>::type view_res_t;
  typedef typename impl::view_of<dist_block_op1_t>::type view_op1_t;
  typedef typename impl::view_of<dist_block_op2_t>::type view_op2_t;
  typedef typename impl::view_of<dist_block_op3_t>::type view_op3_t;

  // map0 is not distributed (effectively).
  map0_t  map0(Block_dist(1), Block_dist(1));

  // Non-distributed view to check results.
  view0_t chk1(create_view<view0_t>(dom, map0));
  view0_t chk2(create_view<view0_t>(dom, map0));

  // Distributed views for actual parallel-expression.
  view_res_t Z1(create_view<view_res_t>(dom, T(0), map_res));
  view_res_t Z2(create_view<view_res_t>(dom, T(0), map_res));
  view_op1_t A (create_view<view_op1_t>(dom, T(3), map_op1));
  view_op2_t B (create_view<view_op2_t>(dom, T(4), map_op2));
  view_op3_t C (create_view<view_op3_t>(dom, T(5), map_op3));

  impl::Communicator& comm = impl::default_communicator();

  foreach_point(A, Set_identity<Dim>(dom, 2, 1));
  foreach_point(B, Set_identity<Dim>(dom, 3, 2));
  foreach_point(C, Set_identity<Dim>(dom, 4, 1));

  for (int l=0; l<loop; ++l)
  {
    foreach_point(Z1, Set_identity<Dim>(dom));
    foreach_point(Z2, Set_identity<Dim>(dom));

    Z1 = A * B + C;
    Z2 = ma(A, B, C);
    // Calls:
    //    vsip::impl::par_expr(Z, A + B);
    // from dispatch_assign.hpp

    // Squirrel result away to check later.
    chk1 = Z1;
    chk2 = Z2;
  }


  // Check results.
  comm.barrier();

  if (map0.subblock() != no_subblock)
  {
    typename view0_t::local_type local_view = chk1.local();

    // Check that local_view is in fact the entire view.
    test_assert(extent(local_view) == extent(dom));

    // Check that each value is correct.
    bool good = true;
    Length<Dim> ext = extent(local_view);
    for (Index<Dim> idx; valid(ext,idx); next(ext, idx))
    {
      T val = T();
      for (dimension_type d=0; d<Dim; ++d)
      {
	val *= local_view.size(d);
	val += idx[d];
      }
      T av = T(2)*val + T(1);
      T bv = T(3)*val + T(2);
      T cv = T(4)*val + T(1);
      T expected_value = av*bv+cv;

      if (get(local_view, idx) != expected_value)
      {
	std::cout << "FAIL: index: " << idx
		  << "  expected " << expected_value
		  << "  got "      << get(local_view, idx)
		  << std::endl;
	good = false;
      }
    }

    // cout << "CHECK: " << (good ? "good" : "BAD") << endl;
    test_assert(good);
  }
}



// Test capture and evaluation of distributed expression with two
// binary-operators.

template <typename       T,
	  dimension_type Dim,
	  typename       MapRes,
	  typename       MapOp1,
	  typename       MapOp2,
	  typename       MapOp3>
void
test_distributed_expr3_capture(
  Domain<Dim> dom,
  MapRes      map_res,
  MapOp1      map_op1,
  MapOp2      map_op2,
  MapOp3      map_op3,
  int         loop)

{
  typedef Map<Block_dist, Block_dist> map0_t;

  typedef typename impl::Row_major<Dim>::type order_type;

  typedef Dense<Dim, T, order_type, map0_t> dist_block0_t;
  typedef Dense<Dim, T, order_type, MapRes> dist_block_res_t;
  typedef Dense<Dim, T, order_type, MapOp1> dist_block_op1_t;
  typedef Dense<Dim, T, order_type, MapOp2> dist_block_op2_t;
  typedef Dense<Dim, T, order_type, MapOp3> dist_block_op3_t;

  typedef typename impl::view_of<dist_block0_t>::type view0_t;
  typedef typename impl::view_of<dist_block_res_t>::type view_res_t;
  typedef typename impl::view_of<dist_block_op1_t>::type view_op1_t;
  typedef typename impl::view_of<dist_block_op2_t>::type view_op2_t;
  typedef typename impl::view_of<dist_block_op3_t>::type view_op3_t;

  // map0 is not distributed (effectively).
  map0_t  map0(Block_dist(1), Block_dist(1));

  // Non-distributed view to check results.
  view0_t chk(create_view<view0_t>(dom, map0));

  // Distributed views for actual parallel-expression.
  view_res_t Z(create_view<view_res_t>(dom, T(0), map_res));
  view_op1_t A(create_view<view_op1_t>(dom, T(3), map_op1));
  view_op2_t B(create_view<view_op2_t>(dom, T(4), map_op2));
  view_op3_t C(create_view<view_op3_t>(dom, T(5), map_op3));

  impl::Communicator& comm = impl::default_communicator();

  foreach_point(A, Set_identity<Dim>(dom, 2, 1));
  foreach_point(B, Set_identity<Dim>(dom, 3, 2));
  foreach_point(C, Set_identity<Dim>(dom, 4, 1));

  Prebound_expr expr(Z, A*B+C);

  for (int l=0; l<loop; ++l)
  {
    foreach_point(Z, Set_identity<Dim>(dom));

    expr(); // Z = A * B + C;

    // Squirrel result away to check later.
    chk = Z;
  }


  // Check results.
  comm.barrier();

  if (map0.subblock() != no_subblock)
  {
    typename view0_t::local_type local_view = chk.local();

    // Check that local_view is in fact the entire view.
    test_assert(extent(local_view) == extent(dom));

    // Check that each value is correct.
    bool good = true;
    Length<Dim> ext = extent(local_view);
    for (Index<Dim> idx; valid(ext,idx); next(ext, idx))
    {
      T expected_value = T();
      for (dimension_type d=0; d<Dim; ++d)
      {
	expected_value *= local_view.size(d);
	expected_value += idx[d];
      }
      T av = T(2)*expected_value + T(1);
      T bv = T(3)*expected_value + T(2);
      T cv = T(4)*expected_value + T(1);
      expected_value = av*bv+cv;

      if (get(local_view, idx) != expected_value)
      {
	std::cout << "FAIL: index: " << idx
		  << "  expected " << expected_value
		  << "  got "      << get(local_view, idx)
		  << std::endl;
	good = false;
      }
    }

    // cout << "CHECK: " << (good ? "good" : "BAD") << endl;
    test_assert(good);
  }
}




// Test several distributed vector cases for a given type and parallel
// assignment implementation.

template <typename T>
void
test_vector_assign(int loop)
{
  processor_type np = num_processors();

  Map<Block_dist>  map_1(Block_dist(1));
  Map<Block_dist>  map_2(Block_dist(2 <= np ? 2 : np));
  Map<Block_dist>  map_4(Block_dist(4 <= np ? 4 : np));
  Map<Block_dist>  map_np = Map<Block_dist>(Block_dist(np));

  test_distributed_expr<T>(
    Domain<1>(16),
    map_1, map_1, map_1,
    loop);

  test_distributed_expr<T>(
    Domain<1>(16),
    map_np, map_np, map_np,
    loop);

  test_distributed_expr<T>(
    Domain<1>(16),
    map_1, map_1, map_2,
    loop);

  test_distributed_expr<T>(
    Domain<1>(16),
    map_4, map_4, map_4,
    loop);

  test_distributed_expr3<T>(
    Domain<1>(16),
    map_1,
    map_1,
    map_1,
    map_1,
    loop);

  test_distributed_expr3_capture<T>(
    Domain<1>(16),
    map_1,
    map_1,
    map_1,
    map_1,
    loop);

#if VSIP_DIST_LEVEL >= 3
  Map<Cyclic_dist> map_c1(Cyclic_dist(np,1));
  Map<Cyclic_dist> map_c2(Cyclic_dist(np,2));
  Map<Cyclic_dist> map_c3(Cyclic_dist(np,3));

  test_distributed_expr<T>(
    Domain<1>(16),
    map_4, map_2, map_c1,
    loop);

  test_distributed_expr<T>(
    Domain<1>(16),
    map_c1, map_c2, map_c3,
    loop);

  test_distributed_expr3<T>(
    Domain<1>(16),
    map_2, map_c1, map_c2, map_c3,
    loop);

  test_distributed_expr3<T>(
    Domain<1>(16),
    map_4,
    map_2,
    map_1,
    map_c1,
    loop);

  test_distributed_expr3_capture<T>(
    Domain<1>(16),
    map_2, map_c1, map_c2, map_c3,
    loop);

  test_distributed_expr3_capture<T>(
    Domain<1>(16),
    map_4,
    map_2,
    map_1,
    map_c1,
    loop);
#endif // VSIP_DIST_LEVEL >= 3
}



template <typename                  T>
void
test_matrix_assign(int loop)
{
  length_type np = num_processors();
  length_type nr = (processor_type)floor(sqrt((double)np));
  length_type nc = (processor_type)floor((double)np/nr);

  Map<Block_dist, Block_dist> map_1(Block_dist(1), Block_dist(1));
  Map<Block_dist, Block_dist> map_r(Block_dist(np), Block_dist(1));
  Map<Block_dist, Block_dist> map_c(Block_dist(1),  Block_dist(np));
  // Map<Block_dist, Block_dist> map_x(Block_dist(nr), Block_dist(nc));
  Map<Block_dist, Block_dist> map_x = Map<>(Block_dist(nr), Block_dist(nc));


  test_distributed_expr<T>(
    Domain<2>(16, 16),
    map_1, map_1, map_1,
    loop);

  test_distributed_expr<T>(
    Domain<2>(16, 16),
    map_x, map_x, map_x,
    loop);

  test_distributed_expr<T>(
    Domain<2>(16, 16),
    map_x, map_r, map_c,
    loop);

#if VSIP_DIST_LEVEL >= 3
  Map<Cyclic_dist, Cyclic_dist> map_se(Cyclic_dist(nr, 1), Cyclic_dist(nc, 1));

  test_distributed_expr<T>(
    Domain<2>(16, 16),
    map_x, map_se, map_c,
    loop);
#endif // VSIP_DIST_LEVEL >= 3
}



int
main(int argc, char** argv)
{
  vsipl vpp(argc, argv);

  int loop = argc > 1 ? atoi(argv[1]) : 1;

#if 0
  // Enable this section for easier debugging.
  impl::Communicator& comm = impl::default_communicator();
  pid_t pid = getpid();

  std::cout << "rank: "   << comm.rank()
	    << "  size: " << comm.size()
	    << "  pid: "  << pid
	    << std::endl;

  // Stop each process, allow debugger to be attached.
  if (comm.rank() == 0) getchar();
  comm.barrier();
#endif

  test_distributed_expr<float>(
    Domain<1>(16),
    Global_map<1>(),
    Global_map<1>(),
    Global_map<1>(),
    loop);

  test_distributed_expr<float>(
    Domain<1>(16),
    Replicated_map<1>(),
    Replicated_map<1>(),
    Replicated_map<1>(),
    loop);

  test_vector_assign<float>(loop);
  test_matrix_assign<float>(loop);
  
  return 0;
}
