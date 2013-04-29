//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   VSIPL++ Library: Unit tests for distributed DDA

#include <vsip/vector.hpp>
#include <vsip/dda.hpp>
#include <vsip/initfin.hpp>
#include <vsip/map.hpp>
#include <vsip/selgen.hpp>

#include <vsip_csl/test.hpp>
#include <vsip_csl/output.hpp>
#include <vsip_csl/diagnostics.hpp>

using namespace vsip;
using vsip_csl::equal;

using vsip::impl::Any_type;
using vsip::impl::Strided;
using vsip::impl::adjust_layout;
using vsip::impl::is_same;
using vsip::impl::conditional;
using vsip::impl::scalar_of;
using vsip::impl::Aligned_allocator;
using vsip::impl::Is_local_map;
using vsip::impl::is_complex;



/***********************************************************************
  Definitions
***********************************************************************/

/// Test high-level data interface to 1-dimensional block.

template <typename T,
	  typename MapT,
	  typename GivenLP,
	  typename ReqLP>
void
test_1_ext(int expect_cost)
{
  length_type const size = 10;

  typedef Strided<1, T, GivenLP, MapT> block_type;

  typedef T value_type;
  typedef typename adjust_layout<value_type, ReqLP, GivenLP>::type use_LP;
  typedef vsip::impl::Storage<use_LP::storage_format, T> storage_type;

  block_type block(size, 0.0);

  value_type val0 =  1.0f;
  value_type val1 =  2.78f;
  value_type val2 =  3.14f;
  value_type val3 = -1.5f;

  // Place values in block.
  block.put(0, val0);
  block.put(1, val1);

  typedef typename
      conditional<use_LP::storage_format == interleaved_complex,
		  Aligned_allocator<T>,
		  Aligned_allocator<typename scalar_of<T>::type> >
      ::type alloc_type;
  alloc_type alloc;
  typename storage_type::type buffer = storage_type::allocate(alloc, size);

  {
    vsip::dda::Data<block_type, dda::inout, use_LP> raw(block, buffer);

    assert(raw.cost() == expect_cost);

    // Check properties of DDI.
    test_assert(raw.stride(0) == 1);
    test_assert(raw.size(0) == size);

    typename storage_type::type data = raw.ptr();

    // Check that block values are reflected.
    test_assert(equal(storage_type::get(data, 0), val0));
    test_assert(equal(storage_type::get(data, 1), val1));

    // Place values in raw data.
    storage_type::put(data, 1, val2);
    storage_type::put(data, 2, val3);
  }

  storage_type::deallocate(alloc, buffer, size);

  // Check that raw data values are reflected.
  test_assert(equal(block.get(1), val2));
  test_assert(equal(block.get(2), val3));
}



// Determine expected cost for dda::Data_dist.

template <typename T,
	  typename MapT,
	  typename OrderT,
	  typename use_LP,
	  typename GivenLP>
int expected_cost()
{
  bool is_local_equiv = Is_local_map<MapT>::value ||
                        is_same<MapT, Replicated_map<1> >::value;

  bool same_order = is_same<OrderT, typename use_LP::order_type>::value;

  bool same_complex_fmt = 
    !is_complex<T>::value || GivenLP::storage_format == use_LP::storage_format;

  return (is_local_equiv && same_order && same_complex_fmt) ? 0 : 2;
}



template <typename T,
	  typename OrderT,
	  typename MapT,
	  typename ReqLP>
void
test_1_dense(MapT const& map)
{
  length_type const size = 10;

  typedef Dense<1, T, OrderT, MapT> block_type;

  typedef typename get_block_layout<block_type>::type GivenLP;

  typedef T value_type;
  typedef typename adjust_layout<value_type, ReqLP, GivenLP>::type use_LP;
  typedef vsip::impl::Storage<use_LP::storage_format, T> storage_type;

  block_type block(size, T(), map);

  value_type val0 =  1.0f;
  value_type val1 =  2.78f;
  value_type val2 =  3.14f;
  value_type val3 = -1.5f;

  // Place values in block.
  block.put(0, val0);
  block.put(1, val1);

  typedef typename
      conditional<use_LP::storage_format == interleaved_complex,
               Aligned_allocator<T>,
               Aligned_allocator<typename scalar_of<T>::type> >
      ::type alloc_type;

  alloc_type alloc;
  typename storage_type::type buffer = storage_type::allocate(alloc, size);

  {
    vsip::dda::Data<block_type, dda::inout, use_LP> raw(block, buffer);

    assert((raw.cost() == expected_cost<T, MapT, OrderT, use_LP, GivenLP>()));

    // Check properties of DDI.
    test_assert(raw.stride(0) == 1);
    test_assert(raw.size(0) == size);

    typename storage_type::type data = raw.ptr();

    // Check that block values are reflected.
    test_assert(equal(storage_type::get(data, 0), val0));
    test_assert(equal(storage_type::get(data, 1), val1));

    // Place values in raw data.
    storage_type::put(data, 1, val2);
    storage_type::put(data, 2, val3);
  }

  storage_type::deallocate(alloc, buffer, size);

  // Check that raw data values are reflected.
  test_assert(equal(block.get(1), val2));
  test_assert(equal(block.get(2), val3));
}



template <typename T,
	  typename OrderT,
	  typename MapT,
	  typename ReqLP>
void
test_1_dense_const(MapT const& map)
{
  length_type const size = 10;

  typedef Dense<1, T, OrderT, MapT> block_type;

  typedef typename get_block_layout<block_type>::type GivenLP;

  typedef T value_type;
  typedef typename adjust_layout<value_type, ReqLP, GivenLP>::type use_LP;
  typedef vsip::impl::Storage<use_LP::storage_format, T> storage_type;

  block_type block(size, T(), map);

  value_type val0 =  1.0f;
  value_type val1 =  2.78f;

  // Place values in block.
  block.put(0, val0);
  block.put(1, val1);

  typedef typename
      conditional<use_LP::storage_format == interleaved_complex,
               Aligned_allocator<T>,
               Aligned_allocator<typename scalar_of<T>::type> >
      ::type alloc_type;

  alloc_type alloc;
  typename storage_type::type buffer = storage_type::allocate(alloc, size);

  {
    block_type const& ref = block;
    vsip::dda::Data<block_type, dda::in, use_LP> raw(ref, buffer);
    assert((raw.cost() == expected_cost<T, MapT, OrderT, use_LP, GivenLP>()));

    // Check properties of DDI.
    test_assert(raw.stride(0) == 1);
    test_assert(raw.size(0) == size);

    typename storage_type::const_type data = raw.ptr();

    // Check that block values are reflected.
    test_assert(equal(storage_type::get(data, 0), val0));
    test_assert(equal(storage_type::get(data, 1), val1));
  }

  storage_type::deallocate(alloc, buffer, size);
}



// Helper function to test dda::Data_dist access to an expression block.
// Deduces type of expression block.
//

template <typename T,
	  typename OrderT,
	  typename MapT,
	  typename ReqLP,
	  typename ViewT>
void
test_1_expr_helper(ViewT view, T val0, T val1, bool alloc_buffer)
{
  typedef typename ViewT::block_type block_type;
  typedef typename ViewT::value_type value_type;

  typedef typename get_block_layout<block_type>::type GivenLP;

  typedef typename adjust_layout<value_type, ReqLP, GivenLP>::type use_LP;
  typedef vsip::impl::Storage<use_LP::storage_format, T> storage_type;

  typedef typename
      conditional<use_LP::storage_format == interleaved_complex,
               Aligned_allocator<T>,
               Aligned_allocator<typename scalar_of<T>::type> >
      ::type alloc_type;

  length_type size = view.size();

  alloc_type alloc;
  typedef typename storage_type::type ptr_type;
  ptr_type buffer = alloc_buffer ? storage_type::allocate(alloc, size)
                                 : ptr_type();

  {
    dda::Data<block_type, dda::in, use_LP> raw(view.block(), buffer);

    // Because block is an expression block, access requires a copy.
    assert(raw.cost() == 2);

    // Check properties of DDI.
    test_assert(raw.stride(0) == 1);
    test_assert(raw.size(0) == size);

    typename dda::Data<block_type, dda::in, use_LP>::ptr_type data = raw.ptr();

    // Check that block values are reflected.
    test_assert(equal(storage_type::get(data, 0), val0));
    test_assert(equal(storage_type::get(data, 1), val1));
  }

  if (alloc_buffer) storage_type::deallocate(alloc, buffer, size);
}



// Test dda::Data_dist access to a simple expression block.
// (vector + vector).

template <typename T,
	  typename OrderT,
	  typename MapT,
	  typename ReqLP>
void
test_1_expr_1(MapT const& map)
{
  typedef Dense<1, T, OrderT, MapT> block_type;

  length_type const size = 10;

  T val0 =  1.0f;
  T val1 =  2.78f;

  Vector<T, block_type> view1(size, T(), map);
  Vector<T, block_type> view2(size, T(), map);

  // Place values in block.
  view1.put(0, val0);
  view2.put(1, val1);

  test_1_expr_helper<T, OrderT, MapT, ReqLP>(view1 + view2, val0, val1, true);
  test_1_expr_helper<T, OrderT, MapT, ReqLP>(view1 + view2, val0, val1, false);
}



// Test dda::Data_dist access to a more complex expression block.
// (vector + vector)(subset).

template <typename T,
	  typename OrderT,
	  typename MapT,
	  typename ReqLP>
void
test_1_expr_2(MapT const& map)
{
  typedef Dense<1, T, OrderT, MapT> block_type;

  length_type const size = 10;

  T val0 =  1.0f;
  T val1 =  2.78f;

  Vector<T, block_type> view1(size, T(), map);
  Vector<T, block_type> view2(size, T(), map);

  // Place values in block.
  view1.put(0, val0);
  view2.put(1, val1);

  test_1_expr_helper<T, OrderT, MapT, ReqLP>((view1 + view2)(Domain<1>(8)),
					     val0, val1, true);
  test_1_expr_helper<T, OrderT, MapT, ReqLP>((view1 + view2)(Domain<1>(8)),
					     val0, val1, false);
}



// Test dda::Data_dist access to an 'magsq(v)' expression block.

template <typename T,
	  typename OrderT,
	  typename MapT,
	  typename ReqLP>
void
test_1_expr_3(MapT const& map)
{
  typedef Dense<1, T, OrderT, MapT> block_type;

  length_type const size = 10;

  T val0 =  1.0f;
  T val1 =  2.78f;

  Vector<T, block_type> view(size, T(), map);

  // Place values in block.
  view.put(0, val0);
  view.put(1, val1);

  typedef typename vsip::impl::scalar_of<T>::type scalar_type;

  test_1_expr_helper<scalar_type, OrderT, MapT, ReqLP>(
		magsq(view), magsq(val0), magsq(val1), true);
  test_1_expr_helper<scalar_type, OrderT, MapT, ReqLP>(
		magsq(view), magsq(val0), magsq(val1), false);
}



void
test()
{
  typedef Layout<1, row1_type, dense, interleaved_complex> LP_1rdi;
  typedef Layout<1, row1_type, dense, split_complex> LP_1rds;

  typedef Layout<1, Any_type, any_packing, interleaved_complex> LP_1xxi;
  typedef Layout<1, Any_type, any_packing, split_complex> LP_1xxs;

  test_1_ext<float,          Local_map, LP_1rdi, LP_1xxi >(0);
  test_1_ext<float,          Local_map, LP_1rds, LP_1xxi >(0);
  test_1_ext<float,          Local_map, LP_1rdi, LP_1xxs >(0);
  test_1_ext<float,          Local_map, LP_1rds, LP_1xxs >(0);

  test_1_ext<complex<float>, Local_map, LP_1rdi, LP_1xxi >(0);
  test_1_ext<complex<float>, Local_map, LP_1rds, LP_1xxi >(2);
  test_1_ext<complex<float>, Local_map, LP_1rdi, LP_1xxs >(2);
  test_1_ext<complex<float>, Local_map, LP_1rds, LP_1xxs >(0);

  Local_map lmap;

  test_1_expr_1<float,          row1_type, Local_map, LP_1xxi >(lmap);
  test_1_expr_1<complex<float>, row1_type, Local_map, LP_1xxi >(lmap);
  test_1_expr_1<complex<float>, row1_type, Local_map, LP_1xxs >(lmap);

  test_1_expr_2<float,          row1_type, Local_map, LP_1xxi >(lmap);
  test_1_expr_2<complex<float>, row1_type, Local_map, LP_1xxi >(lmap);
  test_1_expr_2<complex<float>, row1_type, Local_map, LP_1xxs >(lmap);

  test_1_expr_3<float,          row1_type, Local_map, LP_1xxi >(lmap);
  test_1_expr_3<complex<float>, row1_type, Local_map, LP_1xxi >(lmap);
  test_1_expr_3<complex<float>, row1_type, Local_map, LP_1xxs >(lmap);


  Replicated_map<1> gmap;

  test_1_dense<float,          row1_type, Replicated_map<1>, LP_1xxi >(gmap);
  test_1_dense<float,          row1_type, Replicated_map<1>, LP_1xxs >(gmap);
  test_1_dense<complex<float>, row1_type, Replicated_map<1>, LP_1xxi >(gmap);
  test_1_dense<complex<float>, row1_type, Replicated_map<1>, LP_1xxs >(gmap);

  test_1_dense_const<float,          row1_type, Replicated_map<1>, LP_1xxi >(gmap);
  test_1_dense_const<float,          row1_type, Replicated_map<1>, LP_1xxs >(gmap);
  test_1_dense_const<complex<float>, row1_type, Replicated_map<1>, LP_1xxi >(gmap);
  test_1_dense_const<complex<float>, row1_type, Replicated_map<1>, LP_1xxs >(gmap);

  test_1_expr_1<float,          row1_type, Replicated_map<1>, LP_1xxi >(gmap);
  test_1_expr_1<complex<float>, row1_type, Replicated_map<1>, LP_1xxi >(gmap);
  test_1_expr_1<complex<float>, row1_type, Replicated_map<1>, LP_1xxs >(gmap);

  test_1_expr_2<float,          row1_type, Replicated_map<1>, LP_1xxi >(gmap);
  test_1_expr_2<complex<float>, row1_type, Replicated_map<1>, LP_1xxi >(gmap);
  test_1_expr_2<complex<float>, row1_type, Replicated_map<1>, LP_1xxs >(gmap);

  test_1_expr_3<float,          row1_type, Replicated_map<1>, LP_1xxi >(gmap);
  test_1_expr_3<complex<float>, row1_type, Replicated_map<1>, LP_1xxi >(gmap);
  test_1_expr_3<complex<float>, row1_type, Replicated_map<1>, LP_1xxs >(gmap);


  Map<Block_dist> map(num_processors());

  test_1_dense<float,          row1_type, Map<Block_dist>, LP_1xxi >(map);
  test_1_dense<float,          row1_type, Map<Block_dist>, LP_1xxs >(map);
  test_1_dense<complex<float>, row1_type, Map<Block_dist>, LP_1xxi >(map);
  test_1_dense<complex<float>, row1_type, Map<Block_dist>, LP_1xxs >(map);

  test_1_dense_const<float,          row1_type, Map<Block_dist>, LP_1xxi>(map);
  test_1_dense_const<float,          row1_type, Map<Block_dist>, LP_1xxs>(map);
  test_1_dense_const<complex<float>, row1_type, Map<Block_dist>, LP_1xxi>(map);
  test_1_dense_const<complex<float>, row1_type, Map<Block_dist>, LP_1xxs>(map);

  test_1_expr_1<float,          row1_type, Map<Block_dist>, LP_1xxi >(map);
  test_1_expr_1<complex<float>, row1_type, Map<Block_dist>, LP_1xxi >(map);
  test_1_expr_1<complex<float>, row1_type, Map<Block_dist>, LP_1xxs >(map);

  test_1_expr_2<float,          row1_type, Map<Block_dist>, LP_1xxi >(map);
  test_1_expr_2<complex<float>, row1_type, Map<Block_dist>, LP_1xxi >(map);
  test_1_expr_2<complex<float>, row1_type, Map<Block_dist>, LP_1xxs >(map);

  test_1_expr_3<float,          row1_type, Map<Block_dist>, LP_1xxi >(map);
  test_1_expr_3<complex<float>, row1_type, Map<Block_dist>, LP_1xxi >(map);
  test_1_expr_3<complex<float>, row1_type, Map<Block_dist>, LP_1xxs >(map);
}



int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  test();
}
