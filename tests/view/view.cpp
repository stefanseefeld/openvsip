//
// Copyright (c) 2005, 2006, 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#ifndef ILLEGALCASE
#  define ILLEGALCASE 0
#endif

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>
#include <cassert>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/domain.hpp>
#include <ovxx/length.hpp>
#include <ovxx/domain_utils.hpp>
#include <test.hpp>
#include <storage.hpp>

using namespace ovxx;

/// Write a vector to a stream.

template <typename T,
	  typename Block>
inline
std::ostream&
operator<<(std::ostream &out, const_Vector<T, Block> vec) VSIP_NOTHROW
{
  for (vsip::index_type i=0; i<vec.size(); ++i)
    out << "  " << i << ": " << vec.get(i) << "\n";
  return out;
}



/// Write a matrix to a stream.

template <typename T,
	  typename Block>
inline
std::ostream&
operator<<(
  std::ostream&		       out,
  vsip::const_Matrix<T, Block> v)
  VSIP_NOTHROW
{
  for (vsip::index_type r=0; r<v.size(0); ++r)
  {
    out << "  " << r << ":";
    for (vsip::index_type c=0; c<v.size(1); ++c)
      out << "  " << v.get(r, c);
    out << std::endl;
  }
  return out;
}



// Check the reported size of a vector against expected values.

template <typename T,
	  typename Block>
void
check_size(
  const_Vector<T, Block> view,
  Domain<1> const&       dom)
{
  test_assert(view.length() == dom.length());
  test_assert(view.size()   == dom.length());
  test_assert(view.size(0)  == dom.length());
}



// Check the reported size of a matrix against expected values.

template <typename T,
	  typename Block>
void
check_size(
  const_Matrix<T, Block> view,
  Domain<2> const&       dom)
{
  test_assert(view.size()  == dom[0].length() * dom[1].length());
  test_assert(view.size(0) == dom[0].length());
  test_assert(view.size(1) == dom[1].length());
}



// Fill vector with sequence of values.

template <typename T,
	  typename Block>
void
fill_view(
  Vector<T, Block> view,
  int              k,
  Index<1>	   offset,
  Domain<1>        /* dom */)
{
  for (index_type i=0; i<view.size(0); ++i)
    view.put(i, T(k*(i + offset[0])+1));
}



// Fill matrix with a sequence of values.

// Values are generated with a slope of K.

template <typename T,
	  typename Block>
void
fill_view(
  Matrix<T, Block> view,
  int              k,
  Index<2>         offset,
  Domain<2>        dom)
{
  for (index_type r=0; r<view.size(0); ++r)
    for (index_type c=0; c<view.size(1); ++c)
    {
      index_type i = (r+offset[0])*dom[1].length() + (c+offset[1]);
      view.put(r, c, T(k*i+1));
    }
}



template <typename T,
	  typename Block>
void
fill_view(
  Vector<T, Block> view,
  int              k)
{
  fill_view(view, k, Index<1>(0), Domain<1>(view.size(0)));
}



template <typename T,
	  typename Block>
void
fill_view(
  Matrix<T, Block> view,
  int              k)
{
  fill_view(view, k, Index<2>(0, 0), Domain<2>(view.size(0), view.size(1)));
}



// Class to fill block with values based on K.
// Partial specializations are provided for dimension equal 1 and 2.

template <dimension_type Dim,
	  typename       Block>
struct Fill_block;

template <typename       Block>
struct Fill_block<1, Block>
{
  static void fill(
    Block&           blk,
    int              k,
    Index<1> const&  offset,
    Domain<1> const& /*dom*/)
  {
    typedef typename Block::value_type T;

    for (index_type i=0; i<blk.size(1, 0); ++i)
    {
      blk.put(i, T(k*(i+offset[0])+1));
    }
  }

  static Index<1> origin() { return Index<1>(0); }

  static Domain<1> size(Block& blk) { return Domain<1>(blk.size(1, 0)); }
};



template <typename       Block>
struct Fill_block<2, Block>
{
  static void fill(
    Block&           blk,
    int              k,
    Index<2> const&  offset,
    Domain<2> const& dom)
  {
    typedef typename Block::value_type T;

    for (index_type r=0; r<blk.size(2, 0); ++r)
      for (index_type c=0; c<blk.size(2, 1); ++c)
      {
	index_type i = (r+offset[0])*dom[1].length() + (c+offset[1]);
	blk.put(r, c, T(k*i+1));
      }
  }

  static Index<2> origin() { return Index<2>(0, 0); }

  static Domain<2> size(Block& blk)
    { return Domain<2>(blk.size(2, 0), blk.size(2, 1)); }
};



// fill_block -- fill a block with sequence of values.
//
// Values are generated with a slope of K.

template <dimension_type Dim,
	  typename       Block>
void
fill_block(Block& blk, int k)
{
  Fill_block<Dim, Block>::fill(blk, k,
			      Fill_block<Dim, Block>::origin(),
			      Fill_block<Dim, Block>::size(blk));
}




// Test values in view against sequence.

template <typename T,
	  typename Block>
void
test_view(const_Vector<T, Block> vec, int k)
{
  for (index_type i=0; i<vec.size(0); ++i)
    test_assert(equal(vec.get(i), T(k*i+1)));
}



// test_view -- test values in view against sequence.
//
// Asserts that view values match those generated by a call to
// fill_view or fill_block with the same k value.

template <typename T,
	  typename Block>
void
test_view(const_Matrix<T, Block> v, int k)
{
  for (index_type r=0; r<v.size(0); ++r)
    for (index_type c=0; c<v.size(1); ++c)
    {
      index_type i = r*v.size(1) + c;
      test_assert(equal(v.get(r, c), T(k*i+1)));
    }
}



// check_vector -- check values in vector against sequence.
//
// Checks that vector values match those generated by a call to
// fill_vector or fill_block with the same k value.  Rather than
// triggering test_assertion failure, check_vector returns a boolean
// pass/fail that can be used to cause an test_assertion failure in
// the caller.

template <typename T,
	  typename Block>
bool
check_view(const_Vector<T, Block> vec, int k)
{
  for (index_type i=0; i<vec.size(0); ++i)
    if (!equal(vec.get(i), T(k*i+1)))
      return false;
  return true;
}



// check_view -- check values in view against sequence.
//
// Checks that view values match those generated by a call to
// fill_view or fill_block with the same k value.  Rather than
// triggering test_assertion failure, check_view returns a boolean
// pass/fail that can be used to cause an test_assertion failure in
// the caller.

template <typename T,
	  typename Block>
bool
check_view(
  const_Matrix<T, Block> view,
  int                    k,
  Index<2>               offset,
  Domain<2>              dom)
{
  for (index_type r=0; r<view.size(0); ++r)
    for (index_type c=0; c<view.size(1); ++c)
    {
      index_type i = (r+offset[0])*dom[1].length() + (c+offset[1]);
      if (!equal(view.get(r, c), T(k*i+1)))
	return false;
    }
  return true;
}



template <typename T,
	  typename Block>
bool
check_view(
  const_Matrix<T, Block> view,
  int                    k)
{
  return check_view(view, k, Index<2>(0, 0),
		    Domain<2>(view.size(0), view.size(1)));
}



// Check that all elements of a view have the same const values


template <typename View>
bool
check_view_const(
  View                      view,
  typename View::value_type scalar)
{
  dimension_type const dim = View::dim;
  Length<dim> ext = extent(view);
  for (Index<dim> idx; valid(ext,idx); next(ext, idx))
  {
    if (!equal(get(view, idx), scalar))
      return false;
  }
  return true;
}



// check_not_alias -- check that two views are not aliased.
//
// Changes made to one should not effect the other.

template <template <typename, typename> class View1,
	  template <typename, typename> class View2,
	  typename T1,
	  typename T2,
	  typename Block1,
	  typename Block2>
void
check_not_alias(
  View1<T1, Block1>& view1,
  View2<T2, Block2>& view2)
{
  test_assert((View1<T1, Block1>::dim == View2<T2, Block2>::dim));
  dimension_type const dim = View1<T1, Block1>::dim;

  fill_block<dim>(view1.block(), 2);
  fill_block<dim>(view2.block(), 3);

  // Make sure that updates to view2 do not affect view1.
  test_assert(check_view(view1, 2));

  // And visa-versa.
  fill_block<dim>(view1.block(), 4);
  test_assert(check_view(view2, 3));
}



// check_alias -- check that two views are aliased.
//
// Changes made to one should effect the other.

template <template <typename, typename> class View1,
	  template <typename, typename> class View2,
	  typename T1,
	  typename T2,
	  typename Block1,
	  typename Block2>
void
check_alias(
  View1<T1, Block1>& view1,
  View2<T2, Block2>& view2)
{
  test_assert((View1<T1, Block1>::dim == View2<T2, Block2>::dim));
  dimension_type const dim = View1<T1, Block1>::dim;

  fill_block<dim>(view1.block(), 2);
  test_assert(check_view(view1, 2));
  test_assert(check_view(view2, 2));

  fill_block<dim>(view2.block(), 3);
  test_assert(check_view(view1, 3));
  test_assert(check_view(view2, 3));
}



/***********************************************************************
  Definitions - View Test Cases.
***********************************************************************/

// -------------------------------------------------------------------- //
// Test cases for view get/put.
//
// The following functions are defined:
//   tc_get()
//		Test case for view get().
//   test_get_type() / test_get()
//		Wrappers for different views and types.
// -------------------------------------------------------------------- //

template <typename       Storage,
	  dimension_type Dim>
void
tc_get(Domain<Dim> const& dom)
{
  Storage stor(dom);

  check_size(stor.view, dom);
  fill_block<Dim>(stor.block(), 2);
  test_view(stor.view, 2);
}



template <typename       T,
	  dimension_type Dim>
void
test_get_type(Domain<Dim> const& dom)
{
  tc_get<     Storage<Dim, T> >(dom);
  tc_get<Const_storage<Dim, T> >(dom);
}



template <dimension_type Dim>
void test_get(Domain<Dim> const& dom)
{
  test_get_type<scalar_f> (dom);
  test_get_type<cscalar_f>(dom);
  test_get_type<int>      (dom);
  test_get_type<short>    (dom);
}



// -------------------------------------------------------------------- //
// Test case for get/put.
//
// The following functions are defined:
//   tc_getput
//		Test a view's get() and put() member functions.
//   test_getput_type() / test_getput()
//		Wrappers for different views and types.
// -------------------------------------------------------------------- //

template <typename       Storage,
	  dimension_type Dim>
void
tc_getput(Domain<Dim> const& dom)
{
  Storage stor(dom);

  check_size(stor.view, dom);
  fill_view(stor.view, 2);
  test_view(stor.view, 2);
}



template <typename T,
	  dimension_type Dim>
void
test_getput_type(Domain<Dim> const& dom)
{
  tc_getput<Storage<Dim, T> >(dom);
#if (ILLEGALCASE == 1)
  // const_Views should not provide put().
  tc_getput<Const_storage<Dim, T> >(dom);
#endif
}



template <dimension_type Dim>
void test_getput(Domain<Dim> const& dom)
{
  test_getput_type<scalar_f> (dom);
  test_getput_type<cscalar_f>(dom);
  test_getput_type<int>      (dom);
  test_getput_type<short>    (dom);
}



// -------------------------------------------------------------------- //
// Test case for view copy constructor.
//
// The following functions are defined:
//   tc_copy_cons()
//		Test copy construction of one view from another.
//   test_copy_cons_type(), test_copy_cons()
//		Wrappers for different views and types.
// -------------------------------------------------------------------- //

template <typename       Storage1,
	  typename       Storage2,
	  dimension_type Dim>
void
tc_copy_cons(Domain<Dim> const& dom, int k)
{
  Storage1 stor1(dom);

  fill_block<Dim>(stor1.block(), k);

  typename Storage2::view_type view2(stor1.view);

  test_view(view2, k);

  check_alias(stor1.view, view2);
}



template <typename       T1,
	  typename       T2,
	  dimension_type Dim>
void
test_copy_cons_type(Domain<Dim> const& dom, int k)
{
  tc_copy_cons<     Storage<Dim, T1>,      Storage<Dim, T2> >(dom, k);
  tc_copy_cons<     Storage<Dim, T1>, Const_storage<Dim, T2> >(dom, k);
  tc_copy_cons<Const_storage<Dim, T1>, Const_storage<Dim, T2> >(dom, k);

#if (ILLEGALCASE == 2)
  // It should be illegal to construct a View from a const_View.
  tc_copy_cons<Const_storage<Dim, T1>,      Storage<Dim, T2>>(dom, k);
#endif
}



template <dimension_type Dim>
void
test_copy_cons(Domain<Dim> const& dom, int k)
{
  test_copy_cons_type< scalar_f,  scalar_f>(dom, k);
  test_copy_cons_type<cscalar_f, cscalar_f>(dom, k);
  test_copy_cons_type<      int,       int>(dom, k);
  test_copy_cons_type<    short,     short>(dom, k);
}



// -------------------------------------------------------------------- //
// Test case for view assignment
//
// The following functions are defined:
//   tc_assign()
//		Test assignment of one view to another.
//   test_assign_type(), test_assign()
//		Wrappers for different views and types.
// -------------------------------------------------------------------- //

template <typename       Storage1,
	  typename       Storage2,
	  dimension_type Dim>
void
tc_assign(Domain<Dim> const& dom, int k)
{
  Storage1 stor1(dom);
  Storage2 stor2(dom);
  Storage2 stor2b(dom);

  fill_block<Dim>(stor1.block(), k);

  stor2.view = stor1.view;
  test_assert(check_view(stor2.view, k));

  check_not_alias(stor1.view, stor2.view);


  fill_block<Dim>(stor1.block(), k+1);
  stor2b.view = stor2.view = stor1.view;
  test_assert(check_view(stor2.view,  k+1));
  test_assert(check_view(stor2b.view, k+1));
}



template <typename       T1,
	  typename       T2,
	  dimension_type Dim>
void
test_assign_type(Domain<Dim> const& dom, int k)
{
  typedef typename row_major<Dim>::type row_t;
  typedef typename col_major<Dim>::type col_t;

  tc_assign<     Storage<Dim, T1, row_t>, Storage<Dim, T2, row_t> >(dom, k);
  tc_assign<     Storage<Dim, T1, row_t>, Storage<Dim, T2, col_t> >(dom, k);
  tc_assign<     Storage<Dim, T1, col_t>, Storage<Dim, T2, row_t> >(dom, k);
  tc_assign<     Storage<Dim, T1, col_t>, Storage<Dim, T2, col_t> >(dom, k);

  tc_assign<     Storage<Dim, T1>, Storage<Dim, T2> >(dom, k);
  tc_assign<Const_storage<Dim, T1>, Storage<Dim, T2> >(dom, k);

#if (ILLEGALCASE == 3)
  tc_assign<     Storage<Dim, T1>, Const_storage<Dim, T2> >(dom, k);
#endif
#if (ILLEGALCASE == 4)
  tc_assign<Const_storage<Dim, T1>, Const_storage<Dim, T2> >(dom, k);
#endif
}



template <dimension_type Dim>
void
test_assign(Domain<Dim> const& dom, int k)
{
  test_assign_type<float, float>(dom, k);
  test_assign_type<int, float>(dom, k);
}



// -------------------------------------------------------------------- //
// Test case for view assignment from scalar
//
// The following functions are defined:
//   tc_assign()
//		Test assignment of one view to another.
//   test_assign_type(), test_assign()
//		Wrappers for different views and types.
// -------------------------------------------------------------------- //

template <typename       Storage,
	  typename       T,
	  dimension_type Dim>
void
tc_assign_scalar(
  Domain<Dim> const& dom,
  int                k)
{
  Storage stor1(dom);
  Storage stor2(dom);

  stor1.view = T(k);

  test_assert(check_view_const(stor1.view, T(k)));

  stor1.view = stor2.view = T(k+1);

  test_assert(check_view_const(stor1.view, T(k+1)));
  test_assert(check_view_const(stor2.view, T(k+1)));
}



template <typename       T1,
	  typename       T2,
	  dimension_type Dim>
void
test_assign_scalar_type(Domain<Dim> const& dom, int k)
{
  typedef typename row_major<Dim>::type row_t;
  typedef typename col_major<Dim>::type col_t;

  tc_assign_scalar<Storage<Dim, T1, row_t>, T2>(dom, k);
}



template <dimension_type Dim>
void
test_assign_scalar(Domain<Dim> const& dom, int k)
{
  test_assign_scalar_type<float, float>(dom, k);
  // test_assign_scalar_type<int, float>(dom, k);
}



// -------------------------------------------------------------------- //
// Test cases for passing views as parameters to functions
//
// The following functions are defined:
//   tc_sum_const()
//		Function taking a const_View parameter (returns the
//		sum of the view's elements).
//   tc_sum()
//		Function taking a View parameter (returns the sum of
//		the view's elements).
//   tc_call_sum_const()
//		Tests passing a view as a parameter when a const View
//		is expected.
//   tc_call_sum()
//		Tests passing a view as a parameter when a non-const
//		View is expected.
//   test_call_type(), test_call()
//		Wrappers for different views and types.
// -------------------------------------------------------------------- //

template <typename T,
	  typename Block>
T
tc_sum_const(const_Vector<T, Block> vec)
{
  T sumval = T();
  for (index_type i=0; i<vec.size(0); ++i)
    sumval += vec.get(i);
  return sumval;
}



template <typename T,
	  typename Block>
T
tc_sum(Vector<T, Block> vec)
{
  T sumval = T();
  for (index_type i=0; i<vec.size(0); ++i)
    sumval += vec.get(i);
  return sumval;
}



template <typename T,
	  typename Block>
T
tc_sum_const(const_Matrix<T, Block> v)
{
  T sumval = T();
  for (index_type r=0; r<v.size(0); ++r)
    for (index_type c=0; c<v.size(1); ++c)
      sumval += v.get(r, c);
  return sumval;
}



template <typename T,
	  typename Block>
T
tc_sum(Matrix<T, Block> v)
{
  T sumval = T();
  for (index_type r=0; r<v.size(0); ++r)
    for (index_type c=0; c<v.size(1); ++c)
      sumval += v.get(r, c);
  return sumval;
}



template <typename       Storage,
	  dimension_type Dim>
void
tc_call_sum_const(Domain<Dim> const& dom, int k)
{
  Storage stor1(dom);

  typedef typename Storage::value_type T;

  fill_block<Dim>(stor1.block(), k);
  T sum = tc_sum_const(stor1.view);

  length_type len = stor1.view.size(); // dom[0].length() * dom[1].length();
  if (!equal(sum, T(k*len*(len-1)/2+len)))
  {
    std::cout << "tc_call_sum_const -- fail\n"
	      << "  expected: " << T(k*len*(len-1)/2+len) << '\n'
	      << "  got     : " << sum << std::endl;
  }
  test_assert(equal(sum, T(k*len*(len-1)/2+len)));
}



template <typename       Storage,
	  dimension_type Dim>
void
tc_call_sum(Domain<Dim> const& dom, int k)
{
  Storage stor1(dom);

  typedef typename Storage::value_type T;

  fill_block<Dim>(stor1.block(), k);
  T sum = tc_sum(stor1.view);

  length_type len = stor1.view.size(); // dom[0].length() * dom[1].length();
  test_assert(equal(sum, T(k*len*(len-1)/2+len)));
}



template <typename       T,
	  dimension_type Dim>
void
test_call_type(Domain<Dim> const& dom, int k)
{
  tc_call_sum_const<Const_storage<Dim, T> >(dom, k);
  tc_call_sum_const<     Storage<Dim, T> >(dom, k);

#if (ILLEGALCASE == 5)
  // should not be able to pass a const_Matrix to a routine
  // expecting a Matrix.
  tc_call_sum<Const_storage<Dim, T> >(dom, k);
#endif
  tc_call_sum<     Storage<Dim, T> >(dom, k);
}



template <dimension_type Dim>
void
test_call(Domain<Dim> const& dom, int k)
{
  test_call_type<scalar_f> (dom, k);
  test_call_type<cscalar_f>(dom, k);
  test_call_type<int>      (dom, k);
  test_call_type<short>    (dom, k);
}



// -------------------------------------------------------------------- //
// Test cases for returning a view from a function.
//
// The following functions are defined:
//   return_view()
//		Returns a view, populated with a fill_block
//   tc_assign_return()
//		Test case: assign returned view to existing view.
//   tc_cons_return()
//		Test case: construct new view from returned view.
//   test_return_type()
//		Calls tc_assign_return() and tc_cons_return() test
//		cases for a given type.
//   test_return()
//		Test for view return.  Calls test_return_type() for
//		several types.
// -------------------------------------------------------------------- //



// Returen view with domain DOM, filled with fill_block(k).

template <typename       Storage,
	  dimension_type Dim>
typename Storage::view_type
return_view(Domain<Dim> const& dom, int k)
{
  Storage stor(dom);

  // put_nth(stor.block(), 0, val)
  fill_block<Dim>(stor.block(), k);

  return stor.view;
}



// Assign a view from the value of a function returning a view.

template <typename       Storage1,
	  typename       Storage2,
	  dimension_type Dim>
void
tc_assign_return(Domain<Dim> const& dom, int k)
{
  Storage1 stor1(dom, typename Storage1::value_type());

  // test_assert(view1.get(0, 0) != val || val == T());

  stor1.view = return_view<Storage2>(dom, k);

  test_assert(check_view(stor1.view, k));
}



// Construct a view from the value of a function returning a view.

template <typename       Storage1,
	  typename       Storage2,
	  dimension_type Dim>
void
tc_cons_return(Domain<Dim> const& dom, int k)
{
  typename Storage1::view_type view1(return_view<Storage2>(dom, k));

  test_assert(check_view(view1, k));
}



// Test tc_cons_assign() and tc_cons_return for a given type.

template <typename       T,
	  dimension_type Dim>
void
test_return_type(Domain<Dim> const& dom, int k)
{
  tc_assign_return<Storage<Dim, T>,      Storage<Dim, T> >(dom, k);
  tc_assign_return<Storage<Dim, T>, Const_storage<Dim, T> >(dom, k);

#if ILLEGALCASE == 6
  // Illegal: you cannot assign to a const_Matrix
  tc_assign_return<float, const_Matrix, const_Matrix>(dom, k);
#endif
#if ILLEGALCASE == 7
  // Illegal: you cannot assign to a const_Matrix
  tc_assign_return<float, const_Matrix,       Matrix>(dom, k);
#endif

  tc_cons_return<     Storage<Dim, T>,      Storage<Dim, T> >(dom, k);
  tc_cons_return<     Storage<Dim, T>, Const_storage<Dim, T> >(dom, k);
  tc_cons_return<Const_storage<Dim, T>,      Storage<Dim, T> >(dom, k);
  tc_cons_return<Const_storage<Dim, T>, Const_storage<Dim, T> >(dom, k);
}



// Test view return.

template <dimension_type Dim>
void
test_return(Domain<Dim> const& dom, int k)
{
  test_return_type<scalar_f> (dom, k);
  test_return_type<cscalar_f>(dom, k);
  test_return_type<int>      (dom, k);
  test_return_type<short>    (dom, k);
}



// -------------------------------------------------------------------- //
// Test case for same-dimensional subviews (call-operator, get()).
//
// The following functions are defined:
//   tc_subview()
//		Test a 1-dim or 2-dim subview for consistency.
//   test_subview_type(), test_subview_domsub(), test_subview()
//		Wrappers for different views, types, and sizes.
// -------------------------------------------------------------------- //

template <typename       Storage>
void
tc_subview(
  Domain<1> const& dom,
  Domain<1> const& sub,
  int                k)
{
  typedef typename Storage::value_type T;
  dimension_type const dim = 1;

  for (dimension_type d=0; d<dim; ++d)
  {
    test_assert(sub[d].first()     >= dom[d].first());
    test_assert(sub[d].impl_last() <= dom[d].impl_last());
  }

  Storage stor(dom);

  fill_block<dim>(stor.block(), k);

  typename Storage::view_type::subview_type        subv = stor.view(sub);
  typename Storage::view_type::const_subview_type csubv = stor.view.get(sub);

  for (index_type i=0; i<subv.size(); ++i)
  {
    index_type parent_i = sub.impl_nth(i);

    test_assert(stor.view.get(parent_i) ==  subv.get(i));
    test_assert(stor.view.get(parent_i) == csubv.get(i));

    T val = stor.view.get(parent_i) + T(1);
    stor.block().put(parent_i, val);

    test_assert(stor.view.get(parent_i) ==  val);
    test_assert(stor.view.get(parent_i) ==  subv.get(i));
    test_assert(stor.view.get(parent_i) == csubv.get(i));
  }
}



template <typename       Storage>
void
tc_subview(
  Domain<2> const& dom,
  Domain<2> const& sub,
  int              k)
{
  typedef typename Storage::value_type T;
  dimension_type const dim = 2;

  for (dimension_type d=0; d<dim; ++d)
  {
    test_assert(sub[d].first()     >= dom[d].first());
    test_assert(sub[d].impl_last() <= dom[d].impl_last());
  }

  Storage stor(dom);

  fill_block<dim>(stor.block(), k);

  typename Storage::view_type::subview_type        subv = stor.view(sub);
  typename Storage::view_type::const_subview_type csubv = stor.view.get(sub);

  for (index_type r=0; r<subv.size(0); ++r)
  {
    for (index_type c=0; c<subv.size(1); ++c)
    {
      index_type par_r = sub[0].impl_nth(r);
      index_type par_c = sub[1].impl_nth(c);

      test_assert(stor.view.get(par_r, par_c) ==  subv.get(r, c));
      test_assert(stor.view.get(par_r, par_c) == csubv.get(r, c));

      T val = stor.view.get(par_r, par_c) + T(1);
      stor.block().put(par_r, par_c, val);
      
      test_assert(stor.view.get(par_r, par_c) ==  val);
      test_assert(stor.view.get(par_r, par_c) ==  subv.get(r, c));
      test_assert(stor.view.get(par_r, par_c) == csubv.get(r, c));
    }
  }
}



template <typename       T,
	  dimension_type Dim>
void
test_subview_type(Domain<Dim> const& dom, Domain<Dim> const& sub, int k)
{

  tc_subview<     Storage<Dim, T> >(dom, sub, k);
  tc_subview<Const_storage<Dim, T> >(dom, sub, k);
}



template <dimension_type Dim>
void
test_subview_domsub(Domain<Dim> const& dom, Domain<Dim> const& sub, int k)
{
  test_subview_type<scalar_f> (dom, sub, k);
  test_subview_type<cscalar_f>(dom, sub, k);
  test_subview_type<int>      (dom, sub, k);
  test_subview_type<short>    (dom, sub, k);
}


void
test_subview()
{
  test_subview_domsub(Domain<1>(10), Domain<1>(0, 1, 3), 3);
  test_subview_domsub(Domain<1>(10), Domain<1>(5, 1, 3), 3);
  test_subview_domsub(Domain<1>(10), Domain<1>(0, 2, 3), 3);
  test_subview_domsub(Domain<1>(10), Domain<1>(5, 2, 3), 3);
  test_subview_domsub(Domain<1>(256), Domain<1>(5, 5, 40), 3);

  test_subview_domsub(Domain<2>(10, 10), Domain<2>(3, 3), 3);
  test_subview_domsub(Domain<2>(10, 10),
		      Domain<2>(Domain<1>(5, 1, 3), Domain<1>(5, 1, 3)), 3);
  test_subview_domsub(Domain<2>(10, 10),
		      Domain<2>(Domain<1>(0, 2, 3), Domain<1>(5, 1, 3)), 3);
}
  


// Wrap non-subview test cases.

template <dimension_type Dim>
void
test_all(Domain<Dim> const& dom, int k)
{
  test_get(dom);
  test_getput(dom);
  test_copy_cons(dom, k);
  test_assign(dom, k);
  test_assign_scalar(dom, k);
  test_call(dom, k);
  test_return(dom, k);

}



int
main(int argc, char** argv)
{
  vsip::vsipl init(argc, argv);

  test_all(Domain<1>(1),   3);
  test_all(Domain<1>(10),  3);
  test_all(Domain<1>(257), 3);
  test_all(Domain<1>(328), 3);

  test_all(Domain<2>(1,  10),  3);
  test_all(Domain<2>(10,  1),  3); // 9
  test_all(Domain<2>(32, 32),  3); // 9


  // These fail the function call test because the element sum exceeds
  // the dynamic range of float:
  // test_all(Domain<2>(17, 257), 3); // 37
  // test_all(Domain<2>(1023, 10), 3); // 13

  test_subview();
}
