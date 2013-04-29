//
// Copyright (c) 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#ifndef TESTS_TEST_COMMON_HPP
#define TESTS_TEST_COMMON_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/complex.hpp>
#include <vsip/vector.hpp>
#include <vsip/map.hpp>

#define VERBOSE      0
#define DO_ASSERT    1
#define PROVIDE_SHOW 0

#if VERBOSE || PROVIDE_SHOW
#  include <iostream>
#endif



/***********************************************************************
  Definitions
***********************************************************************/

template <typename T>
struct Value_class
{
  static T exec(vsip::index_type idx, int k)
  {
    return T(k*idx);
  }
};

template <typename T>
struct Value_class<vsip::complex<T> >
{
  static vsip::complex<T> exec(vsip::index_type idx, int k)
  {
    return vsip::complex<T>(k*idx, k*idx+1);
  }
};



template <typename T>
inline T
value(vsip::index_type idx, int k)
{
  return Value_class<T>::exec(idx, k);
}



template <typename T>
inline T
value(vsip::index_type row, vsip::index_type col, int k)
{
  return T(100*k*row + col);
}



template <typename T>
inline T
value(
  vsip::index_type idx0,
  vsip::index_type idx1,
  vsip::index_type idx2,
  int              k)
{
  return T(k*(10000*idx0 + 100*idx1 + idx2));
}



template <typename T,
	  typename BlockT>
void
setup(vsip::Vector<T, BlockT> vec, int k)
{
  using vsip::no_subblock;
  using vsip::index_type;
  typedef T value_type;

  if (subblock(vec) != no_subblock)
  {
    for (index_type li=0; li<vec.local().size(); ++li)
    {
      index_type gi = global_from_local_index(vec, 0, li); 
      vec.local().put(li, value<T>(gi, k));
    }
  }
}



template <typename T,
	  typename BlockT>
void
setup(vsip::Matrix<T, BlockT> view, int k)
{
  using vsip::no_subblock;
  using vsip::index_type;
  typedef T value_type;

  if (subblock(view) != no_subblock)
  {
    for (index_type lr=0; lr<view.local().size(0); ++lr)
      for (index_type lc=0; lc<view.local().size(1); ++lc)
      {
	index_type gr = global_from_local_index(view, 0, lr); 
	index_type gc = global_from_local_index(view, 1, lc); 
	view.local().put(lr, lc, value<T>(gr, gc, k));
      }
  }
}



template <typename T,
	  typename BlockT>
void
setup(vsip::Tensor<T, BlockT> view, int k)
{
  using vsip::no_subblock;
  using vsip::index_type;
  typedef T value_type;

  if (subblock(view) != no_subblock)
  {
    for (index_type l0=0; l0<view.local().size(0); ++l0)
      for (index_type l1=0; l1<view.local().size(1); ++l1)
	for (index_type l2=0; l2<view.local().size(2); ++l2)
	{
	  index_type g0 = global_from_local_index(view, 0, l0); 
	  index_type g1 = global_from_local_index(view, 1, l1); 
	  index_type g2 = global_from_local_index(view, 2, l2); 
	  view.local().put(l0, l1, l2, value<T>(g0, g1, g2, k));
	}
  }
}



template <typename T,
	  typename BlockT>
void
check(vsip::const_Vector<T, BlockT> vec, int k, int shift=0)
{
  using vsip::no_subblock;
  using vsip::index_type;
  typedef T value_type;

#if VERBOSE
  std::cout << "check(k=" << k << ", shift=" << shift << "):"
	    << std::endl;
#endif
  if (subblock(vec) != no_subblock)
  {
    for (index_type li=0; li<vec.local().size(); ++li)
    {
      index_type gi = global_from_local_index(vec, 0, li); 
#if VERBOSE
      std::cout << " - " << li << "  gi:" << gi << " = "
		<< vec.local().get(li)
		<< "  exp: " << value<T>(gi + shift, k)
		<< std::endl;
#endif
#if DO_ASSERT
      test_assert(vec.local().get(li) == value<T>(gi + shift, k));
#endif
    }
  }
}



template <typename T,
	  typename BlockT>
void
check(vsip::const_Matrix<T, BlockT> view, int k, int rshift=0, int cshift=0)
{
  using vsip::no_subblock;
  using vsip::index_type;
  typedef T value_type;

#if VERBOSE
  std::cout << "check(k=" << k << ", rshift=" << rshift
	    << ", cshift=" << cshift << "):"
	    << std::endl;
#endif
  if (subblock(view) != no_subblock)
  {
    for (index_type lr=0; lr<view.local().size(0); ++lr)
      for (index_type lc=0; lc<view.local().size(1); ++lc)
      {
	index_type gr = global_from_local_index(view, 0, lr); 
	index_type gc = global_from_local_index(view, 1, lc); 
#if VERBOSE
	std::cout << " - " << lr << ", " << lc << "  g:"
		  << gr << ", " << gc << " = "
		  << view.local().get(lr, lc)
		  << "  exp: " << value<T>(gr+rshift, gc + cshift, k)
		  << std::endl;
#endif
#if DO_ASSERT
	test_assert(view.local().get(lr, lc) ==
		    value<T>(gr+rshift, gc+cshift, k));
#endif
      }
  }
}



template <typename T,
	  typename BlockT>
void
check(
  vsip::const_Tensor<T, BlockT> view,
  int                           k,
  int                           shift0=0,
  int                           shift1=0,
  int                           shift2=0)
{
  using vsip::no_subblock;
  using vsip::index_type;
  typedef T value_type;

#if VERBOSE
  std::cout << "check(k=" << k
	    << ", shift0=" << shift0
	    << ", shift1=" << shift1
	    << ", shift2=" << shift2 << "):"
	    << std::endl;
#endif
  if (subblock(view) != no_subblock)
  {
    for (index_type l0=0; l0<view.local().size(0); ++l0)
      for (index_type l1=0; l1<view.local().size(1); ++l1)
	for (index_type l2=0; l2<view.local().size(2); ++l2)
        {
	  index_type g0 = global_from_local_index(view, 0, l0); 
	  index_type g1 = global_from_local_index(view, 1, l1); 
	  index_type g2 = global_from_local_index(view, 2, l2); 
#if VERBOSE
	  std::cout << " - "
		    << l0 << ", " << l1 << ", " << l2 << "  g:"
		    << g0 << ", " << g1 << ", " << g2 << " = "
		    << view.local().get(l0, l1, l2)
		    << "  exp: " << value<T>(g0+shift0, g1 + shift1,
					     g2+shift2, k)
		    << std::endl;
#endif
#if DO_ASSERT
	test_assert(view.local().get(l0, l1, l2) ==
		    value<T>(g0+shift0, g1+shift1, g2+shift2, k));
#endif
      }
  }
}



template <typename T,
	  typename BlockT>
void
check_row_vector(vsip::const_Vector<T, BlockT> view, int row, int k=1)
{
  using vsip::no_subblock;
  using vsip::index_type;
  typedef T value_type;

  if (subblock(view) != no_subblock)
  {
    for (index_type lc=0; lc<view.local().size(0); ++lc)
    {
      index_type gc = global_from_local_index(view, 0, lc); 
      test_assert(view.local().get(lc) ==
		  value<T>(row, gc, k));
      }
  }
}



template <typename T,
	  typename BlockT>
void
check_col_vector(vsip::const_Vector<T, BlockT> view, int col, int k=0)
{
  using vsip::no_subblock;
  using vsip::index_type;
  typedef T value_type;

  if (subblock(view) != no_subblock)
  {
    for (index_type lr=0; lr<view.local().size(0); ++lr)
    {
      index_type gr = global_from_local_index(view, 0, lr); 
      test_assert(view.local().get(lr) == value<T>(gr, col, k));
    }
  }
}



#if PROVIDE_SHOW
template <typename T,
	  typename BlockT>
void
show(vsip::const_Vector<T, BlockT> vec)
{
  using vsip::no_subblock;
  using vsip::index_type;
  typedef T value_type;

  std::cout << "[" << vsip::local_processor() << "] " << "show\n";
  if (subblock(vec) != no_subblock)
  {
    for (index_type li=0; li<vec.local().size(); ++li)
    {
      index_type gi = global_from_local_index(vec, 0, li); 
      std::cout << "[" << vsip::local_processor() << "] "
		<< li << "  gi:" << gi << " = "
		<< vec.local().get(li)
		<< std::endl;
    }
  }
  else
    std::cout << "[" << vsip::local_processor() << "] "
	      << "show: no local subblock\n";
}



template <typename T,
	  typename BlockT>
void
show(vsip::const_Matrix<T, BlockT> view)
{
  using vsip::no_subblock;
  using vsip::index_type;
  typedef T value_type;

  std::cout << "[" << vsip::local_processor() << "] " << "show\n";
  if (subblock(view) != no_subblock)
  {
    for (index_type lr=0; lr<view.local().size(0); ++lr)
      for (index_type lc=0; lc<view.local().size(1); ++lc)
      {
	index_type gr = global_from_local_index(view, 0, lr); 
	index_type gc = global_from_local_index(view, 1, lc); 
	std::cout << "[" << vsip::local_processor() << "] "
		  << lr << "," << lc
		  << "  g:" << gr << "," << gc << " = "
		  << view.local().get(lr, lc)
		  << std::endl;
      }
  }
  else
    std::cout << "[" << vsip::local_processor() << "] "
	      << "show: no local subblock\n";
}
#endif

#undef VERBOSE

#endif // TESTS_TEST_COMMON_HPP
