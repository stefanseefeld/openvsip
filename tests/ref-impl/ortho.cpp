/***********************************************************************

  File:   ortho.cpp
  Author: Jules Bergmann, CodeSourcery, LLC.
  Date:   12/04/2005

  Contents: Tests for Cottel issues

Copyright 2005 Georgia Tech Research Corporation, all rights reserved.

A non-exclusive, non-royalty bearing license is hereby granted to all
Persons to copy, distribute and produce derivative works for any
purpose, provided that this copyright notice and following disclaimer
appear on All copies: THIS LICENSE INCLUDES NO WARRANTIES, EXPRESSED
OR IMPLIED, WHETHER ORAL OR WRITTEN, WITH RESPECT TO THE SOFTWARE OR
OTHER MATERIAL INCLUDING, BUT NOT LIMITED TO, ANY IMPLIED WARRANTIES
OF MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE, OR ARISING
FROM A COURSE OF PERFORMANCE OR DEALING, OR FROM USAGE OR TRADE, OR OF
NON-INFRINGEMENT OF ANY PATENTS OF THIRD PARTIES. THE INFORMATION IN
THIS DOCUMENT SHOULD NOT BE CONSTRUED AS A COMMITMENT OF DEVELOPMENT
BY ANY OF THE ABOVE PARTIES.

The US Government has a license under these copyrights, and this
Material may be reproduced by or for the US Government.
***********************************************************************/

/***********************************************************************
  Included Files
***********************************************************************/

#include <cstdlib>
#include <vsip/domain.hpp>
#include <vsip/initfin.hpp>
#include <vsip/vector.hpp>

#include "test.hpp"
#include "test-util.hpp"

using namespace vsip;


template <dimension_type Dim,
	  typename       T>
class ViewOfDim;

template <typename T>
class ViewOfDim<1, T>
{
public:
   typedef Vector<T> type;

   ViewOfDim(length_type N)        : view(N) {}
   ViewOfDim(length_type N, T val) : view(N, val) {}

public:
   type view;
};

template <typename T>
class ViewOfDim<2, T>
{
public:
   typedef Matrix<T> type;

   ViewOfDim(length_type N)        : view(N, N) {}
   ViewOfDim(length_type N, T val) : view(N, N, val) {}

public:
   type view;
};


/***********************************************************************
  Function Definitions
***********************************************************************/

// Test 'view = scalar' assignment (Issue #2)

template <dimension_type Dim,
	  typename       T>
void
test_view_scalar_assn(T const& val1, T const& val2)
{
   length_type N = 5;

   ViewOfDim<Dim, T> view(N, val1);

   for (index_type i=0; i<view.view.size(); ++i)
      insist(equal(get_nth(view.view, i), val1));

   view.view = val2;

   for (index_type i=0; i<view.view.size(); ++i)
      insist(equal(get_nth(view.view, i), val2));
}



// Test 'view op= scalar' assignment (Issue #3)

template <dimension_type Dim,
	  typename       T1,
	  typename       T2>
void
test_view_op_assn(T1 const& val1, T2 const& val2)
{
   length_type N = 5;

   ViewOfDim<Dim, T1> view(N, val1);

   for (index_type i=0; i<view.view.size(); ++i)
      insist(equal(get_nth(view.view, i), val1));

   // test +=

   view.view += val2;

   for (index_type i=0; i<view.view.size(); ++i)
      insist(equal(get_nth(view.view, i), val1 + val2));

   // test -=

   view.view = val1;
   view.view -= val2;

   for (index_type i=0; i<view.view.size(); ++i)
      insist(equal(get_nth(view.view, i), val1 - val2));

   // test *=

   view.view = val1;
   view.view *= val2;

   for (index_type i=0; i<view.view.size(); ++i)
      insist(equal(get_nth(view.view, i), val1 * val2));

   // test /=

   view.view = val1;
   view.view /= val2;

   for (index_type i=0; i<view.view.size(); ++i)
      insist(equal(get_nth(view.view, i), val1 / val2));
}



template <dimension_type Dim,
	  typename       T1,
	  typename       T2>
void
test_add(T1 const& val1, T2 const& val2)
{
   typedef typename Promotion<T1, T2>::type RT;

   length_type N = 5;

   ViewOfDim<Dim, T1> view1(N, val1);
   ViewOfDim<Dim, T2> view2(N, val2);
   ViewOfDim<Dim, RT> viewR(N, RT());

   RT valR = vsip::add(val1, val2);

   viewR.view = view1.view + view2.view;

   for (index_type i=0; i<viewR.view.size(); ++i)
      insist(equal(get_nth(viewR.view, i), valR));
}



template <dimension_type Dim,
	  typename       T1,
	  typename       T2>
void
test_sub(T1 const& val1, T2 const& val2)
{
   typedef typename Promotion<T1, T2>::type RT;

   length_type N = 5;

   ViewOfDim<Dim, T1> view1(N, val1);
   ViewOfDim<Dim, T2> view2(N, val2);
   ViewOfDim<Dim, RT> viewR(N, RT());

   RT valR = vsip::sub(val1, val2);

   viewR.view = view1.view - view2.view;

   for (index_type i=0; i<viewR.view.size(); ++i)
      insist(equal(get_nth(viewR.view, i), valR));
}


template <dimension_type Dim,
	  typename       T1,
	  typename       T2>
void
test_mul(T1 const& val1, T2 const& val2)
{
   typedef typename Promotion<T1, T2>::type RT;

   length_type N = 5;

   ViewOfDim<Dim, T1> view1(N, val1);
   ViewOfDim<Dim, T2> view2(N, val2);
   ViewOfDim<Dim, RT> viewR(N, RT());

   RT valR = vsip::mul(val1, val2);

   viewR.view = view1.view * view2.view;

   for (index_type i=0; i<viewR.view.size(); ++i)
      insist(equal(get_nth(viewR.view, i), valR));
}



template <dimension_type Dim,
	  typename       T1,
	  typename       T2>
void
test_div(T1 const& val1, T2 const& val2)
{
   typedef typename Promotion<T1, T2>::type RT;

   length_type N = 5;

   ViewOfDim<Dim, T1> view1(N, val1);
   ViewOfDim<Dim, T2> view2(N, val2);
   ViewOfDim<Dim, RT> viewR(N, RT());

   RT valR = vsip::div(val1, val2);

   viewR.view = view1.view / view2.view;

   for (index_type i=0; i<viewR.view.size(); ++i)
      insist(equal(get_nth(viewR.view, i), valR));
}



template <dimension_type Dim,
	  typename       T>
void
test_neg(T const& val1)
{
   length_type N = 5;

   ViewOfDim<Dim, T> view1(N, val1);
   ViewOfDim<Dim, T> viewR(N, T());

   T valR = vsip::neg(val1);

   viewR.view = -view1.view;

   for (index_type i=0; i<viewR.view.size(); ++i)
      insist(equal(get_nth(viewR.view, i), valR));
}



int
main (int argc, char** argv)
{
  vsip::vsipl	init(argc, argv);

  // Issue #2 ---------------------------------------------------------	//
  // Pre-existing cases.
  test_view_scalar_assn<1, scalar_f>(1.f, 2.f);
  test_view_scalar_assn<1, scalar_i>(1, 2);
  test_view_scalar_assn<1, cscalar_f>(1.f, 2.f);
  test_view_scalar_assn<2, scalar_f>(1.f, 2.f);
  test_view_scalar_assn<2, scalar_i>(1, 2);
  test_view_scalar_assn<2, cscalar_f>(1.f, 2.f);

  // New, orthoganol functionality.
  test_view_scalar_assn<1, bool>(true, false);
  test_view_scalar_assn<1, index_type>(1, 2);
  test_view_scalar_assn<1, Index<1> >(Index<1>(1), Index<1>(2));
  test_view_scalar_assn<1, Index<2> >(Index<2>(1, 2), Index<2>(3, 4));
  test_view_scalar_assn<2, bool>(true, false);

  // Not supported. (tvcpp doesn't provide cscalar_i vector)
  // test_view_scalar_assn<1, cscalar_i>(); 
  // test_view_scalar_assn<2, cscalar_i>(); 


  // Issue #3 ---------------------------------------------------------	//
  // Pre-existing cases.
  // (Note: cscalar_f /= cscalar_f wasn't implemented)
  test_view_op_assn<1, cscalar_f,  scalar_f>(cscalar_f(1.f, 2.f), 2.f);
  test_view_op_assn<1, cscalar_f, cscalar_f>(cscalar_f(1.f, 3.f),
					     cscalar_f(2.f, 0.f));
  test_view_op_assn<2, cscalar_f,  scalar_f>(cscalar_f(1.f, 2.f), 2.f);
  test_view_op_assn<2, cscalar_f, cscalar_f>(cscalar_f(1.f, 3.f),
					     cscalar_f(2.f, 0.f));

  // cscalar_i not supported by tvcpp

  // Issue #X ---------------------------------------------------------	//
  test_add<1,  scalar_i,  scalar_i>(1, 2);
  test_add<1,  scalar_f,  scalar_f>(1.f, 2.f);
  test_add<1,  scalar_f, cscalar_f>(1.f, 2.f);
  test_add<1, cscalar_f,  scalar_f>(1.f, 2.f);	// New case.
  test_add<1, cscalar_f, cscalar_f>(1.f, 2.f);

  test_add<2,  scalar_i,  scalar_i>(1, 2);	// New case.
  test_add<2,  scalar_f,  scalar_f>(1.f, 2.f);
  test_add<2,  scalar_f, cscalar_f>(1.f, 2.f);
  test_add<2, cscalar_f,  scalar_f>(1.f, 2.f);	// New case.
  test_add<2, cscalar_f, cscalar_f>(1.f, 2.f);

  test_sub<1,  scalar_i,  scalar_i>(1, 2);
  test_sub<1,  scalar_f,  scalar_f>(1.f, 2.f);
  test_sub<1,  scalar_f, cscalar_f>(1.f, 2.f);
  test_sub<1, cscalar_f,  scalar_f>(1.f, 2.f);	// New case.
  test_sub<1, cscalar_f, cscalar_f>(1.f, 2.f);

  test_sub<2,  scalar_i,  scalar_i>(1, 2);	// New case.
  test_sub<2,  scalar_f,  scalar_f>(1.f, 2.f);
  test_sub<2,  scalar_f, cscalar_f>(1.f, 2.f);
  test_sub<2, cscalar_f,  scalar_f>(1.f, 2.f);	// New case.
  test_sub<2, cscalar_f, cscalar_f>(1.f, 2.f);

  test_mul<1,  scalar_f,  scalar_f>(1.f, 2.f);
  test_mul<1,  scalar_f, cscalar_f>(1.f, 2.f);
  test_mul<1, cscalar_f,  scalar_f>(1.f, 2.f);	// New case.
  test_mul<1, cscalar_f, cscalar_f>(1.f, 2.f);

  test_mul<2,  scalar_f,  scalar_f>(1.f, 2.f);
  test_mul<2,  scalar_f, cscalar_f>(1.f, 2.f);
  test_mul<2, cscalar_f,  scalar_f>(1.f, 2.f);	// New case.
  test_mul<2, cscalar_f, cscalar_f>(1.f, 2.f);

  test_div<1,  scalar_f,  scalar_f>(1.f, 2.f);
  test_div<1,  scalar_f, cscalar_f>(1.f, 2.f);
  test_div<1, cscalar_f,  scalar_f>(1.f, 2.f);
  test_div<1, cscalar_f, cscalar_f>(1.f, 2.f);

  test_div<2,  scalar_f,  scalar_f>(1.f, 2.f);
  test_div<2,  scalar_f, cscalar_f>(1.f, 2.f);
  test_div<2, cscalar_f,  scalar_f>(1.f, 2.f);
  test_div<2, cscalar_f, cscalar_f>(1.f, 2.f);

  test_neg<1,  scalar_i>(1);
  test_neg<1,  scalar_f>(1.f);
  test_neg<1, cscalar_f>(1.f);
  test_neg<2,  scalar_i>(1); // New case.
  test_neg<2,  scalar_f>(1.f);
  test_neg<2, cscalar_f>(1.f);

  return EXIT_SUCCESS;
}
