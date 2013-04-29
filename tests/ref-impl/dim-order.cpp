/***********************************************************************

  File:   dim-order.cpp
  Author: Jules Bergmann, CodeSourcery, LLC.
  Date:   11/12/2004

  Contents: Test cases for dimension order.

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
#include "test.hpp"
#include <vsip/domain.hpp>
#include <vsip/initfin.hpp>
#include <vsip/vector.hpp>
#include "test-util.hpp"

using namespace std;
using namespace vsip;



/***********************************************************************
  Function Definitions
***********************************************************************/



// -------------------------------------------------------------------- //
// mat_value -- produce deterministic matrix values based on index.
template <typename T>
T mat_value(int i0, int i1)
{
   return (T)(100*i0 + i1);
}



// -------------------------------------------------------------------- //
// check_fix0_view -- Check a "row" vector sub-view of a matrix for
//                    corret values.
//
// Notes:
//  - "fix0" refers to holding the 0th matrix dimension constant.
//    I.e. matrix(fix, free)
//  - Expected values are defined by the mat_value() function.
// -------------------------------------------------------------------- //
template <typename T,
	  typename Block>
void check_fix0_view(
   const_Vector<T, Block>	vec,
   int				i0,
   int				offset=0)
{
   for (length_type i1=0; i1<vec.size(); ++i1) {
      insist(equal(vec.get(i1), mat_value<T>(i0, i1+offset)));
      }
}



// -------------------------------------------------------------------- //
// check_fix1_view -- Check a "col" vector sub-view of a matrix for
//                    corret values.
//
// Notes:
//  - "fix1" refers to holding the 1th matrix dimension constant.
//    I.e. matrix(free, fix)
//  - Expected values are defined by the mat_value() function.
// -------------------------------------------------------------------- //
template <typename T,
	  typename Block>
void check_fix1_view(
   const_Vector<T, Block>	vec,
   int				i1,
   int				offset=0)
{
   for (length_type i0=0; i0<vec.size(); ++i0) {
      insist(equal(vec.get(i0), mat_value<T>(i0+offset, i1)));
      }
}



// -------------------------------------------------------------------- //
// test_vec_subviews -- Test row and column vector subviews of a matrix
//
// Description:
//  1. Populate matrix with data from mat_value() function.
//  2. Use check_fixX_view() to check row/col subviews for
//     correctness.
// -------------------------------------------------------------------- //
template <typename T,
	  typename Block>
void
test_vec_subviews(
   Matrix<T, Block>	mat)
{
   length_type const	size0 = mat.size(0);
   length_type const	size1 = mat.size(1);

   for (length_type i0=0; i0<size0; ++i0) {
      for (length_type i1=0; i1<size1; ++i1) {
	 mat.put(i0, i1, mat_value<T>(i0, i1));
	 }
      }

   // check "row" vector sub-views
   for (length_type i0=0; i0<size0; ++i0) {
      check_fix0_view(mat.row(i0), i0);
      check_fix0_view(mat.row(i0).get(Domain<1>(size0-1)), i0);
      check_fix0_view(mat.row(i0).get(Domain<1>(1,1,size0-2)), i0, 1);
      }

   // check "col" vector sub-views
   for (length_type i1=0; i1<size1; ++i1) {
      check_fix1_view(mat.col(i1), i1);
      check_fix1_view(mat.col(i1).get(Domain<1>(size0-1)), i1);
      check_fix1_view(mat.col(i1).get(Domain<1>(1,1,size0-2)), i1, 1);
      }
}



// -------------------------------------------------------------------- //
// test_1 -- test various matrix combinations for a given data type
//           and dimension order (specified as template parameters).
//
template <typename T,
	  typename Order>
void
test_1()
{
   length_type const	size0 = 15;
   length_type const	size1 = 25;
   
   Matrix<T, Dense<2, T, Order> >	mat(size0,size1);

   // base case: regular matrix
   mat = 0.f;
   test_vec_subviews(mat);

   // stress case: matrix subview
   mat = 0.f;
   test_vec_subviews(mat(Domain<2>(Domain<1>(5,1,5),
				   Domain<1>(10,1,10))) );

   // stress case: matrix subview-subview
   mat = 0.f;
   test_vec_subviews(mat(Domain<2>(Domain<1>(5,1,5),
				   Domain<1>(10,1,10)))
		        (Domain<2>(Domain<1>(0,1,4),
				       Domain<1>(0,1,9))) );

   // stress case: matrix subview-subview plus shift
   mat = 0.f;
   test_vec_subviews(mat(Domain<2>(Domain<1>(5,1,5),
				   Domain<1>(10,1,10)))
		        (Domain<2>(Domain<1>(1,1,4),
				   Domain<1>(1,1,9))) );

#if 0
   // stress case: matrix subview-subview plus shift/stride
   mat = 0.f;
   test_vec_subviews(mat(Domain<2>(Domain<1>(5,1,5),
				   Domain<1>(10,1,10)))
		        (Domain<2>(Domain<1>(1,2,2),
				   Domain<1>(1,3,2))) );
#endif
}



// -------------------------------------------------------------------- //
int
main (int argc, char** argv)
{
   vsip::vsipl	init(argc, argv);

   test_1<scalar_f, row2_type>();
   test_1<scalar_f, col2_type>();
}
