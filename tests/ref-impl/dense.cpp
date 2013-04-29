/***********************************************************************

  File:   dense.cpp
  Author: Jeffrey D. Oldham, CodeSourcery, LLC.
  Date:   06/24/2002

  Contents: Very simple tests of the Dense block class.

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
#include <vsip/initfin.hpp>
#include <vsip/dense.hpp>
#include <vsip/domain.hpp>
#include <vsip/support.hpp>

/***********************************************************************
  Function Definitions
***********************************************************************/

// Increment/decrement refcount of a block reference.
// (Should not cause block to be freed).
template <typename Block>
void
tc_mutable_refcount(Block& block)
{
   block.increment_count();
   block.decrement_count();
}



// Test that increment/decrement_count work despite Block const-ness.
template <vsip::dimension_type Dim,
	  typename          T>
void
tc_mutable_refcount_dim(vsip::Domain<Dim> const& dom)
{
   typedef vsip::Dense<Dim, T> block_type;

   block_type*	block = new block_type(dom, T(3));

   tc_mutable_refcount<const block_type>(*block);
   tc_mutable_refcount<block_type>(*block);
}


void test_mutable_refcount()
{
   tc_mutable_refcount_dim<1, vsip::scalar_f> (vsip::Domain<1>(7));
   tc_mutable_refcount_dim<1, vsip::cscalar_f>(vsip::Domain<1>(7));
   tc_mutable_refcount_dim<2, vsip::scalar_f> (vsip::Domain<2>(3, 7));
   tc_mutable_refcount_dim<2, vsip::cscalar_f>(vsip::Domain<2>(3, 7));
}



int
main (int argc, char** argv)
{
  vsip::vsipl	init(argc, argv);

  test_mutable_refcount();

  /* Test Dense<1, scalar_f>.  */
  vsip::Dense<>	denseDoubleOne ((vsip::Domain<1> (7)),
				static_cast<vsip::scalar_f>(3.4));
  insist (denseDoubleOne.user_storage () == vsip::no_user_format);
  for (vsip::index_type i = 0; i < 7; ++i)
    insist (equal (denseDoubleOne.get (i), static_cast<vsip::scalar_f>(3.4)));
  denseDoubleOne.put (2, 3.0);
  insist (equal (denseDoubleOne.get (1), static_cast<vsip::scalar_f>(3.4)));
  insist (equal (denseDoubleOne.get (2), static_cast<vsip::scalar_f>(3.0)));
  insist (equal (denseDoubleOne.get (3), static_cast<vsip::scalar_f>(3.4)));
  for (vsip::index_type i = 0; i < 7; ++i)
    insist (equal (denseDoubleOne.get (i), denseDoubleOne.get (i)));
  denseDoubleOne.put (2, static_cast<vsip::scalar_f>(3.4));
  for (vsip::index_type i = 0; i < 7; ++i)
    insist (equal (denseDoubleOne.get (i), static_cast<vsip::scalar_f>(3.4)));
  insist (denseDoubleOne.size () == denseDoubleOne.size (1, 0));
  insist (denseDoubleOne.size () == 7);

  /* Test a user-specified storage Dense<1, scalar_f>.  */
  vsip::length_type const
		block_length = 7;
			/* The length of the block to create and test.  */
  vsip::scalar_f*
		array = new vsip::scalar_f[block_length];
  vsip::Dense<> denseDoubleUser (block_length, array);
  insist (denseDoubleUser.admitted () == false);
  insist (denseDoubleUser.user_storage () == vsip::array_format);
  denseDoubleUser.admit ();
  insist (denseDoubleUser.admitted () == true);
  insist (denseDoubleUser.size () == denseDoubleUser.size (1, 0));
  insist (denseDoubleUser.size () == block_length);
  for (vsip::index_type i = 0; i < block_length; ++i)
    denseDoubleUser.put (i, static_cast<vsip::scalar_f>(3.4));
  for (vsip::index_type i = 0; i < block_length; ++i)
    insist (equal (denseDoubleUser.get (i), static_cast<vsip::scalar_f>(3.4)));
  denseDoubleUser.release (false);
  vsip::scalar_f* tmp_array;
  denseDoubleUser.find (tmp_array);
  insist (tmp_array == array);
  denseDoubleUser.admit (false);
  for (vsip::index_type i = 0; i < block_length; ++i)
    insist (equal (denseDoubleUser.get (i), static_cast<vsip::scalar_f>(3.4)));
  denseDoubleUser.release (false);
  denseDoubleUser.rebind (tmp_array);
  denseDoubleUser.admit (false);
  for (vsip::index_type i = 0; i < block_length; ++i)
    insist (equal (denseDoubleUser.get (i), static_cast<vsip::scalar_f>(3.4)));
  delete [] array;

  /* Test Dense<1, cscalar_f>.  */
  vsip::Dense<1, vsip::cscalar_f>
		denseComplexOne (block_length,
				 vsip::cscalar_f (3.4, 3.4));
  for (vsip::index_type i = 0; i < block_length; ++i)
    insist (equal (denseComplexOne.get (i), vsip::cscalar_f (3.4, 3.4)));
  denseComplexOne.put (2, vsip::cscalar_f (3.4, 0.0));
  for (vsip::index_type i = 0; i < block_length; ++i)
    if (i == 2) {
      insist (equal (denseComplexOne.get (i), vsip::cscalar_f (3.4, 0.0)));
      }
    else {
      insist (equal (denseComplexOne.get (i), vsip::cscalar_f (3.4, 3.4)));
    }
  insist (denseDoubleOne.size () == denseDoubleOne.size (1, 0));
  insist (denseDoubleOne.size () == block_length);

  /* Test a user-specified storage Dense<1, cscalar_f>.  */
  array = new vsip::scalar_f[2 * block_length];
  vsip::Dense<1, vsip::cscalar_f>
		denseComplexUser (block_length, array);
  insist (denseComplexUser.admitted () == false);
  insist (denseComplexUser.user_storage () == vsip::interleaved_format);
  denseComplexUser.admit ();
  insist (denseComplexUser.admitted () == true);
  insist (denseComplexUser.size () == denseComplexUser.size (1, 0));
  insist (denseComplexUser.size () == block_length);
  for (vsip::index_type i = 0; i < block_length; ++i)
    denseComplexUser.put (i, vsip::cscalar_f(3.4, 3.4));
  for (vsip::index_type i = 0; i < block_length; ++i)
    insist (equal (denseComplexUser.get (i), vsip::cscalar_f(3.4, 3.4)));
  denseComplexUser.release (false);
  denseComplexUser.find (tmp_array);
  insist (tmp_array == array);
  denseComplexUser.admit (false);
  for (vsip::index_type i = 0; i < block_length; ++i)
    insist (equal (denseComplexUser.get (i), vsip::cscalar_f(3.4, 3.4)));
  denseComplexUser.release (false);
  denseComplexUser.rebind (tmp_array);
  denseComplexUser.admit (false);
  for (vsip::index_type i = 0; i < block_length; ++i)
    insist (equal (denseComplexUser.get (i), vsip::cscalar_f(3.4, 3.4)));
  delete [] array;

  /* Test Dense<1, scalar_i>.  */

  vsip::Dense<1, vsip::scalar_i>
		denseIntOne ((vsip::Domain<1> (7)), 3);
  for (vsip::index_type i = 0; i < 7; ++i)
    insist (equal (denseIntOne.get (i), 3));
  denseIntOne.put (2, 4);
  insist (equal (denseIntOne.get (1), 3));
  insist (equal (denseIntOne.get (2), 4));
  insist (equal (denseIntOne.get (3), 3));
  for (vsip::index_type i = 0; i < 7; ++i)
    insist (equal (denseIntOne.get (i), denseIntOne.get (i)));
  denseIntOne.put (2, 3);
  for (vsip::index_type i = 0; i < 7; ++i)
    insist (equal (denseIntOne.get (i), 3));
  insist (denseIntOne.size () == denseIntOne.size (1, 0));
  insist (denseIntOne.size () == 7);
  insist (denseIntOne.dim == static_cast<vsip::dimension_type>(1));

  /* Test Dense<1, Index<2> >.  */

  vsip::Dense<1, vsip::Index<2> >
		denseIndex2 (7, vsip::Index<2>(2, 3));
  for (vsip::index_type i = 0; i < 7; ++i)
    insist (equal (denseIndex2.get (i), vsip::Index<2>(2, 3)));
  denseIndex2.put (2, vsip::Index<2>(3, 2));
  insist (equal (denseIndex2.get (1), vsip::Index<2>(2, 3)));
  insist (equal (denseIndex2.get (2), vsip::Index<2>(3, 2)));
  insist (equal (denseIndex2.get (3), vsip::Index<2>(2, 3)));

  /* Test Dense<2, scalar_f>.  */

  vsip::Dense<2, vsip::scalar_f>
		denseDoubleTwo (vsip::Domain<2> (3, 4),
				static_cast<vsip::scalar_f>(3.4));
  for (vsip::index_type i = 0; i < 3; ++i)
    for (vsip::index_type j = 0; j < 4; ++j)
      insist (equal (denseDoubleTwo.get (i, j),
		     static_cast<vsip::scalar_f>(3.4)));
  for (vsip::index_type i = 0; i < 3 * 4; ++i)
    insist (equal (denseDoubleTwo.get (i), static_cast<vsip::scalar_f>(3.4)));
  denseDoubleTwo.put (1, 1, 3.0);
  denseDoubleTwo.put (1, 2.0);
  for (vsip::index_type i = 0; i < 3; ++i)
    for (vsip::index_type j = 0; j < 4; ++j) {
      insist (equal (denseDoubleTwo.get (i, j), 
		     i == 1 && j == 1 ? static_cast<vsip::scalar_f>(3.0) :
		     (i == 0 && j == 1 ? static_cast<vsip::scalar_f>(2.0) :
		      static_cast<vsip::scalar_f>(3.4))));
      insist (equal (denseDoubleTwo.get (i, j), denseDoubleTwo.get (i, j)));
    }
  for (vsip::index_type i = 0; i < 3; ++i) {
    insist (equal (denseDoubleTwo.get (i), 
		   i == 4 ? static_cast<vsip::scalar_f>(3.0) :
		   (i == 1 ? static_cast<vsip::scalar_f>(2.0) :
		    static_cast<vsip::scalar_f>(3.4))));
    insist (equal (denseDoubleTwo.get (i), denseDoubleTwo.get (i)));
  }
  insist (denseDoubleTwo.size (2, 0) == 3);
  insist (denseDoubleTwo.size (2, 1) == 4);
  insist (denseDoubleTwo.size (1, 0) == 3*4);
  insist (denseDoubleTwo.size () == 3*4);
  insist (denseDoubleTwo.dim == static_cast<vsip::dimension_type>(2));

  /* Test a user-specified storage Dense<2, scalar_f>.  */
  vsip::length_type const
		number_rows = 3;
			/* The number of matrix rows.  */
  vsip::length_type const
		number_cols = 4;
			/* The number of matrix columns.  */
  array = new vsip::scalar_f[number_rows * number_cols];
  vsip::Dense<2>
		denseDoubleTwoUser (vsip::Domain<2>(number_rows, number_cols),
				    array);
  insist (denseDoubleTwoUser.admitted () == false);
  insist (denseDoubleTwoUser.user_storage () == vsip::array_format);
  denseDoubleTwoUser.admit ();
  insist (denseDoubleTwoUser.admitted () == true);
  insist (denseDoubleTwoUser.size () == denseDoubleTwoUser.size (1, 0));
  insist (denseDoubleTwoUser.size () == number_rows * number_cols);
  for (vsip::index_type i = 0; i < number_rows; ++i)
    for (vsip::index_type j = 0; j < number_cols; ++j)
      denseDoubleTwoUser.put (i, j, static_cast<vsip::scalar_f>(3.4));
  for (vsip::index_type i = 0; i < number_rows; ++i)
    for (vsip::index_type j = 0; j < number_cols; ++j)
      insist (equal (denseDoubleTwoUser.get (i, j),
		     static_cast<vsip::scalar_f>(3.4)));
  denseDoubleTwoUser.release (false);
  denseDoubleTwoUser.find (tmp_array);
  insist (tmp_array == array);
  denseDoubleTwoUser.admit (false);
  for (vsip::index_type i = 0; i < number_rows; ++i)
    for (vsip::index_type j = 0; j < number_cols; ++j)
      insist (equal (denseDoubleTwoUser.get (i, j),
		     static_cast<vsip::scalar_f>(3.4)));
  denseDoubleUser.release (false);
  denseDoubleUser.rebind (tmp_array);
  denseDoubleUser.admit (false);
  for (vsip::index_type i = 0; i < number_rows; ++i)
    for (vsip::index_type j = 0; j < number_cols; ++j)
      insist (equal (denseDoubleTwoUser.get (i, j),
		     static_cast<vsip::scalar_f>(3.4)));
  delete [] array;

  /* Test a Dense<2, scalar_f> with column-major storage order.  */
  vsip::Dense<2, vsip::scalar_f, vsip::col2_type>
		denseDoubleCol (vsip::Domain<2>(number_rows, number_cols),
				static_cast<vsip::scalar_f>(3.4));
  insist (denseDoubleCol.size () == denseDoubleCol.size (1, 0));
  insist (denseDoubleCol.size () == number_rows * number_cols);
  insist (denseDoubleCol.size (2, 0) == number_rows);
  insist (denseDoubleCol.size (2, 1) == number_cols);
  denseDoubleCol.put (2, 0, static_cast<vsip::scalar_f>(3.0));
  insist (equal (denseDoubleCol.get (2, 0),
		 static_cast<vsip::scalar_f>(3.0)));
  insist (equal (denseDoubleCol.get (2),
		 static_cast<vsip::scalar_f>(3.0)));
  
  /* Test a Dense given a Domain with non-zero offsets and non-unit
     strides.  */

  vsip::Dense<2, vsip::scalar_i>
		denseIntTwo ((vsip::Domain<2>
			      ((::vsip::Domain<1> (2, 2, 3)),
			       (::vsip::Domain<1> (3, -1, 2)))));
  denseIntTwo.put (2, 1, 1);
  insist (equal (denseIntTwo.get (2, 1), 1));
  insist (equal (denseIntTwo.get (2, 1), denseIntTwo.get (5)));
  insist (denseIntTwo.dim == static_cast<vsip::dimension_type>(2));

  return EXIT_SUCCESS;
}
