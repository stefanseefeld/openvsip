/***********************************************************************

  File:   matrix-const.cpp
  Author: Jules Bergmann, CodeSourcery, LLC.
  Date:   11/19/2004

  Contents: Test cases related to Matrix / const_Matrix classes.

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


#include <iostream>
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

int const	dim = 2;

template <typename T>
struct block_type {
   typedef Dense<dim, T>
		type;
};


template <int N, typename T>
struct new_block;


template <typename T>
struct new_block<1, T> {
   Dense<1, T>* alloc(int N) { return new Dense<1, T>(N, T()); }
};



template <typename T>
struct new_block<2, T> {
   Dense<2, T>* alloc(int N) { return new Dense<2, T>(N, N, T()); }
};
		


/* ==================================================================== *
 * Test #1 -- test basic put/get member functions for Matrix Views
 *
 * Functions
 *  - test_put_get_case() tests the put() and get() member functions
 *                        for a View<T>.  If View does not defined
 *                        put(), this routine should result in a
 *                        compilation error.
 *  - test_put_get_set() calls test_put_get_case() for each VSIPL++
 *                       data type, leaving View as a template parameter.
 *  - test_put_get_Matrix() calls test_put_get_set() for Matrix
 *  - test_put_get_const_Matrix() calls test_put_get_set() for const_Matrix
 *
 * Note: const_Matrix test should not compile.  Special test setup
 *       is required to run these tests.
 * ==================================================================== */



// -------------------------------------------------------------------- //
template <typename			      T,
	  template <typename, typename> class View>
int
test_put_get_case()
{
   typedef View<T, Dense<dim, T> >
		view_type;

   int const	N   = 10;
   view_type	mat(N, N, T());

   mat.put(1, 0, TestRig<T>::test_value1());
   mat.put(2, 3, TestRig<T>::test_value2());

   T chk1 = mat.get(1, 0);
   T chk2 = mat.get(2, 3);
   
   return chk1 == TestRig<T>::test_value1() &&
	  chk2 == TestRig<T>::test_value2();
}



// -------------------------------------------------------------------- //
template <template <typename, typename> class View>
void
test_put_get_set()
{
#if !defined(PARTIAL_COMPILE) || defined(TEST_PUT_GET_1)
   insist((test_put_get_case< scalar_f, View>()));
#endif
#if !defined(PARTIAL_COMPILE) || defined(TEST_PUT_GET_2)
   insist((test_put_get_case< scalar_i, View>()));
#endif
#if !defined(PARTIAL_COMPILE) || defined(TEST_PUT_GET_3)
   insist((test_put_get_case<cscalar_f, View>()));
#endif
#if !defined(PARTIAL_COMPILE) || defined(TEST_PUT_GET_4)
   insist((test_put_get_case<     bool, View>()));
#endif
#if !defined(PARTIAL_COMPILE) || defined(TEST_PUT_GET_5)
   // insist((test_put_get_case<  index_type, View>())); // not valid for Matrix
#endif
}



// -------------------------------------------------------------------- //
void
test_put_get_Matrix()
{
   test_put_get_set<Matrix>();
}



// -------------------------------------------------------------------- //
void
test_put_get_const_Matrix()
{
#if defined(PARTIAL_COMPILE)
   test_put_get_set<const_Matrix>();
#endif
}



/* ==================================================================== *
 * Test #2 -- Test Matrix and const_Matrix copy constructors
 *
 * Three Functions:
 *  - test_copy_cons() tests the copy constructor View1<T1>(View2<T2>).
 *  - wrap_copy_cons() calls test_copy_cons() for the 4 permutations of
 *                     {Matrix, const_Matrix}^2.  Types are left as
 *		       template parameters.
 *  - test() calls wrap_copy_cons() for each permissible VSIPL type.
 * ==================================================================== */


// -------------------------------------------------------------------- //
template <typename			      T1,
	  typename			      T2,
	  template <typename, typename> class View1,
	  template <typename, typename> class View2>
int
test_copy_cons_case()
{
   int const	N	= 10;
   Dense<dim, T1>*
		block	= new Dense<2, T1>(Domain<2>(N, N), T1());
		
   put_origin(block, TestRig<T1>::test_value1());

   View1<T1, Dense<dim, T1> >	view1(*block);
   View2<T2, Dense<dim, T2> >	view2(view1);

   block->decrement_count(); // allow view1 to take ownership

   return 
      get_origin(view2) == (T2)TestRig<T1>::test_value1();
}



// -------------------------------------------------------------------- //
template <typename T>
void
test_copy_cons_set()
{
   insist((test_copy_cons_case<T, T, Matrix,             Matrix>()));
   insist((test_copy_cons_case<T, T, Matrix,       const_Matrix>()));
   insist((test_copy_cons_case<T, T, const_Matrix,       Matrix>()));
   insist((test_copy_cons_case<T, T, const_Matrix, const_Matrix>()));
}



template <typename T1,
	  typename T2>
void
test_copy_cons_set()
{
   insist((test_copy_cons_case<T1, T2, Matrix,             Matrix>()));
   insist((test_copy_cons_case<T1, T2, const_Matrix,       Matrix>()));

   // It is only possible to construct a const_Vector from a Vector if:
   //  - They have the same value type (T1 == T2)
   //  - They have the same block type
   // insist((test_copy_cons_case<T1, T2, Matrix,       const_Matrix>()));
   // insist((test_copy_cons_case<T1, T2, const_Matrix, const_Matrix>()));
}



// -------------------------------------------------------------------- //
void test_copy_cons()
{
   test_copy_cons_set<scalar_f>();
   // test_copy_cons_set<scalar_i, scalar_f>(); // not req by spec

   test_copy_cons_set<scalar_i>();
   // test_copy_cons_set<scalar_f, scalar_i>(); // not req by spec

   test_copy_cons_set<cscalar_f>();
   // test_copy_cons_set< scalar_f, cscalar_f>(); // NRBS

   test_copy_cons_set<bool>();
   // test_copy_cons_set<scalar_f, bool>(); // not allowable (but op= is)

   // test_copy_cons_set<index_type, index_type>(); // Matrix<index_type> NRBS
}



/* ==================================================================== *
 * Test #3 -- Test Matrix and const_Matrix assignment
 *
 * Functions:
 *  - test_assign_case() tests assignment operator for
 *                       View2<T1> = View2<T2>
 *  - test_assign_set()  calls test_assign for the permissible combinations
 *                       of Matrix and const_Matrix, leaving types as
 *			 template parameters.
 *  - test_assign() calls test_assign_set for 
 * ==================================================================== */


// -------------------------------------------------------------------- //
template <typename			      T1,
	  typename			      T2,
	  template <typename, typename> class View1,
	  template <typename, typename> class View2>
int
test_assign_case()
{
   int const	 N	= 10;
   Dense<dim, T1>*
		block	= new Dense<2, T1>(Domain<2>(N, N), T1());

   put_nth(block, 1, TestRig<T1>::test_value1());
   put_nth(block, 2, TestRig<T1>::test_value2());

   View1<T1, Dense<dim, T1> >	view1(*block);
   View2<T2, Dense<dim, T2> >	view2(N, N);
   View2<T2, Dense<dim, T2> >	view3(N-1, N-1);

   block->decrement_count(); // allow view1 to take ownership

   view2 = view1;
   view3 = view1(Domain<dim>(N-1, N-1));

   if (get_nth(view2, 1) != (T2)TestRig<T1>::test_value1()) {
      cout << "test_assign_case: (view2) miscompare\n"
	   << "   got       " << get_nth(view2, 1) << endl
	   << "   expecting " << (T1)TestRig<T1>::test_value1() << endl;
      }
   if (get_nth(view3, 1) != (T2)TestRig<T1>::test_value1()) {
      cout << "test_assign_case: (view3) miscompare\n"
	   << "   got       " << get_nth(view3, 1) << endl
	   << "   expecting " << (T1)TestRig<T1>::test_value1() << endl;
      }

   return (get_nth(view2, 1) == (T2)TestRig<T1>::test_value1()) &&
          (get_nth(view2, 2) == (T2)TestRig<T1>::test_value2()) &&
          (get_nth(view3, 1) == (T2)TestRig<T1>::test_value1()) &&
          (get_nth(view3, 2) == (T2)TestRig<T1>::test_value2());
}


template <typename T1, typename T2>
void test_assign_set()
{
   insist((test_assign_case<T1, T2,       Matrix,       Matrix>()));
   insist((test_assign_case<T1, T2, const_Matrix,       Matrix>()));
#ifdef ILLEGAL2_1
   insist((test_assign_case<T1, T2,       Matrix, const_Matrix>()));
#endif
#ifdef ILLEGAL2_2
   insist((test_assign_case<T1, T2, const_Matrix, const_Matrix>()));
#endif
}





// -------------------------------------------------------------------- //
void test_assign()
{
   test_assign_set< scalar_f,  scalar_f>();
   test_assign_set< scalar_i,  scalar_i>();
   test_assign_set<cscalar_f, cscalar_f>();
   test_assign_set<     bool,      bool>();
}



/* ==================================================================== *
 * Test #5 -- test example const_Matrix function sum()
 *
 * Functions:
 *  - sum() - sums the values in a const_Matrix<T>.
 *  - test_sum_case() - tests sum() on a View<T>.  (If View != const_View
 *                      then a conversion is necessary).
 *  - test_sum() - calls test_sum_case() for permissible combinations
 *                 of VSIPL++ types and views (Matrix or const_Matrix).
 * ==================================================================== */



// -------------------------------------------------------------------- //
template <typename T,
	  typename Block>
T
sum(const_Matrix<T, Block> mat)
{
   T total = T();

   for (int r=mat.size(0)-1; r>=0; --r)
      for (int c=mat.size(1)-1; c>=0; --c)
	 total += mat.get(r, c);

   return total;
}



// -------------------------------------------------------------------- //
template <typename			      T,
	  template <typename, typename> class View>
int
test_sum_case()
{
   typedef Dense<dim, T> Block;
		
   int const	N	= 10;
   Block*	block	= new Block(Domain<dim>(N, N), T());

   block->put(1, 2, TestRig<T>::test_value1());
   block->put(2, 0, TestRig<T>::test_value2());

   View<T, Block>	view1(*block);

   block->decrement_count(); // allow view1 to take ownership
		

   T total = sum(view1);

   if (!equal(total, TestRig<T>::test_value1() + TestRig<T>::test_value2())) {
      cout << "test_sum_case: miscompare\n"
	   << "          got       " << total << endl
	   << "          expecting "
	   << (TestRig<T>::test_value1() + TestRig<T>::test_value2())
	   << endl;
      }
      
   return equal(total, TestRig<T>::test_value1() + TestRig<T>::test_value2());
}



// -------------------------------------------------------------------- //
void
test_sum()
{
   insist((test_sum_case<scalar_f,       Matrix>()));
   insist((test_sum_case<scalar_f, const_Matrix>()));

   insist((test_sum_case<scalar_i,       Matrix>()));
   insist((test_sum_case<scalar_i, const_Matrix>()));

   insist((test_sum_case<cscalar_f,       Matrix>()));
   insist((test_sum_case<cscalar_f, const_Matrix>()));
}





/* ==================================================================== *
 * Illegal example #3
 *
 * bad_const_1() directly modifies a const_Matrix.
 * This should not compile
 * ==================================================================== */
#ifdef ILLEGAL3
template <typename T, typename Block>
void bad_const_1(const_Matrix<T, Block> vec)
{
   vec.put(1, T());
}
template void bad_const_1<scalar_f, Dense<1, scalar_f> >(
   const_Matrix<scalar_f, Dense<1, scalar_f> >);
#endif



/* ==================================================================== *
 * Illegal example #4
 *
 * bad_const_2() passes a const_Matrix to modify(), which expects a
 * non-const Matrix.
 *
 * bad_const_2() should not compile
 * ==================================================================== */
template <typename T, typename Block>
void modify(Matrix<T, Block> vec)
{
   vec.put(1, T());
}

#ifdef ILLEGAL4
template <typename T, typename Block>
void bad_const_2(const_Matrix<T, Block> vec)
{
   modify(vec);
}
template void bad_const_2<scalar_f, Dense<1, scalar_f> >(
   const_Matrix<scalar_f, Dense<1, scalar_f> >);
#endif



/* ==================================================================== *
 * Test #6 -- test nonconst subviews
 *
 * Functions:
 *  - test_nonconst_subview_case() - tests that a view's non-const
 *        subview (returned by the "()" operator) has a working put()
 *        member function.
 *  - test_nonconst_subview() - calls test_nonconst_subview_case() for
 *        each VSIPL++ data type.
 * ==================================================================== */

// -------------------------------------------------------------------- //
template <typename			      T,
	  template <typename, typename> class View>
int
test_nonconst_subview_case()
{
   int const N = 10;

   View<T, Dense<dim, T> >	vec(N, N);

   vec(Domain<2>(N-8, N-8)).put(1, 1, TestRig<T>::test_value1());

   return vec.get(1, 1) == TestRig<T>::test_value1();
}


// -------------------------------------------------------------------- //
void
test_nonconst_subview()
{
   insist((test_nonconst_subview_case< scalar_i, Matrix>()));
   insist((test_nonconst_subview_case< scalar_f, Matrix>()));
   insist((test_nonconst_subview_case<cscalar_f, Matrix>()));
   insist((test_nonconst_subview_case<     bool, Matrix>()));
}



/* ==================================================================== *
 * Test #7 -- test const subviews
 *
 * Functions:
 *  - test_const_subview_case() - tests that a view's const
 *        subview (returned by the "get(Domain)" function) can be
 *        assigned and returns proper values from it's get(index_type)
 *        member function.
 *  - test_const_subview() - calls test_nonconst_subview_case() for
 *        each VSIPL++ data type.
 * ==================================================================== */


// -------------------------------------------------------------------- //
template <typename			      T,
	  template <typename, typename> class View>
int
test_const_subview_case()
{
   int const	N	= 10;
   Dense<dim, T>*
		block	= new Dense<dim, T>(Domain<dim>(N, N), T());

   block->put(1, 2, TestRig<T>::test_value1());
   block->put(2, 0, TestRig<T>::test_value2());

   typedef View<T, Dense<dim, T> >			   view_type;
   typedef typename View<T, Dense<dim, T> >::subview_type	   subview_type;
   typedef typename View<T, Dense<dim, T> >::const_subview_type const_subview_type;

   Domain<dim>	dom(N-2, N-2);

   view_type	view1(*block);
   const_subview_type	sub(view1.get(dom));

   block->decrement_count(); // allow view1 to take ownership

   return sub.get(1, 2) == TestRig<T>::test_value1()
       && sub.get(2, 0) == TestRig<T>::test_value2();
}



// -------------------------------------------------------------------- //
void
test_const_subview()
{
   insist((test_const_subview_case< scalar_i,       Matrix>()));
   insist((test_const_subview_case< scalar_i, const_Matrix>()));
   insist((test_const_subview_case< scalar_f,       Matrix>()));
   insist((test_const_subview_case< scalar_f, const_Matrix>()));
   insist((test_const_subview_case<cscalar_f,       Matrix>()));
   insist((test_const_subview_case<cscalar_f, const_Matrix>()));
   insist((test_const_subview_case<     bool,       Matrix>()));
   insist((test_const_subview_case<     bool, const_Matrix>()));
}



// -------------------------------------------------------------------- //
int
main (int argc, char** argv)
{
   vsipl	init(argc, argv);

   test_put_get_Matrix();
   test_put_get_const_Matrix();

   test_copy_cons();
   test_assign();
   test_sum();

   test_nonconst_subview();
   test_const_subview();
}
