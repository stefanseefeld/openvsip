/***********************************************************************

  File:   vector-const.cpp
  Author: Jules Bergmann, CodeSourcery, LLC.
  Date:   11/1/2004

  Contents: Test cases related to Vector / const_Vector classes.

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

/* ==================================================================== *
 * Test #1 -- test basic put/get member functions for Vector Views
 *
 * Functions
 *  - test_put_get_case() tests the put() and get() member functions
 *                        for a View<T>.  If View does not defined
 *                        put(), this routine should result in a
 *                        compilation error.
 *  - test_put_get_set() calls test_put_get_case() for each VSIPL++
 *                       data type, leaving View as a template parameter.
 *  - test_put_get_Vector() calls test_put_get_set() for Vector
 *  - test_put_get_const_Vector() calls test_put_get_set() for const_Vector
 *
 * Note: const_Vector test should not compile.  Special test setup
 *       is required to run these tests.
 * ==================================================================== */

// -------------------------------------------------------------------- //
template <typename			      T,
	  template <typename, typename> class View>
int
test_put_get_case()
{
   View<T, Dense<1, T> >	vec(10);

   vec.put(1, TestRig<T>::test_value1());
   vec.put(2, TestRig<T>::test_value2());

   T chk1 = vec.get(1);
   T chk2 = vec.get(2);
   
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
   insist((test_put_get_case<  index_type, View>()));
#endif
}



// -------------------------------------------------------------------- //
void
test_put_get_Vector()
{
   test_put_get_set<Vector>();
}



// -------------------------------------------------------------------- //
void
test_put_get_const_Vector()
{
#if defined(PARTIAL_COMPILE)
   test_put_get_set<const_Vector>();
#endif
}



/* ==================================================================== *
 * Test #3 -- Test Vector and const_Vector copy constructors
 *
 * Three Functions:
 *  - test_copy_cons() tests the copy constructor View1<T1>(View2<T2>).
 *  - wrap_copy_cons() calls test_copy_cons() for the 4 permutations of
 *                     {Vector, const_Vector}^2.  Types are left as
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
   Dense<1, T1>* block	= new Dense<1, T1>(N, T1());

   block->put(1, TestRig<T1>::test_value1());

   View1<T1, Dense<1, T1> >	vec1(*block);
   View2<T2, Dense<1, T2> >	vec2(vec1);

   block->decrement_count(); // allow vec1 to take ownership

   return 
      vec2.get(1) == (T2)TestRig<T1>::test_value1();
}



// -------------------------------------------------------------------- //
// Construct view with same value type
template <typename T>
void
test_copy_cons_set()
{
   insist((test_copy_cons_case<T, T, Vector,             Vector>()));
   insist((test_copy_cons_case<T, T, const_Vector,       Vector>()));
   insist((test_copy_cons_case<T, T, const_Vector, const_Vector>()));
   insist((test_copy_cons_case<T, T, Vector,       const_Vector>()));
}



// Construct view with different value type
template <typename T1,
	  typename T2>
void
test_copy_cons_set()
{
   insist((test_copy_cons_case<T1, T2, Vector,             Vector>()));
   insist((test_copy_cons_case<T1, T2, const_Vector,       Vector>()));

   // It is only possible to construct a const_Vector from a Vector if:
   //  - They have the same value type (T1 == T2)
   //  - They have the same block type
   // insist((test_copy_cons_case<T1, T2, Vector,       const_Vector>()));
   // insist((test_copy_cons_case<T1, T2, const_Vector, const_Vector>()));
}



// -------------------------------------------------------------------- //
void test_copy_cons()
{
   test_copy_cons_set<scalar_f>();
   test_copy_cons_set<scalar_i, scalar_f>();

   test_copy_cons_set<scalar_i>();
   // test_copy_cons_set<scalar_f, scalar_i>(); // warning: assign int to float

   test_copy_cons_set<cscalar_f>();
   // test_copy_cons_set< scalar_f, cscalar_f>(); // not allowable

   test_copy_cons_set<bool>();
   // test_copy_cons_set<scalar_f, bool>(); // not allowable (but op= is)

   test_copy_cons_set<index_type>();
}



/* ==================================================================== *
 * Test #4 -- Test Vector and const_Vector assignment
 *
 * Functions:
 *  - test_assign_case() tests assignment operator for
 *                       View2<T1> = View2<T2>
 *  - test_assign_set()  calls test_assign for the permissible combinations
 *                       of Vector and const_Vector, leaving types as
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
   Dense<1, T1>* block	= new Dense<1, T1>(N, T1());

   block->put(1, TestRig<T1>::test_value2());
   block->put(2, TestRig<T1>::test_value3());

   View1<T1, Dense<1, T1> >	vec1(*block);
   View2<T2, Dense<1, T2> >	vec2(N);
   View2<T2, Dense<1, T2> >	vec3(N-1);

   block->decrement_count(); // allow vec1 to take ownership

   vec2 = vec1;
   vec3 = vec1(Domain<1>(N-1));

   if (vec2.get(1) != (T2)(TestRig<T1>::test_value2())) {
      cout << "test_assign_case: (vec2) miscompare\n"
	   << "   got       " << vec2.get(1) << endl
	   << "   expecting " << (T2)(TestRig<T1>::test_value2()) << endl;
      }
   if (vec3.get(1) != (T2)(TestRig<T1>::test_value2())) {
      cout << "test_assign_case: (vec3) miscompare\n"
	   << "   got       " << vec3.get(1) << endl
	   << "   expecting " << (T2)(TestRig<T1>::test_value2()) << endl;
      }

   return (vec2.get(1) == (T2)TestRig<T1>::test_value2()) &&
          (vec3.get(1) == (T2)TestRig<T1>::test_value2());
}


template <typename T1, typename T2>
void test_assign_set()
{
   insist((test_assign_case<T1, T2,       Vector,       Vector>()));
   insist((test_assign_case<T1, T2, const_Vector,       Vector>()));
#ifdef ILLEGAL2_1
   insist((test_assign_case<T1, T2,       Vector, const_Vector>()));
#endif
#ifdef ILLEGAL2_2
   insist((test_assign_case<T1, T2, const_Vector, const_Vector>()));
#endif
}





// -------------------------------------------------------------------- //
void test_assign()
{
   test_assign_set< scalar_f,  scalar_f>();
   test_assign_set< scalar_i,  scalar_f>();
   test_assign_set< scalar_i,  scalar_i>();
   // test_assign_set< scalar_f,  scalar_i>(); // warning: assign float to int

   test_assign_set<cscalar_f, cscalar_f>();
   // test_assign_set< scalar_f, cscalar_f>(); // not supported

   test_assign_set<     bool,      bool>();
   test_assign_set< scalar_f,      bool>();
   // wrap_assign< scalar_i,      bool>(); // not supported

   test_assign_set<  index_type,   index_type>();
}



/* ==================================================================== *
 * Test #5 -- test example const_Vector function sum()
 *
 * Functions:
 *  - sum() - sums the values in a const_Vector<T>.
 *  - test_sum_case() - tests sum() on a View<T>.  (If View != const_View
 *                      then a conversion is necessary).
 *  - test_sum() - calls test_sum_case() for permissible combinations
 *                 of VSIPL++ types and views (Vector or const_Vecotr).
 * ==================================================================== */



// -------------------------------------------------------------------- //
template <typename T,
	  typename Block>
T
sum(const_Vector<T, Block> vec)
{
   T total = T();

   for (int i=vec.length()-1; i>=0; --i)
      total += vec.get(i);

   return total;
}



// -------------------------------------------------------------------- //
template <typename			      T,
	  template <typename, typename> class View>
int
test_sum_case()
{
   typedef Dense<1, T> Block;
		
   int const	N	= 10;
   Block*	block	= new Block(N);

   for (int i=0; i<N; ++i)
      block->put(i, T());
   block->put(1, TestRig<T>::test_value1());
   block->put(2, TestRig<T>::test_value2());

   View<T, Block>	vec1(*block);

   block->decrement_count(); // allow vec1 to take ownership
		

   T total = sum(vec1);

   if (!equal(total, TestRig<T>::test_value1() + TestRig<T>::test_value2())) {
      cout << "test_sum_case: miscompare\n"
	   << "          got       " << total << endl
	   << "          expecting "
	   << (TestRig<T>::test_value1() + TestRig<T>::test_value2())
	   << endl;
      }
      
   return equal(total, TestRig<T>::test_value1() + TestRig<T>::test_value2());
// return total == (TestRig<T>::test_value1() + TestRig<T>::test_value2());
}



// -------------------------------------------------------------------- //
void
test_sum()
{
   insist((test_sum_case<scalar_f,       Vector>()));
   insist((test_sum_case<scalar_f, const_Vector>()));

   insist((test_sum_case<scalar_i,       Vector>()));
   insist((test_sum_case<scalar_i, const_Vector>()));

   insist((test_sum_case<cscalar_f,       Vector>()));
   insist((test_sum_case<cscalar_f, const_Vector>()));
}





/* ==================================================================== *
 * Illegal example #3
 *
 * bad_const_1() directly modifies a const_Vector.
 * This should not compile
 * ==================================================================== */
#ifdef ILLEGAL3
template <typename T, typename Block>
void bad_const_1(const_Vector<T, Block> vec)
{
   vec.put(1, T());
}
template void bad_const_1<scalar_f, Dense<1, scalar_f> >(
   const_Vector<scalar_f, Dense<1, scalar_f> >);
#endif



/* ==================================================================== *
 * Illegal example #4
 *
 * bad_const_2() passes a const_Vector to modify(), which expects a
 * non-const Vector.
 *
 * bad_const_2() should not compile
 * ==================================================================== */
template <typename T, typename Block>
void modify(Vector<T, Block> vec)
{
   vec.put(1, T());
}

#ifdef ILLEGAL4
template <typename T, typename Block>
void bad_const_2(const_Vector<T, Block> vec)
{
   modify(vec);
}
template void bad_const_2<scalar_f, Dense<1, scalar_f> >(
   const_Vector<scalar_f, Dense<1, scalar_f> >);
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

   View<T, Dense<1, T> >	vec(N);

   vec(Domain<1>(N-8)).put(1, TestRig<T>::test_value1());

   return vec.get(1) == TestRig<T>::test_value1();
}


// -------------------------------------------------------------------- //
void
test_nonconst_subview()
{
   insist((test_nonconst_subview_case< scalar_i, Vector>()));
   insist((test_nonconst_subview_case< scalar_f, Vector>()));
   insist((test_nonconst_subview_case<cscalar_f, Vector>()));
   insist((test_nonconst_subview_case<     bool, Vector>()));
   insist((test_nonconst_subview_case<  index_type, Vector>()));
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
   Dense<1, T>*	block	= new Dense<1, T>(N);

   block->put(1, TestRig<T>::test_value1());
   block->put(2, TestRig<T>::test_value2());

   typedef View<T, Dense<1, T> >			   view_type;
   typedef typename View<T, Dense<1, T> >::subview_type	   subview_type;
   typedef typename View<T, Dense<1, T> >::const_subview_type const_subview_type;

   Domain<1>	dom(N-2);

   view_type	vec1(*block);
   const_subview_type	sub(vec1.get(dom));

   block->decrement_count(); // allow vec1 to take ownership

   return sub.get(1) == TestRig<T>::test_value1() &&
          sub.get(2) == TestRig<T>::test_value2();
}



// -------------------------------------------------------------------- //
void
test_const_subview()
{
   insist((test_const_subview_case< scalar_i,       Vector>()));
   insist((test_const_subview_case< scalar_i, const_Vector>()));
   insist((test_const_subview_case< scalar_f,       Vector>()));
   insist((test_const_subview_case< scalar_f, const_Vector>()));
   insist((test_const_subview_case<cscalar_f,       Vector>()));
   insist((test_const_subview_case<cscalar_f, const_Vector>()));
   insist((test_const_subview_case<     bool,       Vector>()));
   insist((test_const_subview_case<     bool, const_Vector>()));
   insist((test_const_subview_case<  index_type,       Vector>()));
   insist((test_const_subview_case<  index_type, const_Vector>()));
}



// -------------------------------------------------------------------- //
int
main (int argc, char** argv)
{
  vsip::vsipl	init(argc, argv);

   test_put_get_Vector();
   test_put_get_const_Vector();

   test_copy_cons();
   test_assign();
   test_sum();

   test_nonconst_subview();
   test_const_subview();
}
