/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    refcount.cpp
    @author  Jules Bergmann
    @date    2005-01-24
    @brief   VSIPL++ Library: Unit tests for Ref_count class.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>
#include <vsip/core/refcount.hpp>
#include <vsip_csl/test.hpp>

using namespace std;
using namespace vsip;


/***********************************************************************
  Definitions
***********************************************************************/

/// Test class for Ref_count base class.
class Test_class
  : public impl::Ref_count<Test_class>
{
  // Constructors.
public:
  Test_class() : ptr_(NULL) { ++count_; }
  Test_class(Test_class* ptr) : ptr_(ptr) { ++count_; ptr_->increment_count(); }

  ~Test_class() { --count_; if (ptr_) ptr_->decrement_count(); }

  // Note: class does not define copy-ctor or assignment, however
  //   defaults are disabled since Ref_count makes them private.

  // Global static data (for test and instrumentation).
public:
  static int count_;

  // Member data.
private:
  Test_class* ptr_;
};



/***********************************************************************
  Definitions
***********************************************************************/

int Test_class::count_ = 0;


/// Simple testcase with two reference-counted objects.
void 
test_simple()
{
  test_assert(Test_class::count_ == 0);

  Test_class* tc1 = new Test_class(); // tc1 refcount starts at 1
  Test_class* tc2 = new Test_class(); // tc2 refcount starts at 1

  test_assert(Test_class::count_ == 2);

  tc1->decrement_count();

  test_assert(Test_class::count_ == 1);

  tc2->decrement_count();

  test_assert(Test_class::count_ == 0);
}



/// Utility function to destroy a reference to a Test_class object.

/// Decrements object reference count and then overwrites the object
/// pointer with NULL.

void
deref(Test_class*& tc)
{
  tc->decrement_count();
  tc = NULL;
}



/// Create chain of reference-counted objects and free.

void 
test_chain_1()
{
  test_assert(Test_class::count_ == 0);

  Test_class* tc_0 = new Test_class();
  Test_class* tc_1 = new Test_class(tc_0); deref(tc_0);
  Test_class* tc_2 = new Test_class(tc_1); deref(tc_1);
  Test_class* tc_3 = new Test_class(tc_2); deref(tc_2);
  Test_class* tc_4 = new Test_class(tc_3); deref(tc_3);

  // tc_4 reference count is 1

  test_assert(Test_class::count_ == 5);

  tc_4->decrement_count();

  test_assert(Test_class::count_ == 0);
}



/// Create chain of reference-counted objects that share tail, and
/// free.
void 
test_chain_2(int tc)
{
  test_assert(Test_class::count_ == 0);

  Test_class* tc_0 = new Test_class();
  Test_class* tc_1 = new Test_class(tc_0); deref(tc_0);
  Test_class* tc_2 = new Test_class(tc_1);
  Test_class* tc_3 = new Test_class(tc_2); deref(tc_2);
  Test_class* tc_4 = new Test_class(tc_3); deref(tc_3);

  // tc_4 reference count is 1

  test_assert(Test_class::count_ == 5);

  Test_class* tc_2x = new Test_class(tc_1); deref(tc_1);
  // tc_2x reference count is 1

  test_assert(Test_class::count_ == 6);


  if (tc == 0)
  {
    tc_4->decrement_count();

    test_assert(Test_class::count_ == 3);

    tc_2x->decrement_count();

    test_assert(Test_class::count_ == 0);
  }
  else
  {
    tc_2x->decrement_count();

    test_assert(Test_class::count_ == 5);

    tc_4->decrement_count();

    test_assert(Test_class::count_ == 0);
  }
}



/// Test that Ref_count prevents derived class from using default copy
/// constructor.
///
/// Test passes if compilation fails with -DILLEGAL_1
void 
test_copy_cons()
{
  test_assert(Test_class::count_ == 0);
  {
    Test_class tc1;	// create object on stack
#ifdef ILLEGAL_1
    Test_class tc2(tc1);	// copy it
#endif
  }
  test_assert(Test_class::count_ == 0);
}



/// Test that Ref_count prevents derived class from using default
/// assignment.
///
/// Test passes if compilation fails with -DILLEGAL_1
void 
test_assign()
{
  test_assert(Test_class::count_ == 0);
  {
    Test_class tc1;	// create object on stack
#ifdef ILLEGAL_2
    Test_class tc2 = tc1;	// assign it
#endif
  }
  test_assert(Test_class::count_ == 0);
}



int
main()
{
  test_simple();
  test_chain_1();
  test_chain_2(0);
  test_chain_2(1);

  test_copy_cons();
  test_assign();
}
