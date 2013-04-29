//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/signal.hpp>
#include <vsip/random.hpp>

#include <vsip_csl/test.hpp>

using namespace std;
using namespace vsip;
using vsip_csl::equal;


/***********************************************************************
  Definitions
***********************************************************************/

// Test C = ite(A >= b, A, T(0)) threshold
//
// This C1 and C2 variations of this are dispatched to SAL vthresx

template <typename T>
void
test_ge_threshold_0(length_type size)
{
  Rand<T> r(0);

  Vector<T> A(size);
  T         b = T(0.5);
  Vector<T> C1(size);
  Vector<T> C2(size);
  Vector<T> C3(size);
  Vector<T> C4(size);

  A = r.randu(size);
  A.put(0, b); // force boundary condition.

  C1 = ite(A >= b, A,    T(0));
  C2 = ite(A <  b, T(0), A);
  C3 = ite(b <= A, A,    T(0));
  C4 = ite(b >  A, T(0), A);

  for (index_type i=0; i<size; ++i)
  {
    test_assert(equal(C1.get(i), A.get(i) >= b ? A.get(i) : T(0)));
    test_assert(equal(C2.get(i), A.get(i) >= b ? A.get(i) : T(0)));
    test_assert(equal(C3.get(i), A.get(i) >= b ? A.get(i) : T(0)));
    test_assert(equal(C4.get(i), A.get(i) >= b ? A.get(i) : T(0)));
  }
}



// Test C = ite(A >= b, A, T(b)) threshold
//
// This C1 and C2 variations of this are dispatched to SAL vthrx

template <typename T>
void
test_ge_threshold_b(length_type size)
{
  Rand<T> r(0);

  Vector<T> A(size);
  T         b = T(0.5);
  Vector<T> C1(size);
  Vector<T> C2(size);
  Vector<T> C3(size);
  Vector<T> C4(size);

  A = r.randu(size);
  A.put(0, b); // force boundary condition.

  C1 = ite(A >= b, A, b);
  C2 = ite(A <  b, b, A);
  C3 = ite(b <= A, A, b);
  C4 = ite(b >  A, b, A);

  for (index_type i=0; i<size; ++i)
  {
    test_assert(equal(C1.get(i), A.get(i) >= b ? A.get(i) : b));
    test_assert(equal(C2.get(i), A.get(i) >= b ? A.get(i) : b));
    test_assert(equal(C3.get(i), A.get(i) >= b ? A.get(i) : b));
    test_assert(equal(C4.get(i), A.get(i) >= b ? A.get(i) : b));
  }
}



// Test C = ite(A OP B, T(1), T(0)) threshold
//
// This variations are dispatched to SAL lv{eq,ne,gt,ge,lt,le}x

#define TEST_LVOP(NAME, OP)						\
template <typename T>							\
void									\
test_l ## NAME (length_type size)					\
{									\
  Rand<T> r(0);								\
									\
  Vector<T> A(size);							\
  Vector<T> B(size);							\
  Vector<T> C(size);							\
									\
  A = r.randu(size);							\
  B = r.randu(size);							\
									\
  A.put(0, B.get(0));							\
									\
  C = ite(A OP B, T(1), T(0));						\
									\
  for (index_type i=0; i<size; ++i)					\
  {									\
    if (!equal(C.get(i), (A.get(i) OP B.get(i) ? T(1) : T(0))))		\
    {									\
      std::cerr << "TEST_LVOP FAILED: i = " << i << std::endl		\
		<< "  C.get(i): " << C.get(i) << std::endl		\
		<< "  A.get(i): " << A.get(i) << std::endl		\
		<< "  B.get(i): " << B.get(i) << std::endl		\
		<< "  expected: "					\
		<< (A.get(i) OP B.get(i) ? T(1) : T(0)) << std::endl;	\
    }									\
    test_assert(equal(C.get(i), (A.get(i) OP B.get(i) ? T(1) : T(0))));	\
  }									\
}

TEST_LVOP(veq, ==)
TEST_LVOP(vne, !=)
TEST_LVOP(vgt, >)
TEST_LVOP(vge, >=)
TEST_LVOP(vlt, >)
TEST_LVOP(vle, >=)



template <typename T>
void
test_type(length_type size)
{
  test_ge_threshold_0<T>(size);
  test_ge_threshold_b<T>(size);

  test_lveq<T>(size);
  test_lvne<T>(size);
  test_lvge<T>(size);
  test_lvgt<T>(size);
  test_lvle<T>(size);
  test_lvlt<T>(size);
}


/***********************************************************************
  Main
***********************************************************************/

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  test_type<float>(16);
  test_type<float>(17);
#if VSIP_IMPL_TEST_DOUBLE
  test_type<double>(19);
#endif
  test_type<int>(21);

  return 0;
}
