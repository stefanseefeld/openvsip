/* Copyright (c) 2008 by CodeSourcery, LLC.  All rights reserved. */


/** @file    tests/view_offset.cpp
    @author  Jules Bergmann
    @date    2008-02-22
    @brief   VSIPL++ Library: Regression test for small (less than SIMD
             width), unaligned element-wise vector operations that triggered
	     a bug in the built-in generic SIMD routines.
     
*/

/***********************************************************************
  Included Files
***********************************************************************/

#define VERBOSE 0

#if VERBOSE
#  include <iostream>
#endif

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/domain.hpp>
#include <vsip/random.hpp>

#include <vsip_csl/test.hpp>

using namespace vsip;
using namespace vsip_csl;


/***********************************************************************
  Definitions - Utility Functions
***********************************************************************/

template <typename T>
void
test_vadd(
  length_type len,
  length_type offset1,
  length_type offset2,
  length_type offset3)
{
  Rand<T> gen(0, 0);

  Vector<T> big_A(len + offset1);
  Vector<T> big_B(len + offset2);
  Vector<T> big_Z(len + offset3);

  typename Vector<T>::subview_type A = big_A(Domain<1>(offset1, 1, len));
  typename Vector<T>::subview_type B = big_B(Domain<1>(offset2, 1, len));
  typename Vector<T>::subview_type Z = big_Z(Domain<1>(offset3, 1, len));

  A = gen.randu(len);
  B = gen.randu(len);

  Z = A + B;

  for (index_type i=0; i<len; ++i)
  {
    test_assert(Almost_equal<T>::eq(Z.get(i), A.get(i) + B.get(i)));
  }
}



template <typename T>
void
test_vma_cSC(
  length_type len,
  length_type offset1,
  length_type offset2,
  length_type offset3)
{
  typedef typename vsip::impl::scalar_of<T>::type ST;

  Rand<ST> rgen(0, 0);
  Rand<T>  cgen(0, 0);

  Vector<ST> big_B(len + offset1);
  Vector<T>  big_C(len + offset2);
  Vector<T>  big_Z(len + offset3);

  T a = 2.0;
  typename Vector<ST>::subview_type B = big_B(Domain<1>(offset1, 1, len));
  typename Vector<T>::subview_type  C = big_C(Domain<1>(offset2, 1, len));
  typename Vector<T>::subview_type  Z = big_Z(Domain<1>(offset3, 1, len));

  B = rgen.randu(len);
  C = cgen.randu(len);

  Z = a * B + C;

  for (index_type i=0; i<len; ++i)
  {
    test_assert(Almost_equal<T>::eq(Z.get(i), a * B.get(i) + C.get(i)));
  }
}



template <typename T>
void
test_vmul(
  length_type len,
  length_type offset1,
  length_type offset2,
  length_type offset3)
{
  Rand<T> gen(0, 0);

  Vector<T> big_A(len + offset1);
  Vector<T> big_B(len + offset2);
  Vector<T> big_Z(len + offset3);

  typename Vector<T>::subview_type A = big_A(Domain<1>(offset1, 1, len));
  typename Vector<T>::subview_type B = big_B(Domain<1>(offset2, 1, len));
  typename Vector<T>::subview_type Z = big_Z(Domain<1>(offset3, 1, len));

  A = gen.randu(len);
  B = gen.randu(len);

  Z = A * B;

  for (index_type i=0; i<len; ++i)
  {
#if VERBOSE
    if (!equal(Z.get(i), A.get(i) * B.get(i)))
    {
      std::cout << "Z(" << i << ")        = " << Z(i) << std::endl;
      std::cout << "A(" << i << ") * B(" << i << ") = "
		<< A(i) * B(i) << std::endl;
    }
#endif
    test_assert(Almost_equal<T>::eq(Z.get(i), A.get(i) * B.get(i)));
  }
}



template <typename T>
void
test_vthresh(
  length_type len,
  length_type offset1,
  length_type offset2,
  length_type offset3)
{
  Rand<T> gen(0, 0);

  Vector<T> big_A(len + offset1);
  Vector<T> big_B(len + offset2);
  Vector<T> big_Z(len + offset3);

  typename Vector<T>::subview_type A = big_A(Domain<1>(offset1, 1, len));
  typename Vector<T>::subview_type B = big_B(Domain<1>(offset2, 1, len));
  typename Vector<T>::subview_type Z = big_Z(Domain<1>(offset3, 1, len));
  T                                k = 0.5;

  A = gen.randu(len);
  B = gen.randu(len);

  Z = ite(A > B, A, k);

  for (index_type i=0; i<len; ++i)
  {
    test_assert(Almost_equal<T>::eq(Z.get(i), A.get(i) > B.get(i) ? A.get(i) : k));
  }
}




template <typename T>
void
test_sweep()
{
  for (index_type i=1; i<=128; ++i)
  {
    // 080222: These broke built-in SIMD functions when i < vector size.
    test_vmul<T>(i, 1, 1, 1);
    test_vadd<T>(i, 1, 1, 1);

    // 080222: This would have been broken if it was being dispatched to.
    test_vma_cSC<T>(i, 1, 1, 1);

    // These work fine.
    test_vmul<T>(i, 0, 0, 0);
    test_vmul<T>(i, 1, 0, 0);
    test_vmul<T>(i, 0, 1, 0);
    test_vmul<T>(i, 0, 0, 1);

    test_vadd<T>(i, 0, 0, 0);
    test_vadd<T>(i, 1, 0, 0);
    test_vadd<T>(i, 0, 1, 0);
    test_vadd<T>(i, 0, 0, 1);

    test_vma_cSC<T>(i, 0, 0, 0);
    test_vma_cSC<T>(i, 1, 0, 0);
    test_vma_cSC<T>(i, 0, 1, 0);
    test_vma_cSC<T>(i, 0, 0, 1);
  }
}

template <typename T>
void
test_sweep_real()
{
  for (index_type i=1; i<=128; ++i)
  {
    test_vthresh<T>(i, 1, 1, 1);

    test_vthresh<T>(i, 0, 0, 0);
    test_vthresh<T>(i, 1, 0, 0);
    test_vthresh<T>(i, 0, 1, 0);
    test_vthresh<T>(i, 0, 0, 1);
  }
}




int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  test_sweep<float          >();
  test_sweep<complex<float> >();

  test_sweep_real<float>();
}
