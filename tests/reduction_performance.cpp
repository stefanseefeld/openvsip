/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Performance tests for math reductions.

#include <iostream>
#include <vsip/initfin.hpp>
#include <vsip/matrix.hpp>
#include <vsip/math.hpp>

#include <vsip_csl/math.hpp>
#include <vsip_csl/test.hpp>

using namespace std;
using namespace vsip;


#define TIME_REDUCTION(OP)                            \
template <typename                            T,      \
          template <typename, typename> class ViewT,  \
          typename                            BlockT> \
float                                                 \
time_##OP(ViewT<T, BlockT> view)                      \
{                                                     \
  /* warmup (to remove cache-miss effects) */         \
  OP(view);                                           \
  vsip::impl::profile::Timer t1;                      \
  t1.start();                                         \
    OP(view);                                         \
  t1.stop(); printf("%f\n", t1.delta());              \
  return t1.delta();                                  \
}

TIME_REDUCTION(sumval);
TIME_REDUCTION(sumsqval);
TIME_REDUCTION(meanval);
TIME_REDUCTION(meansqval);

#undef test_assert
#define test_assert(x) (x);

#define TEST_REDUCTION(OP, VIEW1, VIEW2, VIEW3)         \
{                                                       \
  float t1 = time_##OP(VIEW1);                          \
  float t2 = time_##OP(VIEW2);                          \
  float t3 = time_##OP(VIEW3);                          \
  float tol = t1 * .1;                                  \
  test_assert(((t1 - tol) < t2) && (t2 < (t1 + tol)) && \
              ((t1 - tol) < t3) && (t3 < (t1 + tol)));  \
}



// The purpose of this test is to measure the relative performance 
// differences between reductions involving differently-dimensioned
// views, which are supposed to be nearly equivalent in terms of
// performance as 2- and 3-D dense views are re-cast into 1-D views
// and then reduced.
int
main(int argc, char **argv)
{
  vsipl init(argc, argv);

  typedef cscalar_f T;

  length_type const L = 16;
  length_type const M = 128;
  length_type const N = 1024;
  Vector<T> vector(L*M*N, T(2.f));
  Matrix<T> matrix(L*M, N, T(2.f));
  Tensor<T> tensor(L, M, N, T(2.f));

  TEST_REDUCTION(sumval, vector, matrix, tensor);
  TEST_REDUCTION(sumsqval, vector, matrix, tensor);
  TEST_REDUCTION(meanval, vector, matrix, tensor);
  TEST_REDUCTION(meansqval, vector, matrix, tensor);
}
