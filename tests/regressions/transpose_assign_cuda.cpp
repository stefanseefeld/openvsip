//
// Copyright (c) 2010 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Implicit vs. Explicit Transpose and Assignment test
//
// Added test to strengthen test coverage during CUDA evaluator development

#include <vsip/initfin.hpp>
#include <vsip/math.hpp>
#include <vsip/matrix.hpp>
#include <vsip/selgen.hpp>
#include <test.hpp>

using namespace ovxx;

template <typename T>
void
transpose_assign(length_type const M, length_type const N)
{  
  typedef Dense<2, T, row2_type> row_block_type;
  typedef Dense<2, T, col2_type> col_block_type;
  Matrix<T, row_block_type> ar(M, N, T());
  Matrix<T, col_block_type> ac(M, N, T());
  Matrix<T, row_block_type> br(N, M, T());
  Matrix<T, col_block_type> bc(N, M, T());
  Matrix<T, row_block_type> dr(M, N, T());
  Matrix<T, col_block_type> dc(M, N, T());

  for (index_type i = 0; i < M; ++i)
    ar.row(i) = ramp(T(i*N+1), T(1), ar.size(1));
  for (index_type i = 0; i < M; ++i)
    ac.row(i) = ramp(T(i*N+1), T(1), ac.size(1));


  // explicit transpose
  br = ar.transpose();    // r --> r
  bc = ac.transpose();    // c --> c
  test_assert(equal(br, bc));

  // implicit copy
  bc = ar.transpose();    // r --> c
  br = ac.transpose();    // c --> r
  test_assert(equal(br, bc));

  // explicit copy
  dr = ar;                // r --> r
  dc = ac;                // c --> c
  test_assert(equal(dr, dc));

  // implicit transpose
  dc = ar;                // r --> c
  dr = ac;                // c --> r
  test_assert(equal(dr, dc));
}


template <typename T>
void
transpose_cases()
{
  transpose_assign<T>(5, 8);
  transpose_assign<T>(8, 5);
  transpose_assign<T>(16, 16);
  transpose_assign<T>(64, 64);
  transpose_assign<T>(53, 187);
}


int
main(int argc, char **argv, char **foo)
{
  vsipl init(argc, argv);
 
  transpose_cases<scalar_f>();
  transpose_cases<cscalar_f>();

  return 0;
}
