//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Randy Judd's matrix-matric product variations.

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/signal.hpp>
#include <ovxx/output.hpp>
#include "benchmark.hpp"

#define VERBOSE 1

using namespace ovxx;

/***********************************************************************
  Matrix-matrix product variants
***********************************************************************/

// Convenience type to disambiguate between prod() overloads.
template <int I> struct Int_type {};



// direct matrix product using VSIPL function
template <typename T,
	  typename Block1,
	  typename Block2,
	  typename Block3>
void
prod(
  Int_type<1>,
  Matrix<T, Block1> A,
  Matrix<T, Block2> B,
  Matrix<T, Block3> C)
{
  C = vsip::prod(A,B);
}

// prod_2: Alg 1.1.8 Outer Product from G&VL
template <typename T,
	  typename Block1,
	  typename Block2,
	  typename Block3>
void
prod(
  Int_type<2>,
  Matrix<T, Block1> A,
  Matrix<T, Block2> B,
  Matrix<T, Block3> C)
{
  index_type p = A.row(0).size();

  C = outer<T>(1.0, A.col(0), B.row(0));
  for(index_type k=1; k < p; k++)
    C += outer<T>(1.0, A.col(k),B.row(k));
}
 
// prod_3: Alg 1.1.7 Saxpy Version G&VL
template <typename T,
	  typename Block1,
	  typename Block2,
	  typename Block3>
void
prod(
  Int_type<3>,
  Matrix<T, Block1> A,
  Matrix<T, Block2> B,
  Matrix<T, Block3> C)
{
  C = T();

  index_type n = B.row(0).size();
  index_type p = A.row(0).size();

  for(index_type j=0; j < n; j++)
    for(index_type k=0; k < p; k++)
      C.col(j) = A.col(k) * B.get(k,j) + C.col(j);
}

// prod_3c
template <typename T,
	  typename Block1,
	  typename Block2,
	  typename Block3>
void
prod(
  Int_type<4>,
  Matrix<T, Block1> A,
  Matrix<T, Block2> B,
  Matrix<T, Block3> C)
{
  C = T();

  // index_type n = B.row(0).size();
  // index_type p = A.row(0).size();
  index_type const n = B.size(1);
  index_type const p = A.size(1);
  for(index_type j=0; j < n; j++)
    for(index_type k=0; k < p; k++)
      C.col(j) = A.col(k) * B.get(k,j) + C.col(j);
}

// prod_3sv: Alg 1.1.7 Saxpy Version G&VL using subview
template <typename T,
	  typename Block1,
	  typename Block2,
	  typename Block3>
void
prod(
  Int_type<5>,
  Matrix<T, Block1> A,
  Matrix<T, Block2> B,
  Matrix<T, Block3> C)
{
  C = T();

  index_type n = B.row(0).size();
  index_type p = A.row(0).size();

  for(index_type j=0; j < n; j++)
  {
    typename Matrix<T, Block3>::col_type c_col(C.col(j));
    typename Matrix<T, Block2>::col_type b_col(B.col(j));
    for(index_type k=0; k < p; k++)
      c_col = A.col(k) * b_col.get(k) + c_col;
  }
}

// prod_4: Alg 1.1.5 ijk variant G&VL
template <typename T,
	  typename Block1,
	  typename Block2,
	  typename Block3>
void
prod(
  Int_type<6>,
  Matrix<T, Block1> A,
  Matrix<T, Block2> B,
  Matrix<T, Block3> C)
{
  C = T();

  for(index_type i=0; i<A.size(0); i++)
  {
    for(index_type j=0; j<B.size(1); j++)
    {
      for(index_type k=0; k<A.size(1); k++)
      {
	C.put(i,j,A.get(i,k)*B.get(k,j) + C.get(i,j));
      }
    }
  }
}

// prod_4t: Alg 1.1.5 ijk variant G&VL with tmp storage
template <typename T,
	  typename Block1,
	  typename Block2,
	  typename Block3>
void
prod(
  Int_type<7>,
  Matrix<T, Block1> A,
  Matrix<T, Block2> B,
  Matrix<T, Block3> C)
{
  C = T();

  for(index_type i = 0; i<A.size(0); i++){
    for(index_type j=0; j<B.size(1); j++){
      T tmp = T();
      for(index_type k=0; k<A.size(1); k++){
	tmp += A.get(i,k)*B.get(k,j);
      }
      C.put(i,j,tmp);
    }
  }
}
 
// prod_4trc:
template <typename T,
	  typename Block1,
	  typename Block2,
	  typename Block3>
void
prod(
  Int_type<8>,
  Matrix<T, Block1> A,
  Matrix<T, Block2> B,
  Matrix<T, Block3> C)
{
  C = T();

  for(index_type i = 0; i<A.size(0); i++){
    for(index_type j=0; j<B.size(1); j++){
      T tmp=T();
      for(index_type k=0; k<A.size(1); k++){
	tmp += A.row(i).get(k)*B.col(j).get(k);
      }
      C.put(i,j,tmp);
    }
  }
}
 
// prod_4tsv:
template <typename T,
	  typename Block1,
	  typename Block2,
	  typename Block3>
void
prod(
  Int_type<9>,
  Matrix<T, Block1> A,
  Matrix<T, Block2> B,
  Matrix<T, Block3> C)
{
  C = T();

  for(index_type i = 0; i<A.size(0); i++){
    typename Matrix<T, Block1>::row_type a_row(A.row(i));

    for(index_type j=0; j<B.size(1); j++){
      typename Matrix<T, Block2>::col_type b_col(B.col(j));
      T tmp=T();
      for(index_type k=0; k<A.size(1); k++){
	tmp += a_row.get(k)*b_col.get(k);
      }
      C.put(i,j,tmp);
    }
  }
}
 
// prod_4dot:
template <typename T,
	  typename Block1,
	  typename Block2,
	  typename Block3>
void
prod(
  Int_type<10>,
  Matrix<T, Block1> A,
  Matrix<T, Block2> B,
  Matrix<T, Block3> C)
{
  for(index_type i = 0; i<A.size(0); i++){
    typename Matrix<T, Block1>::row_type a_row(A.row(i));
    for(index_type j=0; j<B.size(1); j++){
      C.put(i, j, vsip::dot(a_row, B.col(j)));
    }
  }
}

// Matrix-matrix product benchmark class.

template <int ImplI, typename T>
struct t_prod1 : Benchmark_base
{
  static length_type const Dec = 1;

  char const* what() { return "t_prod1"; }
  float ops_per_point(length_type M)
  {
    length_type N = M;
    length_type P = M;

    float ops = /*M * */ P * N * 
      (ovxx::ops_count::traits<T>::mul + ovxx::ops_count::traits<T>::add);

    return ops;
  }

  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 0; }
  int mem_per_point(length_type M) { return 3*M*M*sizeof(T); }

  void operator()(length_type M, length_type loop, float& time)
  {
    typedef typename scalar_of<T>::type scalar_type;

    length_type N = M;
    length_type P = M;

    Matrix<T>   A (M, N, T(1));
    Matrix<T>   B (N, P, T(1));
    Matrix<T>   Z (M, P, T(1));
    Matrix<T>   chk(M, P);
    Matrix<scalar_type> gauge(M, P);

    timer t1;
    for (index_type l=0; l<loop; ++l)
      prod(Int_type<ImplI>(), A, B, Z);
    time = t1.elapsed();
  }

  t_prod1() {}
};



void
defaults(Loop1P& loop)
{
  loop.loop_start_ = 5000;
  loop.start_ = 4;
  loop.stop_  = 8;
}

int
benchmark(Loop1P& loop, int what)
{
  switch (what)
  {
  case  1: loop(t_prod1< 1, float>()); break;
  case  2: loop(t_prod1< 2, float>()); break;
  case  3: loop(t_prod1< 3, float>()); break;
  case  4: loop(t_prod1< 4, float>()); break;
  case  5: loop(t_prod1< 5, float>()); break;
  case  6: loop(t_prod1< 6, float>()); break;
  case  7: loop(t_prod1< 7, float>()); break;
  case  8: loop(t_prod1< 8, float>()); break;
  case  9: loop(t_prod1< 9, float>()); break;
  case 10: loop(t_prod1<10, float>()); break;

  case 11: loop(t_prod1< 1, complex<float> >()); break;
  case 12: loop(t_prod1< 2, complex<float> >()); break;
  case 13: loop(t_prod1< 3, complex<float> >()); break;
  case 14: loop(t_prod1< 4, complex<float> >()); break;
  case 15: loop(t_prod1< 5, complex<float> >()); break;
  case 16: loop(t_prod1< 6, complex<float> >()); break;
  case 17: loop(t_prod1< 7, complex<float> >()); break;
  case 18: loop(t_prod1< 8, complex<float> >()); break;
  case 19: loop(t_prod1< 9, complex<float> >()); break;
  case 20: loop(t_prod1<10, complex<float> >()); break;

  case  0:
    std::cout
      << "prod_var -- Randy Judd's matrix-matrix product variations\n"
      << "   -1 -- direct matrix product using VSIPL function\n"
      << "   -2 -- prod_2:    Alg 1.1.8 Outer Product from G&VL\n"
      << "   -3 -- prod_3:    Alg 1.1.7 Saxpy Version G&VL\n"
      << "   -4 -- prod_3c:\n"
      << "   -5 -- prod_3sv:  Alg 1.1.7 Saxpy Version G&VL using subview\n"
      << "   -6 -- prod_4:    Alg 1.1.5 ijk variant G&VL\n"
      << "   -7 -- prod_4t:   Alg 1.1.5 ijk variant G&VL with tmp storage\n"
      << "   -8 -- prod_4trc:\n"
      << "   -9 -- prod_4tsv:\n"
      << "  -10 -- prod_4dot:\n"
      << "  Options  -1 through -10 use float matrices.\n"
      << "  Options -11 through -20 are the same as the above, but \n"
      << "                          use complex<float> matrices.\n"
      ;
  default: return 0;
  }
  return 1;
}
