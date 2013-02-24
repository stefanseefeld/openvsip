/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    benchmarks/prod_var.cpp
    @author  Jules Bergmann
    @date    2005-11-07
    @brief   VSIPL++ Library: Randy Judd's matrix-matric product variations.

*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/signal.hpp>

#include <vsip/core/profile.hpp>

#include <vsip_csl/test.hpp>
#include <vsip_csl/test-precision.hpp>
#include <vsip_csl/output.hpp>
#include <vsip_csl/ref_matvec.hpp>

#include "loop.hpp"

#define VERBOSE 1

using namespace vsip;
using namespace vsip_csl;
using namespace ref;


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



/***********************************************************************
  Comparison routine and benchmark driver.
***********************************************************************/

template <typename T0,
	  typename T1,
          typename T2,
          typename Block0,
          typename Block1,
          typename Block2>
void
check_prod(
  Matrix<T0, Block0> test,
  Matrix<T1, Block1> chk,
  Matrix<T2, Block2> gauge)
{
  typedef typename Promotion<T0, T1>::type return_type;
  typedef typename vsip::impl::Scalar_of<return_type>::type scalar_type;

  Index<2> idx;
  scalar_type err = maxval(((mag(chk - test)
			     / Precision_traits<scalar_type>::eps)
			    / gauge),
			   idx);

#if VERBOSE
  if (err >= 10.0)
  {
    std::cout << "test  =\n" << test;
    std::cout << "chk   =\n" << chk;
    std::cout << "gauge =\n" << gauge;
    std::cout << "err = " << err << std::endl;
  }
#endif

  test_assert(err < 10.0);
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
      (vsip::impl::Ops_info<T>::mul + vsip::impl::Ops_info<T>::add);

    return ops;
  }

  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 0; }
  int mem_per_point(length_type M) { return 3*M*M*sizeof(T); }

  void operator()(length_type M, length_type loop, float& time)
  {
    typedef typename vsip::impl::Scalar_of<T>::type scalar_type;

    length_type N = M;
    length_type P = M;

    Matrix<T>   A (M, N, T(1));
    Matrix<T>   B (N, P, T(1));
    Matrix<T>   Z (M, P, T(1));
    Matrix<T>   chk(M, P);
    Matrix<scalar_type> gauge(M, P);

    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      prod(Int_type<ImplI>(), A, B, Z);
    t1.stop();

    chk   = ref::prod(A, B);
    gauge = ref::prod(mag(A), mag(B));

    for (index_type i=0; i<gauge.size(0); ++i)
      for (index_type j=0; j<gauge.size(1); ++j)
	if (!(gauge(i, j) > scalar_type()))
	  gauge(i, j) = scalar_type(1);

    check_prod(Z, chk, gauge );
    
    time = t1.delta();
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


template <> float  Precision_traits<float>::eps = 0.0;
template <> double Precision_traits<double>::eps = 0.0;

int
test(Loop1P& loop, int what)
{
  Precision_traits<float>::compute_eps();
  Precision_traits<double>::compute_eps();

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
