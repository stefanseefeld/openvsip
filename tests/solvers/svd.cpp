//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/tensor.hpp>
#include <vsip/solvers.hpp>
#include <ovxx/dispatcher/diagnostics.hpp>
#include <test.hpp>
#include "common.hpp"

#define VERBOSE  0
#if !defined(DO_FULL)
#  if VSIP_IMPL_TEST_LEVEL >= 2
#    define DO_FULL  1
#  else
#    define DO_FULL  0
#  endif
#endif

using namespace ovxx;
using vsip::impl::trans_or_herm;

template <typename T,
	  typename Block>
typename scalar_of<T>::type norm_1(const_Vector<T, Block> v)
{
  return sumval(mag(v));
}



/// Matrix norm-1

template <typename T,
	  typename Block>
typename scalar_of<T>::type
norm_1(const_Matrix<T, Block> m)
{
  typedef typename scalar_of<T>::type scalar_type;
  scalar_type norm = sumval(mag(m.col(0)));

  for (index_type j=1; j<m.size(1); ++j)
  {
    norm = std::max(norm, sumval(mag(m.col(j))));
  }

  return norm;
}



/// Matrix norm-infinity

template <typename T,
	  typename Block>
typename scalar_of<T>::type
norm_inf(const_Matrix<T, Block> m)
{
  return norm_1(m.transpose());
}

template <typename T,
	  typename Block1,
	  typename Block2>
void
compare_view(vsip::const_Vector<T, Block1>           a,
	     vsip::const_Vector<T, Block2>           b,
	     typename scalar_of<T>::type thresh
  )
{
  typedef typename scalar_of<T>::type scalar_type;

  vsip::Index<1> idx;
  scalar_type err = vsip::maxval((mag(a - b) / test::precision<scalar_type>::eps), idx);

  if (err > thresh)
  {
    for (vsip::index_type r=0; r<a.size(0); ++r)
	test_assert(equal(a.get(r), b.get(r)));
  }
}



template <typename T,
	  typename Block1,
	  typename Block2>
void
compare_view(vsip::const_Matrix<T, Block1>           a,
	     vsip::const_Matrix<T, Block2>           b,
	     typename scalar_of<T>::type thresh
  )
{
  typedef typename scalar_of<T>::type scalar_type;

  vsip::Index<2> idx;
  scalar_type err = vsip::maxval((mag(a - b) / test::precision<scalar_type>::eps), idx);

  if (err > thresh)
  {
    std::cout << "a = \n" << a;
    std::cout << "b = \n" << b;
    for (vsip::index_type r=0; r<a.size(0); ++r)
      for (vsip::index_type c=0; c<a.size(1); ++c)
	test_assert(equal(a.get(r, c), b.get(r, c)));
  }
}


/***********************************************************************
  svd function tests
***********************************************************************/

template <typename              T,
	  typename              Block0,
	  typename              Block1,
	  typename              Block2,
	  typename              Block3>
void
apply_svd(
  svd<T, by_reference>&    sv,
  Matrix<T, Block0>        a,
  Vector<typename scalar_of<T>::type, Block1> sv_s,
  Matrix<T, Block2>        sv_u,
  Matrix<T, Block3>        sv_v)
{
  length_type m = sv.rows();
  length_type n = sv.columns();
  length_type p = std::min(m, n);
  length_type u_columns = sv.ustorage() == svd_uvfull ? m : p;
  length_type v_rows    = sv.vstorage() == svd_uvfull ? n : p;

  sv.decompose(a, sv_s);
  if (sv.ustorage() != svd_uvnos)
    sv.u(0, u_columns-1, sv_u);
  if (sv.vstorage() != svd_uvnos)
    sv.v(0, v_rows-1,    sv_v);
}



template <typename              T,
	  typename              Block0,
	  typename              Block1,
	  typename              Block2,
	  typename              Block3>
void
apply_svd(svd<T, by_value>&        sv,
	  Matrix<T, Block0>        a,
	  Vector<typename scalar_of<T>::type, Block1> sv_s,
	  Matrix<T, Block2>        sv_u,
	  Matrix<T, Block3>        sv_v)
{
  length_type m = sv.rows();
  length_type n = sv.columns();
  length_type p = std::min(m, n);
  length_type u_columns = sv.ustorage() == svd_uvfull ? m : p;
  length_type v_rows    = sv.vstorage() == svd_uvfull ? n : p;

  sv_s = sv.decompose(a);
  if (sv.ustorage() != svd_uvnos)
    sv_u = sv.u(0, u_columns-1);
  if (sv.vstorage() != svd_uvnos)
    sv_v = sv.v(0, v_rows-1);
}



template <mat_op_type       tr,
	  product_side_type ps,
	  typename          T,
	  typename          Block0,
	  typename          Block1>
void
apply_svd_produ(svd<T, by_reference>&    sv,
		const_Matrix<T, Block0>  b,
		Matrix<T, Block1>        produ)
{
  sv.template produ<tr, ps>(b, produ);
}



template <mat_op_type       tr,
	  product_side_type ps,
	  typename          T,
	  typename          Block0,
	  typename          Block1>
void
apply_svd_produ(svd<T, by_value>&        sv,
		const_Matrix<T, Block0>  b,
		Matrix<T, Block1>        produ)
{
  produ = sv.template produ<tr, ps>(b);
}



template <mat_op_type       tr,
	  product_side_type ps,
	  typename          T,
	  typename          Block0,
	  typename          Block1>
void
apply_svd_prodv(svd<T, by_reference>&    sv,
		const_Matrix<T, Block0>  b,
		Matrix<T, Block1>        prodv)
{
  sv.template prodv<tr, ps>(b, prodv);
}



template <mat_op_type       tr,
	  product_side_type ps,
	  typename          T,
	  typename          Block0,
	  typename          Block1>
void
apply_svd_prodv(svd<T, by_value>&        sv,
		const_Matrix<T, Block0>  b,
		Matrix<T, Block1>        prodv)
{
  prodv = sv.template prodv<tr, ps>(b);
}



template <return_mechanism_type RtM,
	  typename              T,
	  typename              Block>
void
test_svd(storage_type     ustorage,
	 storage_type     vstorage,
	 Matrix<T, Block> a,
	 length_type      loop)
{
  typedef typename scalar_of<T>::type scalar_type;

  length_type m = a.size(0);
  length_type n = a.size(1);

  length_type p = std::min(m, n);
  test_assert(m > 0 && n > 0);

  length_type u_cols = ustorage == svd_uvfull ? m : p;
  length_type v_cols = vstorage == svd_uvfull ? n : p;

  Vector<scalar_type> sv_s(p);		// singular values
  Matrix<T>     sv_u(m, u_cols);	// U matrix
  Matrix<T>     sv_v(n, v_cols);	// V matrix

  svd<T, RtM> sv(m, n, ustorage, vstorage);

  test_assert(sv.rows()     == m);
  test_assert(sv.columns()  == n);
  test_assert(sv.ustorage() == ustorage);
  test_assert(sv.vstorage() == vstorage);

  for (index_type i=0; i<loop; ++i)
  {
    apply_svd(sv, a, sv_s, sv_u, sv_v);

    // Check that sv_sv is non-increasing.
    for (index_type i=0; i<p-1; ++i)
      test_assert(sv_s(i) >= sv_s(i+1));

    // Check that product of u, s, v equals a.
    if (ustorage != svd_uvnos && vstorage != svd_uvnos)
    {
      Matrix<T> sv_sm(m, n, T());
      sv_sm.diag() = sv_s;

      Matrix<T> chk(m, n);
      if (ustorage == svd_uvfull && vstorage == svd_uvfull)
      {
	chk = prod(prod(sv_u, sv_sm), trans_or_herm(sv_v));
      }
      else
      {
	chk = prod(prod(sv_u(Domain<2>(m, p)), sv_sm(Domain<2>(p, p))),
		   trans_or_herm(sv_v(Domain<2>(n, p))));
      }

      // When using LAPACK, the error E when computing the
      // bi-diagonal decomposition Q, B, P
      //
      //   A + E = Q B herm(P)
      //
      // Is
      //
      //   norm-2(E) = c(n) eps norm-2(A)
      //
      // Where
      //
      //   c(n) is a "modestly increasing function of n", and
      //   eps is the machine precision.   
      //
      // Computing norm-2(A) is expensive, so we use the relationship:
      //
      //   norm-2(A) <= sqrt(norm-1(A) norm-inf(A))

      Index<2> idx;
      scalar_type err = maxval((mag(chk - a)
				/ test::precision<scalar_type>::eps),
			     idx);
      scalar_type errx = maxval(mag(chk - a), idx);
      scalar_type norm_est = std::sqrt(norm_1(a) * norm_inf(a));
      
      err  = err / norm_est;
      errx = errx / norm_est;

#if VERBOSE
      std::cout << "a    = " << '\n' << a  << '\n'
		<< "sv_s = " << '\n' << sv_s << '\n'
		<< "sv_u = " << '\n' << sv_u << '\n'
		<< "sv_v = " << '\n' << sv_v << '\n'
		<< "chk  = " << '\n' << chk << '\n'
		<< "err = " << err << "   " << "norm = " << norm_est << '\n'
		<< "eps = " << test::precision<scalar_type>::eps << '\n'
		<< "p:" << p << "   " << "err = " << err   << "   "
		<< "errx = " << errx << std::endl;
#endif

      if (err > 5.0)
      {
	for (index_type r=0; r<m; ++r)
	  for (index_type c=0; c<n; ++c)
	      test_assert(equal(chk(r, c), a(r, c)));
      }
    }

    const length_type chk_single_uv = 2;

    if (ustorage != svd_uvnos)
    {
      length_type u_cols = (ustorage == svd_uvfull) ? m : p;

      Matrix<T> in_m (m,      m,    T());
      Matrix<T> in_p (u_cols, u_cols, T());

      Matrix<T> pu_nl(m,      u_cols, T());
      Matrix<T> pu_tl(u_cols, m,      T());
      Matrix<T> pu_nr(m,      u_cols, T());
      Matrix<T> pu_tr(u_cols, m,      T());

      Vector<T> zero_m(m, T());
      Vector<T> zero_p(u_cols, T());

      index_type pos = 0;
      for (index_type i=0; i<chk_single_uv; ++i, pos = (17*pos+5)%u_cols)
      {
	in_m(pos, pos) = T(1);
	in_p(pos, pos) = T(1);
      
	apply_svd_produ<mat_ntrans,            mat_lside>(sv, in_p, pu_nl);
	apply_svd_produ<Test_traits<T>::trans, mat_lside>(sv, in_m, pu_tl);
	apply_svd_produ<mat_ntrans,            mat_rside>(sv, in_m, pu_nr);
	apply_svd_produ<Test_traits<T>::trans, mat_rside>(sv, in_p, pu_tr);

	compare_view(pu_nl.col(pos), sv_u.col(pos), 5.0);
	compare_view(pu_tl.col(pos), trans_or_herm(sv_u).col(pos), 5.0);
	compare_view(pu_nr.row(pos), sv_u.row(pos), 5.0);
	compare_view(pu_tr.row(pos), trans_or_herm(sv_u).row(pos), 5.0);

	for (index_type j=0; j<u_cols; ++j)
	{
	  if (j != pos)
	  {
	    compare_view(pu_nl.col(j), zero_m, 5.0);
	    compare_view(pu_tl.col(j), zero_p, 5.0);
	    compare_view(pu_nr.row(j), zero_p, 5.0);
	    compare_view(pu_tr.row(j), zero_m, 5.0);
	  }
	}
	in_m(pos, pos) = T();
	in_p(pos, pos) = T();
      }
    }

    if (vstorage != svd_uvnos)
    {
      length_type v_cols = (vstorage == svd_uvfull) ? n : p;

      Matrix<T> in_p (v_cols, v_cols, T());
      Matrix<T> in_n (n,      n,      T());

      Matrix<T> pv_nl(n,      v_cols, T());
      Matrix<T> pv_tl(v_cols, n,      T());
      Matrix<T> pv_nr(n,      v_cols, T());
      Matrix<T> pv_tr(v_cols, n,      T());
      
      Vector<T> zero_n(n,      T());
      Vector<T> zero_p(v_cols, T());
      
      index_type pos = 0;
      for (index_type i=0; i<chk_single_uv; ++i, pos = (17*pos+5)%v_cols)
      {
	in_p(pos, pos) = T(1);
	in_n(pos, pos) = T(1);
      
	apply_svd_prodv<mat_ntrans,            mat_lside>(sv, in_p, pv_nl);
	apply_svd_prodv<Test_traits<T>::trans, mat_lside>(sv, in_n, pv_tl);
	apply_svd_prodv<mat_ntrans,            mat_rside>(sv, in_n, pv_nr);
	apply_svd_prodv<Test_traits<T>::trans, mat_rside>(sv, in_p, pv_tr);

	compare_view(pv_nl.col(pos), sv_v.col(pos), 5.0);
	compare_view(pv_tl.col(pos), trans_or_herm(sv_v).col(pos), 5.0);
	compare_view(pv_nr.row(pos), sv_v.row(pos), 5.0);
	compare_view(pv_tr.row(pos), trans_or_herm(sv_v).row(pos), 5.0);
	
	for (index_type j=0; j<v_cols; ++j)
	{
	  if (j != pos)
	  {
	    compare_view(pv_nl.col(j), zero_n, 5.0);
	    compare_view(pv_tl.col(j), zero_p, 5.0);
	    compare_view(pv_nr.row(j), zero_p, 5.0);
	    compare_view(pv_tr.row(j), zero_n, 5.0);
	  }
	}
	in_p(pos, pos) = T();
	in_n(pos, pos) = T();
      }
    }

    // Solver a different problem next iteration.
    a(0, 0) = a(0, 0) + T(1);
  }
}



// Description:

template <return_mechanism_type RtM,
	  typename              T>
void
test_svd_ident(storage_type ustorage,
	       storage_type vstorage,
	       length_type  m,
	       length_type  n,
	       length_type  loop)
{
  typedef typename scalar_of<T>::type scalar_type;

  length_type p = std::min(m, n);
  test_assert(m > 0 && n > 0);

  Matrix<T>     a(m, n);
  Vector<scalar_type> sv_s(p);	// singular values
  Matrix<T>     sv_u(m, m);	// U matrix
  Matrix<T>     sv_v(n, n);	// V matrix

  // Setup a.
  a        = T();
  a.diag() = T(1);
  if (p > 0) a(0, 0)  = Test_traits<T>::value1();
  if (p > 2) a(2, 2)  = Test_traits<T>::value2();
  if (p > 3) a(3, 3)  = Test_traits<T>::value3();

  test_svd<RtM>(ustorage, vstorage, a, loop);
}



template <return_mechanism_type RtM,
	  typename              T>
void
test_svd_rand(storage_type ustorage,
	      storage_type vstorage,
	      length_type  m,
	      length_type  n,
	      length_type  loop)
{
  typedef typename scalar_of<T>::type scalar_type;

  length_type p = std::min(m, n);
  test_assert(m > 0 && n > 0);

  Matrix<T>     a(m, n);
  Vector<scalar_type> sv_s(p);	// singular values
  Matrix<T>     sv_u(m, m);	// U matrix
  Matrix<T>     sv_v(n, n);	// U matrix

  // Setup a.
  test::randm(a);

  test_svd<RtM>(ustorage, vstorage, a, loop);
}



template <return_mechanism_type RtM,
	  typename              T>
void
svd_cases(storage_type ustorage,
	  storage_type vstorage,
	  length_type  loop,
	  bool         m_lt_n,
	  true_type)
{
  test_svd_ident<RtM, T>(ustorage, vstorage, 1, 1, loop);
  test_svd_ident<RtM, T>(ustorage, vstorage, 9, 1, loop);

  test_svd_ident<RtM, T>(ustorage, vstorage, 5,   5, loop);
  test_svd_ident<RtM, T>(ustorage, vstorage, 16,  5, loop);

  if (m_lt_n)
  {
    test_svd_ident<RtM, T>(ustorage, vstorage, 1, 7, loop);
    test_svd_ident<RtM, T>(ustorage, vstorage, 3,  20, loop);
  }

  test_svd_rand<RtM, T>(ustorage, vstorage, 5, 5, loop);
  test_svd_rand<RtM, T>(ustorage, vstorage, 5, 3, loop);

  if (m_lt_n)
  {
    test_svd_rand<RtM, T>(ustorage, vstorage, 3, 5, loop);
  }

#if DO_FULL
  test_svd_rand<RtM, T>(ustorage, vstorage, 17, 5, loop);
  test_svd_rand<RtM, T>(ustorage, vstorage, 25, 27, loop);
  test_svd_rand<RtM, T>(ustorage, vstorage, 32, 32, loop);
  test_svd_rand<RtM, T>(ustorage, vstorage, 32, 10, loop);

  if (m_lt_n)
  {
    test_svd_rand<RtM, T>(ustorage, vstorage, 5, 17, loop);
    test_svd_rand<RtM, T>(ustorage, vstorage, 17, 19, loop);
    test_svd_rand<RtM, T>(ustorage, vstorage, 8, 32, loop);
  }
#endif
}



template <return_mechanism_type RtM,
	  typename              T>
void
svd_cases(storage_type, storage_type, length_type, bool, false_type)
{
}



// Front-end function for svd_cases.

template <return_mechanism_type RtM,
	  typename              T>
void
svd_cases(storage_type ustorage,
	  storage_type vstorage,
	  length_type  loop)
{
  using namespace dispatcher;

  // Test m less-than n cases. (SAL backend doesn't support this.)
  bool m_lt_n = true;

  svd_cases<RtM, T>(ustorage, vstorage, loop, m_lt_n,
		    integral_constant<bool,
		    is_operation_supported<op::svd, T>::value>());
}



template <return_mechanism_type RtM>
void
svd_types(storage_type ustorage,
	  storage_type vstorage,
	  length_type  loop)
{
  svd_cases<RtM, float>(ustorage, vstorage, loop);
  svd_cases<RtM, double>(ustorage, vstorage, loop);
  svd_cases<RtM, complex<float> >(ustorage, vstorage, loop);
  svd_cases<RtM, complex<double> >(ustorage, vstorage, loop);
}



template <return_mechanism_type RtM>
void
svd_storage(length_type  loop)
{
  svd_types<RtM>(svd_uvfull, svd_uvfull, loop);
  svd_types<RtM>(svd_uvpart, svd_uvfull, loop);
  svd_types<RtM>(svd_uvnos,  svd_uvfull, loop);
  svd_types<RtM>(svd_uvfull, svd_uvpart, loop);
  svd_types<RtM>(svd_uvpart, svd_uvpart, loop);
  svd_types<RtM>(svd_uvnos,  svd_uvpart, loop);
  svd_types<RtM>(svd_uvfull, svd_uvnos, loop);
  svd_types<RtM>(svd_uvpart, svd_uvnos, loop);
  svd_types<RtM>(svd_uvnos,  svd_uvnos, loop);
}
  
int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  test::precision<float>::init();
  test::precision<double>::init();

  length_type loop = 2;

  svd_storage<by_reference>(loop);
  svd_storage<by_value>    (loop);
}
