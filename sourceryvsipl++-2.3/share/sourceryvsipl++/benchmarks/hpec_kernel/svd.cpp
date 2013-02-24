/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    benchmarks/hpec-kernel/svd.cpp
    @author  Don McCoy
    @date    2006-04-06
    @brief   VSIPL++ Library: Singular Value Decomposition - High Performance 
             Embedded Computing (HPEC) Kernel-Level Benchmarks
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <cassert>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/tensor.hpp>
#include <vsip/solvers.hpp>
#include <vsip/random.hpp>
#include <vsip_csl/test-precision.hpp>

#include "benchmarks.hpp"

using namespace std;
using namespace vsip;
using vsip_csl::Precision_traits;

/***********************************************************************
  Support
***********************************************************************/

#define SVD_DEFAULT_STORAGE_TYPE  vsip::svd_uvfull

template <typename T,
	  typename Block>
typename vsip::impl::Scalar_of<T>::type
norm_1(const_Vector<T, Block> v)
{
  return sumval(mag(v));
}



/// Matrix norm-1

template <typename T,
	  typename Block>
typename vsip::impl::Scalar_of<T>::type
norm_1(const_Matrix<T, Block> m)
{
  typedef typename vsip::impl::Scalar_of<T>::type scalar_type;
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
typename vsip::impl::Scalar_of<T>::type
norm_inf(const_Matrix<T, Block> m)
{
  return norm_1(m.transpose());
}


template <typename T,
          typename Block0,
          typename Block1,
          typename Block2,
          typename Block3>
void
apply_svd(
  svd<T, by_reference>&    sv,
  Matrix<T, Block0>        a,
  Vector<scalar_f, Block1> sv_s,
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


template <typename T,
          typename Block>
void
test_svd(
  storage_type     ustorage,
  storage_type     vstorage,
  Matrix<T, Block> a,
  length_type      loop,
  float&           time)
{
  using vsip::impl::trans_or_herm;
  typedef typename vsip::impl::Scalar_of<T>::type scalar_type;

  length_type m = a.size(0);
  length_type n = a.size(1);

  length_type p = std::min(m, n);
  test_assert(m > 0 && n > 0);

  length_type u_cols = ustorage == svd_uvfull ? m : p;
  length_type v_cols = vstorage == svd_uvfull ? n : p;

  Vector<float> sv_s(p);                // singular values
  Matrix<T>     sv_u(m, u_cols);        // U matrix
  Matrix<T>     sv_v(n, v_cols);        // V matrix

  svd<T, by_reference> sv(m, n, ustorage, vstorage);

  test_assert(sv.rows()     == m);
  test_assert(sv.columns()  == n);
  test_assert(sv.ustorage() == ustorage);
  test_assert(sv.vstorage() == vstorage);


  vsip::impl::profile::Timer t1;
    
  t1.start();
  for (index_type l=0; l<loop; ++l)
  {
    apply_svd(sv, a, sv_s, sv_u, sv_v);
  }
  t1.stop();
  time = t1.delta();

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

    Index<2> idx;
    scalar_type err = maxval((mag(chk - a)
                               / Precision_traits<scalar_type>::eps),
                             idx);
    scalar_type norm_est = std::sqrt(norm_1(a) * norm_inf(a));
    err  = err / norm_est;

    if (err > 10.0)
    {
      for (index_type r=0; r<m; ++r)
        for (index_type c=0; c<n; ++c)
          test_assert(equal(chk(r, c), a(r, c)));
    }
  } 
}




/***********************************************************************
  svd function tests
***********************************************************************/

template <typename T>
struct t_svd_base : public Benchmark_base
{
  t_svd_base(storage_type ust, storage_type vst)
    : ust_(ust), vst_(vst) {}

protected:
  int svd_ops(length_type m, length_type n, length_type s)
  {
    // Workload calculations are taken from from impl_decompose()
    // in opt/lapack/svd.hpp:

    // step 1
    int step_one_ops = ((m >= n) ?
                        (4*n*n*(3*m - n))/3/s :
                        (4*m*m*(3*n - m))/3/s );
    if (impl::Is_complex<T>::value == true)
      step_one_ops *= 4;

    // step 2
    int step_two_ops = 0;
    if ( this->ust_ != svd_uvnos )
      step_two_ops += ((m >= n) ?
                       (4*n*(3*m*m - 3*m*n + n*n))/3/s :
                       (4*m*m*m)/3/s );
    if ( this->vst_ != svd_uvnos )
      step_two_ops += ((m >= n) ?
                       (4*n*n*n)/3/s :
                       (4*m*(3*n*n - 3*m*n + m*m))/3/s );
    if (impl::Is_complex<T>::value == true)
      step_two_ops *= 4;

    // step 3
    length_type nru    = (this->ust_ != svd_uvnos) ? m : 0;
    length_type ncvt   = (this->vst_ != svd_uvnos) ? n : 0;
    int rorc = impl::Is_complex<T>::value ? 2 : 1;
    int step_three_ops = (n*n + rorc*(6*n*n * nru + 6*n*n * ncvt))/s;

    return (int)(step_one_ops + step_two_ops + step_three_ops); 
  }

  // Member data.
protected:
  storage_type ust_;			// U storage type
  storage_type vst_;			// V storage type
};



template <typename T>
struct t_svd_sweep_n : public t_svd_base<T>
{
  char const *what() { return "t_svd_sweep_n"; }
  int ops_per_point(length_type size)
    { return this->svd_ops(this->m_, size, size); }
  int riob_per_point(length_type) { return -1; }
  int wiob_per_point(length_type) { return -1; }
  int mem_per_point(length_type size) 
    { return this->m_ * size * sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    typedef Matrix<T>  view_type;
    view_type a(this->m_, size);

    // Initialize
    a = T();

    // Create some test data
    Rand<T> gen(0, 0);
    a = gen.randu(this->m_, size);


    // Run the test and time it
    test_svd<T>( this->ust_, this->vst_, a, loop, time );
  }


  t_svd_sweep_n(length_type m,
    storage_type ust = SVD_DEFAULT_STORAGE_TYPE, 
    storage_type vst = SVD_DEFAULT_STORAGE_TYPE )
   : t_svd_base<T>(ust, vst), m_(m) {}

public:
  // Member data
  length_type const m_;
};


template <typename              T>
struct t_svd_sweep_m : public t_svd_base<T>
{
  char const *what() { return "t_svd_sweep_m"; }
  int ops_per_point(length_type size) 
    { return this->svd_ops(size, this->n_, size); }
  int riob_per_point(length_type) { return -1*sizeof(T); }
  int wiob_per_point(length_type) { return -1*sizeof(T); }
  int mem_per_point(length_type size) 
    { return size * this->n_ * sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    typedef Matrix<T>  view_type;
    view_type a(size, this->n_);

    // Initialize
    a = T();

    // Create some test data
    Rand<T> gen(0, 0);
    a = gen.randu(size, this->n_);


    // Run the test and time it
    test_svd<T>( this->ust_, this->vst_, a, loop, time );
  }


  t_svd_sweep_m(length_type n, 
    storage_type ust = SVD_DEFAULT_STORAGE_TYPE, 
    storage_type vst = SVD_DEFAULT_STORAGE_TYPE )
   : t_svd_base<T>(ust, vst), n_(n) {}

public:
  // Member data
  length_type const n_;
};





template <typename              T>
struct t_svd_sweep_fixed_aspect : public t_svd_base<T>
{
  char const *what() { return "t_svd_sweep_fixed_aspect"; }
  int ops_per_point(length_type size) 
    { return this->svd_ops(size * this->m_, size * this->n_, size); }
  int riob_per_point(length_type) { return -1*sizeof(T); }
  int wiob_per_point(length_type) { return -1*sizeof(T); }
  int mem_per_point(length_type size) 
    { return size * this->m_ * this->n_ * sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    typedef Matrix<T>  view_type;

    // Create some test data
    Rand<T> gen(0, 0);
    view_type a = gen.randu(size * this->m_, size * this->n_);

    // Run the test and time it
    test_svd<T>( this->ust_, this->vst_, a, loop, time );
  }


  t_svd_sweep_fixed_aspect(length_type m, length_type n, 
    storage_type ust = SVD_DEFAULT_STORAGE_TYPE, 
    storage_type vst = SVD_DEFAULT_STORAGE_TYPE )
   : t_svd_base<T>(ust, vst), m_(m), n_(n) {}

public:
  // Member data
  length_type const m_;
  length_type const n_;
};


void
defaults(Loop1P& loop)
{
  loop.progression_ = linear;
  loop.prog_scale_ = 10;
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
    // The number of rows are fixed and given as the argument
  case  1: loop(t_svd_sweep_n<float          >(500));    break;
  case  2: loop(t_svd_sweep_n<float          >(180));    break;
  case  3: loop(t_svd_sweep_n<float          >(150));    break;
  case 11: loop(t_svd_sweep_n<complex<float> >(500));    break;
  case 12: loop(t_svd_sweep_n<complex<float> >(180));    break;
  case 13: loop(t_svd_sweep_n<complex<float> >(150));    break;

    // Now the same is done for columns
  case 21: loop(t_svd_sweep_m<float          >(100));    break;
  case 22: loop(t_svd_sweep_m<float          >( 60));    break;
  case 23: loop(t_svd_sweep_m<float          >(150));    break;
  case 31: loop(t_svd_sweep_m<complex<float> >(100));    break;
  case 32: loop(t_svd_sweep_m<complex<float> >( 60));    break;
  case 33: loop(t_svd_sweep_m<complex<float> >(150));    break;

    // These use a fixed ratio of (rows, columns)
  case 41: loop(
    t_svd_sweep_fixed_aspect<complex<float> >(1, 1));    break;
  case 42: loop(
    t_svd_sweep_fixed_aspect<complex<float> >(3, 1));    break;
  case 43: loop(
    t_svd_sweep_fixed_aspect<complex<float> >(1, 3));    break;
  case 51: loop(t_svd_sweep_fixed_aspect
    <complex<float> >(1, 1, svd_uvnos, svd_uvnos));      break;
  case 52: loop(t_svd_sweep_fixed_aspect
    <complex<float> >(3, 1, svd_uvnos, svd_uvnos));      break;
  case 53: loop(t_svd_sweep_fixed_aspect
    <complex<float> >(1, 3, svd_uvnos, svd_uvnos));      break;
  case  0:
    std::cout
      << "svd -- Singular Value Decomposition\n"
      << "  Fixed number of rows.\n"
      << "   -1: float           rows=500\n"
      << "   -2: float           rows=180\n"
      << "   -3: float           rows=150\n"
      << "  -11: complex<float>  rows=500\n"
      << "  -12: complex<float>  rows=180\n"
      << "  -13: complex<float>  rows=150\n"
      << "\n"
      << "  Fixed number of columns.\n"
      << "  -21: float           cols=100\n"
      << "  -22: float           cols= 60\n"
      << "  -23: float           cols=150\n"
      << "  -31: complex<float>  cols=100\n"
      << "  -32: complex<float>  cols= 60\n"
      << "  -33: complex<float>  cols=150\n"
      << "\n"
      << "  Fixed ratio of rows to columns.\n"
      << "  -41: complex<float>  1 : 1  storage=default\n"
      << "  -42: complex<float>  3 : 1  storage=default\n"
      << "  -43: complex<float>  1 : 3  storage=default\n"
      << "  -51: complex<float>  1 : 1  storage=svd_uvnos\n"
      << "  -52: complex<float>  3 : 1  storage=svd_uvnos\n"
      << "  -53: complex<float>  1 : 3  storage=svd_uvnos\n"
      ;
  default: 
    return 0;
  }
  return 1;
}
 
