//
// Copyright (c) 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_cvsip_solver_hpp_
#define ovxx_cvsip_solver_hpp_

#include <ovxx/cvsip/common.hpp>
#include <complex>

namespace ovxx
{
namespace cvsip
{

template <typename T>
struct solver_traits { static bool const valid = false;};

#if OVXX_CVSIP_HAVE_FLOAT
template<> struct solver_traits<float>
{
  static bool const valid = true;

  typedef vsip_mview_f matrix_type;
  typedef vsip_lu_f    lud_type;
  typedef vsip_qr_f    qr_type;
  typedef vsip_chol_f  chol_type;

  static lud_type *lu_create(length_type n) 
  {
    lud_type *s = vsip_lud_create_f(n);
    if (!s) VSIP_IMPL_THROW(std::bad_alloc());
    return s;
  }
  static void lu_destroy(lud_type *s) { vsip_lud_destroy_f(s);}
  static int lu_decompose(lud_type *s, matrix_type *m) { return vsip_lud_f(s, m);}
  static int lu_solve(lud_type *s, mat_op_type o, matrix_type *m) 
  { return vsip_lusol_f(s, mat_op(o), m);}

  static qr_type *qr_create(length_type n, length_type m, storage_type s)
  {
    qr_type *qr = vsip_qrd_create_f(n, m, storage(s));
    if (!qr) VSIP_IMPL_THROW(std::bad_alloc());
    return qr;
  }
  static void qr_destroy(qr_type *qr) { vsip_qrd_destroy_f(qr);}
  static int qr_decompose(qr_type *qr, matrix_type *m) { return vsip_qrd_f(qr, m);}
  static int qr_solve(qr_type *qr, vsip_qrd_prob p, matrix_type *m)
  { return vsip_qrsol_f(qr, p, m);}
  static int qr_solve_r(qr_type *qr, mat_op_type o, float t, matrix_type *m)
  { return vsip_qrdsolr_f(qr, mat_op(o), t, m);}
  static int qr_prodq(qr_type *qr, mat_op_type o, product_side_type s, matrix_type *m)
  { return vsip_qrdprodq_f(qr, mat_op(o), product_side(s), m); }
  static chol_type *chol_create(length_type n, mat_uplo ul) 
  {
    chol_type *s = vsip_chold_create_f(ul == 0 ? VSIP_TR_LOW : VSIP_TR_UPP, n);
    if (!s) VSIP_IMPL_THROW(std::bad_alloc());
    return s;
  }
  static void chol_destroy(chol_type *s) { vsip_chold_destroy_f(s);}
  static int chol_decompose(chol_type *s, matrix_type *m)
  { return vsip_chold_f(s, m);}
  static int chol_solve(chol_type *s, matrix_type *m)
  { return vsip_cholsol_f(s, m);}
};

template<> struct solver_traits<std::complex<float> >
{
  static bool const valid = true;

  typedef vsip_cmview_f matrix_type;
  typedef vsip_clu_f    lud_type;
  typedef vsip_cqr_f    qr_type;
  typedef vsip_cchol_f  chol_type;

  static lud_type *lu_create(length_type n) 
  {
    lud_type *s = vsip_clud_create_f(n);
    if (!s) VSIP_IMPL_THROW(std::bad_alloc());
    return s;
  }
  static void lu_destroy(lud_type *s) { vsip_clud_destroy_f(s);}
  static int lu_decompose(lud_type *s, matrix_type *m) { return vsip_clud_f(s, m);}
  static int lu_solve(lud_type *s, mat_op_type o, matrix_type *m) 
  { return vsip_clusol_f(s, mat_op(o), m);}

  static qr_type *qr_create(length_type n, length_type m, storage_type s) 
  {
    qr_type *qr = vsip_cqrd_create_f(n, m, storage(s));
    if (!qr) VSIP_IMPL_THROW(std::bad_alloc());
    return qr;
  }
  static void qr_destroy(qr_type *qr) { vsip_cqrd_destroy_f(qr);}
  static int qr_decompose(qr_type *qr, matrix_type *m) { return vsip_cqrd_f(qr, m);}
  static int qr_solve(qr_type *qr, vsip_qrd_prob p, matrix_type *m)
  { return vsip_cqrsol_f(qr, p, m);}
  static int qr_solve_r(qr_type *qr, mat_op_type o,
                        std::complex<float> t, matrix_type *m)
  {
    vsip_cscalar_f tt = {t.real(), t.imag()};
    return vsip_cqrdsolr_f(qr, mat_op(o), tt, m);
  }
  static int qr_prodq(qr_type *qr, mat_op_type o, product_side_type s, matrix_type *m)
  { return vsip_cqrdprodq_f(qr, mat_op(o), product_side(s), m); }

  static chol_type *chol_create(length_type n, mat_uplo ul) 
  {
    chol_type *s = vsip_cchold_create_f(ul == 0 ? VSIP_TR_LOW : VSIP_TR_UPP, n);
    if (!s) VSIP_IMPL_THROW(std::bad_alloc());
    return s;
  }
  static void chol_destroy(chol_type *s) { vsip_cchold_destroy_f(s);}
  static int chol_decompose(chol_type *s, matrix_type *m)
  { return vsip_cchold_f(s, m);}
  static int chol_solve(chol_type *s, matrix_type *m)
  { return vsip_ccholsol_f(s, m);}
};
#endif

#if OVXX_CVSIP_HAVE_DOUBLE
template<> struct solver_traits<double>
{
  static bool const valid = true;

  typedef vsip_mview_d matrix_type;
  typedef vsip_lu_d    lud_type;
  typedef vsip_qr_d    qr_type;
  typedef vsip_chol_d  chol_type;

  static lud_type *lu_create(length_type n) 
  {
    lud_type *s = vsip_lud_create_d(n);
    if (!s) VSIP_IMPL_THROW(std::bad_alloc());
    return s;
  }
  static void lu_destroy(lud_type *s) { vsip_lud_destroy_d(s);}
  static int lu_decompose(lud_type *s, matrix_type *m) { return vsip_lud_d(s, m);}
  static int lu_solve(lud_type *s, mat_op_type o, matrix_type *m) 
  { return vsip_lusol_d(s, mat_op(o), m);}

  static qr_type *qr_create(length_type n, length_type m, storage_type s) 
  {
    qr_type *qr = vsip_qrd_create_d(n, m, storage(s));
    if (!qr) VSIP_IMPL_THROW(std::bad_alloc());
    return qr;
  }
  static void qr_destroy(qr_type *qr) { vsip_qrd_destroy_d(qr);}
  static int qr_decompose(qr_type *qr, matrix_type *m) { return vsip_qrd_d(qr, m);}
  static int qr_solve(qr_type *qr, vsip_qrd_prob p, matrix_type *m)
  { return vsip_qrsol_d(qr, p, m);}
  static int qr_solve_r(qr_type *qr, mat_op_type o, double t, matrix_type *m)
  { return vsip_qrdsolr_d(qr, mat_op(o), t, m);}
  static int qr_prodq(qr_type *qr, mat_op_type o, product_side_type s, matrix_type *m)
  { return vsip_qrdprodq_d(qr, mat_op(o), product_side(s), m); }

  static chol_type *chol_create(length_type n, mat_uplo ul) 
  {
    chol_type *s = vsip_chold_create_d(ul == 0 ? VSIP_TR_LOW : VSIP_TR_UPP, n);
    if (!s) VSIP_IMPL_THROW(std::bad_alloc());
    return s;
  }
  static void chol_destroy(chol_type *s) { vsip_chold_destroy_d(s);}
  static int chol_decompose(chol_type *s, matrix_type *m)
  { return vsip_chold_d(s, m);}
  static int chol_solve(chol_type *s, matrix_type *m)
  { return vsip_cholsol_d(s, m);}
};

template<> struct solver_traits<std::complex<double> >
{
  static bool const valid = true;

  typedef vsip_cmview_d matrix_type;
  typedef vsip_clu_d    lud_type;
  typedef vsip_cqr_d    qr_type;
  typedef vsip_cchol_d  chol_type;

  static lud_type *lu_create(length_type n) 
  {
    lud_type *s = vsip_clud_create_d(n);
    if (!s) VSIP_IMPL_THROW(std::bad_alloc());
    return s;
  }
  static void lu_destroy(lud_type *s) { vsip_clud_destroy_d(s);}
  static int lu_decompose(lud_type *s, matrix_type *m) { return vsip_clud_d(s, m);}
  static int lu_solve(lud_type *s, mat_op_type o, matrix_type *m) 
  { return vsip_clusol_d(s, mat_op(o), m);}

  static qr_type *qr_create(length_type n, length_type m, storage_type s) 
  {
    qr_type *qr = vsip_cqrd_create_d(n, m, storage(s));
    if (!qr) VSIP_IMPL_THROW(std::bad_alloc());
    return qr;
  }
  static void qr_destroy(qr_type *qr) { vsip_cqrd_destroy_d(qr);}
  static int qr_decompose(qr_type *qr, matrix_type *m) { return vsip_cqrd_d(qr, m);}
  static int qr_solve(qr_type *qr, vsip_qrd_prob p, matrix_type *m)
  { return vsip_cqrsol_d(qr, p, m);}
  static int qr_solve_r(qr_type *qr, mat_op_type o,
                        std::complex<double> t, matrix_type *m)
  {
    vsip_cscalar_d tt = {t.real(), t.imag()};
    return vsip_cqrdsolr_d(qr, mat_op(o), tt, m);
  }
  static int qr_prodq(qr_type *qr, mat_op_type o, product_side_type s, matrix_type *m)
  { return vsip_cqrdprodq_d(qr, mat_op(o), product_side(s), m); }

  static chol_type *chol_create(length_type n, mat_uplo ul) 
  {
    chol_type *s = vsip_cchold_create_d(ul == 0 ? VSIP_TR_LOW : VSIP_TR_UPP, n);
    if (!s) VSIP_IMPL_THROW(std::bad_alloc());
    return s;
  }
  static void chol_destroy(chol_type *s) { vsip_cchold_destroy_d(s);}
  static int chol_decompose(chol_type *s, matrix_type *m)
  { return vsip_cchold_d(s, m);}
  static int chol_solve(chol_type *s, matrix_type *m)
  { return vsip_ccholsol_d(s, m);}
};
#endif

} // namespace ovxx::cvsip
} // namespace ovxx

#endif
