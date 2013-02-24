/* Copyright (c) 2010 by CodeSourcery.  All rights reserved. */

/// VSIPL++ Library: SVD solver example.
///
/// This example illustrates how to use the VSIPL++ SVD Solver
/// to factorize a matrix.  This method is useful for computing
/// the pseudoinverse of a matrix and for least-squares fitting
/// of data, to name but two applications.
///
/// Briefly, a matrix M of size m x n may be decomposed into
/// the product of an m x m unitary matrix U, a diagonal matrix
/// S and the conjugate transpose of an n x n unitary matrix V.
/// The non-zero elements of S are known as the "singular values"
/// and are uniquely determined by M.
///
/// This example[1] takes the matrix
///
///       1  0  0  0  2
///  M =  0  0  3  0  0
///       0  0  0  0  0
///       0  4  0  0  0
///
/// and computes the singular values
///
///  S =  4  3  sqrt(5)  0
///
/// It then verifies the product
///
///  U S V* 
///
/// produces the original input matrix.  U and V are then multiplied by
/// their transpose to produce an identity matrix (1's on the diagonal, 
/// 0 otherwise).  This shows that they are, in fact, unitary.
///
/// [1] http://en.wikipedia.org/wiki/Singular_value_decomposition#Example

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/matrix.hpp>
#include <vsip/solvers.hpp>
#include <vsip/support.hpp>

#include <vsip_csl/output.hpp>
#include <vsip_csl/test.hpp>


using namespace vsip;

template <typename T>
void
svd_by_ref()
{
  length_type const m = 4;
  length_type const n = 5;

  Matrix<T> a(m, n, T());
  a.put(0, 0, 1.f);
  a.put(0, 4, 2.f);
  a.put(1, 2, 3.f);
  a.put(3, 1, 4.f);


  // Create views needed to hold the factorized parts
  // of the input matrix.  The exact size of views required
  // depends on the storage method chosen.
  storage_type ustorage = svd_uvfull;
  storage_type vstorage = svd_uvfull;
  length_type p = std::min(m, n);
  length_type u_cols = ustorage == svd_uvfull ? m : p;
  length_type v_cols = vstorage == svd_uvfull ? n : p;
  length_type v_rows = vstorage == svd_uvfull ? n : p;

  Vector<float> sv_s(p);		// singular values
  Matrix<T>     sv_u(m, u_cols);	// U matrix
  Matrix<T>     sv_v(n, v_cols);	// V matrix


  // Create the SVD object and find singular values.
  svd<T, by_reference> sv(m, n, ustorage, vstorage);
  sv.decompose(a, sv_s);


  // Obtain U and V parts.
  if (sv.ustorage() != svd_uvnos)
    sv.u(0, u_cols - 1, sv_u);
  if (sv.vstorage() != svd_uvnos)
    sv.v(0, v_rows - 1, sv_v);

  Matrix<T> sv_sm(m, n, T());
  sv_sm.diag() = sv_s;

  std::cout << " Input = " << std::endl << a << std::endl;
  std::cout << " S = " << std::endl << sv_s << std::endl;
  std::cout << " U = " << std::endl << sv_u << std::endl;
  std::cout << " V* = " << std::endl << sv_v.transpose() << std::endl;

  // Verify that the product USV* gives back the original inputs.
  Matrix<T> chk(m, n);
  chk = prod(prod(sv_u, sv_sm), trans(sv_v));

  std::cout << " U S V* = " << std::endl << chk << std::endl;
  test_assert(vsip_csl::view_equal(a, chk));


  // Verify that U and V are unitary, also demonstrate use of
  // the product functions.
  // 
  // [Note the '.template' keyword is required in this case because the
  // member functions produ/prodv have their own template parameters.]
  // 
  Matrix<T> u_identity(m, u_cols);
  sv.template produ<mat_ntrans, mat_lside>(sv_u.transpose(), u_identity);
  std::cout << " U U* = " << std::endl << u_identity << std::endl;

  Matrix<T> v_identity(n, v_cols);
  sv.template prodv<mat_trans, mat_lside>(sv_v, v_identity);
  std::cout << " V V* = " << std::endl << v_identity << std::endl;
}



int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  svd_by_ref<float>();
}
