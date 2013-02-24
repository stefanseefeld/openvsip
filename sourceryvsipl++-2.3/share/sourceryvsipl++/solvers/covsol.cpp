/* Copyright (c) 2010 by CodeSourcery.  All rights reserved. */

/// VSIPL++ Library: Covariance solver example.
///
/// This example illustrates how to use the VSIPL++ Covariance
/// Solver facility to solve the covariance system
/// 
///   A' A x = b
/// 
/// for x.  A is MxN; b and x are NxP.
/// 
/// The first step uses the QR method (see that example) to
/// decompose A into matrices Q and R.  The system becomes
///
///   R' Q' Q R x = b
///
/// which simplifies to
///
///  R' R x = b
///
/// The second step solves the system
/// 
///   R' c = b
/// 
/// for c, also NxP.  Finally it solves
/// 
///   R x = c
/// 
/// for x.
/// 
/// The expected result is
///   x = 
///     0:  13.3333
///     1:  -6.33333
///     2:  -9.22222

#include <iostream>
#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/signal.hpp>
#include <vsip/math.hpp>
#include <vsip/solvers.hpp>
#include <vsip_csl/output.hpp>


using namespace vsip;


int
main(int argc, char **argv)
{
  vsipl init(argc, argv);

  length_type m =  3;   // Rows in A.
  length_type n =  3;	// Cols in A; rows in b.
  length_type p =  1;	// Cols in b.

  // Create some inputs, a 3x3 matrix and a 3x1 matrix.
  Matrix<float> A(m, n);
  Matrix<float> b(n, p);

  // Initialize the inputs.
  A(0, 0) =  1; A(0, 1) = -2; A(0, 2) =  3;
  A(1, 0) =  4; A(1, 1) =  0; A(1, 2) =  6;
  A(2, 0) =  2; A(2, 1) = -1; A(2, 2) =  3;

  b(0, 0) =  1; b(0, 1) = -2; b(0, 2) = -1;
  
  // Create a 3x1 matrix for output.
  Matrix<float> x(n, p);

  // Solve A' A x = b for x.
  covsol(A, b, x);

  // Display the results.
  std::cout
    << std::endl
    << "Covariance Solver example" << std::endl
    << "-------------------------" << std::endl
    << "A = " << std::endl << A << std::endl
    << "b = " << std::endl << b << std::endl
    << "x = " << std::endl << x << std::endl;

  return 0;
}
