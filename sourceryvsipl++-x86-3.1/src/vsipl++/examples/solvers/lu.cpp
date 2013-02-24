/* Copyright (c) 2010 by CodeSourcery.  All rights reserved. */

/// VSIPL++ Library: LU solver example.
///
/// This example illustrates how to use the VSIPL++ LU
/// Solver facility to solve the small example provided by
/// The Mathworks at the URL
///   http://www.mathworks.com/access/helpdesk/help/toolbox/dspblks/ug/f14-131737.html
///
/// Given a matrix equation A X = B, solve for X.  A is n-by-n;
/// B and X are n-by-p.  The "LU" method first decomposes A
/// into an n-by-n lower triangular matrix L and an n-by-n
/// upper triangular matrix U, such that A = L U.  That is,
/// L has only zeroes above the diagonal, U has only zeroes
/// below the diagonal.
///
/// To solve the original equation, which is now L U X = B,
/// first solve the equation L Y = B for Y.  Second, solve
/// the equation U X = Y for X.
///
/// The LU method is efficient when solving with several B
/// matrices and one A.  That is, A is decomposed only once.
///
/// The expected result is
///   x =
///     0:  -2
///     1:  0
///     2:  1

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

  length_type n =  3;	// Rows and cols in A; rows in b.
  length_type p =  1;	// Cols in b.

  // Create some inputs, a 3x3 matrix and a 3x1 matrix.
  Matrix<float> A(n, n);
  Matrix<float> b(n, p);

  // Initialize the inputs.
  A(0, 0) =  1; A(0, 1) = -2; A(0, 2) =  3;
  A(1, 0) =  4; A(1, 1) =  0; A(1, 2) =  6;
  A(2, 0) =  2; A(2, 1) = -1; A(2, 2) =  3;
  b(0, 0) =  1; b(1, 0) = -2; b(2, 0) = -1;
  
  // Create a 3x1 matrix for output.
  Matrix<float> x(n, p);

  // Build the solver.
  lud<float, by_reference> lu(n);

  // Factor A.
  lu.decompose(A);

  // Solve A x = b.  Do not transpose A.
  lu.solve<mat_ntrans>(b, x);

  // Display the results.
  std::cout
    << std::endl
    << "LU Solver example" << std::endl
    << "-----------------" << std::endl
    << "A = " << std::endl << A << std::endl
    << "b = " << std::endl << b << std::endl
    << "x = " << std::endl << x << std::endl;

  return 0;
}
