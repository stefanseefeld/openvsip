/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/// Description: VSIPL++ Library: Custom evaluator.
///
/// This example illustrates the process of creating a custom evaluator
/// for a VSIPL++ standard function -- in this case, a fixed-size 4x4
/// matrix product that would be amenable to SIMD optimizations.

#include <vsip/initfin.hpp>
#include <vsip/matrix.hpp>
#include <vsip/math.hpp>
#include <vsip_csl/profile.hpp>
#include <iostream>
#include <vsip_csl/c++0x.hpp>


// ====================================================================
// Define a backend function for floating point matrix products of the
// form Z = AB, where A is of size 4*4 and B is of size 4*N.  This
// function is called from the Evaluator::exec method defined below,
// and is a stand-in for an implementation that might be defined in an
// external library or other user-supplied code.  In this example, we
// use VSIPL++ blocks to pass the data between the Evaluator::exec
// method and the backend function.
namespace example
{
template <typename ZBlockType,
          typename ABlockType,
	  typename BBlockType>
void 
mprod_4x4_backend(ZBlockType& Z, ABlockType const& A, BBlockType const& B)
{
  // A simple matrix product implementation.  In real code, a highly-
  // optimized implementation would go here instead.
  for (vsip::index_type i=0; i<Z.size(2, 0); ++i)
    for (vsip::index_type j=0; j<Z.size(2, 1); ++j)
    {
      float sum = 0;
      for (vsip::index_type k=0; k<A.size(2, 1); ++k)
      {
	sum += A.get(i, k) * B.get(k, j);
      }
      Z.put(i, j, sum);
    }

  // For purposes of the example, we'll also print a diagnostic message
  // to the console to indicate that this backend function has been
  // called.
  std::cout << " examples::mprod_4x4_backend called." << std::endl;
}

} // namespace example


// ====================================================================
// Provide a specialization of the Evaluator class that maps matrix
// products for 4x4 matrices to the above backend.
namespace vsip_csl
{
namespace dispatcher
{
template <typename ZBlockType,
          typename ABlockType,
	  typename BBlockType>
struct Evaluator<op::prod, be::user,
                 void(ZBlockType&, ABlockType const&, BBlockType const&),
		 typename enable_if_c<ABlockType::dim == 2 &&
				      BBlockType::dim == 2>::type>
{
  // Block sizes are not known at compile time, so this evaluator is
  // potentially valid for any arguments with float value_types.
  static bool const ct_valid =
    is_same<float, typename ABlockType::value_type>::value &&
    is_same<float, typename BBlockType::value_type>::value &&
    is_same<float, typename ZBlockType::value_type>::value;

  // The matrix sizes are checked at runtime.  We only need to check
  // the size of A; we can assume that the sizes of the matrices are
  // properly conformant with each other, as this is checked (with
  // asserts) by the library before the dispatcher is called.  Note
  // that here we are dealing with blocks, not views, and thus
  // must give the view dimension explicitly to the size() operator.
  static bool
  rt_valid(ZBlockType&, ABlockType const& A, BBlockType const&)
  {
    return A.size(2, 0) == 4 && A.size(2, 1) == 4;
  }

  // The exec method performs the actual computation.  Here, this
  // simply calls the implementation we defined earlier, but this could
  // include other logic or even a complete implementation.  Note that
  // the exec function gets blocks, not views, as its inputs.
  static void
  exec(ZBlockType& Z, ABlockType const& A, BBlockType const& B)
  {
    // We include a profiling statement here so that calls to this
    // backend will show up in the profile traces.  This is optional,
    // but very useful for validating the dispatch code.  See the
    // dispatch examples for more details.
    index_type const op_count = 4 * 4 * B.size(2, 1);
    profile::Scope<vsip_csl::profile::matvec>
      scope("mprod_4x4_backend", op_count);
  
    // Now, the call to the backend function.
    example::mprod_4x4_backend(Z, A, B);
  }
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl


// ====================================================================
// Main Program
int 
main(int argc, char **argv)
{
  using namespace vsip;

  // Initialize the Sourcery VSIPL++ library.
  vsipl init(argc, argv);

  // Set up some example inputs and outputs.
  Matrix<float> A4(4, 4, 1.0);
  Matrix<float> B4(4, 8, 2.0);
  Matrix<float> Z4(4, 8);

  // Do a matrix product, which should use the backend defined above.
  std::cout << "Executing Z4 = prod(A4, B4)." << std::endl;
  Z4 = prod(A4, B4);
  
  // Repeat for different-sized matrices.  As the sizes of these
  // matrices do not match the new evaluator we defined, these will
  // use the standard backend.
  Matrix<float> A8(8, 4, 1.0);
  Matrix<float> B8(4, 8, 2.0);
  Matrix<float> Z8(8, 8);
  std::cout << "Executing Z8 = prod(A8, B8)." << std::endl;
  Z8 = prod(A8, B8);
}
