/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/// Description: VSIPL++ Library: Custom evaluator.
///
/// This example builds on the ``mprod.cpp`` example, illustrating
/// the process of creating a custom evaluator that references its
/// data directly through data pointers, as might be necessary for
/// interfacing to external libraries or using SIMD vectors.

#include <vsip/initfin.hpp>
#include <vsip/matrix.hpp>
#include <vsip/math.hpp>
#include <vsip_csl/dda.hpp>
#include <vsip_csl/c++0x.hpp>
#include <vsip_csl/profile.hpp>
#include <iostream>


// ====================================================================
// Define a backend function for floating point matrix products of the
// form Z = AB, where A is of size 4*4 and B is of size 4*N.  This
// function is called from the Evaluator::exec method defined below.
namespace example
{
void 
mprod_4x4_backend(float* Z, float *A, float*B, size_t n)
{
  // The inner loop of this backend would typically be an unrolled
  // target-specific SIMD vector implementation, but for simplicity we
  // will continue to use a straightforward elementwise version in this
  // example.
  for (size_t i=0; i<n; i++)
    for (size_t j=0; j<4; j++)
    {
      float sum = 0;
      for (size_t k=0; k<4; k++)
      {
        sum += A[i*4+k] * B[k*4+j];
      }
      Z[i*4+j] = sum;
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
  // potentially valid for any arguments with float value_types and
  // appropriate data layouts.
  static bool const ct_valid =
    // Check that all blocks have float value_type.
    is_same<float, typename ABlockType::value_type>::value &&
    is_same<float, typename BBlockType::value_type>::value &&
    is_same<float, typename ZBlockType::value_type>::value &&
    // check that all blocks have row-major layout, as required by the
    // indexing in the backend function.
    is_same<typename dda::Block_layout<ZBlockType>::order_type,
	    row2_type>::value && 
    is_same<typename dda::Block_layout<ABlockType>::order_type,
	    row2_type>::value && 
    is_same<typename dda::Block_layout<BBlockType>::order_type,
	    row2_type>::value && 
    // Check that direct access is supported without data copies.  This
    // cost value will be greater than zero if data copies are required.
    dda::Ext_data_cost<ZBlockType>::value == 0 &&
    dda::Ext_data_cost<ABlockType>::value == 0 &&
    dda::Ext_data_cost<BBlockType>::value == 0;

  // We check the size and stride of the blocks at runtime.
  static bool
  rt_valid(ZBlockType& Z, ABlockType const& A, BBlockType const& B)
  {
    // We only need to check the size of A, as the library contains
    // asserts that check that the sizes of the matrices are conformant
    // with each other.
    bool size_ok = A.size(2, 0) == 4 && A.size(2, 1) == 4;

    // In some cases (e.g., subview blocks) the data stride, like the
    // data size, may be unknown at compile time.  Thus we must check
    // this as well, which requires going through the Ext_data API.
    dda::Ext_data<ZBlockType> ext_Z(const_cast<ZBlockType&>(Z));
    dda::Ext_data<ABlockType> ext_A(const_cast<ABlockType&>(A));
    dda::Ext_data<BBlockType> ext_B(const_cast<BBlockType&>(B));

    bool strides_ok = ext_Z.stride(1) == 1 &&
		      ext_A.stride(1) == 1 &&
		      ext_B.stride(1) == 1;

    return size_ok && strides_ok;
  }

  // The exec method performs the actual computation.
  static void
  exec(ZBlockType& Z, ABlockType const& A, BBlockType const& B)
  {
    // As in mprod.cpp, we start with a profiling statement so that
    // calls to this backend will show up in the profile traces.
    index_type const op_count = 4 * 4 * B.size(2, 1);
    profile::Scope<vsip_csl::profile::matvec>
      scope("mprod_4x4_backend", op_count);

    // First, we generate Ext_data objects that will provide us with
    // data pointers that we can pass to the (non-VSIPL++) backend
    // function.
    dda::Ext_data<ZBlockType> ext_Z(const_cast<ZBlockType&>(Z));
    dda::Ext_data<ABlockType> ext_A(const_cast<ABlockType&>(A));
    dda::Ext_data<BBlockType> ext_B(const_cast<BBlockType&>(B));

    // Then we call the matrix product backend we defined above with
    // these data pointers and the length of the B array.
    example::mprod_4x4_backend(ext_Z.data(), ext_A.data(), ext_B.data(),
			       (size_t) B.size(2,1));
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
