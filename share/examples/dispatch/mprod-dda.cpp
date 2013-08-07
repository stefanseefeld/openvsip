/* Copyright (c) 2009, 2011 CodeSourcery, Inc.  All rights reserved. */

/* Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

       * Redistributions of source code must retain the above copyright
         notice, this list of conditions and the following disclaimer.
       * Redistributions in binary form must reproduce the above
         copyright notice, this list of conditions and the following
         disclaimer in the documentation and/or other materials
         provided with the distribution.
       * Neither the name of CodeSourcery nor the names of its
         contributors may be used to endorse or promote products
         derived from this software without specific prior written
         permission.

   THIS SOFTWARE IS PROVIDED BY CODESOURCERY, INC. "AS IS" AND ANY
   EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
   PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL CODESOURCERY BE LIABLE FOR
   ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
   BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
   OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
   EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  */

/// Description: VSIPL++ Library: Custom evaluator.
///
/// This example builds on the ``mprod.cpp`` example, illustrating
/// the process of creating a custom evaluator that references its
/// data directly through data pointers, as might be necessary for
/// interfacing to external libraries or using SIMD vectors.

#include <vsip/initfin.hpp>
#include <vsip/matrix.hpp>
#include <vsip/math.hpp>
#include <vsip/dda.hpp>
#include <iostream>


// ====================================================================
// Define a backend function for floating point matrix products of the
// form Z = AB, where A is of size 4*4 and B is of size 4*N.  This
// function is called from the Evaluator::exec method defined below.
namespace example
{
void 
mprod_4x4_backend(float *Z, float const *A, float const *B, size_t n)
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
namespace ovxx
{
namespace dispatcher
{
template <typename ZBlockType,
          typename ABlockType,
	  typename BBlockType>
struct Evaluator<op::prod, be::user,
  void(ZBlockType&, ABlockType const&, BBlockType const&),
  typename enable_if<ABlockType::dim == 2 && BBlockType::dim == 2>::type>
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
    is_same<typename get_block_layout<ZBlockType>::order_type,
	    row2_type>::value && 
    is_same<typename get_block_layout<ABlockType>::order_type,
	    row2_type>::value && 
    is_same<typename get_block_layout<BBlockType>::order_type,
	    row2_type>::value && 
    // Check that direct access is supported without data copies.  This
    // cost value will be greater than zero if data copies are required.
    dda::Data<ZBlockType, dda::out>::ct_cost == 0 &&
    dda::Data<ABlockType, dda::in>::ct_cost == 0 &&
    dda::Data<BBlockType, dda::in>::ct_cost == 0;

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
    // this as well, which requires going through the DDA API.
    dda::Data<ZBlockType, dda::out> data_Z(Z);
    dda::Data<ABlockType, dda::in> data_A(A);
    dda::Data<BBlockType, dda::in> data_B(B);

    bool strides_ok = data_Z.stride(1) == 1 &&
		      data_A.stride(1) == 1 &&
		      data_B.stride(1) == 1;

    return size_ok && strides_ok;
  }

  // The exec method performs the actual computation.
  static void
  exec(ZBlockType& Z, ABlockType const& A, BBlockType const& B)
  {
    // First, we generate DDA objects that will provide us with
    // data pointers that we can pass to the (non-VSIPL++) backend
    // function.
    dda::Data<ZBlockType, dda::out> data_Z(Z);
    dda::Data<ABlockType, dda::in> data_A(A);
    dda::Data<BBlockType, dda::in> data_B(B);

    // Then we call the matrix product backend we defined above with
    // these data pointers and the length of the B array.
    example::mprod_4x4_backend(data_Z.ptr(), data_A.ptr(), data_B.ptr(),
			       (size_t) B.size(2,1));
  }
};

} // namespace ovxx::dispatcher
} // namespace ovxx


// ====================================================================
// Main Program
int 
main(int argc, char **argv)
{
  using namespace vsip;

  // Initialize the library.
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
