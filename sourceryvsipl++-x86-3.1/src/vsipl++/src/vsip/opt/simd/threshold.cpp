/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#include <vsip/opt/simd/threshold.hpp>

namespace vsip
{
namespace impl
{
namespace simd
{

#if !VSIP_IMPL_INLINE_LIBSIMD

template <template <typename,typename> class O,
          typename T>
void 
threshold(T* Z, T const *A, T const *B, T k, length_type n)
{
  static bool const Is_vectorized =
    is_algorithm_supported<T, false, Alg_threshold>::value &&
    Binary_operator_map<T,O>::is_supported;
  Threshold<T, O, Is_vectorized>::exec(Z,A,B,k,n);
}

#define DECL_THRESH(O)   		 				  \
template void threshold<O>(float *Z, float const *A, float const *B, float k, length_type n); \
template void threshold<O>(double *Z, double const *A, double const *B, double k, length_type n);

DECL_THRESH(expr::op::Gt)
DECL_THRESH(expr::op::Lt)
DECL_THRESH(expr::op::Ge)
DECL_THRESH(expr::op::Le)

#undef DECL_THRESH

#endif

}
}
}
