/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Reductions implementation for CUDA
#ifndef VSIP_OPT_CUDA_REDUCTIONS_HPP
#define VSIP_OPT_CUDA_REDUCTIONS_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/core/domain_utils.hpp>
#include <vsip/core/signal/types.hpp>
#include <vsip/core/reductions/functors.hpp>
#include <vsip_csl/profile.hpp>

#include <vsip/opt/cuda/dda.hpp>
#include <vsip/opt/dispatch.hpp>

namespace vsip
{
namespace impl
{
namespace cuda
{

template <typename T, typename R>
R
reduce(T const *, length_type nrows, length_type ncols,
       index_type &row_return_idx, index_type &col_return_idx, int cmd);

template <>
float
reduce<float, float>(float const *, length_type nrows, length_type ncols,
            index_type &row_return_idx, index_type &col_return_idx, int cmd);

template <>
complex<float>
reduce<std::complex<float>, std::complex<float> >(complex<float> const *,
        length_type nrows, length_type ncols, index_type &row_return_idx,
        index_type &col_return_idx, int cmd);

template <>
float
reduce<std::complex<float>, float>(complex<float> const *, length_type nrows,
                               length_type ncols, index_type &row_return_idx,
                               index_type &col_return_idx, int cmd);

template <template <typename> class ReduceT>
struct Reduction 
{
  static bool const is_supported = false;
};

template <>
struct Reduction<Sum_value> 
{
  static bool const is_supported = true;
  static int const value = reduce_sum;
};

template <>
struct Reduction<Sum_sq_value> 
{
  static bool const is_supported = true;
  static int const value = reduce_sum_sq;
};

template <>
struct Reduction<Mean_value> 
{
  static bool const is_supported = true;
  static int const value = reduce_mean;
};

template <>
struct Reduction<Mean_magsq_value> 
{
  static bool const is_supported = true;
  static int const value = reduce_mean_magsq;
};

template <>
struct Reduction<Max_magsq_value> 
{
  static bool const is_supported = true;
  static int const value = reduce_max_magsq;
};

template <>
struct Reduction<Min_magsq_value> 
{
  static bool const is_supported = true;
  static int const value = reduce_min_magsq;
};

template <>
struct Reduction<Max_mag_value> 
{
  static bool const is_supported = true;
  static int const value = reduce_max_mag;
};

template <>
struct Reduction<Min_mag_value> 
{
  static bool const is_supported = true;
  static int const value = reduce_min_mag;
};

template <>
struct Reduction<Max_value> 
{
  static bool const is_supported = true;
  static int const value = reduce_max;
};

template <>
struct Reduction<Min_value> 
{
  static bool const is_supported = true;
  static int const value = reduce_min;
};

} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{
template <template <typename> class R, typename T, typename Block>
struct Evaluator<op::reduce<R>, be::cuda,
		 void(T &, Block const &, row1_type, impl::Int_type<1>)>
{
  static char const* name() { return "CUDA_reduction";}

  static bool const ct_valid = 
    impl::cuda::Reduction<R>::is_supported &&
    !impl::is_expr_block<Block>::value &&
    (is_same<T, float>::value || is_same<T, complex<float> >::value);

  static bool rt_valid(T &, Block const &b, row1_type, impl::Int_type<1>)
  {
    return true;
  }
  
  static void exec(T &result, Block const &b, row1_type, impl::Int_type<1>)
  {
    typedef typename Block::value_type VT;
    typedef typename R<VT>::result_type RT;

    index_type row_i, col_i;

    impl::cuda::dda::Data<Block, dda::in> dev_b(b);
    result = impl::cuda::reduce<VT, RT>(dev_b.ptr(), 1, dev_b.size(0),
			       row_i, col_i, impl::cuda::Reduction<R>::value);
  }
};

template <template <typename> class R, typename T, typename Block>
struct Evaluator<op::reduce_idx<R>, be::cuda,
		 void(T &, Block const &, Index<1>&, row1_type)>
{
  static char const* name() { return "CUDA_reduction_idx";}

  static bool const ct_valid = 
    impl::cuda::Reduction<R>::is_supported &&
    !impl::is_expr_block<Block>::value &&
    (is_same<T, float>::value || is_same<T, complex<float> >::value);

  static bool rt_valid(T &, Block const &b, Index<1>&, row1_type)
  {
    return true;
  }
  
  static void exec(T &result, Block const &b, Index<1> &idx, row1_type)
  {
    typedef typename Block::value_type VT;
    typedef typename R<VT>::result_type RT;

    index_type row_i, col_i;

    impl::cuda::dda::Data<Block, dda::in> dev_b(b);
    result = impl::cuda::reduce<VT, RT>(dev_b.ptr(), 1, dev_b.size(0),
			       row_i, col_i, impl::cuda::Reduction<R>::value);

    idx = Index<1>(col_i);
  }
};

template <template <typename> class R, typename T, typename Block>
struct Evaluator<op::reduce<R>, be::cuda,
		 void(T &, Block const &, row2_type, impl::Int_type<2>)>
{
  static char const* name() { return "CUDA_reduction";}

  static bool const ct_valid = 
    impl::cuda::Reduction<R>::is_supported &&
    !impl::is_expr_block<Block>::value &&
    (is_same<T, float>::value || is_same<T, complex<float> >::value);

  static bool rt_valid(T &, Block const &b, row2_type, impl::Int_type<2>)
  {
    impl::cuda::dda::Data<Block, dda::in> dev_b(b);

    return(dev_b.stride(1) == 1 && dev_b.stride(0) == dev_b.size(1));
  }
  
  static void exec(T &result, Block const &b, row2_type, impl::Int_type<2>)
  {
    typedef typename Block::value_type VT;
    typedef typename R<VT>::result_type RT;

    index_type row_i, col_i;

    impl::cuda::dda::Data<Block, dda::in> dev_b(b);
    result = impl::cuda::reduce<VT, RT>(dev_b.ptr(), dev_b.size(0), dev_b.size(1),
			       row_i, col_i, impl::cuda::Reduction<R>::value);
  }
};

template <template <typename> class R, typename T, typename Block>
struct Evaluator<op::reduce_idx<R>, be::cuda,
		 void(T &, Block const &, Index<2>&, row2_type)>
{
  static char const* name() { return "CUDA_reduction_idx";}

  static bool const ct_valid = 
    impl::cuda::Reduction<R>::is_supported &&
    !impl::is_expr_block<Block>::value &&
    (is_same<T, float>::value || is_same<T, complex<float> >::value);

  static bool rt_valid(T &, Block const &b, Index<2>&, row2_type)
  {
    impl::cuda::dda::Data<Block, dda::in> dev_b(b);

    return(dev_b.stride(1) == 1 && dev_b.stride(0) == dev_b.size(1));
  }
  
  static void exec(T &result, Block const &b, Index<2> &idx, row2_type)
  {
    typedef typename Block::value_type VT;
    typedef typename R<VT>::result_type RT;

    index_type row_i, col_i;

    impl::cuda::dda::Data<Block, dda::in> dev_b(b);
    result = impl::cuda::reduce<VT, RT>(dev_b.ptr(), dev_b.size(0), dev_b.size(1),
			       row_i, col_i, impl::cuda::Reduction<R>::value);

    idx = Index<2>(row_i, col_i);
  }
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif
