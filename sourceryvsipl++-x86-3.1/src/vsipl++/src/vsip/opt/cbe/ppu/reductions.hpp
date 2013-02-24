/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
#ifndef VSIP_OPT_CBE_PPU_REDUCTIONS_HPP
#define VSIP_OPT_CBE_PPU_REDUCTIONS_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/opt/cbe/ppu/task_manager.hpp>
#include <vsip/opt/cbe/ppu/util.hpp>
#include <vsip/core/reductions/functors.hpp>
#include <vsip_csl/profile.hpp>

namespace vsip
{
namespace impl
{
namespace cbe
{
enum reduction_type { sum, sumsq, mean, meansq};

// vector reductions
float
reduce(float const *, length_type len, int cmd);

complex<float>
reduce(complex<float> const *, length_type len, int cmd);

complex<float>
reduce(std::pair<float const *,float const *> const &, length_type len, int cmd);

// matrix reductions
float
reduce_matrix(float const* A, length_type rows, index_type row_stride, 
  length_type cols, int cmd);

complex<float>
reduce_matrix(complex<float> const* A, length_type rows, index_type row_stride, 
  length_type cols, int cmd);

std::complex<float>
reduce_matrix(std::pair<float const *,float const *> const &A, length_type rows, 
  index_type row_stride, length_type cols, int cmd);


// helper functions used in compile-time checks
template <template <typename> class ReduceT>
struct Reduction 
{
  static bool const is_supported = false;
};

template <>
struct Reduction<Sum_value> 
{
  static bool const is_supported = true;
  static int const value = sum;
};

template <>
struct Reduction<Sum_sq_value> 
{
  static bool const is_supported = true;
  static int const value = sumsq;
};

template <>
struct Reduction<Mean_value> 
{
  static bool const is_supported = true;
  static int const value = mean;
};

template <>
struct Reduction<Mean_magsq_value> 
{
  static bool const is_supported = true;
  static int const value = meansq;
};



template <typename T>
struct Extract
{
  static T const& apply(complex<float> const &);
};

template <> 
struct Extract<float>
{
  static float const&
  apply(complex<float> const &v) { return v.real(); }
};

template <> 
struct Extract<complex<float> >
{
  static complex<float> const&
  apply(complex<float> const &v) { return v; }
};



// traits class used for run-time checks
template <dimension_type D, 
          typename       T>
struct Dispatch_threshold
{ static unsigned int const value = 1024; };

template <>
struct Dispatch_threshold<1, float>
{ static unsigned int const value = 16384; };

template <>
struct Dispatch_threshold<2, float>
{ static unsigned int const value = 8192; };

template <>
struct Dispatch_threshold<2, std::complex<float> >
{ static unsigned int const value = 4096; };


} // namespace vsip::impl::cbe
} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{

template <template <typename> class ReduceT, 
          typename                  T, 
          typename                  Block>
struct Evaluator<op::reduce<ReduceT>, be::cbe_sdk,
		 void(T &, Block const &, row1_type, impl::Int_type<1>)>
{
  static char const* name() { return "be_cbe_sdk_vector_reduction ";}

  static bool const ct_valid = 
    // verify this particular reduction is supported
    impl::cbe::Reduction<ReduceT>::is_supported &&
    // the argument may not be an expression
    !impl::is_expr_block<Block>::value &&
    // only real or complex (split or interleaved) single-precision allowed
    (is_same<T, float>::value || impl::is_same<T, complex<float> >::value) &&
     // check that direct access is supported
    dda::Data<Block, dda::in>::ct_cost == 0;

  static bool rt_valid(T &, Block const &b, row1_type, impl::Int_type<1>)
  {
    dda::Data<Block, dda::in> data_b(b);
    typedef vsip::impl::cbe::Dispatch_threshold<Block::dim, T>
      threshold_type;

    return data_b.stride(0) == 1 &&
      // length must be great enough to outweigh the data transfer costs
      data_b.size(0) >= threshold_type::value && 
      // the beginning address must be suitable for DMA
      impl::cbe::is_dma_addr_ok(data_b.ptr()) &&
      // there must be sufficient number of spes available
      impl::cbe::Task_manager::instance()->num_spes() > 0;
  }
  
  static void exec(T &result, Block const &b, row1_type, impl::Int_type<1>)
  {
    profile::event<profile::dispatch>("cbe::reduce");
    dda::Data<Block, dda::in> data_b(b);

    typedef typename Block::value_type block_value_type;
    block_value_type reduction_result;

    reduction_result = impl::cbe::reduce(
      data_b.ptr(), data_b.size(0), impl::cbe::Reduction<ReduceT>::value);

    // Reductions involving squares of complex values give real results.
    // For these cases, the real component contains the desired value.
    // This metafunction extracts the right value in all cases.
    result = vsip::impl::cbe::Extract<T>::apply(reduction_result);
  }
};



template <template <typename> class ReduceT, 
          typename                  T, 
          typename                  Block>
struct Evaluator<op::reduce<ReduceT>, be::cbe_sdk,
		 void(T &, Block const &, row2_type, impl::Int_type<2>)>
{
  static char const* name() { return "be_cbe_sdk_matrix_reduction ";}

  static bool const ct_valid = 
    // verify this particular reduction is supported
    impl::cbe::Reduction<ReduceT>::is_supported &&
    // the argument may not be an expression
    !impl::is_expr_block<Block>::value &&
    // only real or complex (split or interleaved) single-precision allowed
    (is_same<T, float>::value || impl::is_same<T, complex<float> >::value) &&
    // check that direct access is supported
    dda::Data<Block, dda::in>::ct_cost == 0;

  static bool rt_valid(T &, Block const &b, row2_type, impl::Int_type<2>)
  {
    dda::Data<Block, dda::in> data_b(b);
    typedef vsip::impl::cbe::Dispatch_threshold<Block::dim, T>
      threshold_type;

    return 
      // must be unit-stride between columns
      data_b.stride(1) == 1 &&
      // length of rows must be great enough to outweigh the data transfer costs
      data_b.size(1) >= threshold_type::value && 
      // the beginning address and the stride between rows must be suitable for DMA
      impl::cbe::is_dma_addr_ok(data_b.ptr()) &&
      impl::cbe::is_dma_stride_ok<T>(data_b.stride(0)) &&
      // there must be sufficient number of spes available
      impl::cbe::Task_manager::instance()->num_spes() > 0;
  }
  
  static void exec(T &result, Block const &b, row2_type, impl::Int_type<2>)
  {
    profile::event<profile::dispatch>("cbe::reduce_matrix");

    dda::Data<Block, dda::in> data_b(b);
    length_type rows = data_b.size(0);
    index_type row_stride = data_b.stride(0);
    length_type cols = data_b.size(1);

    typedef typename Block::value_type block_value_type;
    block_value_type reduction_result;

    reduction_result = impl::cbe::reduce_matrix(
      data_b.ptr(), rows, row_stride, cols, impl::cbe::Reduction<ReduceT>::value);

    // Reductions involving squares of complex values give real results.
    // For these cases, the real component contains the desired value.
    // This metafunction extracts the right value in all cases.
    result = vsip::impl::cbe::Extract<T>::apply(reduction_result);
  }
};
 


} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif
