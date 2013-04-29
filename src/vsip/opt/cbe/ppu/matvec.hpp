/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
#ifndef VSIP_OPT_CBE_PPU_MATVEC_HPP
#define VSIP_OPT_CBE_PPU_MATVEC_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/opt/expr/assign_fwd.hpp>
#include <vsip/opt/cbe/ppu/task_manager.hpp>
#include <vsip/opt/cbe/ppu/util.hpp>
#include <vsip_csl/profile.hpp>

namespace vsip
{
namespace impl
{
namespace cbe
{

float dot(float const *A, float const *B, length_type, bool conj);
complex<float> 
dot(complex<float> const *A, complex<float> const *B, length_type, bool conj);
complex<float> dot(std::pair<float const *,float const *> const &A,
		   std::pair<float const *,float const *> const &B, length_type,
		   bool conj);

} // namespace vsip::impl::cbe
} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{
template <typename T, typename Block1, typename Block2>
struct Evaluator<op::dot, be::cbe_sdk,
		 T(Block1 const &, Block2 const &)>
{
  static char const* name() { return "Cbe_Sdk_dot"; }

  static bool const ct_valid = 
    !impl::is_expr_block<Block1>::value &&
    !impl::is_expr_block<Block2>::value &&
    (is_same<T, float>::value || impl::is_same<T, complex<float> >::value) &&
    is_same<T, typename Block1::value_type>::value &&
    is_same<T, typename Block2::value_type>::value &&
    impl::is_split_block<Block1>::value == impl::is_split_block<Block2>::value &&
     // check that direct access is supported
    dda::Data<Block1, dda::in>::ct_cost == 0 &&
    dda::Data<Block2, dda::in>::ct_cost == 0;

  static bool rt_valid(Block1 const &b1, Block2 const &b2)
  {
    dda::Data<Block1, dda::in> data_b1(b1);
    dda::Data<Block2, dda::in> data_b2(b2);

    return data_b1.stride(0) == 1 && data_b2.stride(0) == 1 &&
      impl::cbe::is_dma_addr_ok(data_b1.ptr()) &&
      impl::cbe::is_dma_addr_ok(data_b2.ptr()) &&
      impl::cbe::Task_manager::instance()->num_spes() > 0;
  }
  
  static T exec(Block1 const &b1, Block2 const &b2)
  {
    profile::event<profile::dispatch>("cbe::dot");
    dda::Data<Block1, dda::in> data_b1(b1);
    dda::Data<Block2, dda::in> data_b2(b2);
    return impl::cbe::dot(data_b1.ptr(), data_b2.ptr(), data_b1.size(0), false);
  }
};

template <typename T, typename Block1, typename Block2>
struct Evaluator<op::dot, be::cbe_sdk,
  T(Block1 const &, expr::Unary<expr::op::Conj, Block2, true> const &)>
{
  typedef expr::Unary<expr::op::Conj, Block2, true> cj_block_type;

  static char const* name() { return "Cbe_Sdk_cjdot"; }

  static bool const ct_valid = 
    !impl::is_expr_block<Block1>::value &&
    !impl::is_expr_block<Block2>::value &&
    (is_same<T, float>::value || impl::is_same<T, complex<float> >::value) &&
    is_same<T, typename Block1::value_type>::value &&
    is_same<T, typename Block2::value_type>::value &&
    impl::is_split_block<Block1>::value == impl::is_split_block<Block2>::value &&
     // check that direct access is supported
    dda::Data<Block1, dda::in>::ct_cost == 0 &&
    dda::Data<Block2, dda::in>::ct_cost == 0;

  static bool rt_valid(Block1 const &b1, cj_block_type const &b2)
  {
    dda::Data<Block1, dda::in> data_b1(b1);
    dda::Data<Block2, dda::in> data_b2(b2.arg());

    return data_b1.stride(0) == 1 && data_b2.stride(0) == 1 &&
      impl::cbe::is_dma_addr_ok(data_b1.ptr()) &&
      impl::cbe::is_dma_addr_ok(data_b2.ptr()) &&
      impl::cbe::Task_manager::instance()->num_spes() > 0;
  }
  
  static T exec(Block1 const &b1, cj_block_type const &b2)
  {
    profile::event<profile::dispatch>("cbe::dot");
    dda::Data<Block1, dda::in> data_b1(b1);
    dda::Data<Block2, dda::in> data_b2(b2.arg());
    return impl::cbe::dot(data_b1.ptr(), data_b2.ptr(), data_b1.size(0), true);
  }
};



} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif
