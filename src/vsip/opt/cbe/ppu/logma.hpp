/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
#ifndef VSIP_OPT_CBE_PPU_LOGMA_HPP
#define VSIP_OPT_CBE_PPU_LOGMA_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/opt/expr/assign_fwd.hpp>
#include <vsip/core/adjust_layout.hpp>
#include <vsip/opt/cbe/ppu/task_manager.hpp>
#include <vsip/opt/cbe/ppu/util.hpp>

namespace vsip_csl
{
namespace expr
{
namespace op
{
// This struct is needed only to select the threshold for the fused
// log-multiply-add kernel.
template <typename T1, typename T2, typename T3>
struct Lma
{};
} 
}
}


namespace vsip
{
namespace impl
{
namespace cbe
{

/// log10(A) * b + c
void vlma(float const* A, float const b, float const c, float* R, length_type len);


template <>
struct Size_threshold<vsip_csl::expr::op::Lma<float, float, float> >
{
  static length_type const value = 4096;
};


} // namespace vsip::impl::cbe
} // namespace vsip::impl
} // namespace vsip

#endif
