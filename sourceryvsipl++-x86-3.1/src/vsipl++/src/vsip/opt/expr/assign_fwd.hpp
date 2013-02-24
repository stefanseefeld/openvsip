/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/expr/assign_fwd.hpp
    @author  Stefan Seefeld
    @date    2009-07-15
    @brief   VSIPL++ Library: block expression assignment.
*/

#ifndef VSIP_OPT_EXPR_ASSIGN_FWD_HPP
#define VSIP_OPT_EXPR_ASSIGN_FWD_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/core/config.hpp>
#include <vsip/core/expr/scalar_block.hpp>
#include <vsip/core/expr/unary_block.hpp>
#include <vsip/core/expr/binary_block.hpp>
#include <vsip/core/expr/ternary_block.hpp>
#include <vsip/core/expr/vmmul_block.hpp>
#include <vsip/core/expr/evaluation.hpp>
#include <vsip/core/block_traits.hpp>
#include <vsip/opt/dispatch.hpp>
#include <vsip/core/dispatch_tags.hpp>

namespace vsip_csl
{

namespace dispatcher
{
namespace op
{
template <dimension_type D> struct assign;
}

template <dimension_type D>
struct List<op::assign<D> >
{
  typedef Make_type_list<be::user,
			 be::cuda,
			 be::intel_ipp,
			 be::cml,
			 be::transpose,
			 be::mercury_sal,
			 be::cbe_sdk,
			 VSIP_IMPL_SIMD_TAG_LIST
#if VSIP_IMPL_ENABLE_EVAL_DENSE_EXPR
			 be::dense_expr,
#endif
			 be::copy,
			 be::op_expr,
			 be::simd_loop_fusion,
			 be::simd_unaligned_loop_fusion,
			 be::fc_expr,
			 be::rbo_expr,
			 be::mdim_expr,
			 be::loop_fusion>::type type;
};


} // namespace vsip_csl::dispatcher
} // namespace vsip_csl


#endif
