/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/sal/eval_reductions.hpp
    @author  Jules Bergmann
    @date    2006-05-30
    @brief   VSIPL++ Library: Reduction functions returning indices.
	     [math.fns.reductidx].

*/

#ifndef VSIP_IMPL_SAL_EVAL_REDUCTIONS_HPP
#define VSIP_IMPL_SAL_EVAL_REDUCTIONS_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/opt/dispatch.hpp>
#include <vsip/core/reductions/functors.hpp>
#include <vsip/opt/sal/eval_util.hpp>
#include <vsip/opt/sal/reductions.hpp>

namespace vsip
{
namespace impl
{
namespace sal
{

template <template <typename> class ReduceT,
	  typename                  T>
struct Is_reduct_supported
{
  static bool const value = false;
};

#define VSIP_IMPL_REDUCT_SUP(OP, T)					\
template <> struct Is_reduct_supported<OP, T>				\
{ static bool const value = true; };

VSIP_IMPL_REDUCT_SUP(impl::Sum_value, float*)
VSIP_IMPL_REDUCT_SUP(impl::Sum_value, double*)

VSIP_IMPL_REDUCT_SUP(impl::Sum_sq_value, float*)
VSIP_IMPL_REDUCT_SUP(impl::Sum_sq_value, double*)

VSIP_IMPL_REDUCT_SUP(impl::Mean_value, float*)
VSIP_IMPL_REDUCT_SUP(impl::Mean_value, double*)

VSIP_IMPL_REDUCT_SUP(impl::Mean_magsq_value, float*)
VSIP_IMPL_REDUCT_SUP(impl::Mean_magsq_value, double*)

VSIP_IMPL_REDUCT_SUP(impl::Max_value, float*)
VSIP_IMPL_REDUCT_SUP(impl::Max_value, double*)

VSIP_IMPL_REDUCT_SUP(impl::Min_value, float*)
VSIP_IMPL_REDUCT_SUP(impl::Min_value, double*)

#undef VSIP_IMPL_REDUCT_SUP

} // namespace vsip::impl::sal
} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{

#define REDUCTION(OP, SALFCN)						\
template <typename T,                                                   \
          typename Block>                                               \
 struct Evaluator<op::reduce<OP>, be::mercury_sal,			\
                 void(T&, Block const&, row1_type, impl::Int_type<1>)>  \
{                                                                       \
  typedef typename Block::map_type                        map_type;     \
  typedef typename impl::sal::Effective_value_type<Block>::type eff_t;	\
                                                                        \
  static bool const ct_valid =                                          \
    !impl::is_expr_block<Block>::value &&				\
    impl::Is_local_map<map_type>::value &&				\
    impl::sal::Is_reduct_supported<OP, eff_t>::value &&			\
    dda::Data<Block, dda::in>::ct_cost == 0;				\
                                                                        \
  static bool rt_valid(T&, Block const&, row1_type, impl::Int_type<1>)  \
  { return true; }                                                      \
                                                                        \
  static void exec(T& r, Block const& blk, row1_type, impl::Int_type<1>)\
  {                                                                     \
    using namespace impl;                                               \
    sal::DDA_wrapper<Block, dda::in> ext(blk);		                \
    SALFCN(typename sal::DDA_wrapper<Block, dda::in>::sal_type(ext), r, blk.size());\
  }                                                                     \
};

REDUCTION(impl::Sum_value,        impl::sal::sumval)
REDUCTION(impl::Sum_sq_value,     impl::sal::sumsqval)
REDUCTION(impl::Mean_value,       impl::sal::meanval)
REDUCTION(impl::Mean_magsq_value, impl::sal::meanmagsqval)

#undef REDUCTION

#define REDUCTION_IDX(OP, SALFCN)					\
template <typename T, typename Block>                                   \
 struct Evaluator<op::reduce_idx<OP>, be::mercury_sal,	     	        \
		  void(T&, Block const&, Index<1>&, row1_type)>		\
{                                                                       \
  typedef typename Block::map_type map_type;                            \
  typedef typename impl::sal::Effective_value_type<Block>::type eff_t;	\
                                                                        \
  static bool const ct_valid =                                          \
    !impl::is_expr_block<Block>::value &&                               \
     impl::Is_local_map<map_type>::value &&                             \
     impl::sal::Is_reduct_supported<OP, eff_t>::value &&                \
    dda::Data<Block, dda::in>::ct_cost == 0;				\
                                                                        \
  static bool rt_valid(T&, Block const&, Index<1>&, row1_type)          \
  { return true; }                                                      \
                                                                        \
  static void exec(T& r, Block const& blk, Index<1>& idx, row1_type)    \
  {                                                                     \
    using namespace impl;                                               \
    sal::DDA_wrapper<Block, dda::in> ext(blk);			\
    int i;                                                              \
    SALFCN(typename sal::DDA_wrapper<Block, dda::in>::sal_type(ext), r,i,blk.size()); \
    idx = Index<1>(i);                                                  \
  }                                                                     \
};

REDUCTION_IDX(impl::Max_value, impl::sal::maxval)
REDUCTION_IDX(impl::Min_value, impl::sal::minval)

REDUCTION_IDX(impl::Max_mag_value, impl::sal::maxmgval)
REDUCTION_IDX(impl::Min_mag_value, impl::sal::minmgval)

#undef REDUCTION_IDX

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif // VSIP_IMPL_SAL_REDUCTIONS_HPP
