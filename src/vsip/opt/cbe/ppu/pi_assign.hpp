/* Copyright (c) 2010 by CodeSourcery.  All rights reserved. */

/// Description
///   PI assignment evaluators for CBE.

#ifndef vsip_opt_cbe_ppu_pi_assign_hpp_
#define vsip_opt_cbe_ppu_pi_assign_hpp_

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/signal.hpp>
#include <cml_core.h>

namespace vsip_csl
{
namespace dispatcher
{
template <typename B1, typename I, template <typename> class O, typename B2>
struct Evaluator<op::pi_assign, be::cml,
  void(pi::Call<B1, I, whole_domain_type> &,
       pi::Unary<O, pi::Call<B2, I, whole_domain_type> > const &)>
{
  typedef pi::Call<B1, I, whole_domain_type> LHS;
  typedef pi::Unary<O, pi::Call<B2, I, whole_domain_type> > RHS;

  // TODO: validate the input.
  // - check that O<Call<B2>::result_type> is actually a convolution, as well as whether
  // - make sure data is well aligned for low-level CML calls.
  static bool const ct_valid = true;
  static bool rt_valid(LHS &, RHS const &) { return true;}
  static void exec(LHS &lhs, RHS const &rhs)
  {
    typedef typename pi::Call<B2, I, whole_domain_type>::result_type block_type;
    typedef O<block_type> operation_type;

    Convolution<const_Vector, nonsym, support_min, float> &conv =
      rhs.operation().operation();

    // Access kernel data through non-standard "kernel()" member function.
    float *kernel = conv.kernel().block().ptr();

    alf_task_handle_t task = cml_core_conv1d_min1_setup_f();

    for (index_type i = 0; i != lhs.block().size(2, 0); ++i)
    {
      typename pi::Call<B1, I, whole_domain_type>::result_type lhs_block(lhs.apply(i));
      typename pi::Call<B2, I, whole_domain_type>::result_type rhs_block(rhs.arg().apply(i));
      cml_async_conv1d_min1_f(task, kernel,
 			      rhs_block.ptr(), lhs_block.ptr(),
 			      1, 4 /*kernel_size*/, lhs_block.size(1, 0));// output_size);
    }
    cml_core_conv1d_min1_destroy_f(task);
  }
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif
