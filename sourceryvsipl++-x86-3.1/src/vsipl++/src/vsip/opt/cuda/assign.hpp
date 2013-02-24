/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
#ifndef VSIP_OPT_CUDA_ASSIGN_HPP
#define VSIP_OPT_CUDA_ASSIGN_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/opt/expr/assign_fwd.hpp>
#include <vsip/opt/cuda/eval_vmmul.hpp>
#include <vsip/opt/cuda/eval_fastconv.hpp>
#include <vsip/opt/cuda/eval_fftmul.hpp>
#include <vsip/opt/cuda/unary.hpp>
#include <vsip/opt/cuda/binary.hpp>
#include <vsip/opt/cuda/ternary.hpp>
#include <vsip/opt/cuda/bindings.hpp>
#include <vsip/opt/cuda/copy.hpp>
#include <vsip/opt/cuda/dda.hpp>
#include <vsip/dda.hpp>
#include <vsip/core/domain_utils.hpp>

namespace vsip
{
namespace impl
{
namespace cuda
{

/// Data<Scalar<...> > isn't instantiable (see issue 808).
/// To temporarily work around the issue we use this filter:
template <typename B>
struct is_scalar_block { static bool const value = false;};
template <typename B>
struct is_scalar_block<B const> : is_scalar_block<B> {};
template <dimension_type D, typename T>
struct is_scalar_block<vsip_csl::expr::Scalar<D, T> > { static bool const value = true;};

} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{

// These evaluators have the ability to handle compound as well as simple
// expressions.  The former are evaluated recursively by using temporary
// dense views with a value type and dimension order that matches the
// sub-expression being evaluated.  
//
// A certain amount of indirection in the exec() functions is necessary
// to make this work:
//
//   From Evaluator<>::exec(), the Inspect_expr* class is used to
//   determine if a sub-expression needs evaluating into a temporary.
//
//   The proxy_exec() function then checks for aliased arguments 
//   (identical input arguments, e.g. a = a * a) and then invokes
//   the appropriate apply_op_*() function to complete the computation.


/// Unary expression assignment evaluator
template <typename LHS, template <typename> class Operator, typename Block>
struct Evaluator<op::assign<1>, be::cuda,
		 void(LHS &, expr::Unary<Operator, Block, true> const &)>
{
  static char const *name() { return "cuda-unary";}

  typedef expr::Unary<Operator, Block, true> RHS;

  typedef typename impl::adjust_layout_dim<
  1, typename get_block_layout<LHS>::type>::type
    lhs_layout;
  typedef impl::cuda::dda::Data<LHS, dda::out, lhs_layout> lhs_dda_type;

  typedef typename impl::adjust_layout_dim<
  1, typename get_block_layout<Block>::type>::type
    block_layout;
  typedef impl::cuda::dda::Data<Block, dda::in, block_layout> block1_dda_type;

  typedef impl::cuda::Unary<Operator,
    void(typename block1_dda_type::ptr_type,
	 typename lhs_dda_type::ptr_type,
	 length_type)> operation_type;

  static bool const ct_valid = 
    // ensure the cuda backend supports this operation
    operation_type::is_supported &&
    // check that direct access is supported
    dda::Data<LHS, dda::out>::ct_cost == 0 &&
    // the input is either a block that supports direct access, 
    // or it must be an expression that will get evaluated
    (dda::Data<Block, dda::in>::ct_cost == 0 ||
     (impl::is_expr_block<Block>::value && 
      !impl::cuda::is_scalar_block<Block>::value));

  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    typedef Operator<typename Block::value_type> op_type;
    // Note that we assume the argument list doesn't mix interleaved and
    // split complex.
    typedef
      impl::cuda::Size_threshold<op_type,
                                 impl::is_split_block<LHS>::value> threshold_type;
    
    // Note that the direct data access object is the one in the 
    // vsip:: namespace, not the vsip::impl::cuda:: namespace.  Using
    // this version means we do not have to check for aliased pointers.
    // Using the cuda::dda::Data class would mean we would have to check,
    // due to restrictions on having more than one such object constructed
    // on a given block at a time.
    dda::Data<LHS, dda::out, lhs_layout>  dda_lhs(lhs);
    bool lhs_valid = 
      // check if LHS is unit stride
      (dda_lhs.stride(0) == 1) &&
      // check that it is over the threshold, 
      // or that the data is already in device memory
      (dda_lhs.size(0) >= threshold_type::value || 
        impl::cuda::has_valid_device_ptr(rhs.arg()));

    // Assume the RHS is has a valid stride if it is an expression, as
    // it will be a temporary and will be dense.
    bool rhs_valid = true;
    if (!impl::is_expr_block<Block>::value)
    {
      // Since it is not an expression, check for unit stride
      dda::Data<Block, dda::in, block_layout> dda_b(rhs.arg());
      rhs_valid = (dda_b.stride(0) == 1);
    }

    return lhs_valid && rhs_valid;
  }

  // This handles the actual execution of operations whose input 
  // and output blocks are not the same.
  template <typename B1>
  static void apply_op_no_alias(LHS &lhs, B1 const &rhs)
  {
    typedef typename impl::adjust_layout_dim<
        1, typename get_block_layout<B1>::type>::type b1_layout_type;
    typedef impl::cuda::dda::Data<B1, dda::in, b1_layout_type> b1_dda_type;

    lhs_dda_type lhs_dda(lhs);
    b1_dda_type b1_dda(rhs);
    operation_type::exec(b1_dda.ptr(), lhs_dda.ptr(), lhs.size());
  }

  // This handles wrapped blocks (e.g. subviews).
  template <typename B1>
  static void proxy_exec(LHS &lhs, B1 const &arg)
  {
    apply_op_no_alias(lhs, arg);
  }

  // In most cases, the block types match on both sides.
  static void proxy_exec(LHS &lhs, LHS const &arg)
  {
    // LHS = op LHS
    if (impl::is_same_ptr(&lhs, const_cast<LHS*>(&arg)))
    {
      impl::cuda::dda::Data<LHS, dda::inout, lhs_layout> lhs_dda(lhs);
      operation_type::exec(lhs_dda.ptr(), lhs_dda.ptr(), lhs.size());
    }
    // no alias
    else
      apply_op_no_alias(lhs, arg);
  }

  // The default version of this handles cases where the 
  // argument is _not_ an expression, which may be handled as
  // described above.
  template <typename B1,
            bool arg1_is_expr = impl::is_expr_block<Block>::value>
  struct Inspect_expr1
  {
    static void eval(LHS& lhs, B1 const& arg)
    {
      proxy_exec(lhs, arg);
    }
  };

  // This specialization captures cases where the argument _is_
  // an expression.
  template <typename B1>
  struct Inspect_expr1<B1, true>
  {
    // The inner argument is evaluated by assigning it to a temporary.
    // This will dispatch the expression to whatever handler is appropriate
    // (whether in this backend or another) and return the result.
    static void eval(LHS& lhs, B1 const& arg)
    {
      typedef Dense<LHS::dim, typename LHS::value_type, 
        typename get_block_layout<LHS>::order_type> tmp_block_type;
      tmp_block_type tmp(impl::block_domain<LHS::dim>(lhs));
      impl::assign<1>(tmp, arg);

      apply_op_no_alias(lhs, tmp);
    }
  };

  // This wrapper class provides for two paths the code can take.
  // 
  // If the argument is an expression, a temporary is created to hold
  // the result after evaluation.
  //
  // If not, then the overloaded proxy_exec() function is called, which further
  // discriminates based on whether it is an actual block or a sub-block,
  // and, if it is an actual block, handle the case where the operation
  // is being done in place (i.e. the input and output alias one another).
  static void exec(LHS &lhs, RHS const &rhs)
  { 
    Inspect_expr1<Block>::eval(lhs, rhs.arg());
  }
};


/// Binary expression assignment evaluator
template <typename LHS,
          template <typename, typename> class Operator,
	  typename Block1,
	  typename Block2>
struct Evaluator<op::assign<1>, be::cuda,
		 void(LHS &, expr::Binary<Operator, Block1, Block2, true> const &)>
{
  static char const *name() { return "cuda-binary";}

  typedef expr::Binary<Operator, Block1, Block2, true> RHS;

  typedef typename impl::adjust_layout_dim<
    1, typename get_block_layout<LHS>::type>::type lhs_layout;
  typedef impl::cuda::dda::Data<LHS, dda::out, lhs_layout> lhs_dda_type;

  typedef impl::cuda::dda::Data<LHS, dda::in, lhs_layout> lhs_in_dda_type;

  typedef Dense<LHS::dim, typename LHS::value_type, 
    typename get_block_layout<LHS>::order_type> tmp_block_type;

  typedef typename impl::adjust_layout_dim<
    1, typename get_block_layout<Block1>::type>::type block1_layout;
  typedef impl::cuda::dda::Data<Block1, dda::in, block1_layout> block1_dda_type;

  typedef typename impl::adjust_layout_dim<
    1, typename get_block_layout<Block2>::type>::type  block2_layout;
  typedef impl::cuda::dda::Data<Block2, dda::in, block2_layout> block2_dda_type;

  typedef impl::cuda::Binary<Operator,
    void(typename block1_dda_type::ptr_type,
	 typename block2_dda_type::ptr_type,
	 typename lhs_dda_type::ptr_type,
	 length_type)> operation_type;

  static bool const ct_valid = 
    // ensure the cuda backend supports this operation
    operation_type::is_supported &&
    // check that direct access is supported
    dda::Data<LHS, dda::out>::ct_cost == 0 &&
    // the inputs are either blocks that supports direct access, 
    // or they must be expressions that will get evaluated
    (dda::Data<Block1, dda::in>::ct_cost == 0 || 
     (impl::is_expr_block<Block1>::value &&
      !impl::cuda::is_scalar_block<Block1>::value)) &&
    (dda::Data<Block2, dda::in>::ct_cost == 0 ||
     (impl::is_expr_block<Block2>::value &&
      !impl::cuda::is_scalar_block<Block2>::value));

  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    typedef Operator<typename Block1::value_type,
                     typename Block2::value_type> op_type;
    // Note that we assume the argument list doesn't mix interleaved and
    // split complex.
    typedef
      impl::cuda::Size_threshold<op_type,
                                 impl::is_split_block<LHS>::value> threshold_type;
    // check if all data is unit stride
    dda::Data<LHS, dda::out, lhs_layout>  ext_lhs(lhs);
    dda::Data<Block1, dda::in, block1_layout> ext_l(rhs.arg1());
    dda::Data<Block2, dda::in, block2_layout> ext_r(rhs.arg2());
    return (ext_lhs.size(0) >= threshold_type::value ||
	    impl::cuda::has_valid_device_ptr(rhs.arg1()) ||
	    impl::cuda::has_valid_device_ptr(rhs.arg2())) &&
           ext_lhs.stride(0) == 1 &&
           ext_l.stride(0) == 1 &&
           ext_r.stride(0) == 1;
  }

  template <typename B1, typename B2>
  static void apply_op_no_alias(LHS &lhs, B1 const &arg1, B2 const &arg2)
  {
    typedef typename impl::adjust_layout_dim<
      1, typename get_block_layout<B1>::type>::type b1_layout;
    typedef impl::cuda::dda::Data<B1, dda::in, block1_layout> b1_dda_type;

    typedef typename impl::adjust_layout_dim<
      1, typename get_block_layout<B2>::type>::type b2_layout;
    typedef impl::cuda::dda::Data<B2, dda::in, block2_layout> b2_dda_type;

    lhs_dda_type lhs_dda(lhs);
    b1_dda_type b1_dda(arg1);
    b2_dda_type b2_dda(arg2);
    operation_type::exec(b1_dda.ptr(),
			 b2_dda.ptr(),
			 lhs_dda.ptr(), lhs.size());
  }

  template <typename B1, typename B2>
  static void proxy_exec(LHS &lhs, B1 const &arg1, B2 const &arg2)
  {
    apply_op_no_alias(lhs, arg1, arg2);
  }

  template <typename B2>
  static void proxy_exec(LHS &lhs, LHS const &arg1, B2 const &arg2)
  {
    // LHS = LHS op ARG2
    if (impl::is_same_ptr(&lhs, const_cast<LHS*>(&arg1)))
    {
      typedef typename impl::adjust_layout_dim<
        1, typename get_block_layout<B2>::type>::type b2_layout;
      typedef impl::cuda::dda::Data<B2, dda::in, b2_layout> b2_dda_type;

      impl::cuda::dda::Data<LHS, dda::inout, lhs_layout> lhs_dda(lhs);
      b2_dda_type b2_dda(arg2);
      operation_type::exec(lhs_dda.ptr(),
			   b2_dda.ptr(),
			   lhs_dda.ptr(), lhs.size());
    }
    else
      apply_op_no_alias(lhs, arg1, arg2);
  }

  template <typename B1>
  static void proxy_exec(LHS &lhs, B1 const &arg1, LHS const &arg2)
  {
    // LHS = ARG1 op LHS
    if (impl::is_same_ptr(&lhs, const_cast<LHS*>(&arg2)))
    {
      typedef typename impl::adjust_layout_dim<
        1, typename get_block_layout<B1>::type>::type b1_layout;
      typedef impl::cuda::dda::Data<B1, dda::in, b1_layout> b1_dda_type;

      impl::cuda::dda::Data<LHS, dda::inout, lhs_layout> lhs_dda(lhs);
      b1_dda_type b1_dda(arg1);
      operation_type::exec(b1_dda.ptr(),
			   lhs_dda.ptr(),
			   lhs_dda.ptr(), lhs.size());
    }
    else
      apply_op_no_alias(lhs, arg1, arg2);
  }

  static void proxy_exec(LHS &lhs, LHS const &arg1, LHS const &arg2)
  {
    // LHS = LHS op LHS
    if (impl::is_same_ptr(&lhs, const_cast<LHS*>(&arg1)))
    {
      impl::cuda::dda::Data<LHS, dda::inout, lhs_layout> lhs_dda(lhs);
      lhs_in_dda_type b2_dda(arg2);
      operation_type::exec(lhs_dda.ptr(),
			   b2_dda.ptr(),
			   lhs_dda.ptr(), lhs.size());
    }
    else if (impl::is_same_ptr(&lhs, const_cast<LHS*>(&arg2)))
    {
      impl::cuda::dda::Data<LHS, dda::inout, lhs_layout> lhs_dda(lhs);
      lhs_in_dda_type b1_dda(arg1);
      operation_type::exec(b1_dda.ptr(),
			   lhs_dda.ptr(),
			   lhs_dda.ptr(), lhs.size());
    }
    else
      apply_op_no_alias(lhs, arg1, arg2);
  }

  // The default version of this handles cases where the 
  // second argument is not an expression
  template <typename B1,
            typename B2,
            bool arg2_is_expr = impl::is_expr_block<B2>::value>
  struct Inspect_expr2
  {
    static void eval(LHS& lhs, B1 const& arg1, B2 const& arg2)
    {
      proxy_exec(lhs, arg1, arg2);
    }
  };

  // This specialization captures cases where the second argument is
  // an expression.
  template <typename B1, typename B2>
  struct Inspect_expr2<B1, B2, true>
  {
    static void eval(LHS& lhs, B1 const& arg1, B2 const& arg2)
    {
      // The second argument is evaluated by assigning it to a temporary.
      tmp_block_type tmp(impl::block_domain<LHS::dim>(lhs));
      impl::assign<1>(tmp, arg2);

      proxy_exec(lhs, arg1, tmp);
    }
  };

  // The default version of this handles cases where the 
  // first argument is not an expression
  template <typename B1, 
            typename B2,
            bool arg1_is_expr = impl::is_expr_block<B1>::value>
  struct Inspect_expr1
  {
    static void eval(LHS& lhs, B1 const& arg1, B2 const& arg2)
    {
      Inspect_expr2<B1, B2>::eval(lhs, arg1, arg2);
    }
  };

  // This specialization captures cases where the first argument is
  // an expression.
  template <typename B1, typename B2>
  struct Inspect_expr1<B1, B2, true>
  {
    static void eval(LHS& lhs, B1 const& arg1, B2 const& arg2)
    {
      // The first argument is evaluated by assigning it to a temporary.
      tmp_block_type tmp(impl::block_domain<LHS::dim>(lhs));
      impl::assign<1>(tmp, arg1);

      Inspect_expr2<tmp_block_type, B2>::eval(lhs, tmp, arg2);
    }
  };

  static void exec(LHS &lhs, RHS const &rhs)
  {
    // As in the Unary assignment case, the arguments may or may not be
    // expression blocks.  The arguments are checked in turn and evaluated
    // into temporaries if needed.
    Inspect_expr1<Block1, Block2>::eval(lhs, rhs.arg1(), rhs.arg2());
  }
};


/// Ternary expression assignment evaluator
template <typename LHS,
          template <typename, typename, typename> class Operator,
	  typename Block1,
	  typename Block2,
	  typename Block3>
struct Evaluator<op::assign<1>, be::cuda,
		 void(LHS &,
		      expr::Ternary<Operator, Block1, Block2, Block3, true> const &)>
{
  static char const *name() { return "cuda-ternary";}

  typedef expr::Ternary<Operator, Block1, Block2, Block3, true> RHS;

  typedef typename impl::adjust_layout_dim<
    1, typename get_block_layout<LHS>::type>::type
    lhs_layout;
  typedef impl::cuda::dda::Data<LHS, dda::out, lhs_layout> lhs_dda_type;

  typedef impl::cuda::dda::Data<LHS, dda::in, lhs_layout> lhs_in_dda_type;

  typedef Dense<LHS::dim, typename LHS::value_type, 
    typename get_block_layout<LHS>::order_type> tmp_block_type;

  typedef typename impl::adjust_layout_dim<
    1, typename get_block_layout<Block1>::type>::type
    block1_layout;
  typedef impl::cuda::dda::Data<Block1, dda::in, block1_layout> block1_dda_type;

  typedef typename impl::adjust_layout_dim<
    1, typename get_block_layout<Block2>::type>::type
    block2_layout;
  typedef impl::cuda::dda::Data<Block2, dda::in, block2_layout> block2_dda_type;

  typedef typename impl::adjust_layout_dim<
    1, typename get_block_layout<Block3>::type>::type
    block3_layout;
  typedef impl::cuda::dda::Data<Block3, dda::in, block3_layout> block3_dda_type;

  typedef impl::cuda::Ternary<Operator,
    void(typename block1_dda_type::ptr_type,
	 typename block2_dda_type::ptr_type,
	 typename block3_dda_type::ptr_type,
	 typename lhs_dda_type::ptr_type,
	 length_type)> operation_type;

  static bool const ct_valid = 
    operation_type::is_supported &&
    // check that direct access is supported
    dda::Data<LHS, dda::out>::ct_cost == 0 &&
    (dda::Data<Block1, dda::in>::ct_cost == 0 ||
     (impl::is_expr_block<Block1>::value &&
      !impl::cuda::is_scalar_block<Block1>::value)) &&
    (dda::Data<Block2, dda::in>::ct_cost == 0 ||
     (impl::is_expr_block<Block2>::value &&
      !impl::cuda::is_scalar_block<Block2>::value)) &&
    (dda::Data<Block3, dda::in>::ct_cost == 0 ||
     (impl::is_expr_block<Block3>::value &&
      !impl::cuda::is_scalar_block<Block3>::value));

  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    typedef Operator<typename Block1::value_type,
                     typename Block2::value_type,
		     typename Block3::value_type> op_type;
    // Note that we assume the argument list doesn't mix interleaved and
    // split complex.
    typedef
      impl::cuda::Size_threshold<op_type,
                                 impl::is_split_block<LHS>::value> threshold_type;
    // check if all data is unit stride
    dda::Data<LHS, dda::out, lhs_layout>  ext_lhs(lhs);
    dda::Data<Block1, dda::in, block1_layout> ext_1(rhs.arg1());
    dda::Data<Block2, dda::in, block2_layout> ext_2(rhs.arg2());
    dda::Data<Block3, dda::in, block3_layout> ext_3(rhs.arg3());
    return (ext_lhs.size(0) >= threshold_type::value ||
	    impl::cuda::has_valid_device_ptr(rhs.arg3()) ||
	    impl::cuda::has_valid_device_ptr(rhs.arg2()) ||
	    impl::cuda::has_valid_device_ptr(rhs.arg1())) &&
           ext_lhs.stride(0) == 1 &&
           ext_1.stride(0) == 1 &&
           ext_2.stride(0) == 1 &&
           ext_3.stride(0) == 1;
  }

  template <typename B1, typename B2, typename B3>
  static void apply_op_no_alias(LHS &lhs,
			    B1 const &arg1,
			    B2 const &arg2,
			    B3 const &arg3)
  {
    typedef typename impl::adjust_layout_dim<
      1, typename get_block_layout<B1>::type>::type b1_layout;
    typedef impl::cuda::dda::Data<B1, dda::in, b1_layout> b1_dda_type;

    typedef typename impl::adjust_layout_dim<
      1, typename get_block_layout<B2>::type>::type b2_layout;
    typedef impl::cuda::dda::Data<B2, dda::in, b2_layout> b2_dda_type;

    typedef typename impl::adjust_layout_dim<
      1, typename get_block_layout<B3>::type>::type b3_layout;
    typedef impl::cuda::dda::Data<B3, dda::in, b3_layout> b3_dda_type;

    lhs_dda_type lhs_dda(lhs);
    b1_dda_type b1_dda(arg1);
    b2_dda_type b2_dda(arg2);
    b3_dda_type b3_dda(arg3);
    operation_type::exec(b1_dda.ptr(),
			 b2_dda.ptr(),
			 b3_dda.ptr(),
			 lhs_dda.ptr(), lhs.size());
  }

  template <typename B2, typename B3>
  static void apply_op_alias_a1(LHS &lhs, B2 const &arg2, B3 const &arg3)
  {
    typedef typename impl::adjust_layout_dim<
      1, typename get_block_layout<B2>::type>::type b2_layout;
    typedef impl::cuda::dda::Data<B2, dda::in, b2_layout> b2_dda_type;

    typedef typename impl::adjust_layout_dim<
      1, typename get_block_layout<B3>::type>::type b3_layout;
    typedef impl::cuda::dda::Data<B3, dda::in, b3_layout> b3_dda_type;

    impl::cuda::dda::Data<LHS, dda::inout, lhs_layout> lhs_dda(lhs);
    b2_dda_type b2_dda(arg2);
    b3_dda_type b3_dda(arg3);
    operation_type::exec(lhs_dda.ptr(),
			 b2_dda.ptr(),
			 b3_dda.ptr(),
			 lhs_dda.ptr(), lhs.size());
  }

  template <typename B1, typename B3>
  static void apply_op_alias_a2(LHS &lhs, B1 const &arg1, B3 const &arg3)
  {
    typedef typename impl::adjust_layout_dim<
      1, typename get_block_layout<B1>::type>::type b1_layout;
    typedef impl::cuda::dda::Data<B1, dda::in, b1_layout> b1_dda_type;

    typedef typename impl::adjust_layout_dim<
      1, typename get_block_layout<B3>::type>::type b3_layout;
    typedef impl::cuda::dda::Data<B3, dda::in, b3_layout> b3_dda_type;

    impl::cuda::dda::Data<LHS, dda::inout, lhs_layout> lhs_dda(lhs);
    b1_dda_type b1_dda(arg1);
    b3_dda_type b3_dda(arg3);
    operation_type::exec(b1_dda.ptr(),
			 lhs_dda.ptr(),
			 b3_dda.ptr(),
			 lhs_dda.ptr(), lhs.size());
  }

  template <typename B1, typename B2>
  static void apply_op_alias_a3(LHS &lhs, B1 const &arg1, B2 const &arg2)
  {
    typedef typename impl::adjust_layout_dim<
      1, typename get_block_layout<B1>::type>::type b1_layout;
    typedef impl::cuda::dda::Data<B1, dda::in, b1_layout> b1_dda_type;

    typedef typename impl::adjust_layout_dim<
      1, typename get_block_layout<B2>::type>::type b2_layout;
    typedef impl::cuda::dda::Data<B2, dda::in, b2_layout> b2_dda_type;

    impl::cuda::dda::Data<LHS, dda::inout, lhs_layout> lhs_dda(lhs);
    b1_dda_type b1_dda(arg1);
    b2_dda_type b2_dda(arg2);
    operation_type::exec(b1_dda.ptr(),
			 b2_dda.ptr(),
			 lhs_dda.ptr(),
			 lhs_dda.ptr(), lhs.size());
  }

  template <typename B1, typename B2, typename B3>
  static void proxy_exec(LHS &lhs, B1 const &arg1, B2 const &arg2, B3 const &arg3)
  {
    apply_op_no_alias(lhs, arg1, arg2, arg3);
  }

  template <typename B2, typename B3>
  static void proxy_exec(LHS &lhs, LHS const &arg1, B2 const &arg2, B3 const &arg3)
  {
    if (impl::is_same_ptr(&lhs, const_cast<LHS*>(&arg1)))
      apply_op_alias_a1(lhs, arg2, arg3);
    else
      apply_op_no_alias(lhs, arg1, arg2, arg3);
  }

  template <typename B1, typename B3>
  static void proxy_exec(LHS &lhs, B1 const &arg1, LHS const &arg2, B3 const &arg3)
  {
    if (impl::is_same_ptr(&lhs, const_cast<LHS*>(&arg2)))
      apply_op_alias_a2(lhs, arg1, arg3);
    else
      apply_op_no_alias(lhs, arg1, arg2, arg3);
  }

  template <typename B1, typename B2>
  static void proxy_exec(LHS &lhs, B1 const &arg1, B2 const &arg2, LHS const &arg3)
  {
    if (impl::is_same_ptr(&lhs, const_cast<LHS*>(&arg3)))
      apply_op_alias_a3(lhs, arg1, arg2);
    else
      apply_op_no_alias(lhs, arg1, arg2, arg3);
  }

  template <typename B3>
  static void proxy_exec(LHS &lhs, LHS const &arg1, LHS const &arg2, B3 const &arg3)
  {
    if (impl::is_same_ptr(&lhs, const_cast<LHS*>(&arg1)))
      apply_op_alias_a1(lhs, arg2, arg3);
    else if (impl::is_same_ptr(&lhs, const_cast<LHS*>(&arg2)))
      apply_op_alias_a2(lhs, arg1, arg3);
    else
      apply_op_no_alias(lhs, arg1, arg2, arg3);
  }

  template <typename B2>
  static void proxy_exec(LHS &lhs, LHS const &arg1, B2 const &arg2, LHS const &arg3)
  {
    if (impl::is_same_ptr(&lhs, const_cast<LHS*>(&arg1)))
      apply_op_alias_a1(lhs, arg2, arg3);
    else if (impl::is_same_ptr(&lhs, const_cast<LHS*>(&arg3)))
      apply_op_alias_a3(lhs, arg1, arg2);
    else
      apply_op_no_alias(lhs, arg1, arg2, arg3);
  }

  template <typename B1>
  static void proxy_exec(LHS &lhs, B1 const &arg1, LHS const &arg2, LHS const &arg3)
  {
    if (impl::is_same_ptr(&lhs, const_cast<LHS*>(&arg2)))
      apply_op_alias_a2(lhs, arg1, arg3);
    else if (impl::is_same_ptr(&lhs, const_cast<LHS*>(&arg3)))
      apply_op_alias_a3(lhs, arg1, arg2);
    else
      apply_op_no_alias(lhs, arg1, arg2, arg3);
  }

  static void proxy_exec(LHS &lhs, LHS const &arg1, LHS const &arg2, LHS const &arg3)
  {
    if (impl::is_same_ptr(&lhs, const_cast<LHS*>(&arg1)))
      apply_op_alias_a1(lhs, arg2, arg3);
    else if (impl::is_same_ptr(&lhs, const_cast<LHS*>(&arg2)))
      apply_op_alias_a2(lhs, arg1, arg3);
    else if (impl::is_same_ptr(&lhs, const_cast<LHS*>(&arg3)))
      apply_op_alias_a3(lhs, arg1, arg2);
    else
      apply_op_no_alias(lhs, arg1, arg2, arg3);
  }


  // The default version of this handles cases where the 
  // third argument is not an expression
  template <typename B1,
            typename B2,
            typename B3,
            bool arg3_is_expr = impl::is_expr_block<B3>::value>
  struct Inspect_expr3
  {
    static void eval(LHS& lhs, B1 const& arg1, B2 const& arg2, B3 const& arg3)
    {
      proxy_exec(lhs, arg1, arg2, arg3);
    }
  };

  // This specialization captures cases where the third argument is
  // an expression.
  template <typename B1, typename B2, typename B3>
  struct Inspect_expr3<B1, B2, B3, true>
  {
    static void eval(LHS& lhs, B1 const& arg1, B2 const& arg2, B3 const& arg3)
    {
      // The third argument is evaluated by assigning it to a temporary.
      tmp_block_type tmp(impl::block_domain<LHS::dim>(lhs));
      impl::assign<1>(tmp, arg3);

      proxy_exec(lhs, arg1, arg2, tmp);
    }
  };


  // The default version of this handles cases where the 
  // second argument is not an expression
  template <typename B1,
            typename B2,
            typename B3,
            bool arg2_is_expr = impl::is_expr_block<B2>::value>
  struct Inspect_expr2
  {
    static void eval(LHS& lhs, B1 const& arg1, B2 const& arg2, B3 const& arg3)
    {
      Inspect_expr3<B1, B2, B3>::eval(lhs, arg1, arg2, arg3);
    }
  };

  // This specialization captures cases where the second argument is
  // an expression.
  template <typename B1, typename B2, typename B3>
  struct Inspect_expr2<B1, B2, B3, true>
  {
    static void eval(LHS& lhs, B1 const& arg1, B2 const& arg2, B3 const& arg3)
    {
      // The second argument is evaluated by assigning it to a temporary.
      tmp_block_type tmp(impl::block_domain<LHS::dim>(lhs));
      impl::assign<1>(tmp, arg2);

      Inspect_expr3<B1, tmp_block_type, B3>::eval(lhs, arg1, tmp, arg3);
    }
  };


  // The default version of this handles cases where the 
  // first argument is not an expression
  template <typename B1, 
            typename B2,
            typename B3,
            bool arg1_is_expr = impl::is_expr_block<B1>::value>
  struct Inspect_expr1
  {
    static void eval(LHS& lhs, B1 const& arg1, B2 const& arg2, B3 const& arg3)
    {
      Inspect_expr2<B1, B2, B3>::eval(lhs, arg1, arg2, arg3);
    }
  };

  // This specialization captures cases where the first argument is
  // an expression.
  template <typename B1, typename B2, typename B3>
  struct Inspect_expr1<B1, B2, B3, true>
  {
    static void eval(LHS& lhs, B1 const& arg1, B2 const& arg2, B3 const& arg3)
    {
      // The first argument is evaluated by assigning it to a temporary.
      tmp_block_type tmp(impl::block_domain<LHS::dim>(lhs));
      impl::assign<1>(tmp, arg1);

      Inspect_expr2<tmp_block_type, B2, B3>::eval(lhs, tmp, arg2, arg3);
    }
  };


  static void exec(LHS &lhs, RHS const &rhs)
  {
    // The arguments here may or may not be expression blocks.  The arguments 
    // are checked in turn and evaluated into temporaries if needed.
    Inspect_expr1<Block1, Block2, Block3>::eval(lhs, rhs.arg1(), rhs.arg2(), rhs.arg3());
  }
};




template <typename LHS, typename RHS>
struct Evaluator<op::assign<2>, be::cuda, void(LHS &, RHS const &)>
{
  static char const *name() { return "Expr_CUDA";}
  
  typedef typename LHS::value_type value_type;

  static bool const ct_valid = 
    is_same<value_type, typename RHS::value_type>::value &&
    (is_same<value_type, float>::value || is_same<value_type, complex<float> >::value) &&
    !impl::is_expr_block<RHS>::value && 
    is_packing_unit_stride<get_block_layout<LHS>::packing>::value &&
    is_packing_unit_stride<get_block_layout<RHS>::packing>::value;

  static bool rt_valid(LHS &, RHS const &) { return true;}

  static bool const same_order = 
    is_same<typename get_block_layout<LHS>::order_type,
	    typename get_block_layout<RHS>::order_type>::value;

  /// LHS = RHS with no dimension reorder.
  static void exec(LHS &lhs, RHS const &rhs, true_type)
  {

    typedef typename get_block_layout<LHS>::order_type order_type;
    dimension_type const l_dim0 = order_type::impl_dim0;
    dimension_type const l_dim1 = order_type::impl_dim1;

    impl::cuda::dda::Data<LHS, dda::out> lhs_dev(lhs);
    impl::cuda::dda::Data<RHS, dda::in> rhs_dev(rhs);

    impl::cuda::copy(rhs_dev.ptr(), rhs_dev.stride(l_dim0),
		     lhs_dev.ptr(), lhs_dev.stride(l_dim0),
		     lhs_dev.size(l_dim0), lhs_dev.size(l_dim1));
  }
  /// LHS = RHS with dimension reorder.
  static void exec(LHS &lhs, RHS const &rhs, false_type)
  {

    typedef typename get_block_layout<LHS>::order_type lhs_order_type;
    dimension_type const l_dim0 = lhs_order_type::impl_dim0;
    dimension_type const l_dim1 = lhs_order_type::impl_dim1;

    impl::cuda::dda::Data<LHS, dda::out> lhs_dev(lhs);
    impl::cuda::dda::Data<RHS, dda::in> rhs_dev(rhs);

    impl::cuda::transpose(rhs_dev.ptr(), lhs_dev.ptr(),
			  lhs_dev.size(l_dim0), lhs_dev.size(l_dim1));
  }
  static void exec(LHS &lhs, RHS const &rhs)
  {
    typedef integral_constant<bool, same_order> copy_tag;
    exec(lhs, rhs, copy_tag());
  }
}; 

template <typename LHS, typename Block>
struct Evaluator<op::assign<2>, be::cuda, 
  void(LHS &, impl::Transposed_block<Block> const &)>
{
  typedef impl::Transposed_block<Block> RHS;

  typedef typename get_block_layout<LHS>::order_type order_type;

  typedef typename LHS::value_type lhs_value_type;
  typedef typename RHS::value_type rhs_value_type;

  static char const *name() { return "Expr_CUDA_Trans (copy)";}

  static bool const is_rhs_expr   = impl::is_expr_block<RHS>::value;

  static bool const is_lhs_split  = impl::is_split_block<LHS>::value;
  static bool const is_rhs_split  = impl::is_split_block<RHS>::value;

  static int const  lhs_cost      = dda::Data<LHS, dda::out>::ct_cost;
  static int const  rhs_cost      = dda::Data<RHS, dda::in>::ct_cost;

  static bool const ct_valid =
    // check that types are equal
    is_same<rhs_value_type, lhs_value_type>::value &&
    // check that CUDA supports this data type
    impl::cuda::Traits<rhs_value_type>::valid &&
    // check that the source block is not an expression
    !is_rhs_expr &&
    // check that direct access is supported
    lhs_cost == 0 && rhs_cost == 0 &&
    // check complex layout is not split (either real or interleaved are ok)
    !is_lhs_split &&
    !is_rhs_split;

  static bool rt_valid(LHS &lhs, RHS const &rhs)
  { 
    dda::Data<LHS, dda::out> dev_lhs(lhs);
    dda::Data<RHS, dda::in> dev_rhs(rhs);

    return 
      // Both source and destination blocks must be unit-stride
      (dev_lhs.stride(order_type::impl_dim1) == 1) &&
      (dev_rhs.stride(order_type::impl_dim1) == 1);
  }

  static bool const same_order = 
    is_same<typename get_block_layout<LHS>::order_type,
	    typename get_block_layout<RHS>::order_type>::value;

  /// "A = B.transpose()", with A and B having different dimension order.
  static void exec(LHS &lhs, RHS const &rhs, true_type)
  {
    impl::cuda::dda::Data<LHS, dda::out> dev_lhs(lhs);
    impl::cuda::dda::Data<Block, dda::in> dev_rhs(rhs.impl_block());

    impl::cuda::copy(dev_rhs.ptr(), dev_rhs.stride(order_type::impl_dim1),
		     dev_lhs.ptr(), dev_lhs.stride(order_type::impl_dim0),
		     dev_lhs.size(order_type::impl_dim0), dev_lhs.size(order_type::impl_dim1));
  }
  /// "A = B.transpose()", with A and B having the same dimension order.
  static void exec(LHS &lhs, RHS const &rhs, false_type)
  {
    if (impl::is_same_ptr(&lhs, const_cast<Block*>(&rhs.impl_block())))
    {
      assert(lhs.size(2, 0) == lhs.size(2, 1));
      impl::cuda::dda::Data<LHS, dda::inout> dev_lhs(lhs);
      impl::cuda::transpose(dev_lhs.ptr(), dev_lhs.size(0));
    }
    else
    {
      impl::cuda::dda::Data<LHS, dda::out> dev_lhs(lhs);
      impl::cuda::dda::Data<Block, dda::in> dev_rhs(rhs.impl_block());

      impl::cuda::transpose(dev_rhs.ptr(), dev_lhs.ptr(),
			    dev_lhs.size(order_type::impl_dim0),
			    dev_lhs.size(order_type::impl_dim1));
    }
  }
  static void exec(LHS &lhs, RHS const &rhs)
  {
    typedef integral_constant<bool, same_order> copy_tag;
    exec(lhs, rhs, copy_tag());
  }
};

template <typename T, typename O>
struct Evaluator<op::assign<2>, be::cuda, 
  void(impl::Subset_block<Dense<2, T, O> >&, 
       impl::Subset_block<Dense<2, T, O> > const&)>
{
  typedef impl::Subset_block<Dense<2, T, O> > LHS;
  typedef impl::Subset_block<Dense<2, T, O> > RHS;

  typedef O order_type;

  static char const *name() { return "Expr_CUDA_sub_copy (block)";}

  static bool const ct_valid = 
    // TODO: Generalize this evaluator to properly handle arbitrary
    //       dimension order.
    is_same<order_type, tuple<1,0,2> >::value;

  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    // Blocks must be unit stride in the last dimension.  Strides should equal
    // or exceed the bounds of the subblock.  The dimension order of the two
    // blocks must be the same.
    dda::Data<LHS, dda::out> ext_lhs(lhs);
    dda::Data<RHS, dda::in> ext_rhs(rhs);

    dimension_type const dim0 = order_type::impl_dim0;
    dimension_type const dim1 = order_type::impl_dim1;

    return 
      (ext_lhs.stride(dim1) == 1) &&
      (ext_rhs.stride(dim1) == 1) &&
      (ext_lhs.stride(dim0) >= static_cast<stride_type>(ext_lhs.size(dim1))) &&
      (ext_rhs.stride(dim0) >= static_cast<stride_type>(ext_rhs.size(dim1)));
  }

  static void exec(LHS &lhs, RHS const &rhs)
  {
    impl::cuda::dda::Data<Dense<2, T, O>, dda::out> 
      dev_lhs(lhs.impl_block());
    impl::cuda::dda::Data<Dense<2, T, O>, dda::in> dev_rhs(rhs.impl_block());

    // Careful: the subblock 'impl_domain()' return logical indices
    //          while 'stride()' reports physical layout !
    Domain<2> const &lhs_dom = lhs.impl_domain();
    Domain<2> const &rhs_dom = rhs.impl_domain();
    
    index_type rhs_offset = 
      rhs_dom[dim0].first() * dev_rhs.stride(dim0) +
      rhs_dom[dim1].first() * dev_rhs.stride(dim1);

    index_type lhs_offset = 
      lhs_dom[dim0].first() * dev_lhs.stride(dim0) +
      lhs_dom[dim1].first() * dev_lhs.stride(dim1);

    T const *rhs_data = dev_rhs.ptr() + rhs_offset;
    T *lhs_data = dev_lhs.ptr() + lhs_offset;

    impl::cuda::copy(rhs_data, dev_rhs.stride(dim0),
		     lhs_data, dev_lhs.stride(dim0),
		     lhs_dom[dim0].size(), lhs_dom[dim1].size());
  }
};

template <typename T, typename O>
struct Evaluator<op::assign<2>, be::cuda, 
  void(impl::Subset_block<Dense<2, T, O> >&, expr::Scalar<2, T> const&)>
{
  typedef impl::Subset_block<Dense<2, T, O> > LHS;
  typedef expr::Scalar<2, T> RHS;

  typedef O lhs_order_type;

  static char const* name() { return "Expr_CUDA_sub_copy (scalar)"; }

  static bool const ct_valid = 
    is_same<T, float>::value ||
    is_same<T, complex<float> >::value;

  static dimension_type const lhs_dim0 = lhs_order_type::impl_dim0;
  static dimension_type const lhs_dim1 = lhs_order_type::impl_dim1;

  static bool rt_valid(LHS &lhs, RHS const &)
  {
    // Blocks must be unit stride in the last dimension.  Strides should equal
    // or exceed the bounds of the subblock.
    dda::Data<LHS, dda::out> ext_lhs(lhs);
    return 
      (ext_lhs.stride(lhs_dim1) == 1) &&
      (ext_lhs.stride(lhs_dim0) >= static_cast<stride_type>(ext_lhs.size(lhs_dim1)));
  }

  static void exec(LHS &lhs, RHS const &rhs)
  {
    impl::cuda::dda::Data<Dense<2, T, O>, dda::out> 
      dev_lhs(lhs.impl_block());

    // Careful: the subblock 'impl_domain()' return logical indices
    //          while 'stride()' reports physical layout !
    Domain<2> const &lhs_dom = lhs.impl_domain();
    
    index_type offset = 
      lhs_dom[lhs_dim0].first() * dev_lhs.stride(lhs_dim0) +
      lhs_dom[lhs_dim1].first() * dev_lhs.stride(lhs_dim1);

    impl::cuda::assign_scalar(rhs.value(), dev_lhs.ptr() + offset, 
     			      dev_lhs.stride(lhs_dim0) * lhs_dom[lhs_dim0].stride(),
			      lhs_dom[lhs_dim0].size(), lhs_dom[lhs_dim1].size());
  }
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl


#endif
