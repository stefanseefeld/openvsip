/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/ipp/bindings.hpp
    @author  Stefan Seefeld
    @date    2005-08-10
    @brief   VSIPL++ Library: Wrappers and traits to bridge with Intel's IPP.
*/

#ifndef VSIP_IMPL_IPP_HPP
#define VSIP_IMPL_IPP_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/support.hpp>
#include <vsip/opt/expr/assign_fwd.hpp>

namespace vsip
{
namespace impl
{
namespace ipp
{

template <typename Type>
struct Is_type_supported
{
  static bool const value = false;
};

template <>
struct Is_type_supported<float>
{
  static bool const value = true;
};

template <>
struct Is_type_supported<double>
{
  static bool const value = true;
};

template <>
struct Is_type_supported<std::complex<float> >
{
  static bool const value = true;
};

template <>
struct Is_type_supported<std::complex<double> >
{
  static bool const value = true;
};

#define VSIP_IMPL_IPP_DECL_V(FCN, T, IPPFCN, IPPT)			\
void									\
FCN(T const *A, T *Z, length_type len);

#define VSIP_IMPL_IPP_DECL_V_CR(FCN, T, IPPFCN, IPPCT, IPPT)		\
void									\
FCN(complex<T> const *A, T *Z, length_type len);

// Square
VSIP_IMPL_IPP_DECL_V(vsq,  float,           ippsSqr_32f,  Ipp32f)
VSIP_IMPL_IPP_DECL_V(vsq,  double,          ippsSqr_64f,  Ipp64f)
VSIP_IMPL_IPP_DECL_V(vsq,  complex<float>,  ippsSqr_32fc, Ipp32fc)
VSIP_IMPL_IPP_DECL_V(vsq,  complex<double>, ippsSqr_64fc, Ipp64fc)

// Square-root
VSIP_IMPL_IPP_DECL_V(vsqrt, float,           ippsSqrt_32f,  Ipp32f)
VSIP_IMPL_IPP_DECL_V(vsqrt, double,          ippsSqrt_64f,  Ipp64f)
VSIP_IMPL_IPP_DECL_V(vsqrt, complex<float>,  ippsSqrt_32fc, Ipp32fc)
VSIP_IMPL_IPP_DECL_V(vsqrt, complex<double>, ippsSqrt_64fc, Ipp64fc)

// Mag 
VSIP_IMPL_IPP_DECL_V(vmag, float,           ippsAbs_32f,  Ipp32f)
VSIP_IMPL_IPP_DECL_V(vmag, double,          ippsAbs_64f,  Ipp64f)
VSIP_IMPL_IPP_DECL_V_CR(vmag, float,  ippsMagnitude_32f, Ipp32fc, Ipp32f)
VSIP_IMPL_IPP_DECL_V_CR(vmag, double, ippsMagnitude_64f, Ipp64fc, Ipp64f)

// Mag-sq
VSIP_IMPL_IPP_DECL_V(vmagsq,    float,  ippsSqr_32f,  Ipp32f)
VSIP_IMPL_IPP_DECL_V(vmagsq,    double, ippsSqr_64f,  Ipp64f)

// functions for vector addition
void vadd(float const* A, float const* B, float* Z, length_type len);
void vadd(double const* A, double const* B, double* Z, length_type len);
void vadd(std::complex<float> const* A, std::complex<float> const* B,
          std::complex<float>* Z, length_type len);
void vadd(std::complex<double> const* A, std::complex<double> const* B,
          std::complex<double>* Z, length_type len);

// functions for vector copy
void vcopy(float const* A, float* Z, length_type len);
void vcopy(double const* A, double* Z, length_type len);
void vcopy(complex<float> const* A, complex<float>* Z, length_type len);
void vcopy(complex<double> const* A, complex<double>* Z, length_type len);

// functions for vector subtraction
void vsub(float const* A, float const* B, float* Z, length_type len);
void vsub(double const* A, double const* B, double* Z, length_type len);
void vsub(std::complex<float> const* A, std::complex<float> const* B,
          std::complex<float>* Z, length_type len);
void vsub(std::complex<double> const* A, std::complex<double> const* B,
          std::complex<double>* Z, length_type len);

// functions for vector multiply
void vmul(float const* A, float const* B, float* Z, length_type len);
void vmul(double const* A, double const* B, double* Z, length_type len);
void vmul(std::complex<float> const* A, std::complex<float> const* B,
          std::complex<float>* Z, length_type len);
void vmul(std::complex<double> const* A, std::complex<double> const* B,
          std::complex<double>* Z, length_type len);

void svadd(float A, float const* B, float* Z, length_type len);
void svadd(double A, double const* B, double* Z, length_type len);
void svadd(complex<float> A, complex<float> const* B,
	   complex<float>* Z, length_type len);
void svadd(complex<double> A, complex<double> const* B,
	   complex<double>* Z, length_type len);

// sub: scalar - vector
void svsub(float A, float const* B, float* Z, length_type len);
void svsub(double A, double const* B, double* Z, length_type len);
void svsub(complex<float> A, complex<float> const* B,
	   complex<float>* Z, length_type len);
void svsub(complex<double> A, complex<double> const* B,
	   complex<double>* Z, length_type len);

// sub: vector - scalar
void svsub(float const* A, float B, float* Z, length_type len);
void svsub(double const* A, double B, double* Z, length_type len);
void svsub(complex<float> const* A, complex<float> B,
	   complex<float>* Z, length_type len);
void svsub(complex<double> const* A, complex<double> B,
	   complex<double>* Z, length_type len);

void svmul(float A, float const* B, float* Z, length_type len);
void svmul(double A, double const* B, double* Z, length_type len);
void svmul(complex<float> A, complex<float> const* B,
	   complex<float>* Z, length_type len);
void svmul(complex<double> A, complex<double> const* B,
	   complex<double>* Z, length_type len);

// functions for scalar-view division: scalar / vector
void svdiv(float A, float const* B, float* Z, length_type len);

// functions for scalar-view division: vector / scalar
void svdiv(float const* A, float B, float* Z, length_type len);
void svdiv(double const* A, double B, double* Z, length_type len);
void svdiv(complex<float> const* A, complex<float> B,
	   complex<float>* Z, length_type len);
void svdiv(complex<double> const* A, complex<double> B,
	   complex<double>* Z, length_type len);

// functions for vector division
void vdiv(float const* A, float const* B, float* Z, length_type len);
void vdiv(double const* A, double const* B, double* Z, length_type len);
void vdiv(std::complex<float> const* A, std::complex<float> const* B,
          std::complex<float>* Z, length_type len);
void vdiv(std::complex<double> const* A, std::complex<double> const* B,
          std::complex<double>* Z, length_type len);

// Vector zero
void vzero(float*           Z, length_type len);
void vzero(double*          Z, length_type len);
void vzero(complex<float>*  Z, length_type len);
void vzero(complex<double>* Z, length_type len);

// functions for convolution
void conv(float* coeff, length_type coeff_size,
	  float* in,    length_type in_size,
	  float* out);
void conv(double* coeff, length_type coeff_size,
	  double* in,    length_type in_size,
	  double* out);

void
conv_full_2d(
  float*      coeff,
  length_type coeff_rows,
  length_type coeff_cols,
  length_type coeff_row_stride,
  float*      in,
  length_type in_rows,
  length_type in_cols,
  length_type in_row_stride,
  float*      out,
  length_type out_row_stride);

void
conv_full_2d(
  short*      coeff,
  length_type coeff_rows,
  length_type coeff_cols,
  length_type coeff_row_stride,
  short*      in,
  length_type in_rows,
  length_type in_cols,
  length_type in_row_stride,
  short*      out,
  length_type out_row_stride);

void
conv_valid_2d(
  float*      coeff,
  length_type coeff_rows,
  length_type coeff_cols,
  length_type coeff_row_stride,
  float*      in,
  length_type in_rows,
  length_type in_cols,
  length_type in_row_stride,
  float*      out,
  length_type out_row_stride);

void
conv_valid_2d(
  short*      coeff,
  length_type coeff_rows,
  length_type coeff_cols,
  length_type coeff_row_stride,
  short*      in,
  length_type in_rows,
  length_type in_cols,
  length_type in_row_stride,
  short*      out,
  length_type out_row_stride);



template <template <typename, typename> class Operator,
	  typename LHS,
	  typename LBlock,
	  typename RBlock>
struct Evaluator
{
  typedef expr::Binary<Operator, LBlock, RBlock, true> RHS;
  typedef typename LHS::value_type lhs_value_type;
  typedef typename LBlock::value_type l_value_type;
  typedef typename RBlock::value_type r_value_type;

  typedef typename Adjust_layout_dim<
      1, typename Block_layout<LHS>::layout_type>::type
    dst_lp;

  typedef typename Adjust_layout_dim<
      1, typename Block_layout<LBlock>::layout_type>::type
    lblock_lp;

  typedef typename Adjust_layout_dim<
      1, typename Block_layout<RBlock>::layout_type>::type
    rblock_lp;

  static bool const ct_valid = 
    !Is_expr_block<LBlock>::value &&
    !Is_expr_block<RBlock>::value &&
     ipp::Is_type_supported<lhs_value_type>::value &&
     Type_equal<lhs_value_type, l_value_type>::value &&
     Type_equal<lhs_value_type, r_value_type>::value &&
     // check that direct access is supported
     Ext_data_cost<LHS>::value == 0 &&
     Ext_data_cost<LBlock>::value == 0 &&
     Ext_data_cost<RBlock>::value == 0 &&
     /* IPP does not support complex split */
     !Is_split_block<LHS>::value &&
     !Is_split_block<LBlock>::value &&
     !Is_split_block<RBlock>::value;

  
  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    // check if all data is unit stride
    Ext_data<LHS, dst_lp> ext_dst(lhs, SYNC_OUT);
    Ext_data<LBlock, lblock_lp> ext_l(rhs.arg1(), SYNC_IN);
    Ext_data<RBlock, rblock_lp> ext_r(rhs.arg2(), SYNC_IN);
    return (ext_dst.stride(0) == 1 &&
	    ext_l.stride(0) == 1 &&
	    ext_r.stride(0) == 1);
  }
};

template <template <typename, typename> class Operator,
	  typename LHS, typename S, typename B, bool Right>
struct Scalar_evaluator
{
  typedef expr::Binary<Operator, expr::Scalar<1, S>, B, true> RHS;
  typedef typename LHS::value_type lhs_value_type;
  typedef typename Adjust_layout_dim<
      1, typename Block_layout<LHS>::layout_type>::type
    dst_lp;

  typedef typename Adjust_layout_dim<
      1, typename Block_layout<B>::layout_type>::type
    vblock_lp;

  static bool const ct_valid = 
    !Is_expr_block<B>::value &&
     ipp::Is_type_supported<lhs_value_type>::value &&
     Type_equal<lhs_value_type, S>::value &&
     Type_equal<lhs_value_type, typename B::value_type>::value &&
     // check that direct access is supported
     Ext_data_cost<LHS>::value == 0 &&
     Ext_data_cost<B>::value == 0 &&
     // Complex split format is not supported.
     Type_equal<typename Block_layout<LHS>::complex_type,
		Cmplx_inter_fmt>::value;

  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    // check if all data is unit stride
    Ext_data<LHS, dst_lp>  ext_dst(lhs, SYNC_OUT);
    Ext_data<B, vblock_lp> ext_r(rhs.arg2(), SYNC_IN);
    return (ext_dst.stride(0) == 1 && ext_r.stride(0) == 1);
  }

};

template <template <typename, typename> class Operator,
	  typename LHS, typename S, typename B>
struct Scalar_evaluator<Operator, LHS, S, B, false>
{
  typedef expr::Binary<Operator, B, expr::Scalar<1, S>, true>
    RHS;
  typedef typename LHS::value_type lhs_value_type;
  typedef typename Adjust_layout_dim<
      1, typename Block_layout<LHS>::layout_type>::type
    dst_lp;

  typedef typename Adjust_layout_dim<
      1, typename Block_layout<B>::layout_type>::type
    vblock_lp;

  static bool const ct_valid = 
    !Is_expr_block<B>::value &&
     ipp::Is_type_supported<lhs_value_type>::value &&
     Type_equal<lhs_value_type, S>::value &&
     Type_equal<lhs_value_type, typename B::value_type>::value &&
     // check that direct access is supported
     Ext_data_cost<LHS>::value == 0 &&
     Ext_data_cost<B>::value == 0 &&
     // Complex split format is not supported.
     Type_equal<typename Block_layout<LHS>::complex_type,
		Cmplx_inter_fmt>::value;

  static bool rt_valid(LHS &lhs, RHS const &rhs)
  {
    // check if all data is unit stride
    Ext_data<LHS, dst_lp>  ext_dst(lhs, SYNC_OUT);
    Ext_data<B, vblock_lp> ext_l(rhs.arg1(), SYNC_IN);
    return (ext_dst.stride(0) == 1 && ext_l.stride(0) == 1);
  }
};

} // namespace vsip::impl::ipp
} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{

#define VSIP_IMPL_IPP_V_EXPR(OP, FUN)					\
template <typename LHS, typename Block>			                \
struct Evaluator<op::assign<1>, be::intel_ipp,				\
         	 void(LHS &, expr::Unary<OP, Block, true> const &)>	\
{									\
  static char const* name() { return "Expr_IPP_V-" #FUN; }		\
									\
  typedef typename impl::Adjust_layout_dim<				\
  1, typename impl::Block_layout<LHS>::layout_type>::type		\
    dst_lp;								\
									\
  typedef typename impl::Adjust_layout_dim<				\
      1, typename impl::Block_layout<Block>::layout_type>::type		\
    blk_lp;								\
									\
  typedef expr::Unary<OP, Block, true> RHS;			        \
									\
  static bool const ct_valid =						\
    !impl::Is_expr_block<Block>::value &&				\
     impl::ipp::Is_type_supported<typename LHS::value_type>::value &&	\
     /* check that direct access is supported */			\
     impl::Ext_data_cost<LHS>::value == 0 &&				\
     impl::Ext_data_cost<Block>::value == 0 &&				\
     /* IPP does not support complex split */				\
     !impl::Is_split_block<LHS>::value &&				\
     !impl::Is_split_block<Block>::value;				\
  									\
  static bool rt_valid(LHS &lhs, RHS const &rhs)		        \
  {									\
    /* check if all data is unit stride */				\
    impl::Ext_data<LHS, dst_lp> ext_dst(lhs, impl::SYNC_OUT);	        \
    impl::Ext_data<Block, blk_lp> ext_src(rhs.arg(), impl::SYNC_IN);	\
    return (ext_dst.stride(0) == 1 &&					\
	    ext_src.stride(0) == 1);					\
  }									\
  									\
  static void exec(LHS &lhs, RHS const &rhs)			        \
  {									\
    impl::Ext_data<LHS, dst_lp> ext_dst(lhs, impl::SYNC_OUT);	        \
    impl::Ext_data<Block, blk_lp> ext_src(rhs.arg(), impl::SYNC_IN);	\
    FUN(ext_src.data(), ext_dst.data(), lhs.size());   			\
  }									\
};

#define VSIP_IMPL_IPP_V_CR_EXPR(OP, FUN)				\
template <typename LHS, typename Block>			                \
struct Evaluator<op::assign<1>, be::intel_ipp,				\
                 void(LHS &, expr::Unary<OP, Block, true> const &)>	\
{									\
  static char const* name() { return "Expr_IPP_V_CR-" #FUN; }		\
									\
  typedef typename impl::Adjust_layout_dim<				\
  1, typename impl::Block_layout<LHS>::layout_type>::type		\
    dst_lp;								\
									\
  typedef typename impl::Adjust_layout_dim<				\
  1, typename impl::Block_layout<Block>::layout_type>::type		\
    blk_lp;								\
									\
  typedef expr::Unary<OP, Block, true> RHS;			        \
									\
  static bool const ct_valid =						\
    !impl::Is_expr_block<Block>::value &&				\
    impl::ipp::Is_type_supported<typename LHS::value_type>::value &&	\
    /* check that direct access is supported */				\
    impl::Ext_data_cost<LHS>::value == 0 &&     			\
    impl::Ext_data_cost<Block>::value == 0 &&				\
    /* IPP does not support complex split */				\
    !impl::Is_split_block<LHS>::value &&				\
    !impl::Is_split_block<Block>::value;				\
  									\
  static bool rt_valid(LHS &lhs, RHS const &rhs)		        \
  {									\
    /* check if all data is unit stride */				\
    impl::Ext_data<LHS, dst_lp> ext_dst(lhs, impl::SYNC_OUT);           \
    impl::Ext_data<Block, blk_lp> ext_src(rhs.arg(), impl::SYNC_IN);    \
    return (ext_dst.stride(0) == 1 && ext_src.stride(0) == 1);		\
  }									\
  									\
  static void exec(LHS &lhs, RHS const &rhs)			        \
  {									\
    impl::Ext_data<LHS, dst_lp> ext_dst(lhs, impl::SYNC_OUT);	        \
    impl::Ext_data<Block, blk_lp> ext_src(rhs.arg(), impl::SYNC_IN);    \
    FUN(ext_src.data(), ext_dst.data(), rhs.size()			\
    );									\
  }									\
};

VSIP_IMPL_IPP_V_EXPR(expr::op::Sq, impl::ipp::vsq)
VSIP_IMPL_IPP_V_EXPR(expr::op::Sqrt, impl::ipp::vsqrt)
VSIP_IMPL_IPP_V_CR_EXPR(expr::op::Mag, impl::ipp::vmag)
// Don't dispatch for now since only real magsq is supported.
// VSIP_IMPL_IPP_V_CR_EXPR(magsq_functor,  ipp::vmagsq)

#define VSIP_IMPL_IPP_VV_EXPR(OP, FUN)					\
template <typename LHS, typename LBlock, typename RBlock>		\
 struct Evaluator<op::assign<1>, be::intel_ipp,				\
	   void(LHS &, expr::Binary<OP, LBlock, RBlock, true> const &)> \
   : impl::ipp::Evaluator<OP, LHS, LBlock, RBlock>			\
{									\
  static char const* name() { return "Expr_IPP_VV-" #FUN; }		\
									\
  typedef expr::Binary<OP, LBlock, RBlock, true> RHS;		        \
  									\
  typedef typename impl::Adjust_layout_dim<			    	\
      1, typename impl::Block_layout<LHS>::layout_type>::type	        \
    dst_lp;								\
  									\
  typedef typename impl::Adjust_layout_dim<				\
      1, typename impl::Block_layout<LBlock>::layout_type>::type	\
    lblock_lp;								\
  									\
  typedef typename impl::Adjust_layout_dim<			       	\
      1, typename impl::Block_layout<RBlock>::layout_type>::type	\
    rblock_lp;								\
  									\
  static void exec(LHS &lhs, RHS const &rhs)			        \
  {									\
    impl::Ext_data<LHS, dst_lp> ext_dst(lhs, impl::SYNC_OUT);	        \
    impl::Ext_data<LBlock, lblock_lp> ext_l(rhs.arg1(), impl::SYNC_IN);	\
    impl::Ext_data<RBlock, rblock_lp> ext_r(rhs.arg2(), impl::SYNC_IN);	\
    FUN(ext_l.data(), ext_r.data(), ext_dst.data(), lhs.size());	\
  } 									\
};

VSIP_IMPL_IPP_VV_EXPR(expr::op::Add, impl::ipp::vadd)
VSIP_IMPL_IPP_VV_EXPR(expr::op::Sub, impl::ipp::vsub)
VSIP_IMPL_IPP_VV_EXPR(expr::op::Mult, impl::ipp::vmul)
VSIP_IMPL_IPP_VV_EXPR(expr::op::Div, impl::ipp::vdiv)

#define VSIP_IMPL_IPP_SV_EXPR(OP, FCN)					\
template <typename LHS, typename S, typename B>		                \
struct Evaluator<op::assign<1>, be::intel_ipp,				\
  void(LHS &, expr::Binary<OP, expr::Scalar<1, S>, B, true> const &)>  	\
  : impl::ipp::Scalar_evaluator<OP, LHS, S, B, true>	                \
{									\
  static char const* name() { return "Expr_IPP_SV-" #FCN; }		\
									\
  typedef expr::Binary<OP, expr::Scalar<1, S>, B, true> RHS;            \
									\
  typedef typename impl::Adjust_layout_dim<				\
      1, typename impl::Block_layout<LHS>::layout_type>::type	        \
    dst_lp;								\
									\
  typedef typename impl::Adjust_layout_dim<				\
      1, typename impl::Block_layout<B>::layout_type>::type	        \
    vblock_lp;								\
									\
  static void exec(LHS &lhs, RHS const &rhs)				\
  {									\
    impl::Ext_data<LHS, dst_lp>  ext_dst(lhs, impl::SYNC_OUT);		\
    impl::Ext_data<B, vblock_lp> ext_r(rhs.arg2(), impl::SYNC_IN);	\
    FCN(rhs.arg1().value(), ext_r.data(), ext_dst.data(), lhs.size());	\
  }									\
};

#define VSIP_IMPL_IPP_SV_EXPR_FO(OP, FCN)				\
template <typename LHS, typename B>				        \
struct Evaluator<op::assign<1>, be::intel_ipp,				\
  void(LHS &, expr::Binary<OP, expr::Scalar<1, float>, B, true> const &)>\
  : impl::ipp::Scalar_evaluator<OP, LHS, float, B, true>	        \
{									\
  static char const* name() { return "Expr_IPP_SV_FO-" #FCN; }		\
									\
  typedef float S;							\
									\
  typedef expr::Binary<OP, expr::Scalar<1, S>, B, true> RHS;            \
									\
  typedef typename impl::Adjust_layout_dim<				\
  1, typename impl::Block_layout<LHS>::layout_type>::type		\
    dst_lp;								\
									\
  typedef typename impl::Adjust_layout_dim<				\
  1, typename impl::Block_layout<B>::layout_type>::type   		\
    vblock_lp;								\
									\
  static void exec(LHS &lhs, RHS const &rhs)			        \
  {									\
    impl::Ext_data<LHS, dst_lp>  ext_dst(lhs, impl::SYNC_OUT);	        \
    impl::Ext_data<B, vblock_lp> ext_r(rhs.arg2(), impl::SYNC_IN);	\
    FCN(rhs.arg1().value(), ext_r.data(), ext_dst.data(), lhs.size());	\
  }									\
};

#define VSIP_IMPL_IPP_VS_EXPR(OP, FCN)					\
template <typename LHS, typename S, typename B>                         \
struct Evaluator<op::assign<1>, be::intel_ipp,				\
  void(LHS &, expr::Binary<OP, B, expr::Scalar<1, S>, true> const &)>   \
  : impl::ipp::Scalar_evaluator<OP, LHS, S, B, false>	                \
{									\
  static char const* name() { return "Expr_IPP_VS-" #FCN; }		\
									\
  typedef expr::Binary<OP, B, expr::Scalar<1, S>, true> RHS;	        \
									\
  typedef typename impl::Adjust_layout_dim<				\
  1, typename impl::Block_layout<LHS>::layout_type>::type		\
    dst_lp;								\
									\
  typedef typename impl::Adjust_layout_dim<				\
  1, typename impl::Block_layout<B>::layout_type>::type	        	\
    vblock_lp;								\
									\
  static void exec(LHS &lhs, RHS const &rhs)			        \
  {									\
    impl::Ext_data<LHS, dst_lp> ext_dst(lhs, impl::SYNC_OUT);           \
    impl::Ext_data<B, vblock_lp> ext_l(rhs.arg1(), impl::SYNC_IN);	\
    FCN(ext_l.data(), rhs.arg2().value(), ext_dst.data(), lhs.size());	\
  }									\
};

#define VSIP_IMPL_IPP_VS_AS_SV_EXPR(OP, FCN)				\
template <typename LHS, typename S, typename B>                         \
struct Evaluator<op::assign<1>, be::intel_ipp,				\
   void(LHS &, expr::Binary<OP, B, expr::Scalar<1, S>, true> const &)>  \
  : impl::ipp::Scalar_evaluator<OP, LHS, S, B, false>		        \
{									\
  static char const* name() { return "Expr_IPP_VS_AS_SV-" #FCN; }	\
									\
  typedef expr::Binary<OP, B, expr::Scalar<1, S>, true> RHS;            \
									\
  typedef typename impl::Adjust_layout_dim<	       			\
      1, typename impl::Block_layout<LHS>::layout_type>::type		\
    dst_lp;								\
									\
  typedef typename impl::Adjust_layout_dim<			      	\
      1, typename impl::Block_layout<B>::layout_type>::type     	\
    vblock_lp;								\
									\
  static void exec(LHS &lhs, RHS const &rhs)			        \
  {									\
    impl::Ext_data<LHS, dst_lp> ext_dst(lhs, impl::SYNC_OUT);	        \
    impl::Ext_data<B, vblock_lp> ext_l(rhs.arg1(), impl::SYNC_IN);	\
    FCN(rhs.arg2().value(), ext_l.data(), ext_dst.data(), lhs.size());	\
  }									\
};

VSIP_IMPL_IPP_SV_EXPR(expr::op::Add, impl::ipp::svadd)
VSIP_IMPL_IPP_VS_AS_SV_EXPR(expr::op::Add, impl::ipp::svadd)
VSIP_IMPL_IPP_SV_EXPR(expr::op::Sub, impl::ipp::svsub)
VSIP_IMPL_IPP_VS_EXPR(expr::op::Sub, impl::ipp::svsub)
VSIP_IMPL_IPP_SV_EXPR(expr::op::Mult, impl::ipp::svmul)
VSIP_IMPL_IPP_VS_AS_SV_EXPR(expr::op::Mult, impl::ipp::svmul)
VSIP_IMPL_IPP_SV_EXPR_FO(expr::op::Div, impl::ipp::svdiv)
VSIP_IMPL_IPP_VS_EXPR(expr::op::Div, impl::ipp::svdiv)

#undef VSIP_IMPL_IPP_SV_EXPR
#undef VSIP_IMPL_IPP_SV_EXPR_FO
#undef VSIP_IMPL_IPP_VS_EXPR
#undef VSIP_IMPL_IPP_VS_AS_SV_EXPR


} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif
