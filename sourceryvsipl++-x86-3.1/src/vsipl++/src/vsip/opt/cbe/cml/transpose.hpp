/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cbe/cml/transpose.hpp
    @author  Don McCoy
    @date    2008-06-04
    @brief   VSIPL++ Library: Bindings for CML matrix transpose.
*/

#ifndef VSIP_OPT_CBE_CML_TRANSPOSE_HPP
#define VSIP_OPT_CBE_CML_TRANSPOSE_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/metaprogramming.hpp>
#include <vsip/core/block_traits.hpp>
#include <vsip/dda.hpp>
#include <vsip/opt/cbe/cml/traits.hpp>
#include <vsip/opt/cbe/ppu/task_manager.hpp>

#include <cml.h>


/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

namespace impl
{

namespace cml
{

// These macros support scalar and interleaved complex types

#define VSIP_IMPL_CML_TRANS(T, FCN, CML_FCN)		\
inline void						\
FCN(T const * a, ptrdiff_t rsa, ptrdiff_t csa,		\
    T* z, ptrdiff_t rsz, ptrdiff_t csz,			\
    size_t m, size_t n)					\
{							\
  typedef scalar_of<T>::type CML_T;			\
  CML_FCN(reinterpret_cast<CML_T const *>(a), rsa, csa,	\
	  reinterpret_cast<CML_T*>(z), rsz, csz,	\
	  m, n);					\
}

VSIP_IMPL_CML_TRANS(float,               transpose, cml_mtrans_f)
VSIP_IMPL_CML_TRANS(std::complex<float>, transpose, cml_cmtrans_f)
#undef VSIP_IMPL_CML_TRANS


#define VSIP_IMPL_CML_TRANS_UNIT(T, FCN, CML_FCN)  \
inline void                                        \
FCN(T const *a, ptrdiff_t rsa,			   \
    T* z, ptrdiff_t rsz,                           \
    size_t m, size_t n)                            \
{						   \
  typedef scalar_of<T>::type CML_T;		   \
  CML_FCN(reinterpret_cast<CML_T const*>(a), rsa,  \
	  reinterpret_cast<CML_T*>(z), rsz,	   \
	  m, n);				   \
}

VSIP_IMPL_CML_TRANS_UNIT(float,               transpose_unit, cml_mtrans1_f)
VSIP_IMPL_CML_TRANS_UNIT(std::complex<float>, transpose_unit, cml_cmtrans1_f)
#undef VSIP_IMPL_CML_TRANS_UNIT


#define VSIP_IMPL_CML_VCOPY(T, FCN, CML_FCN)		   \
inline void						   \
FCN(T const *a, ptrdiff_t rsa,				   \
    T* z, ptrdiff_t rsz,				   \
    size_t n)						   \
{							   \
  typedef scalar_of<T>::type CML_T;			   \
  CML_FCN(reinterpret_cast<CML_T const*>(a), rsa,	   \
	  reinterpret_cast<CML_T*>(z), rsz,		   \
	  n * (is_complex<T>::value ? 2 : 1));		   \
}

VSIP_IMPL_CML_VCOPY(float,          vcopy, cml_vcopy_f)
VSIP_IMPL_CML_VCOPY(complex<float>, vcopy, cml_vcopy_f)
#undef VSIP_IMPL_CML_VCOPY


// These macros support split complex types only

#define VSIP_IMPL_CML_TRANS_SPLIT(T, FCN, CML_FCN)		     \
inline void							     \
FCN(std::pair<T const *, T const *> a, ptrdiff_t rsa, ptrdiff_t csa, \
    std::pair<T*, T*> z, ptrdiff_t rsz, ptrdiff_t csz,		     \
    size_t m, size_t n)						     \
{								     \
  CML_FCN(a.first, a.second, rsa, csa,				     \
	  z.first, z.second, rsz, csz,				     \
	  m, n);						     \
}

VSIP_IMPL_CML_TRANS_SPLIT(float, transpose, cml_zmtrans_f)
#undef VSIP_IMPL_CML_TRANS_SPLIT


#define VSIP_IMPL_CML_TRANS_UNIT_SPLIT(T, FCN, CML_FCN) \
inline void                                             \
FCN(std::pair<T const *, T const *> a, ptrdiff_t rsa,   \
    std::pair<T*, T*> z, ptrdiff_t rsz,                 \
    size_t m, size_t n)                                 \
{							\
  CML_FCN(a.first, a.second, rsa,			\
	  z.first, z.second, rsz,			\
	  m, n);					\
}

VSIP_IMPL_CML_TRANS_UNIT_SPLIT(float, transpose_unit, cml_zmtrans1_f)
#undef VSIP_IMPL_CML_TRANS_UNIT_SPLIT


#define VSIP_IMPL_CML_VCOPY_SPLIT(T, FCN, CML_FCN)	\
inline void                                             \
FCN(std::pair<T const *, T const *> a, ptrdiff_t rsa,	\
    std::pair<T*, T*> z, ptrdiff_t rsz,                 \
    size_t n)                                           \
{							\
  CML_FCN(a.first, rsa, z.first, rsz, n);		\
  CML_FCN(a.second, rsa, z.second, rsz, n);		\
}

VSIP_IMPL_CML_VCOPY_SPLIT(float, vcopy, cml_vcopy_f)
#undef VSIP_IMPL_CML_VCOPY_SPLIT


} // namespace vsip::impl::cml
} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{

template <typename LHS, typename RHS>
struct Evaluator<op::assign<2>, be::cml, void(LHS &, RHS const &)>
{
  typedef typename get_block_layout<RHS>::order_type rhs_order_type;
  typedef typename get_block_layout<LHS>::order_type lhs_order_type;

  typedef typename LHS::value_type lhs_value_type;
  typedef typename RHS::value_type rhs_value_type;

  static char const* name()
  {
    char s = is_same<rhs_order_type, row2_type>::value ? 'r' : 'c';
    char d = is_same<lhs_order_type, row2_type>::value ? 'r' : 'c';
    if      (s == 'r' && d == 'r')    return "Expr_CML_Trans (rr copy)";
    else if (s == 'r' && d == 'c')    return "Expr_CML_Trans (rc trans)";
    else if (s == 'c' && d == 'r')    return "Expr_CML_Trans (cr trans)";
    else /* (s == 'c' && d == 'c') */ return "Expr_CML_Trans (cc copy)";
  }

  static bool const is_rhs_expr   = impl::is_expr_block<RHS>::value;

  static bool const is_lhs_split  = impl::is_split_block<LHS>::value;
  static bool const is_rhs_split  = impl::is_split_block<RHS>::value;

  static int const  lhs_cost      = dda::Data<LHS, dda::out>::ct_cost;
  static int const  rhs_cost      = dda::Data<RHS, dda::in>::ct_cost;

  static bool const ct_valid =
    // check that CML supports this data type and/or layout
    impl::cml::Cml_supports_block<RHS>::valid &&
    impl::cml::Cml_supports_block<LHS>::valid &&
    // check that types are equal
    is_same<rhs_value_type, lhs_value_type>::value &&
    // check that the source block is not an expression
    !is_rhs_expr &&
    // check that direct access is supported
    lhs_cost == 0 && rhs_cost == 0 &&
    // check complex layout is consistent
    is_lhs_split == is_rhs_split;

  static length_type tunable_threshold()
  {
    if (VSIP_IMPL_TUNE_MODE)
      return 0;
    // Copy is always faster with SPU
    else if (is_same<rhs_order_type, lhs_order_type>::value)
      return 0;
    // Transpose not always faster with SPU
    // mcopy -6
    else if (is_same<lhs_value_type, complex<float> >::value)
      return 128*128;
    // mcopy -2
    else if (is_same<lhs_value_type, float>::value)
      return 256*256;

    return 0;
  }

  static bool rt_valid(LHS &lhs, RHS const &rhs)
  { 
    bool rt = true;

    // If performing a copy, both source and destination blocks
    // must be unit stride and dense.
    if (is_same<rhs_order_type, lhs_order_type>::value)
    {
      dda::Data<LHS, dda::out> dst_ext(lhs);
      dda::Data<RHS, dda::in> src_ext(rhs);

      dimension_type const s_dim0 = rhs_order_type::impl_dim0;
      dimension_type const s_dim1 = rhs_order_type::impl_dim1;
      dimension_type const d_dim0 = lhs_order_type::impl_dim0;
      dimension_type const d_dim1 = lhs_order_type::impl_dim1;

      if (dst_ext.stride(d_dim1) != 1 ||
	  dst_ext.stride(d_dim0) != static_cast<stride_type>(lhs.size(2, d_dim1)) ||
	  src_ext.stride(s_dim1) != 1 ||
	  src_ext.stride(s_dim0) != static_cast<stride_type>(rhs.size(2, s_dim1)))
        rt = false;
    }

    rt &= lhs.size(2, 0) * lhs.size(2, 1) > tunable_threshold();
    rt &= impl::cbe::Task_manager::instance()->num_spes() > 0;

    return rt; 
  }

  static void exec(LHS &lhs, RHS const &rhs, row2_type, row2_type)
  {
    dda::Data<LHS, dda::out> dst_ext(lhs);
    dda::Data<RHS, dda::in> src_ext(rhs);

    if (dst_ext.stride(1) == 1 && src_ext.stride(1) == 1)
    {
      assert(dst_ext.stride(0) == static_cast<stride_type>(lhs.size(2, 1)));
      assert(src_ext.stride(0) == static_cast<stride_type>(rhs.size(2, 1)));

      impl::cml::vcopy(src_ext.ptr(), 1,
		       dst_ext.ptr(), 1,
		       lhs.size(2, 0) * lhs.size(2, 1));
    }
    else
      assert(0);
  }

  static void exec(LHS &lhs, RHS const &rhs, col2_type, col2_type)
  {
    dda::Data<LHS, dda::out> dst_ext(lhs);
    dda::Data<RHS, dda::in> src_ext(rhs);

    if (dst_ext.stride(0) == 1 && src_ext.stride(0) == 1)
    {
      assert(dst_ext.stride(1) == static_cast<stride_type>(lhs.size(2, 0)));
      assert(src_ext.stride(1) == static_cast<stride_type>(rhs.size(2, 0)));

      impl::cml::vcopy(src_ext.ptr(), 1,
		       dst_ext.ptr(), 1,
		       lhs.size(2, 0) * lhs.size(2, 1));
    }
    else
      assert(0);
  }

  static void exec(LHS &lhs, RHS const &rhs, col2_type, row2_type)
  {
    dda::Data<LHS, dda::out> dst_ext(lhs);
    dda::Data<RHS, dda::in> src_ext(rhs);

    if (dst_ext.stride(0) == 1 && src_ext.stride(1) == 1)
    {
      impl::cml::transpose_unit(src_ext.ptr(), src_ext.stride(0),
				dst_ext.ptr(), dst_ext.stride(1),
				lhs.size(2, 1), lhs.size(2, 0));
    }
    else
    {
      impl::cml::transpose(src_ext.ptr(), src_ext.stride(0), src_ext.stride(1),
			   dst_ext.ptr(), dst_ext.stride(1), dst_ext.stride(0),
			   lhs.size(2, 1), lhs.size(2, 0));
    }
  }

  static void exec(LHS &lhs, RHS const &rhs, row2_type, col2_type)
  {
    dda::Data<LHS, dda::out> dst_ext(lhs);
    dda::Data<RHS, dda::in> src_ext(rhs);

    if (dst_ext.stride(1) == 1 && src_ext.stride(0) == 1)
    {
      impl::cml::transpose_unit(src_ext.ptr(), src_ext.stride(1),
				dst_ext.ptr(), dst_ext.stride(0),
				lhs.size(2, 0), lhs.size(2, 1));
    }
    else
    {
      impl::cml::transpose(src_ext.ptr(), src_ext.stride(1), src_ext.stride(0),
			   dst_ext.ptr(), dst_ext.stride(0), dst_ext.stride(1),
			   lhs.size(2, 0), lhs.size(2, 1));
    }
  }

  static void exec(LHS &lhs, RHS const &rhs)
  {
    exec(lhs, rhs, lhs_order_type(), rhs_order_type());
  }
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif // VSIP_OPT_CBE_CML_TRANSPOSE_HPP
