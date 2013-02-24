/* Copyright (c) 2006 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/cvsip/eval_reductions.hpp
    @author  Jules Bergmann
    @date    2006-12-07
    @brief   VSIPL++ Library: Reduction functions using C-VSIP BE.
	     [math.fns.reductions].

*/

#ifndef VSIP_CORE_CVSIP_EVAL_REDUCTIONS_HPP
#define VSIP_CORE_CVSIP_EVAL_REDUCTIONS_HPP

extern "C" {
#include <vsip.h>

// TVCPP vsip.h does not include this decl:
vsip_scalar_vi vsip_vsumval_bl(vsip_vview_bl const* a);
}

#include <vsip/core/dispatch.hpp>
#include <vsip/core/coverage.hpp>
#include <vsip/core/static_assert.hpp>
#include <vsip/core/cvsip/view.hpp>
#include <vsip/core/cvsip/convert_value.hpp>

namespace vsip
{
namespace impl
{
namespace cvsip
{

template <template <typename> class ReduceT,
	  dimension_type            Dim,
	  typename                  T>
struct Reduction_not_implemented_by_cvsip_backend;

template <template <typename> class ReduceT,
	  dimension_type            Dim,
          typename                  T>
struct Reduce_class
{
  static bool const valid = false;

  // Provide a dummy exec() function that generates a compile-time error.
  //
  // When used by the optimized implementation, checking of ct_valid
  // will prevent this function from being instantiated.
  //
  // When used by the reference implementation, since ct_valid is not
  // used, this function will be instantiated if none of the specializations
  // below apply.
  template <typename CvsipViewT>
  static T exec(CvsipViewT* view)
  {
    Compile_time_assert_msg<false,
      Reduction_not_implemented_by_cvsip_backend<ReduceT, Dim, T> >::test();
  }
};

#define VSIP_IMPL_CVSIP_RDX_RT(REDUCET, DIM, T, RT, CVSIP_FCN)		\
template <>								\
struct Reduce_class<REDUCET, DIM, T>					\
{									\
  static bool const valid = true;					\
  static RT exec(cvsip::View_traits<DIM, T>::view_type* view)		\
  {									\
    VSIP_IMPL_COVER_FCN("cvsip_reduce", CVSIP_FCN);			\
    return Convert_value<RT>::to_cpp(CVSIP_FCN(view));			\
  }									\
};

#define VSIP_IMPL_CVSIP_RDX(REDUCET, DIM, T, CVSIP_FCN)			\
  VSIP_IMPL_CVSIP_RDX_RT(REDUCET, DIM, T, T, CVSIP_FCN)

// Reductions marked by [*] are not implemented in TVCPP.

VSIP_IMPL_CVSIP_RDX_RT(Sum_value, 1,   bool, length_type, vsip_vsumval_bl)
VSIP_IMPL_CVSIP_RDX_RT(Sum_value, 2,   bool, length_type, vsip_msumval_bl)

VSIP_IMPL_CVSIP_RDX(Sum_value, 1,    int, vsip_vsumval_i)
VSIP_IMPL_CVSIP_RDX(Sum_value, 1,  float, vsip_vsumval_f)
VSIP_IMPL_CVSIP_RDX(Sum_value, 1, double, vsip_vsumval_d)
//[*] VSIP_IMPL_CVSIP_RDX(Sum_value, 1, complex<float>, vsip_cvsumval_f)

//[*] VSIP_IMPL_CVSIP_RDX(Sum_value, 2, int, int, vsip_msumval_i)
VSIP_IMPL_CVSIP_RDX(Sum_value, 2,  float, vsip_msumval_f)
VSIP_IMPL_CVSIP_RDX(Sum_value, 2, double, vsip_msumval_d)

VSIP_IMPL_CVSIP_RDX(Sum_sq_value, 1,  float, vsip_vsumsqval_f)
VSIP_IMPL_CVSIP_RDX(Sum_sq_value, 1, double, vsip_vsumsqval_d)
VSIP_IMPL_CVSIP_RDX(Sum_sq_value, 2,  float, vsip_msumsqval_f)
VSIP_IMPL_CVSIP_RDX(Sum_sq_value, 2, double, vsip_msumsqval_d)

VSIP_IMPL_CVSIP_RDX(Mean_value, 1, float,           vsip_vmeanval_f)
VSIP_IMPL_CVSIP_RDX(Mean_value, 1, double,          vsip_vmeanval_d)
VSIP_IMPL_CVSIP_RDX(Mean_value, 1, complex<float>,  vsip_cvmeanval_f)
VSIP_IMPL_CVSIP_RDX(Mean_value, 1, complex<double>, vsip_cvmeanval_d)
VSIP_IMPL_CVSIP_RDX(Mean_value, 2, float,           vsip_mmeanval_f)
VSIP_IMPL_CVSIP_RDX(Mean_value, 2, double,          vsip_mmeanval_d)
VSIP_IMPL_CVSIP_RDX(Mean_value, 2, complex<float>,  vsip_cmmeanval_f)
VSIP_IMPL_CVSIP_RDX(Mean_value, 2, complex<double>, vsip_cmmeanval_d)

VSIP_IMPL_CVSIP_RDX_RT(Mean_magsq_value, 1, float,           float,
		       vsip_vmeansqval_f)
VSIP_IMPL_CVSIP_RDX_RT(Mean_magsq_value, 1, double,         double,
		       vsip_vmeansqval_d)
VSIP_IMPL_CVSIP_RDX_RT(Mean_magsq_value, 1, complex<float>,  float,
		       vsip_cvmeansqval_f)
VSIP_IMPL_CVSIP_RDX_RT(Mean_magsq_value, 1, complex<double>,double,
		       vsip_cvmeansqval_d)
VSIP_IMPL_CVSIP_RDX_RT(Mean_magsq_value, 2, float,           float,
		       vsip_mmeansqval_f)
VSIP_IMPL_CVSIP_RDX_RT(Mean_magsq_value, 2, double,         double,
		       vsip_mmeansqval_d)
VSIP_IMPL_CVSIP_RDX_RT(Mean_magsq_value, 2, complex<float>,  float,
		       vsip_cmmeansqval_f)
VSIP_IMPL_CVSIP_RDX_RT(Mean_magsq_value, 2, complex<double>,double,
		       vsip_cmmeansqval_d)

VSIP_IMPL_CVSIP_RDX(All_true, 1, bool,  vsip_valltrue_bl)
VSIP_IMPL_CVSIP_RDX(All_true, 2, bool,  vsip_malltrue_bl)

VSIP_IMPL_CVSIP_RDX(Any_true, 1, bool,  vsip_vanytrue_bl)
VSIP_IMPL_CVSIP_RDX(Any_true, 2, bool,  vsip_manytrue_bl)

#undef VSIP_IMPL_CVSIP_RDX
#undef VSIP_IMPL_CVSIP_RDX_RT

} // namespace vsip::impl::cvsip
} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{

template <template <typename> class ReduceT,
          typename                  T,
	  typename                  Block,
	  typename                  OrderT,
	  int                       Dim>
struct Evaluator<op::reduce<ReduceT>, be::cvsip,
                 void(T&, Block const&, OrderT, impl::Int_type<Dim>)>
{
  typedef typename Block::value_type value_type;

  static bool const ct_valid =
    impl::cvsip::Reduce_class<ReduceT, Dim, value_type>::valid;

  static bool rt_valid(T&, Block const&, OrderT, impl::Int_type<Dim>)
  { return true; }

  static void exec(T& r, Block const& blk, OrderT, impl::Int_type<Dim>)
  {
    dda::Data<Block, dda::in> data(blk);
    impl::cvsip::View_from_data<Dim, value_type, dda::in> view(data);
    
    r = impl::cvsip::Reduce_class<ReduceT, Dim, value_type>::exec(view.view.ptr());
  }
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif
