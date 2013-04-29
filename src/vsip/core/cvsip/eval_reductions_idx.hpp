//
// Copyright (c) 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_CORE_CVSIP_REDUCTIONS_IDX_HPP
#define VSIP_CORE_CVSIP_REDUCTIONS_IDX_HPP

/***********************************************************************
  Included Files
***********************************************************************/

extern "C" {
#include <vsip.h>
}

#include <vsip/core/dispatch.hpp>
#include <vsip/core/static_assert.hpp>
#include <vsip/core/cvsip/view.hpp>
#include <vsip/core/cvsip/convert_value.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

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
struct Reduce_idx_class
{
  static bool const valid = false;

  // Provide a dummy exec() function that generates a compile-time error.
  //
  // When used by the optimized implementation, checking of ct_valid
  // will prevent this function from being called.
  //
  // When used by the reference implementation, since ct_valid is not
  // used, this function will be instantiated if none of the specializations
  // below apply.

  template <typename CvsipViewT>
  // static T exec(typename cvsip::View_traits<Dim, T>::view_type*, Index<Dim>&)
  static T exec(CvsipViewT*, Index<Dim>&)
  {
    Compile_time_assert_msg<false,
      Reduction_not_implemented_by_cvsip_backend<ReduceT, Dim, T> >::test();
  }
};



#define VSIP_IMPL_CVSIP_RDX_IDX_RT(REDUCET, DIM, T, RT, CVSIP_FCN)	\
template <>								\
struct Reduce_idx_class<REDUCET, DIM, T>				\
{									\
  static bool const valid = true;					\
  static RT exec(cvsip::View_traits<DIM, T>::view_type* view,		\
		 Index<DIM>& idx)					\
  {									\
    VSIP_IMPL_COVER_FCN("cvsip_reduce_idx", CVSIP_FCN);			\
    Convert_value<Index<DIM> >::cvsip_type v_idx;			\
    RT rv = Convert_value<RT>::to_cpp(CVSIP_FCN(view, &v_idx));		\
    idx = Convert_value<Index<DIM> >::to_cpp(v_idx);			\
    return rv;								\
  }									\
};

#define VSIP_IMPL_CVSIP_RDX_IDX(REDUCET, DIM, T, CVSIP_FCN)		\
  VSIP_IMPL_CVSIP_RDX_IDX_RT(REDUCET, DIM, T, T, CVSIP_FCN)

// Reductions marked by [*] are not implemented in TVCPP.

VSIP_IMPL_CVSIP_RDX_IDX(Max_value, 1,  float, vsip_vmaxval_f)
VSIP_IMPL_CVSIP_RDX_IDX(Max_value, 1, double, vsip_vmaxval_d)

VSIP_IMPL_CVSIP_RDX_IDX(Max_value, 2,  float, vsip_mmaxval_f)
VSIP_IMPL_CVSIP_RDX_IDX(Max_value, 2, double, vsip_mmaxval_d)

VSIP_IMPL_CVSIP_RDX_IDX(Min_value, 1,  float, vsip_vminval_f)
VSIP_IMPL_CVSIP_RDX_IDX(Min_value, 1, double, vsip_vminval_d)

VSIP_IMPL_CVSIP_RDX_IDX(Min_value, 2,  float, vsip_mminval_f)
VSIP_IMPL_CVSIP_RDX_IDX(Min_value, 2, double, vsip_mminval_d)

VSIP_IMPL_CVSIP_RDX_IDX(Max_mag_value, 1,  float, vsip_vmaxmgval_f)
VSIP_IMPL_CVSIP_RDX_IDX(Max_mag_value, 1, double, vsip_vmaxmgval_d)

VSIP_IMPL_CVSIP_RDX_IDX(Max_mag_value, 2,  float, vsip_mmaxmgval_f)
VSIP_IMPL_CVSIP_RDX_IDX(Max_mag_value, 2, double, vsip_mmaxmgval_d)

VSIP_IMPL_CVSIP_RDX_IDX(Min_mag_value, 1,  float, vsip_vminmgval_f)
VSIP_IMPL_CVSIP_RDX_IDX(Min_mag_value, 1, double, vsip_vminmgval_d)

VSIP_IMPL_CVSIP_RDX_IDX(Min_mag_value, 2,  float, vsip_mminmgval_f)
VSIP_IMPL_CVSIP_RDX_IDX(Min_mag_value, 2, double, vsip_mminmgval_d)


VSIP_IMPL_CVSIP_RDX_IDX_RT(Min_magsq_value,1, complex<float>,float,vsip_vcminmgsqval_f)
VSIP_IMPL_CVSIP_RDX_IDX_RT(Min_magsq_value,1, complex<double>,double,vsip_vcminmgsqval_d)

VSIP_IMPL_CVSIP_RDX_IDX_RT(Min_magsq_value,2, complex<float>,float,vsip_mcminmgsqval_f)
VSIP_IMPL_CVSIP_RDX_IDX_RT(Min_magsq_value,2, complex<double>,double,vsip_mcminmgsqval_d)


VSIP_IMPL_CVSIP_RDX_IDX_RT(Max_magsq_value,1, complex<float>,float,vsip_vcmaxmgsqval_f)
VSIP_IMPL_CVSIP_RDX_IDX_RT(Max_magsq_value,1, complex<double>,double,vsip_vcmaxmgsqval_d)

VSIP_IMPL_CVSIP_RDX_IDX_RT(Max_magsq_value,2, complex<float>,float,vsip_mcmaxmgsqval_f)
VSIP_IMPL_CVSIP_RDX_IDX_RT(Max_magsq_value,2, complex<double>,double,vsip_mcmaxmgsqval_d)

} // namespace vsip::impl::cvsip
} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{

/***********************************************************************
  Evaluators.
***********************************************************************/

template <template <typename> class ReduceT,
          typename                  T,
	  typename                  Block,
	  typename                  OrderT,
	  dimension_type            Dim>
struct Evaluator<op::reduce_idx<ReduceT>, be::cvsip,
                 void(T&, Block const&, Index<Dim>&, OrderT)>
{
  typedef typename Block::value_type value_type;

  static bool const ct_valid =
    impl::cvsip::Reduce_idx_class<ReduceT, Dim, value_type>::valid;

  static bool rt_valid(T&, Block const&, Index<Dim>&, OrderT)
  { return true; }

  static void exec(T& r, Block const& blk, Index<Dim>& idx, OrderT)
  {
    dda::Data<Block, dda::in> data(blk);
    vsip::impl::cvsip::View_from_data<Dim, value_type, dda::in> view(data);

    r = vsip::impl::cvsip::Reduce_idx_class<ReduceT, Dim, value_type>::exec(
		view.view.ptr(),
		idx);
  }
};

} // namespace vsip_csl::dispatcher
} // namespace vsip

#endif
