//
// Copyright (c) 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_cvsip_reductions_idx_hpp_
#define ovxx_cvsip_reductions_idx_hpp_

#include <ovxx/dispatch.hpp>
#include <ovxx/ct_assert.hpp>
#include <ovxx/cvsip/view.hpp>
#include <ovxx/cvsip/converter.hpp>

namespace ovxx
{
namespace cvsip
{

template <template <typename> class R,
	  dimension_type            D,
	  typename                  T>
struct reduction_not_implemented_by_cvsip_backend;

template <template <typename> class R,
	  dimension_type            D,
          typename                  T>
struct reduction_idx
{
  static bool const valid = false;
  template <typename V>
  static T exec(V*, Index<D>&)
  {
    ct_assert_msg<false,
      reduction_not_implemented_by_cvsip_backend<R, D, T> >::test();
  }
};

#define OVXX_CVSIP_RDX_IDX_RT(R, D, T, RT, F)	\
template <>	      				\
struct reduction_idx<R, D, T>  			\
{     						\
  static bool const valid = true;	     	\
  static RT exec(view_traits<D, T>::view_type* view,\
		 Index<D>& idx)      		\
  {   						\
    converter<Index<D> >::vsipl_type v_idx;   	\
    RT rv = converter<RT>::to_vsiplxx(F(view, &v_idx));\
    idx = converter<Index<D> >::to_vsiplxx(v_idx);\
    return rv;		       			\
  }    						\
};

#define OVXX_CVSIP_RDX_IDX(R, D, T, F)		\
  OVXX_CVSIP_RDX_IDX_RT(R, D, T, T, F)

// Reductions marked by [*] are not implemented in TVCPP.

OVXX_CVSIP_RDX_IDX(Max_value, 1,  float, vsip_vmaxval_f)
OVXX_CVSIP_RDX_IDX(Max_value, 1, double, vsip_vmaxval_d)

OVXX_CVSIP_RDX_IDX(Max_value, 2,  float, vsip_mmaxval_f)
OVXX_CVSIP_RDX_IDX(Max_value, 2, double, vsip_mmaxval_d)

OVXX_CVSIP_RDX_IDX(Min_value, 1,  float, vsip_vminval_f)
OVXX_CVSIP_RDX_IDX(Min_value, 1, double, vsip_vminval_d)

OVXX_CVSIP_RDX_IDX(Min_value, 2,  float, vsip_mminval_f)
OVXX_CVSIP_RDX_IDX(Min_value, 2, double, vsip_mminval_d)

OVXX_CVSIP_RDX_IDX(Max_mag_value, 1,  float, vsip_vmaxmgval_f)
OVXX_CVSIP_RDX_IDX(Max_mag_value, 1, double, vsip_vmaxmgval_d)

OVXX_CVSIP_RDX_IDX(Max_mag_value, 2,  float, vsip_mmaxmgval_f)
OVXX_CVSIP_RDX_IDX(Max_mag_value, 2, double, vsip_mmaxmgval_d)

OVXX_CVSIP_RDX_IDX(Min_mag_value, 1,  float, vsip_vminmgval_f)
OVXX_CVSIP_RDX_IDX(Min_mag_value, 1, double, vsip_vminmgval_d)

OVXX_CVSIP_RDX_IDX(Min_mag_value, 2,  float, vsip_mminmgval_f)
OVXX_CVSIP_RDX_IDX(Min_mag_value, 2, double, vsip_mminmgval_d)


OVXX_CVSIP_RDX_IDX_RT(Min_magsq_value,1, complex<float>,float,vsip_vcminmgsqval_f)
OVXX_CVSIP_RDX_IDX_RT(Min_magsq_value,1, complex<double>,double,vsip_vcminmgsqval_d)

OVXX_CVSIP_RDX_IDX_RT(Min_magsq_value,2, complex<float>,float,vsip_mcminmgsqval_f)
OVXX_CVSIP_RDX_IDX_RT(Min_magsq_value,2, complex<double>,double,vsip_mcminmgsqval_d)


OVXX_CVSIP_RDX_IDX_RT(Max_magsq_value,1, complex<float>,float,vsip_vcmaxmgsqval_f)
OVXX_CVSIP_RDX_IDX_RT(Max_magsq_value,1, complex<double>,double,vsip_vcmaxmgsqval_d)

OVXX_CVSIP_RDX_IDX_RT(Max_magsq_value,2, complex<float>,float,vsip_mcmaxmgsqval_f)
OVXX_CVSIP_RDX_IDX_RT(Max_magsq_value,2, complex<double>,double,vsip_mcmaxmgsqval_d)

} // namespace ovxx::cvsip

namespace dispatcher
{

template <template <typename> class R,
          typename                  T,
	  typename                  B,
	  typename                  O,
	  dimension_type            D>
struct Evaluator<op::reduce_idx<R>, be::cvsip,
                 void(T&, B const&, Index<D>&, O)>
{
  typedef typename B::value_type value_type;

  static bool const ct_valid =
    cvsip::reduction_idx<R, D, value_type>::valid;

  static bool rt_valid(T&, B const&, Index<D>&, O)
  { return true;}

  static void exec(T& r, B const& block, Index<D>& idx, O)
  {
    dda::Data<B, dda::in> data(block);
    cvsip::view_from_data<D, value_type, dda::in> view(data);
    r = cvsip::reduction_idx<R, D, value_type>::exec
      (view.view.ptr(), idx);
  }
};

} // namespace ovxx::dispatcher
} // namespace ovxx

#endif
