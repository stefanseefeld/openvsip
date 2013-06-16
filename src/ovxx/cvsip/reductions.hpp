//
// Copyright (c) 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_cvsip_eval_reductions_hpp_
#define ovxx_cvsip_eval_reductions_hpp_

#include <ovxx/dispatch.hpp>
#include <ovxx/ct_assert.hpp>
#include <ovxx/cvsip/converter.hpp>
#include <ovxx/cvsip/view.hpp>
extern "C" 
{
// TVCPP vsip.h does not include this decl:
vsip_scalar_vi vsip_vsumval_bl(vsip_vview_bl const* a);
}

namespace ovxx
{
namespace cvsip
{

template <template <typename> class R, dimension_type D, typename T>
struct reduction_not_implemented_by_cvsip_backend;

template <template <typename> class R, dimension_type D, typename T>
struct reduction
{
  static bool const valid = false;
  template <typename V>
  static T exec(V *view)
  {
    ct_assert_msg<false,
      reduction_not_implemented_by_cvsip_backend<R, D, T> >::test();
  }
};

#define OVXX_CVSIP_RDX_RT(R, D, T, RT, F)		\
template <>					      	\
struct reduction<R, D, T>      				\
{	  						\
  static bool const valid = true;    			\
  static RT exec(view_traits<D, T>::view_type* view)   	\
  {				      			\
    return converter<RT>::to_vsiplxx(F(view)); 		\
  }				   			\
};

#define OVXX_CVSIP_RDX(R, D, T, F)			\
  OVXX_CVSIP_RDX_RT(R, D, T, T, F)

// Reductions marked by [*] are not implemented in TVCPP.

OVXX_CVSIP_RDX_RT(Sum_value, 1,   bool, length_type, vsip_vsumval_bl)
OVXX_CVSIP_RDX_RT(Sum_value, 2,   bool, length_type, vsip_msumval_bl)

OVXX_CVSIP_RDX(Sum_value, 1,    int, vsip_vsumval_i)
OVXX_CVSIP_RDX(Sum_value, 1,  float, vsip_vsumval_f)
OVXX_CVSIP_RDX(Sum_value, 1, double, vsip_vsumval_d)
//[*] OVXX_CVSIP_RDX(Sum_value, 1, complex<float>, vsip_cvsumval_f)

//[*] OVXX_CVSIP_RDX(Sum_value, 2, int, int, vsip_msumval_i)
OVXX_CVSIP_RDX(Sum_value, 2,  float, vsip_msumval_f)
OVXX_CVSIP_RDX(Sum_value, 2, double, vsip_msumval_d)

OVXX_CVSIP_RDX(Sum_sq_value, 1,  float, vsip_vsumsqval_f)
OVXX_CVSIP_RDX(Sum_sq_value, 1, double, vsip_vsumsqval_d)
OVXX_CVSIP_RDX(Sum_sq_value, 2,  float, vsip_msumsqval_f)
OVXX_CVSIP_RDX(Sum_sq_value, 2, double, vsip_msumsqval_d)

OVXX_CVSIP_RDX(Mean_value, 1, float,           vsip_vmeanval_f)
OVXX_CVSIP_RDX(Mean_value, 1, double,          vsip_vmeanval_d)
OVXX_CVSIP_RDX(Mean_value, 1, complex<float>,  vsip_cvmeanval_f)
OVXX_CVSIP_RDX(Mean_value, 1, complex<double>, vsip_cvmeanval_d)
OVXX_CVSIP_RDX(Mean_value, 2, float,           vsip_mmeanval_f)
OVXX_CVSIP_RDX(Mean_value, 2, double,          vsip_mmeanval_d)
OVXX_CVSIP_RDX(Mean_value, 2, complex<float>,  vsip_cmmeanval_f)
OVXX_CVSIP_RDX(Mean_value, 2, complex<double>, vsip_cmmeanval_d)

OVXX_CVSIP_RDX_RT(Mean_magsq_value, 1, float,           float,
		       vsip_vmeansqval_f)
OVXX_CVSIP_RDX_RT(Mean_magsq_value, 1, double,         double,
		       vsip_vmeansqval_d)
OVXX_CVSIP_RDX_RT(Mean_magsq_value, 1, complex<float>,  float,
		       vsip_cvmeansqval_f)
OVXX_CVSIP_RDX_RT(Mean_magsq_value, 1, complex<double>,double,
		       vsip_cvmeansqval_d)
OVXX_CVSIP_RDX_RT(Mean_magsq_value, 2, float,           float,
		       vsip_mmeansqval_f)
OVXX_CVSIP_RDX_RT(Mean_magsq_value, 2, double,         double,
		       vsip_mmeansqval_d)
OVXX_CVSIP_RDX_RT(Mean_magsq_value, 2, complex<float>,  float,
		       vsip_cmmeansqval_f)
OVXX_CVSIP_RDX_RT(Mean_magsq_value, 2, complex<double>,double,
		       vsip_cmmeansqval_d)

OVXX_CVSIP_RDX(All_true, 1, bool,  vsip_valltrue_bl)
OVXX_CVSIP_RDX(All_true, 2, bool,  vsip_malltrue_bl)

OVXX_CVSIP_RDX(Any_true, 1, bool,  vsip_vanytrue_bl)
OVXX_CVSIP_RDX(Any_true, 2, bool,  vsip_manytrue_bl)

#undef OVXX_CVSIP_RDX
#undef OVXX_CVSIP_RDX_RT

} // namespace ovxx::cvsip

namespace dispatcher
{

template <template <typename> class R,
          typename                  T,
	  typename                  B,
	  typename                  O,
	  int                       D>
struct Evaluator<op::reduce<R>, be::cvsip,
                 void(T&, B const&, O, integral_constant<dimension_type, D>)>
{
  typedef typename B::value_type value_type;

  static bool const ct_valid =
    cvsip::reduction<R, D, value_type>::valid;

  static bool rt_valid(T&, B const&, O, integral_constant<dimension_type, D>)
  { return true;}

  static void exec(T& r, B const &block, O, integral_constant<dimension_type, D>)
  {
    dda::Data<B, dda::in> data(block);
    cvsip::view_from_data<D, value_type, dda::in> view(data);
    
    r = cvsip::reduction<R, D, value_type>::exec(view.view.ptr());
  }
};

} // namespace ovxx::dispatcher
} // namespace ovxx

#endif
