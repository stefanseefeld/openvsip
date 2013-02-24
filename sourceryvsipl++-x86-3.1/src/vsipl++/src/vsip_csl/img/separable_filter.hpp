/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip_csl/img/separable_filter.hpp
    @author  Jules Bergmann
    @date    2007-10-04
    @brief   VSIPL++ Library: Image-processing separable filter.

*/

#ifndef VSIP_CSL_IMG_SEPARABLE_FILTER_HPP
#define VSIP_CSL_IMG_SEPARABLE_FILTER_HPP

#include <vsip/opt/dispatch.hpp>
#include <vsip_csl/img/impl/sfilt_gen.hpp>
#if VSIP_IMPL_HAVE_IPP
#  include <vsip_csl/img/impl/sfilt_ipp.hpp>
#endif

namespace vsip_csl
{
namespace dispatcher
{
template <dimension_type D,
	  vsip::support_region_type R,
	  img::edge_handling_type E,
	  unsigned N,
          vsip::alg_hint_type H>
struct List<op::sfilt<D, R, E, N, H> >
{
  typedef Make_type_list<be::user,
			 be::intel_ipp,
			 be::mercury_sal,
			 be::generic>::type type;
};

} // namespace vsip_csl::dispatcher

namespace img
{

template <typename T,
	  vsip::support_region_type R,
	  edge_handling_type E,
	  unsigned N = 0,
	  vsip::alg_hint_type H = vsip::alg_time>
class Separable_filter
{
  typedef typename vsip_csl::dispatcher::Dispatcher<
    vsip_csl::dispatcher::op::sfilt<2, R, E, N, H>, T>::type
  backend_type;
public:
  static vsip::dimension_type const dim = 2;

  template <typename Block1, typename Block2>
  Separable_filter(vsip::Vector<T, Block1> coeff0,	// coeffs for dimension 0
		   vsip::Vector<T, Block2> coeff1,	// coeffs for dimension 1
		   vsip::Domain<2> const&  input_size)
    VSIP_THROW((std::bad_alloc))
    : backend_(coeff0, coeff1, input_size)
  {}

  vsip::Domain<dim> kernel_size() const VSIP_NOTHROW
  { return backend_.kernel_size();}
  vsip::Domain<dim> filter_order() const VSIP_NOTHROW
  { return backend_.filter_order();}
  vsip::Domain<dim> input_size() const VSIP_NOTHROW
  { return backend_.input_size();}
  vsip::Domain<dim> output_size() const VSIP_NOTHROW
  { return backend_.output_size();}

  template <typename Block1, typename Block2>
  vsip::Matrix<T, Block2>
  operator()(vsip::const_Matrix<T, Block1> in, vsip::Matrix<T, Block2> out)
    VSIP_NOTHROW
  {
    backend_.filter(in, out);
    return out;
  }

private:
  Separable_filter(Separable_filter const&) VSIP_NOTHROW;
  Separable_filter& operator=(Separable_filter const&) VSIP_NOTHROW;

  backend_type backend_;
};

} // namespace vsip_csl::img
} // namespace vsip_csl

#endif // VSIP_CSL_IMG_SEPARABLE_FILTER_HPP
