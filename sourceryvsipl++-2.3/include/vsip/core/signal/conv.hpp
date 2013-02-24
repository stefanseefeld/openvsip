/* Copyright (c) 2005, 2006, 2007 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/signal/conv.hpp
    @author  Jules Bergmann
    @date    2005-06-09
    @brief   VSIPL++ Library: Convolution class [signal.conv].
*/

#ifndef VSIP_CORE_SIGNAL_CONV_HPP
#define VSIP_CORE_SIGNAL_CONV_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/core/domain_utils.hpp>
#include <vsip/core/length.hpp>
#include <vsip/core/signal/types.hpp>
#include <vsip/core/profile.hpp>
#include <vsip/core/signal/conv_common.hpp>
#if !VSIP_IMPL_REF_IMPL
# include <vsip/opt/signal/conv_ext.hpp>
# ifdef VSIP_IMPL_CBE_SDK
#  include <vsip/opt/cbe/cml/conv.hpp>
# endif
# if VSIP_IMPL_HAVE_IPP
#  include <vsip/opt/ipp/conv.hpp>
# endif
# if VSIP_IMPL_HAVE_SAL
#  include <vsip/opt/sal/conv.hpp>
# endif
#endif
#if VSIP_IMPL_HAVE_CVSIP
# include <vsip/core/cvsip/conv.hpp>
#endif

/***********************************************************************
  Declarations
***********************************************************************/

#if !VSIP_IMPL_REF_IMPL

namespace vsip_csl
{
namespace dispatcher
{
template <dimension_type D,
          symmetry_type S,
          support_region_type R,
          typename T,
          unsigned int N,
          alg_hint_type H>
struct List<op::conv<D, S, R, T, N, H> >
{
  typedef Make_type_list<be::user,
			 be::cml,
			 be::intel_ipp,
			 be::mercury_sal,
			 be::cvsip,
			 be::generic>::type type;
};
} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif

namespace vsip
{

template <template <typename, typename> class ConstViewT,
	  symmetry_type                       Symm,
	  support_region_type                 Supp,
	  typename                            T = VSIP_DEFAULT_VALUE_TYPE,
	  unsigned                            N_times = 0,
          alg_hint_type                       A_hint = alg_time>
class Convolution
  : public impl::profile::Accumulator<impl::profile::signal>,
#if VSIP_IMPL_REF_IMPL
    public impl::cvsip::Convolution<impl::Dim_of_view<ConstViewT>::dim,
                                    Symm, Supp, float, N_times, A_hint>
#else
    public vsip_csl::dispatcher::Dispatcher<
             vsip_csl::dispatcher::op::conv<
               impl::Dim_of_view<ConstViewT>::dim,
				 Symm,
				 Supp,
				 T,
				 N_times,
				 A_hint> >::type
#endif
{
  // Implementation compile-time constants.
private: 
  static dimension_type const dim = impl::Dim_of_view<ConstViewT>::dim;

  typedef impl::profile::Accumulator<impl::profile::signal> accumulator_type;

#if VSIP_IMPL_REF_IMPL
  typedef impl::cvsip::Convolution<impl::Dim_of_view<ConstViewT>::dim,
                                   Symm, Supp, float, N_times, A_hint> base_type;
#else
  typedef typename vsip_csl::dispatcher::Dispatcher<
    vsip_csl::dispatcher::op::conv<dim,
				   Symm,
				   Supp,
				   T,
				   N_times,
				   A_hint> >::type base_type;
#endif

public:
  template <typename Block>
  Convolution(ConstViewT<T, Block> filter_coeffs,
	      Domain<dim> const&   input_size,
	      length_type          decimation = 1)
    VSIP_THROW((std::bad_alloc))
      : accumulator_type(impl::signal_detail::Description<dim, T>::tag
                         ("Convolution",
                          impl::extent(impl::conv_output_size
                                       (Supp, view_domain(filter_coeffs), 
                                        input_size, decimation)),
                          impl::extent(filter_coeffs)),
                         impl::signal_detail::Op_count_conv<dim, T>::value
                         (impl::extent(impl::conv_output_size
                                       (Supp, view_domain(filter_coeffs),
                                        input_size, decimation)),
                          impl::extent(filter_coeffs))),
        base_type(filter_coeffs, input_size, decimation)
  {
    assert(decimation >= 1);
    assert(Symm == nonsym ? (filter_coeffs.size() <=   input_size.size())
			  : (filter_coeffs.size() <= 2*input_size.size()));
  }

  Convolution(Convolution const&) VSIP_NOTHROW;
  Convolution& operator=(Convolution const&) VSIP_NOTHROW;
  ~Convolution() VSIP_NOTHROW {}

  // Convolution operators.
public:
#if USE_IMPL_VIEWS
  template <template <typename, typename> class V1,
	    template <typename, typename> class V2,
	    typename Block1,
	    typename Block2>
  typename impl::View_of_dim<dim, T, Block2>::type
  operator()(
    impl_const_View<V1, Block1, T, dim> in,
    impl_View<V2, Block2, T, dim>       out)
    VSIP_NOTHROW
  {
    typename accumulator_type::Scope scope(*this);

    for (dimension_type d=0; d<dim; ++d)
      assert(in.size(d) == this->input_size()[d].size());

    for (dimension_type d=0; d<dim; ++d)
      assert(out.size(d) == this->output_size()[d].size());

    convolve(in.impl_view(), out.impl_view());

    return out;
  }
#else
  template <typename Block1,
	    typename Block2>
  Vector<T, Block2>
  operator()(
    const_Vector<T, Block1> in,
    Vector<T, Block2>       out)
    VSIP_NOTHROW
  {
    typename accumulator_type::Scope scope(*this);

    for (dimension_type d=0; d<dim; ++d)
      assert(in.size(d) == this->input_size()[d].size());

    for (dimension_type d=0; d<dim; ++d)
      assert(out.size(d) == this->output_size()[d].size());

    convolve(in, out);

    return out;
  }

  template <typename Block1,
	    typename Block2>
  Matrix<T, Block2>
  operator()(
    const_Matrix<T, Block1> in,
    Matrix<T, Block2>       out)
    VSIP_NOTHROW
  {
    typename accumulator_type::Scope scope(*this);

    for (dimension_type d=0; d<dim; ++d)
      assert(in.size(d) == this->input_size()[d].size());

    for (dimension_type d=0; d<dim; ++d)
      assert(out.size(d) == this->output_size()[d].size());

    convolve(in, out);

    return out;
  }
#endif

  float impl_performance(char const* what) const
  {
    if      (!strcmp(what, "mops"))  return this->mops();
    else if (!strcmp(what, "time"))  return this->total();
    else if (!strcmp(what, "count")) return this->count();
    else return this->base_type::impl_performance(what);
  }
};

} // namespace vsip

#endif // VSIP_CORE_SIGNAL_CONV_HPP
