/* Copyright (c) 2005 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/signal/corr.hpp
    @author  Jules Bergmann
    @date    2005-10-05
    @brief   VSIPL++ Library: Correlation class [signal.correl].
*/

#ifndef VSIP_CORE_SIGNAL_CORR_HPP
#define VSIP_CORE_SIGNAL_CORR_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/core/signal/types.hpp>
#include <vsip/core/profile.hpp>
#include <vsip/core/signal/corr_common.hpp>
#if !VSIP_IMPL_REF_IMPL
# include <vsip/opt/signal/corr_ext.hpp>
# include <vsip/opt/signal/corr_opt.hpp>
# ifdef VSIP_IMPL_CBE_SDK
#  include <vsip/opt/cbe/cml/corr.hpp>
# endif
#endif
#if VSIP_IMPL_HAVE_CVSIP
# include <vsip/core/cvsip/corr.hpp>
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
          support_region_type R,
          typename T,
          unsigned int N,
          alg_hint_type H>
struct List<op::corr<D, R, T, N, H> >
{
  typedef Make_type_list<be::user,
			 be::cml,
			 be::opt,
			 be::cvsip,
			 be::generic>::type type;
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif

namespace vsip
{

template <template <typename, typename> class ConstViewT,
	  support_region_type                 Supp,
	  typename                            T = VSIP_DEFAULT_VALUE_TYPE,
	  unsigned                            N_times = 0,
          alg_hint_type                       A_hint = alg_time>
class Correlation
  : public impl::profile::Accumulator<impl::profile::signal>,
#if VSIP_IMPL_REF_IMPL
    public impl::cvsip::Correlation<impl::Dim_of_view<ConstViewT>::dim,
                                    Supp, T, N_times, A_hint>
#else
    public vsip_csl::dispatcher::Dispatcher<
      vsip_csl::dispatcher::op::corr<impl::Dim_of_view<ConstViewT>::dim,
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
  typedef impl::cvsip::Correlation<dim, Supp, T, N_times, A_hint> base_type;
#else
  typedef typename 
  vsip_csl::dispatcher::Dispatcher<
    vsip_csl::dispatcher::op::corr<dim, Supp, T, N_times, A_hint> >::type base_type;
#endif

public:
  Correlation(Domain<dim> const&   ref_size,
	      Domain<dim> const&   input_size)
    VSIP_THROW((std::bad_alloc))
      : accumulator_type(impl::signal_detail::Description<dim, T>::tag
                         ("Correlation",
                          impl::extent(input_size), impl::extent(ref_size)),
                         impl::signal_detail::Op_count_corr<dim, T>::value
                         (impl::extent(input_size), impl::extent(ref_size))),
        base_type(ref_size, input_size)
  {}

  Correlation(Correlation const& corr) VSIP_NOTHROW;
  Correlation& operator=(Correlation const&) VSIP_NOTHROW;
  ~Correlation() VSIP_NOTHROW {}


  // Accessors
public:
  support_region_type support() const VSIP_NOTHROW     { return Supp; }


  // Correlation operators.
public:
  template <typename Block0,
	    typename Block1,
	    typename Block2>
  Vector<T, Block2>
  operator()(
    bias_type               bias,
    const_Vector<T, Block0> ref,
    const_Vector<T, Block1> in,
    Vector<T, Block2>       out)
    VSIP_NOTHROW
  {
    typename accumulator_type::Scope scope(*this);

    for (dimension_type d=0; d<dim; ++d)
    {
      assert(ref.size(d) == this->reference_size()[d].size());
      assert(in.size(d)  == this->input_size()[d].size());
      assert(out.size(d) == this->output_size()[d].size());
    }

    impl_correlate(bias, ref, in, out);

    return out;
  }

  template <typename Block0,
	    typename Block1,
	    typename Block2>
  Matrix<T, Block2>
  operator()(
    bias_type               bias,
    const_Matrix<T, Block0> ref,
    const_Matrix<T, Block1> in,
    Matrix<T, Block2>       out)
    VSIP_NOTHROW
  {
    typename accumulator_type::Scope scope(*this);

    for (dimension_type d=0; d<dim; ++d)
    {
      assert(ref.size(d) == this->reference_size()[d].size());
      assert(in.size(d)  == this->input_size()[d].size());
      assert(out.size(d) == this->output_size()[d].size());
    }

    impl_correlate(bias, ref, in, out);

    return out;
  }

  float impl_performance(char* what) const
  {
    if      (!strcmp(what, "mops"))  return this->mflops();
    else if (!strcmp(what, "time"))  return this->total();
    else if (!strcmp(what, "count")) return this->count();
    else return this->base_type::impl_performance(what);
  }
};

} // namespace vsip

#endif // VSIP_CORE_SIGNAL_CORR_HPP
