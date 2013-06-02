//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef vsip_impl_signal_corr_hpp_
#define vsip_impl_signal_corr_hpp_

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/impl/signal/types.hpp>
#include <ovxx/signal/corr.hpp>
#if OVXX_HAVE_CVSIP
# include <ovxx/cvsip/corr.hpp>
#endif

namespace ovxx
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
  typedef make_type_list<be::user,
                         be::cuda,
			 be::opt,
			 be::cvsip,
			 be::generic>::type type;
};

} // namespace ovxx::dispatcher
} // namespace ovxx

namespace vsip
{

template <template <typename, typename> class V,
	  support_region_type                 R,
	  typename                            T = VSIP_DEFAULT_VALUE_TYPE,
	  unsigned                            N = 0,
          alg_hint_type                       A = alg_time>
class Correlation
  : public ovxx::dispatcher::Dispatcher<
      ovxx::dispatcher::op::corr<ovxx::dim_of_view<V>::dim,
				 R, T, N, A> >::type
{
  static dimension_type const dim = ovxx::dim_of_view<V>::dim;

  typedef typename 
  ovxx::dispatcher::Dispatcher<
    ovxx::dispatcher::op::corr<dim, R, T, N, A> >::type base_type;

public:
  Correlation(Domain<dim> const&   ref_size,
	      Domain<dim> const&   input_size)
    VSIP_THROW((std::bad_alloc))
  : base_type(ref_size, input_size)
  {}

  Correlation(Correlation const& corr) VSIP_NOTHROW;
  Correlation& operator=(Correlation const&) VSIP_NOTHROW;
  ~Correlation() VSIP_NOTHROW {}

  support_region_type support() const VSIP_NOTHROW { return R;}

  template <typename Block0, typename Block1, typename Block2>
  Vector<T, Block2>
  operator()(bias_type               bias,
	     const_Vector<T, Block0> ref,
	     const_Vector<T, Block1> in,
	     Vector<T, Block2>       out)
    VSIP_NOTHROW
  {
    for (dimension_type d=0; d<dim; ++d)
    {
      OVXX_PRECONDITION(ref.size(d) == this->reference_size()[d].size());
      OVXX_PRECONDITION(in.size(d)  == this->input_size()[d].size());
      OVXX_PRECONDITION(out.size(d) == this->output_size()[d].size());
    }

    this->correlate(bias, ref, in, out);

    return out;
  }

  template <typename Block0, typename Block1, typename Block2>
  Matrix<T, Block2>
  operator()(bias_type               bias,
	     const_Matrix<T, Block0> ref,
	     const_Matrix<T, Block1> in,
	     Matrix<T, Block2>       out)
    VSIP_NOTHROW
  {
    for (dimension_type d=0; d<dim; ++d)
    {
      OVXX_PRECONDITION(ref.size(d) == this->reference_size()[d].size());
      OVXX_PRECONDITION(in.size(d)  == this->input_size()[d].size());
      OVXX_PRECONDITION(out.size(d) == this->output_size()[d].size());
    }

    this->correlate(bias, ref, in, out);
    return out;
  }
};

} // namespace vsip

#endif
