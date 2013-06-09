//
// Copyright (c) 2005, 2006, 2007 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef vsip_impl_signal_conv_hpp_
#define vsip_impl_signal_conv_hpp_

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <ovxx/domain_utils.hpp>
#include <vsip/impl/signal/types.hpp>
#include <ovxx/signal/conv.hpp>
#if OVXX_HAVE_CVSIP
# include <ovxx/cvsip/conv.hpp>
#endif

namespace ovxx
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
  typedef make_type_list<be::user,
			 be::cvsip,
			 be::generic>::type type;
};
} // namespace ovxx::dispatcher
} // namespace ovxx

namespace vsip
{

template <template <typename, typename> class V,
	  symmetry_type                       S,
	  support_region_type                 R,
	  typename                            T = VSIP_DEFAULT_VALUE_TYPE,
	  unsigned                            N = 0,
          alg_hint_type                       H = alg_time>
class Convolution : public ovxx::dispatcher::Dispatcher<
  ovxx::dispatcher::op::conv<ovxx::dim_of_view<V>::dim, S, R, T, N, H> >::type
{
  static dimension_type const dim = ovxx::dim_of_view<V>::dim;

  typedef typename ovxx::dispatcher::Dispatcher<
    ovxx::dispatcher::op::conv<dim, S, R, T, N, H> >::type base_type;

public:
  template <typename Block>
  Convolution(V<T, Block> filter_coeffs,
	      Domain<dim> const&   input_size,
	      length_type          decimation = 1)
    VSIP_THROW((std::bad_alloc)) : base_type(filter_coeffs, input_size, decimation)
  {
    OVXX_PRECONDITION(decimation >= 1);
    OVXX_PRECONDITION(S == nonsym ? (filter_coeffs.size() <=   input_size.size())
		      : (filter_coeffs.size() <= 2*input_size.size()));
  }

  Convolution(Convolution const&) VSIP_NOTHROW;
  Convolution& operator=(Convolution const&) VSIP_NOTHROW;
  ~Convolution() VSIP_NOTHROW {}

  template <typename B1, typename B2>
  Vector<T, B2>
  operator()(const_Vector<T, B1> in, Vector<T, B2> out) VSIP_NOTHROW
  {
    for (dimension_type d=0; d<dim; ++d)
      OVXX_PRECONDITION(in.size(d) == this->input_size()[d].size());
    for (dimension_type d=0; d<dim; ++d)
      OVXX_PRECONDITION(out.size(d) == this->output_size()[d].size());
    this->convolve(in, out);
    return out;
  }

  template <typename Block1, typename Block2>
  Matrix<T, Block2>
  operator()(const_Matrix<T, Block1> in, Matrix<T, Block2> out) VSIP_NOTHROW
  {
    for (dimension_type d=0; d<dim; ++d)
      OVXX_PRECONDITION(in.size(d) == this->input_size()[d].size());
    for (dimension_type d=0; d<dim; ++d)
      OVXX_PRECONDITION(out.size(d) == this->output_size()[d].size());
    this->convolve(in, out);
    return out;
  }
};

} // namespace vsip

#endif
