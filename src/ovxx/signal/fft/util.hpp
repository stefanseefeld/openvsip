//
// Copyright (c) 2006 - 2010 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_signal_fft_util_hpp_
#define ovxx_signal_fft_util_hpp_

#include <vsip/support.hpp>
#include <ovxx/signal/fft/backend.hpp>
#include <ovxx/signal/fft/functor.hpp>
#include <ovxx/strided.hpp>
#include <ovxx/view/traits.hpp>
#include <ovxx/expr.hpp>
#include <ovxx/view/utils.hpp>

namespace ovxx
{
namespace signal
{
namespace fft
{

/// Determine whether the FFT size is a power of two.
inline bool is_power_of_two(unsigned size)
{
  return (size & (size - 1)) == 0;
}
template <dimension_type D>
inline bool 
is_power_of_two(Domain<D> const &dom)
{
  for (dimension_type d = 0; d != D; ++d)
    if (!is_power_of_two(dom[d].size())) return false;
  return true;
}

/// Determine the exponent (forward or inverse) of a given Fft
/// from its parameters.
template <typename I, typename O, int sD> struct exponent;
template <typename T, int sD>
struct exponent<T, complex<T>, sD> { static int const value = -1;};
template <typename T, int sD>
struct exponent<complex<T>, T, sD> { static int const value = 1;};
template <typename T>
struct exponent<T, T, -2> { static int const value = -1;};
template <typename T>
struct exponent<T, T, -1> { static int const value = 1;};

/// Determine the 'special dimension' of a real fft
/// from its parameters.
template <typename I, typename O, int S> 
struct axis { static int const value = S;};
template <typename T, int S>
struct axis<T, T, S> { static int const value = 0;};

/// Device to calculate the size of the input and output blocks
/// for complex and real ffts.
template <dimension_type D, typename I, typename O, int A>
struct io_size
{
  static Domain<D> size(Domain<D> const &dom) { return dom;}
};
template <dimension_type D, typename T, int A>
struct io_size<D, complex<T>, T, A>
{
  static Domain<D> size(Domain<D> const &dom) 
  {
    Domain<D> retn(dom);
    Domain<1> &mod = retn.impl_at(A);
    mod = Domain<1>(0, 1, mod.size() / 2 + 1); 
    return retn;
  }
};

/// Traits class to determine block type returned by Fft and Fftm
/// by_value operators.
template<typename T,
	 typename BlockT,
	 typename MapT = typename BlockT::map_type>
struct result
{
  static dimension_type const dim = BlockT::dim;
  typedef typename
  ovxx::Strided<dim, T, 
		Layout<dim, tuple<0,1,2>,
		       dense,
		       get_block_layout<BlockT>::storage_format>,
		MapT> block_type;

  typedef typename view_of<block_type>::type view_type;

  static view_type create(Domain<dim> const &dom, MapT const& map)
  { return create_view<view_type>(dom, map);}
};

/// Traits class to determine view type returned by Fft for
/// by_value operators with return-block optimization.
template <dimension_type Dim,
	  typename       InT,
	  typename       OutT,
	  typename       ViewT,
	  typename       WorkspaceT,
	  int            S>
struct Result_rbo
{
  typedef expr::op::fft<Dim,
			fft_backend<Dim, InT, OutT, S>,
			WorkspaceT> traits;
  typedef typename traits::template Functor<typename ViewT::block_type> functor_type;
  typedef expr::Unary<traits::template Functor, typename ViewT::block_type> block_type;
  typedef typename view_of<block_type const>::type view_type;
};

template <typename       InT,
	  typename       OutT,
	  typename       BlockT,
	  typename       WorkspaceT,
	  int            Axis,
	  int            Direction>
struct Result_fftm_rbo
{
  static dimension_type const dim = 2;
  typedef const_Matrix<InT, BlockT> in_view_type;
  typedef expr::op::fft<dim, fftm_backend<InT, OutT, Axis, Direction>,
			      WorkspaceT> traits;
  typedef typename traits::template Functor<BlockT> functor_type;
  typedef expr::Unary<traits::template Functor, BlockT> block_type;
  typedef typename view_of<block_type const>::type view_type;
};

} // namespace ovxx::signal::fft
} // namespace ovxx::signal
} // namespace ovxx

#endif
