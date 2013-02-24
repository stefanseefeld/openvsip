/* Copyright (c) 2005 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/signal/freqswap.hpp
    @author  Don McCoy
    @date    2005-11-29
    @brief   VSIPL++ Library: Frequency swap functions [signal.freqswap]
*/

#ifndef VSIP_CORE_SIGNAL_FREQSWAP_HPP
#define VSIP_CORE_SIGNAL_FREQSWAP_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/core/domain_utils.hpp>
#include <vsip/core/block_traits.hpp>
#include <vsip/core/expr/unary_block.hpp>
#include <vsip/core/dispatch.hpp>
#ifndef VSIP_IMPL_REF_IMPL
# include <vsip/opt/dispatch.hpp>
# ifdef VSIP_IMPL_HAVE_CUDA
#  include <vsip/opt/cuda/eval_freqswap.hpp>
# endif
#endif

/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{
namespace impl
{

template <dimension_type> 
struct Freqswap_worker;

/// Specialization for 1D swap
template <>
struct Freqswap_worker<1>
{
  template <typename ResultBlock, typename ArgumentBlock>
  static void apply(ResultBlock &out, ArgumentBlock const &in)
  {
    length_type const M = in.size();

    index_type const ia = M % 2;  // adjustment to get source index

    typename ArgumentBlock::value_type mid = in.get(M / 2);

    for (index_type i = 0, ii = M / 2; i < M / 2; ++i, ++ii)
    {
      // Be careful to allow 'out' to alias 'in'
      typename ArgumentBlock::value_type tmp = in.get(ii + ia);
      out.put(ii, in.get(i));
      out.put(i, tmp);
    }

    // if odd, fill in the last row/column(s)
    if (ia)
      out.put(M-1, mid);
  }
};

/// Specialization for 2D swap
template <>
struct Freqswap_worker<2>
{
  template <typename ResultBlock, typename ArgumentBlock>
  static void apply(ResultBlock &out, ArgumentBlock const &in)
  {
    length_type const M = in.size(2, 0);
    length_type const N = in.size(2, 1);

    index_type const ia = M % 2;  // adjustment to get source index
    index_type const ja = N % 2;

    if (is_same_block(in, out))
    {
      // If in-place, use algorithm that trades O(1) temporary storage
      // for extra copies.

      // First swap rows.
      for (index_type i=0; i < M; ++i)
      {
	typename ArgumentBlock::value_type mid = in.get(i, N / 2);

	for (index_type j = 0, jj = N / 2; j < N / 2; ++j, ++jj)
	{
	  // Be careful to allow 'out' to alias 'in'
	  typename ArgumentBlock::value_type tmp = in.get(i, jj + ja);
	  out.put(i, jj, in.get(i, j));
	  out.put(i, j, tmp);
	}

	// if odd, fill in the last row/column(s)
	if (ja) out.put(i, N-1, mid);
      }

      // Second, swap columns.
      for (index_type j=0; j < N; ++j)
      {
	typename ArgumentBlock::value_type mid = in.get(M / 2, j);

	for (index_type i = 0, ii = M / 2; i < M / 2; ++i, ++ii)
	{
	  // Be careful to allow 'out' to alias 'in'
	  typename ArgumentBlock::value_type tmp = in.get(ii + ia, j);
	  out.put(ii, j, in.get(i, j));
	  out.put(i,  j, tmp);
	}

	// if odd, fill in the last row/column(s)
	if (ia) out.put(M-1, j, mid);
      }
    }
    else
    {
      // equiv. to out[i,j] = in[(M/2 + i) mod M,(N/2 + i) mod N], 
      //   where i = 0 --> M - 1 and j = 0 --> N - 1
      for (index_type i = 0, ii = M / 2; i < M / 2; ++i, ++ii)
      {
	for (index_type j = 0, jj = N / 2; j < N / 2; ++j, ++jj)
	{
	  typename ArgumentBlock::value_type tmp = in.get(ii + ia, jj + ja);
	  out.put(ii, jj, in.get(i, j));
	  out.put(i, j, tmp);
	  tmp = in.get(ii + ia, j);
	  out.put(ii, j, in.get(i, jj + ja));
	  out.put(i, jj, tmp);
	}
      }

      // if odd, fill in the last row/column(s)
      if (ia)
      {
	index_type i = M / 2;
	index_type ii = M - 1;
	for (index_type j = 0, jj = N / 2; j < N / 2; ++j, ++jj)
	{
	  out.put(ii, jj, in.get(i, j));
	  out.put(ii,  j, in.get(i,jj + ja));
	}
      }
      if (ja)
      {
	index_type j = N / 2;
	index_type jj = N - 1;
	for (index_type i = 0, ii = M / 2; i < M / 2; ++i, ++ii)
	{
	  out.put(ii, jj, in.get(i      , j));
	  out.put( i, jj, in.get(ii + ia, j));
	}
      }
      if (ia && ja) out.put(M - 1, N - 1, in.get(M / 2, N / 2));
    }
  }
};
} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{
#ifndef VSIP_IMPL_REF_IMPL
template<>
struct List<op::freqswap>
{
  typedef Make_type_list<be::cuda,
			 be::generic>::type type;
};
#endif

template <typename ResultBlock,
	  typename ArgumentBlock>
struct Evaluator<op::freqswap, be::generic,
                 void(ResultBlock &, ArgumentBlock const &)>
{
  static bool const ct_valid = true;
  static bool rt_valid(ResultBlock &, ArgumentBlock const &) { return true;}

  static void exec(ResultBlock &result, ArgumentBlock const &argument)
  {
    vsip::impl::Freqswap_worker<ArgumentBlock::dim>::apply(result, argument);
  }
};
} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

namespace vsip
{
namespace impl
{

template <typename Block>
class Freqswap
{
  typedef typename View_block_storage<Block>::plain_type block_ref_type;

public:
  typedef typename Block::map_type map_type;
  typedef typename Block::value_type result_type;
  static dimension_type const dim = Block::dim;

  typedef Freqswap<typename Distributed_local_block<Block>::type> local_type;

  Freqswap(block_ref_type in) : in_(in) {}

  length_type size(dimension_type block_dim, dimension_type d) const
  {
    assert(block_dim == dim);
    return in_.size(block_dim, d);
  }
  length_type size() const { return in_.size();}
  map_type const &map() const { return in_.map();}
  Block const &arg() const { return in_;}

  template <typename ResultBlock>
  void apply(ResultBlock &out) const
  {
    namespace d = vsip_csl::dispatcher;
#ifdef VSIP_IMPL_REF_IMPL
    Evaluator<d::op::freqswap, d::be::generic,
      void(ResultBlock&, Block const&)>::exec(out, in_);
#else
    vsip_csl::dispatch<d::op::freqswap, void, ResultBlock &, Block const &>(out, in_);
#endif
  }
  local_type local() const { return local_type(get_local_block(in_));}

private:
  block_ref_type in_;
};

} // namespace impl

/// Swaps halves of a vector, or quadrants of a matrix, to remap zero 
/// frequencies from the origin to the middle.
template <template <typename, typename> class const_View,
          typename T,
          typename B>
const_View<T, impl::expr::Unary<impl::Freqswap, B> const>
freqswap(const_View<T, B> in) VSIP_NOTHROW
{
  typedef impl::expr::Unary<impl::Freqswap, B> const block_type;
  impl::Freqswap<B> fs(in.block());
  return const_View<T, block_type>(block_type(fs));
}

} // namespace vsip

#endif // VSIP_CORE_SIGNAL_FREQSWAP_HPP
