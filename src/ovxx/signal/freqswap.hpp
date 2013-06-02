//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_signal_freqswap_hpp_
#define ovxx_signal_freqswap_hpp_

#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <ovxx/domain_utils.hpp>
#include <ovxx/block_traits.hpp>
#include <ovxx/expr.hpp>
#include <ovxx/dispatch.hpp>

namespace ovxx
{
namespace signal
{
namespace detail
{

template <dimension_type> 
struct Freqswap;

/// Specialization for 1D swap
template <>
struct Freqswap<1>
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
struct Freqswap<2>
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
} // namespace ovxx::signal::detail
} // namespace ovxx::signal

namespace dispatcher
{
template<>
struct List<op::freqswap>
{
  typedef make_type_list<be::cuda,
			 be::generic>::type type;
};

template <typename ResultBlock,
	  typename ArgumentBlock>
struct Evaluator<op::freqswap, be::generic,
                 void(ResultBlock &, ArgumentBlock const &)>
{
  static bool const ct_valid = true;
  static bool rt_valid(ResultBlock &, ArgumentBlock const &) { return true;}

  static void exec(ResultBlock &result, ArgumentBlock const &argument)
  {
    signal::detail::Freqswap<ArgumentBlock::dim>::apply(result, argument);
  }
};
} // namespace ovxx::dispatcher

namespace expr
{
namespace op
{
template <typename B>
class Freqswap
{
  typedef typename block_traits<B>::plain_type block_ref_type;

public:
  typedef typename B::map_type map_type;
  typedef typename B::value_type result_type;
  static dimension_type const dim = B::dim;

  typedef Freqswap<typename distributed_local_block<B>::type> local_type;

  Freqswap(block_ref_type in) : in_(in) {}

  length_type size(dimension_type block_dim, dimension_type d) const
  {
    OVXX_PRECONDITION(block_dim == dim);
    return in_.size(block_dim, d);
  }
  length_type size() const { return in_.size();}
  map_type const &map() const { return in_.map();}
  B const &arg() const { return in_;}

  template <typename B1>
  void apply(B1 &out) const
  {
    namespace d = ovxx::dispatcher;
    dispatch<d::op::freqswap, void, B1 &, B const &>(out, in_);
  }
  local_type local() const { return local_type(get_local_block(in_));}

private:
  block_ref_type in_;
};

} // namespace ovxx::expr::op
} // namespace ovxx::expr
} // namespace ovxx

#endif
