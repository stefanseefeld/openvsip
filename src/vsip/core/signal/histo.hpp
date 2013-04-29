//
// Copyright (c) 2005 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_CORE_SIGNAL_HISTO_HPP
#define VSIP_CORE_SIGNAL_HISTO_HPP

#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/core/dispatch.hpp>
#ifndef VSIP_IMPL_REF_IMPL
# include <vsip/opt/dispatch.hpp>
# ifdef VSIP_IMPL_CBE_SDK
#  include <vsip/opt/cbe/ppu/signal.hpp>
# endif
#endif

namespace vsip
{
namespace impl
{
template <typename T>
inline index_type
hist_bin(T min, T max, T delta, length_type num, T value)
{
  if (value < min) return 0;
  else if (value >= max) return num - 1;
  else return (index_type)(((value - min) / delta) + 1);
} 

template <dimension_type D>
struct Histogram_accumulator;

template <>
struct Histogram_accumulator<1>
{
  template <typename T, typename HBlock, typename DBlock>
  static void exec(T min, T max, HBlock &hist, DBlock const &input)
  {
    length_type num = hist.size();
    T delta = (max - min) / (num - 2);

    for (index_type i = 0; i < input.size(1, 0); ++i)
    {
      index_type n = hist_bin(min, max, delta, num, input.get(i));
      hist.put(n, hist.get(n) + 1);
    }
  }
};

template <>
struct Histogram_accumulator<2>
{
  template <typename T, typename HBlock, typename DBlock>
  static void exec(T min, T max, HBlock &hist, DBlock const &input)
  {
    length_type num = hist.size();
    T delta = (max - min) / (num - 2);

    for (index_type i = 0; i < input.size(2, 0); ++i)
      for (index_type j = 0; j < input.size(2, 1); ++j)
      {
	index_type n = hist_bin(min, max, delta, num, input.get(i, j));
	hist.put(n, hist.get(n) + 1);
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
struct List<op::hist>
{
  typedef Make_type_list<be::user,
			 be::cbe_sdk,
			 be::generic>::type type;
};
#endif

template <typename T, typename HBlock, typename DBlock>
struct Evaluator<op::hist, be::generic, void(T, T, HBlock &, DBlock const &)>
{
  static bool const ct_valid = true;
  static bool rt_valid(T, T, HBlock &, DBlock const &) { return true;}

  static void exec(T min, T max, HBlock &hist, DBlock const &input)
  {
    vsip::impl::Histogram_accumulator<DBlock::dim>::exec(min, max, hist, input);
  }
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

namespace vsip
{


template <template <typename, typename> class const_View = const_Vector,
          typename T = VSIP_DEFAULT_VALUE_TYPE>
class Histogram
{
  typedef Dense<1, int> hist_block_type;
public:
  // Constructor and destructor [signal.histo.constructors]
  Histogram(T min_value, T max_value, length_type num_bin)
    VSIP_THROW((std::bad_alloc))
    : min_(min_value),
      max_(max_value),
      hist_(num_bin, 0)
  {
    assert(min_ < max_);
    assert(num_bin >= 3);
  }

  /// This constructor, albeit not required by the VSIPL++ spec, is
  /// necessary to implement the C-VSIPL bindings. They allow an external
  /// view to be used to accumulate the histogram values over multiple
  /// calls outside any persistent Histogram<> object / state.
  template <typename Block>
  Histogram(T min_value, T max_value, const_Vector<int, Block> hist)
    VSIP_THROW((std::bad_alloc))
    : min_(min_value),
      max_(max_value),
      hist_(hist.size())
  {
    assert(min_ < max_);
    assert(hist_.size() >= 3);
    hist_ = hist;
  }

  ~Histogram() VSIP_NOTHROW
  {}

  // Histogram operators [signal.histo.operators]
  template <typename Block>
  const_Vector<scalar_i>
  operator()(const_Vector<T, Block> input, bool accumulate = false)
    VSIP_NOTHROW
  {
    namespace d = vsip_csl::dispatcher;

    if (accumulate == false) hist_ = 0;

#ifdef VSIP_IMPL_REF_IMPL
    Evaluator<d::op::hist, be::generic, void(T, T, hist_block_type &, Block const&)>::
      exec(min_, max_, hist_.block(), input.block());
#else
    vsip_csl::dispatch<d::op::hist, void, T, T, hist_block_type &, Block const &>
      (min_, max_, hist_.block(), input.block());
#endif
    return hist_;
  }

  template <typename Block>
  const_Vector<scalar_i>
  operator()(const_Matrix<T, Block> input, bool accumulate = false)
    VSIP_NOTHROW
  {
    namespace d = vsip_csl::dispatcher;

    if (accumulate == false) hist_ = 0;

#ifdef VSIP_IMPL_REF_IMPL
    Evaluator<d::op::hist, be::generic, void(T, T, hist_block_type &, Block const&)>::
      exec(min_, max_, hist_.block(), input.block());
#else
    vsip_csl::dispatch<d::op::hist, void, T, T, hist_block_type &, Block const &>
      (min_, max_, hist_.block(), input.block());
#endif

    return hist_;
  }

private:
  T min_;
  T max_;
  Vector<scalar_i> hist_;
};

} // namespace vsip

#endif // VSIP_CORE_SIGNAL_HISTO_HPP
