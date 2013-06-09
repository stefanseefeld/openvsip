//
// Copyright (c) 2005 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef vsip_selgen_hpp_
#define vsip_selgen_hpp_

#include <ovxx/block_traits.hpp>
#include <ovxx/expr/unary.hpp>
#include <ovxx/expr/generator.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>

namespace vsip
{
namespace impl
{

template <typename T, typename B1, typename B2>
length_type
indexbool(const_Vector<T, B1> source, Vector<Index<1>, B2> indices)
{
  index_type cursor = 0;
  for (index_type i = 0; i != source.size(); ++i)
    if (source.get(i) && cursor++ < indices.size())
      indices.put(cursor - 1, Index<1>(i));
  return cursor;
}

template <typename T, typename B1, typename B2>
length_type
indexbool(const_Matrix<T, B1> source, Vector<Index<2>, B2> indices)
{
  index_type cursor = 0;
  for (index_type r = 0; r != source.size(0); ++r)
    for (index_type c = 0; c != source.size(1); ++c)
      if (source.get(r, c) && cursor++ < indices.size())
	indices.put(cursor - 1, Index<2>(r, c));
  return cursor;
}

/// Generator functor for ramp.
template <typename T>
class Ramp_generator
{
  // Typedefs.
public:
  typedef T result_type;

  // Constructor.
public:
  Ramp_generator(T a, T b) : a_(a), b_(b) {}


  // Operator
public:
  T operator()(index_type i) const
  {
    return a_ + T(i)*b_;
  }

  // Member data.
private:
  T a_;
  T b_;
};

} // namespace vsip::impl

template <typename Predicate,
	  typename T1, typename T2,
	  typename B1, typename B2>
index_type
first(index_type begin, Predicate p, 
      const_Vector<T1, B1> v, const_Vector<T2, B2> w) VSIP_NOTHROW
{
  assert(view_domain(v).element_conformant(view_domain(w)));
  if (begin > v.size()) 
    return begin;
  for (index_type i = begin, end = v.size(); i != end; ++i)
    if (p(v.get(i), w.get(i)))
      return i;
  return v.size();
}

template <template <typename, typename> class const_View,
	  typename T, typename B1, typename B2>
length_type
indexbool(const_View<T, B1> source, 
	  Vector<Index<const_View<T, B2>::dim>, B2> indices)
VSIP_NOTHROW
{
  return impl::indexbool(source, indices);
}

/// Returns values from `source`, at positions given by `indices`.
template <template <typename, typename> class const_View,
	  typename T, typename B1, typename B2>
Vector<T, Dense<1, T> >
gather(const_View<T, B1> source,
       const_Vector<Index<const_View<T, B2>::dim>, B2> indices)
VSIP_NOTHROW
{
  Vector<T, Dense<1, T> > result(indices.size());
  for (index_type i = 0; i != indices.size(); ++i)
    result.put(i, get(source, indices.get(i)));
  return result;
}

/// Copies all values from `source` into `destination`, at the index
/// given by `indices`.
template <template <typename, typename> class View,
	  typename T, typename B1, typename B2, typename B3>
void
scatter(const_Vector<T, B1> source,
	const_Vector<Index<View<T, B3>::dim>, B2>  indices,
	View<T, B3> destination)
VSIP_NOTHROW
{
  for (index_type i = 0; i != indices.size(); ++i)
    put(destination, indices.get(i), source.get(i));
}

/// Generate a linear ramp: :equation:`v[i] = a + i * b`
///
/// Requires:
///   :len: to be output vector size (len > 0)
template <typename T>
const_Vector<T, ovxx::expr::Generator<1, impl::Ramp_generator<T> > const>
ramp(T a, T b, length_type len) VSIP_NOTHROW
{
  assert(len > 0);

  typedef impl::Ramp_generator<T> generator_type;
  typedef ovxx::expr::Generator<1, generator_type> const block_type;

  generator_type gen(a, b);
  block_type block(ovxx::Length<1>(len), gen);

  return const_Vector<T, block_type>(block);
}

namespace impl
{
template <typename Tout, typename Tin1>
struct clip_wrapper
{
  template <typename Tin0>
  struct clip_functor
  {
    typedef Tout result_type;
    static char const* name() { return "clip"; }                
    result_type operator()(Tin0 t) const 
    {
      return t <= lower_threshold ? lower_clip_value 
	: t < upper_threshold ? t
	: upper_clip_value;
    }

    Tin1 lower_threshold;
    Tin1 upper_threshold;
    result_type lower_clip_value;
    result_type upper_clip_value;
  };
  template <typename Tin0>
  struct invclip_functor
  {
    typedef Tout result_type;
    static char const* name() { return "invclip"; }                
    result_type operator()(Tin0 t) const 
    {
      return t < lower_threshold ? t
	: t < middle_threshold ? lower_clip_value
	: t <= upper_threshold ? upper_clip_value
	: t;
    }

    Tin1 lower_threshold;
    Tin1 middle_threshold;
    Tin1 upper_threshold;
    result_type lower_clip_value;
    result_type upper_clip_value;
  };
};

template <typename Tout, typename Tin0, typename Tin1, 
	  template <typename, typename> class const_View,
	  typename Block>
struct Clip_return_type
{
  typedef
    const_View<Tout,
	       ovxx::expr::Unary<clip_wrapper<Tout, Tin1>::template clip_functor,
			   Block, true> const>
    type;
};

template <typename Tout, typename Tin0, typename Tin1, 
	  template <typename, typename> class const_View,
	  typename Block>
struct Invclip_return_type
{
  typedef
    const_View<Tout,
	       ovxx::expr::Unary<clip_wrapper<Tout, Tin1>::template invclip_functor,
			   Block, true> const>
    type;
};
  
} // namespace vsip::impl

template <typename Tout, typename Tin0, typename Tin1,
	  template <typename, typename> class const_View,
	  typename Block>
typename impl::Clip_return_type<Tout, Tin0, Tin1, const_View, Block>::type
clip(const_View<Tin0, Block> v, Tin1 lower_threshold, Tin1 upper_threshold,
     Tout lower_clip_value, Tout upper_clip_value)
{
  typedef ovxx::expr::Unary<impl::clip_wrapper<Tout, Tin1>::template clip_functor,
    Block, true> block_type;

  typename impl::clip_wrapper<Tout, Tin1>::template clip_functor<Tin0> functor;
  functor.lower_threshold = lower_threshold;
  functor.upper_threshold = upper_threshold;
  functor.lower_clip_value = lower_clip_value;
  functor.upper_clip_value = upper_clip_value;

  return const_View<Tout, block_type const>(block_type(functor, v.block()));
}

template <typename Tout, typename Tin0, typename Tin1,
	  template <typename, typename> class const_View,
	  typename Block>
typename impl::Invclip_return_type<Tout, Tin0, Tin1, const_View, Block>::type
invclip(const_View<Tin0, Block> v,
	Tin1 lower_threshold, Tin1 middle_threshold, Tin1 upper_threshold,
	Tout lower_clip_value, Tout upper_clip_value)
{
  typedef ovxx::expr::Unary<impl::clip_wrapper<Tout, Tin1>::template invclip_functor,
    Block, true> block_type;

  typename impl::clip_wrapper<Tout, Tin1>::template invclip_functor<Tin0> functor;
  functor.lower_threshold = lower_threshold;
  functor.middle_threshold = middle_threshold;
  functor.upper_threshold = upper_threshold;
  functor.lower_clip_value = lower_clip_value;
  functor.upper_clip_value = upper_clip_value;

  return const_View<Tout, block_type const>(block_type(functor, v.block()));
}

namespace impl
{
/// Generic swapping of the content of two blocks.
template <typename Block1, typename Block2>
struct Swap
{
  static void apply(Block1 &block1, Block2 &block2)
  {
    assert(block1.size() == block2.size());
    for (index_type i = 0; i != block1.size(); ++i)
    {
      typename Block1::value_type tmp = block1.get(i);
      block1.put(i, block2.get(i));
      block2.put(i, tmp);
    }

  }
};
}

template <typename T1, typename T2,
	  template <typename, typename> class View,
	  typename Block1, typename Block2>
inline void
swap(View<T1, Block1> v, View<T2, Block2> w)
{
  impl::Swap<Block1, Block2>::apply(v.block(), w.block());
}

} // namespace vsip

#endif // VSIP_SELGEN_HPP
