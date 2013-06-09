//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_expr_combine_hpp_
#define ovxx_expr_combine_hpp_

#include <ovxx/expr/traversal.hpp>

namespace ovxx
{
namespace expr
{
namespace transform
{
template <typename F, typename B> struct return_type;

// FIXME: For some reason this forward-declaration is required
//        for overload resolution to work a little later in this file.
//        Without it, the parallel/expr.cpp test fails with a compile-time
//        failure.
template <typename F,
	  template <typename, typename, typename> class O,
	  typename B1, typename B2, typename B3, bool E>
typename return_type<F, Ternary<O, B1, B2, B3, E> const>::type
combine(F const &, Ternary<O, B1, B2, B3, E> const &);

template <typename F,
	  template <typename, typename, typename> class O,
	  typename B1, typename B2, typename B3, bool E>
void
apply(F const &, Ternary<O, B1, B2, B3, E> const &);

template <typename F, typename B>
struct return_type
{
  typedef typename F::template return_type<B>::type type;
  typedef typename F::template tree_type<B>::type tree_type;
};

template <typename F, typename B>
typename return_type<F, B>::type
combine(F const &func, B const &block) { return func.apply(block);}

template <typename F, typename B>
void apply(F const &func, B const &block) { func.apply(block);}

template <typename F, dimension_type D, typename T>
struct return_type<F, Scalar<D, T> const>
{
  typedef Scalar<D, T> block_type;
  typedef typename F::template return_type<block_type>::type
  type;
  typedef typename F::template tree_type<block_type>::type
  tree_type;
};

template <typename F, dimension_type D, typename T>
struct return_type<F, Scalar<D, T> >
  : return_type<F, Scalar<D, T> const>
{};

template <typename F, dimension_type D, typename T>
typename return_type<F, Scalar<D, T> const>::type
combine(F const &func, Scalar<D, T> const &block)
{
  return func.apply(block);
}

template <typename F, dimension_type D, typename T>
void
apply(F const &func, Scalar<D, T> const &block)
{
  func.apply(block);
}

template <typename F, typename B, dimension_type D>
struct return_type<F, Sliced<B, D> >
{
  typedef Sliced<B, D> block_type;
  typedef typename F::template return_type<block_type>::type type;
  typedef typename F::template tree_type<block_type>::type tree_type;
};

template <typename F, typename B, dimension_type D>
typename return_type<F, Sliced<B, D> >::type
combine(F const &func, Sliced<B, D> const &block)
{
  return func.apply(block);
}

template <typename F, typename B, dimension_type D1, dimension_type D2>
struct return_type<F, Sliced2<B, D1, D2> >
{
  typedef Sliced2<B, D1, D2> block_type;
  typedef typename F::template return_type<block_type>::type type;
  typedef typename F::template tree_type<block_type>::type tree_type;
};

template <typename F, typename B, dimension_type D1, dimension_type D2>
typename return_type<F, Sliced2<B, D1, D2> >::type
combine(F const &func, Sliced2<B, D1, D2> const &block)
{
  return func.apply(block);
}

template <typename F,
	  template <typename, typename> class O,
	  typename B1, typename B2, bool E>
struct return_type<F, Binary<O, B1, B2, E> const>
{
  typedef Binary<O,
		typename return_type<F, B1>::tree_type,
		typename return_type<F, B2>::tree_type,
		E> const tree_type;
  typedef tree_type type;
};

template <typename F,
	  template <typename, typename> class O,
	  typename B1, typename B2, bool E>
struct return_type<F, Binary<O, B1, B2, E> >
  : return_type<F, Binary<O, B1, B2, E> const>
{};

template <typename F,
	  template <typename, typename> class O,
	  typename B1, typename B2, bool E>
typename return_type<F, Binary<O, B1, B2, E> const>::type
combine(F const &func, Binary<O, B1, B2, E> const &block)
{
  typedef typename 
    return_type<F, Binary<O, B1, B2, E> const>::type
    block_type;

  return block_type(combine(func, block.arg1()),
		    combine(func, block.arg2()));
}

template <typename F,
	  template <typename, typename> class O,
	  typename B1, typename B2, bool E>
void
apply(F const &func, Binary<O, B1, B2, E> const &block)
{
  apply(func, block.arg1());
  apply(func, block.arg2());
}

template <typename F,
	  template <typename, typename, typename> class O,
	  typename B1, typename B2, typename B3, bool E>
struct return_type<F, Ternary<O, B1, B2, B3, E> const>
{
  typedef Ternary<O,
		  typename return_type<F, B1>::tree_type,
		  typename return_type<F, B2>::tree_type,
		  typename return_type<F, B3>::tree_type,
		  E>
    const tree_type;
  typedef tree_type type;
};

template <typename F,
	  template <typename, typename, typename> class O,
	  typename B1, typename B2, typename B3, bool E>
struct return_type<F, Ternary<O, B1, B2, B3, E> >
  : return_type<F, Ternary<O, B1, B2, B3, E> const>
{};

template <typename F,
	  template <typename, typename, typename> class O,
	  typename B1, typename B2, typename B3, bool E>
typename return_type<F, Ternary<O, B1, B2, B3, E> const>::type
combine(F const &func, Ternary<O, B1, B2, B3, E> const &block)
{
  typedef typename 
    return_type<F, Ternary<O, B1, B2, B3, E> const>::type
    block_type;

  return block_type(combine(func, block.arg1()),
		    combine(func, block.arg2()),
		    combine(func, block.arg3()));
}

template <typename F,
	  template <typename, typename, typename> class O,
	  typename B1, typename B2, typename B3, bool E>
void
apply(F const &func, Ternary<O, B1, B2, B3, E> const &block)
{
  apply(func, block.arg1());
  apply(func, block.arg2());
  apply(func, block.arg3());
}

template <typename F, typename B>
struct return_type<F, Transposed<B> >
{
  typedef Transposed<B> block_type;
  typedef typename F::template return_type<block_type>::type
    type;
  typedef typename F::template tree_type<block_type>::type
    tree_type;
};

template <typename F, typename B>
typename return_type<F, Transposed<B> >::type
combine(F const &func, Transposed<B> const &block)
{
  return func.apply(block);
}

template <typename F,
	  template <typename> class O, typename B, bool E>
struct return_type<F, Unary<O, B, E> const>
{
  typedef Unary<O, typename return_type<F, B>::tree_type,
		E> const tree_type;
  typedef tree_type type;
};

template <typename F,
	  template <typename> class O, typename B, bool E>
struct return_type<F, Unary<O, B, E> >
  : return_type<F, Unary<O, B, E> const>
{};

template <typename F,
	  template <typename> class O, typename B, bool E>
typename return_type<F, Unary<O, B, E> const>::type
combine(F const &func, Unary<O, B, E> const &block)
{
  typedef typename return_type<F, Unary<O, B, E> const>::type
    block_type;

  return block_type(combine(func, block.arg()));
}

template <typename F,
	  template <typename> class O, typename B, bool E>
void
apply(F const &func, Unary<O, B, E> const &block)
{
  apply(func, block.arg());
}

template <typename F,
	  dimension_type D, typename B1, typename B2>
struct return_type<F, Vmmul<D, B1, B2> const>
{
  typedef Vmmul<D,
		typename return_type<F, B1>::tree_type,
		typename return_type<F, B2>::tree_type>
    const tree_type;
  typedef tree_type type;
};

template <typename F,
	  dimension_type D, typename B1, typename B2>
struct return_type<F, Vmmul<D, B1, B2> >
{
  typedef Vmmul<D,
		typename return_type<F, B1>::tree_type,
		typename return_type<F, B2>::tree_type>
    const tree_type;
  typedef tree_type type;
};

template <typename F,
	  dimension_type D, typename B1, typename B2>
typename return_type<F, Vmmul<D, B1, B2> const>::type
combine(F const &func, Vmmul<D, B1, B2> const &block)
{
  typedef typename 
    return_type<F, Vmmul<D, B1, B2> const>::type
    block_type;

  return block_type(combine(func, block.get_vblk()),
		    combine(func, block.get_mblk()));
}

template <typename F,
	  dimension_type D, typename B1, typename B2>
void
apply(F const &func, Vmmul<D, B1, B2> const &block)
{
  apply(func, block.get_vblk());
  apply(func, block.get_mblk());
}

template <typename F, dimension_type D, typename G>
struct return_type<F, Generator<D, G> const>
{
  typedef Generator<D, G> block_type;
  typedef typename F::template return_type<block_type>::type type;
  typedef typename F::template tree_type<block_type>::type tree_type;
};

template <typename F, dimension_type D, typename G>
struct return_type<F, Generator<D, G> >
{
  typedef Generator<D, G> block_type;
  typedef typename F::template return_type<block_type>::type type;
  typedef typename F::template tree_type<block_type>::type tree_type;
};

template <typename F, dimension_type D, typename G>
typename return_type<F, Generator<D, G> const>::type
combine(F const &func, Generator<D, G> const &block)
{
  return func.apply(block);
}

template <typename F, dimension_type D, typename G>
void
apply(F const &, Generator<D, G> const &) {}

template <typename F, typename B>
struct return_type<F, Subset<B> >
{
  typedef Subset<B> block_type;
  typedef typename F::template return_type<block_type>::type type;
  typedef typename F::template tree_type<block_type>::type tree_type;
};

template <typename F, typename B>
typename return_type<F, Subset<B> >::type
combine(F const &func, Subset<B> const &block)
{
  return func.apply(block);
}

template <typename F,
	  dimension_type D, typename T, typename O, typename M>
struct return_type<F, vsip::Dense<D, T, O, M> >
{
  typedef vsip::Dense<D, T, O, M> block_type;
  typedef typename F::template return_type<block_type>::type
    type;
  typedef typename F::template tree_type<block_type>::type
    tree_type;
};

template <typename F,
	  dimension_type D, typename T, typename O, typename M>
typename return_type<F, vsip::Dense<D, T, O, M> >::type
combine(F const &func, vsip::Dense<D, T, O, M> const &block)
{
  return func.apply(block);
}

} // namespace ovxx::expr::transform
} // namespace ovxx::expr
} // namespace ovxx

#endif
