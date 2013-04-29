/* Copyright (c) 2010 by CodeSourcery, Inc.  All rights reserved. */

/// Description
///   Parallel Iterator expression support

#ifndef vsip_csl_pi_is_stencil_expr_hpp_
#define vsip_csl_pi_is_stencil_expr_hpp_

#include <vsip_csl/pi/is_linear_expr.hpp>
#include <vsip/core/metaprogramming.hpp>

namespace vsip_csl
{
namespace pi
{
namespace impl
{

// Check whether two references refer to the same object,
// and if so, return its address
template <typename B1, typename B2>
inline bool
is_same_ptr_or_null(B1 const *b1, B2 const *b2, B1 const *&b)
{
  // This function overload is chosen if B1 contains a Call<> terminal.
  // If b2 is non-zero, B2 contains a Call<> terminal, too.
  // Since B1 != B2, we return false.
  if (b2 != 0)
  {
    b = 0;
    return false;
  }
  else
  {
    b = b1;
    return true;
  }
}

template <typename B1, typename B2>
inline bool
is_same_ptr_or_null(B1 const *b1, B2 const *b2, B2 const *&b)
{
  // This function overload is chosen if B2 contains a Call<> terminal.
  // If b1 is non-zero, B1 contains a Call<> terminal, too.
  // Since B1 != B2, we return false.
  if (b1 != 0)
  {
    b = 0;
    return false;
  }
  else
  {
    b = b2;
    return true;
  }
}

template <typename B>
inline bool
is_same_ptr_or_null(B const *b1, B const *b2, B const *&b)
{
  // This function overload is chosen if b1 and b2 contain Call<> terminals.
  // We return true if the two pointers are in fact referring to the same object.
  b = b1 == b2 ? b1 : 0;
  return b;
}
} // namespace vsip_csl::pi::impl

/// A linear expression is a stencil expression
/// if all its call operator nodes are referring
/// to the same block.
template <typename E>
struct is_stencil_expr
{
  typedef void block_type;
  static bool check(E, block_type const *&a)
  {
    a = 0;
    return true;
  }
};

template <typename B, typename I, typename J, typename K>
struct is_stencil_expr<Call<B, I, J, K> >
{
  typedef B block_type;
  static bool check(Call<B, I, J, K> const &c, block_type const *&a)
  {
    a = &c.block();
    return true;
  }
};

template <template <typename, typename> class O, typename A1, typename A2>
struct is_stencil_expr<Binary<O, A1, A2> > 
{
  typedef typename is_stencil_expr<A1>::block_type a1_block_type;
  typedef typename is_stencil_expr<A2>::block_type a2_block_type;
  typedef typename conditional<is_same<a1_block_type, void>::value,
			       a2_block_type,
			       a1_block_type>::type block_type;

  static bool check(Binary<O, A1, A2> const &b, block_type const *&a)
  {
    a1_block_type const *a1;
    a2_block_type const *a2;
    return (is_linear_expr<Binary<O, A1, A2> >::value &&
	    is_stencil_expr<A1>::check(b.arg1(), a1) &&
	    is_stencil_expr<A2>::check(b.arg2(), a2) &&
	    impl::is_same_ptr_or_null(a1, a2, a));
   };
};

} // namespace vsip_csl::pi
} // namespace vsip_csl

#endif
