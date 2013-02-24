/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#ifndef VSIP_CSL_UNWRAP_HPP
#define VSIP_CSL_UNWRAP_HPP

#include <vsip/vector.hpp>

namespace vsip_csl
{
namespace dispatcher
{
namespace op
{
struct unwrap;
}

template<>
struct List<op::unwrap>
{
  typedef Make_type_list<be::opt,
			 be::generic>::type type;
};

/// Generic evaluator.
template <typename Block1,
	  typename Block2>
struct Evaluator<op::unwrap, be::generic,
                 void(Block1 &, Block2 const &)>
{
  static bool const ct_valid = true;
  static bool rt_valid(Block1 &, Block2 const &) { return true;}

  static void exec(Block1 &out, Block2 const &in)
  {
    typedef typename Block2::value_type T;
  
    int offset_count = 0;
    T last = 0.0f;
    
    for (length_type i = 0; i < in.size(); i++)
    {
      T current = in.get(i);
      T delta = current - last;
      last = current;
      if (delta > M_PI || delta < -M_PI)
      {
        if (delta > M_PI)
          offset_count -= (int(delta/M_PI) + 1) / 2;
        else
          offset_count += (int(-delta/M_PI) + 1) / 2;
      }
      out.put(i, current + offset_count * 2 * M_PI);
    }
  }
};

} // namespace vsip_csl::dispatcher


/// Unwrap phase sequences that have had mod(2*pi) applied to them.
template <typename T, typename Block1, typename Block2>
void
unwrap(Vector<T, Block1> out, const_Vector<T, Block2> in)
{
  dispatch<dispatcher::op::unwrap, void, Block1 &, Block2 const &>
    (out.block(), in.block());
}

template <typename T, typename Block1>
void
unwrap(Vector<T, Block1> A)
{
  unwrap(A, A);
}

} // namespace vsip_csl

#endif // VSIP_CSL_UNWRAP_HPP
