/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   VSIPL++ Library: Matrix product benchmark base class

#ifndef BENCHMARKS_MPROD_HPP
#define BENCHMARKS_MPROD_HPP

#include <vsip/support.hpp>

template <typename T>
struct t_mprod_base : Benchmark_base
{
  static vsip::length_type const Dec = 1;

  float ops_total(vsip::length_type M, vsip::length_type N, vsip::length_type P)
  {
    float ops = M * N * P * 
      (vsip::impl::Ops_info<T>::mul + vsip::impl::Ops_info<T>::add);

    return ops;
  }

  float riob_total(vsip::length_type M, vsip::length_type N, vsip::length_type P)
    { return ((M * N) + (N * P)) * sizeof(T); }

  float wiob_total(vsip::length_type M, vsip::length_type P)
    { return M * P * sizeof(T); }

  float mem_total(vsip::length_type M, vsip::length_type N, vsip::length_type P)
    { return riob_total(M, N, P) + wiob_total(M, P); } 
};

#endif // BENCHMARKS_MPROD_HPP
