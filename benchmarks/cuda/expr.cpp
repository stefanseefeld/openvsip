/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Benchmark for compound expressions on CUDA

#include <iostream>
#include <ostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/core/expr/fns_elementwise.hpp>
#include <vsip_csl/profile.hpp>
#include <vsip/opt/cuda/kernels.hpp>
#include <vsip/opt/cuda/dda.hpp>
#include <vsip/opt/type_name.hpp>
#include <vsip/selgen.hpp>

#include <vsip_csl/test.hpp>
#include <vsip_csl/diagnostics.hpp>
#include "loop.hpp"

using namespace vsip;
using vsip_csl::equal;
namespace expr = vsip_csl::expr;

#define DEBUG 0


/***********************************************************************
  Expression test harness
***********************************************************************/

#define DEFINE_EXPR_BENCHMARK(NAME, OP, OP_COUNT)                       \
  template <typename T1,                                                \
            typename T2>                                                \
  struct test_##NAME : Benchmark_base                                   \
  {                                                                     \
    int ops_per_point(length_type)  { return ops_per_element_; }        \
    int riob_per_point(length_type) { return sizeof(T1); }              \
    int wiob_per_point(length_type) { return sizeof(T2); }              \
    int mem_per_point(length_type)  { return sizeof(T1) + sizeof(T2); } \
                                                                        \
    char const* what() { return "this"; }                               \
                                                                        \
    void time_expr(vsip::length_type size, vsip::length_type loop, float& time) \
    {                                                                   \
      Vector<T1> A(size, T1(M_PI));                                     \
      Vector<T2> B(size, T2());                                         \
                                                                        \
      vsip_csl::profile::Timer t1;                                      \
      t1.start();                                                       \
      for (index_type l = 0; l < loop; ++l)                             \
        B = OP(A);                                                      \
      t1.stop();                                                        \
      time = t1.delta();                                                \
                                                                        \
      T2 result;                                                        \
      result = OP(static_cast<T1>(M_PI));                               \
      test_assert(equal(result, B(0)));                                 \
    }                                                                   \
                                                                        \
    void operator()(vsip::length_type size, vsip::length_type loop, float& time) \
    {                                                                   \
      this->time_expr(size, loop, time);                                \
    }                                                                   \
                                                                        \
    test_##NAME()                                                       \
      : ops_per_element_(OP_COUNT)                                      \
    {}                                                                  \
                                                                        \
    void diag()                                                         \
    {                                                                   \
      length_type const size = 8192;                                    \
      Vector<T1> A(size, T1(M_PI));                                     \
      Vector<T2> B(size, T2());                                         \
                                                                        \
      assign_diagnostics(B, OP(A));                                     \
    }                                                                   \
                                                                        \
    int ops_per_element_;                                               \
  };



#define TEST_SIN(IN)  sin(IN)
DEFINE_EXPR_BENCHMARK(sin_f, TEST_SIN, 1)
DEFINE_EXPR_BENCHMARK(sin_cf, TEST_SIN, 2)
#undef TEST_SIN

#define TEST_SIN(IN)  sin(sin(IN))
DEFINE_EXPR_BENCHMARK(double_sin_f, TEST_SIN, 2)
DEFINE_EXPR_BENCHMARK(double_sin_cf, TEST_SIN, 4)
#undef TEST_SIN

#define TEST_SIN(IN)  sin(sin(sin(IN)))
DEFINE_EXPR_BENCHMARK(triple_sin_f, TEST_SIN, 3)
DEFINE_EXPR_BENCHMARK(triple_sin_cf, TEST_SIN, 6)
#undef TEST_SIN

#define TEST_EULER(IN)  sin(IN) + std::complex<float>(0, -1) * cos(IN)
DEFINE_EXPR_BENCHMARK(euler_f, TEST_EULER, 6)
DEFINE_EXPR_BENCHMARK(euler_cf, TEST_EULER, 9)
#undef TEST_EULER


/***********************************************************************
  Benchmark Driver
***********************************************************************/

void
defaults(Loop1P &)
{
}

int
test(Loop1P& loop, int what)
{
  typedef float F;
  typedef complex<float> C;

  switch (what)
  {
    // Template parameters are:  
    //     <Input Type, Output Type>
  case  1: loop(test_sin_f<F, F>()); break;
  case  2: loop(test_double_sin_f<F, F>()); break;
  case  3: loop(test_triple_sin_f<F, F>()); break;
  case  4: loop(test_euler_f<F, C>()); break;

  case 11: loop(test_sin_cf<C, C>()); break;
  case 12: loop(test_double_sin_cf<C, C>()); break;
  case 13: loop(test_triple_sin_cf<C, C>()); break;
  case 14: loop(test_euler_cf<C, C>()); break;

    // help
  default:
    std::cout
      << "CUDA compound expressions -- float input\n"
      << "   -1 -- sin(x)\n"
      << "   -2 -- sin(sin(x))\n"
      << "   -3 -- sin(sin(sin((x)))\n"
      << "   -4 -- sin(x) + i * cos(x)\n"
      << " complex<float> input\n"
      << "  -11 -- sin(x)\n"
      << "  -12 -- sin(sin(x))\n"
      << "  -13 -- sin(sin(sin((x)))\n"
      << "  -14 -- sin(x) + i * cos(x)\n"
      << " Note:  i =: sqrt(-1)\n"
      ;
    return 0;
  }
  return 1;
}
