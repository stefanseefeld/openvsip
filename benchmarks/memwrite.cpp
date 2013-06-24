//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Benchmark for memory write bandwidth.

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/random.hpp>
#include "benchmark.hpp"

using namespace ovxx;

template <typename T>
struct t_memwrite1 : Benchmark_base
{
  char const* what() { return "t_memwrite1"; }
  int ops_per_point(length_type)  { return 1; }
  int riob_per_point(length_type) { return 0; }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 1*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    Vector<T>   view(size, T());
    T           val = T(1);
    
    timer t1;
    for (index_type l=0; l<loop; ++l)
      view = val;
    time = t1.elapsed();

    for(index_type i=0; i<size; ++i)
      test_assert(equal(view.get(i), val));
  }

  void diag()
  {
    using namespace ovxx;
    length_type const size = 256;

    Vector<T>   view(size, T());
    T           val = T(1);

    std::cout << assignment::diagnostics(view, val) << std::endl;
  }
};



// explicit loop

template <typename T>
struct t_memwrite_expl : Benchmark_base
{
  char const* what() { return "t_memwrite_expl"; }
  int ops_per_point(length_type)  { return 1; }
  int riob_per_point(length_type) { return 0; }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 1*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    Vector<T>   view(size, T());
    T           val = T(1);
    
    timer t1;
    for (index_type l=0; l<loop; ++l)
      for (index_type i=0; i<size; ++i)
	view.put(i, val);
    time = t1.elapsed();

    for(index_type i=0; i<size; ++i)
      test_assert(equal(view.get(i), val));
  }
};

void
defaults(Loop1P&)
{
}

int
benchmark(Loop1P& loop, int what)
{
  switch (what)
  {
  case   1: loop(t_memwrite1<float>()); break;
  case   2: loop(t_memwrite_expl<float>()); break;

  case   0:
    std::cout
      << "memwrite -- memory write bandwidth\n"
      << "  -1 -- write a float scalar into all elements of a view\n"
      << "  -2 -- same using an explicit loop\n"
      ;
  default: return 0;
  }
  return 1;
}
