//
// Copyright (c) 2005, 2006, 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/selgen.hpp>
#include <vsip/dense.hpp>
#include <vsip/dda.hpp>
#include <vsip_csl/strided.hpp>
#include <vsip_csl/test.hpp>
#include <vsip_csl/diagnostics.hpp>
#include <iostream>
#include <cassert>

#ifndef VERBOSE
#define VERBOSE 0
#endif

using namespace vsip;
using vsip_csl::equal;

// Sum values in a view via Data interface.
template <typename T, typename Block>
T
dda_sum(const_Vector<T, Block> view)
{
  dda::Data<Block, dda::in> raw(view.block());

#if VERBOSE
  std::cout << vsip_csl::type_name<typename dda::Data<Block, dda::in>::backend_type>() << std::endl;
#endif

  typename Block::value_type const *data = raw.ptr();
  stride_type stride = raw.stride(0);

  typename Block::value_type sum = 0.f;
  
  for (index_type i = 0; i < raw.size(); ++i, data += stride)
    sum += *data;

  return sum;
}

// Sum values in a view via block interface.
template <typename T, typename Block>
T
sum(const_Vector<T, Block> view)
{
  typename Block::value_type sum = 0.f;
  
  for (index_type i = 0; i < view.size(); ++i)
    sum += view.get(i);

  return sum;
}

// Test various view summations.
template <typename Block>
void
test_sum()
{
  typedef typename Block::value_type value_type;
  Vector<value_type, Block> vector(8);
  vector.local() = ramp(0., 1., 8);
  typename Block::value_type s = sum(vector);
  test_assert(equal(s, dda_sum(vector)));
  test_assert(equal(2 * s, dda_sum(vector + vector)));
}

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  test_sum<Dense<1, float> >();
  test_sum<vsip_csl::Strided<1, float> >();
  test_sum<Dense<1, float, tuple<0,1,2>, Map<> > >();
}
