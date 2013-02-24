/* Copyright (c) 2010 by CodeSourcery.  All rights reserved. */

#include <vsip/initfin.hpp>
#include <vsip/dense.hpp>
#include <vsip_csl/udk.hpp>
#include <vsip_csl/dda.hpp>
#include <vsip_csl/test.hpp>
#include <vsip_csl/output.hpp>
#include "udk.hpp"
#include <iostream>

using namespace vsip;
namespace udk = vsip_csl::udk;

void copy(dda::Data<Dense<2>, dda::in> &in,
	  dda::Data<Dense<2>, dda::out> &out)
{
  for (length_type y = 0; y != in.size(0); ++y)
    for (length_type x = 0; x != in.size(1); ++x)
      out.ptr()[x * out.stride(1) + y * out.stride(0)] =
	in.ptr()[x * in.stride(1) + y * in.stride(0)];
}

int
main(int argc, char **argv)
{
  vsipl init(argc, argv);
  
  udk::Task<udk::target::test, 
    udk::tuple<udk::in<Dense<2> >, udk::out<Dense<2> > > > 
    task(copy);
  Matrix<float> input(4, 4, 2.);
  Matrix<float> output(4, 4, 1.);
  task.execute(input, output);
  test_assert(vsip_csl::view_equal(input, output));
}
