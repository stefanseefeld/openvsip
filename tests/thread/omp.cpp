//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include <vsip/initfin.hpp>
#include <test.hpp>
#include <iostream>

using namespace ovxx;

void check_thread()
{
  allocator *a = allocator::get_default();
  // Just make sure the library is properly initialized
  test_assert(a);
}

void check()
{
#if OVXX_ENABLE_OMP
#pragma omp parallel
  check_thread();
#endif
}


int main(int argc, char **argv)
{
  vsipl library(argc, argv);
  check();
}
