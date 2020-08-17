//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <vsip/initfin.hpp>
#include <ovxx/thread.hpp>
#include <test.hpp>
#include <iostream>

using namespace ovxx;

thread_local int tss;

void callable()
{
  std::cout << "callable" << std::endl;
}

struct C
{
  void operator()()
  {
    std::cout << "callable 2" << std::endl;
    tss = 24;
    test_assert(tss == 24);
  }
};

int main(int argc, char **argv)
{
  vsipl library(argc, argv);
  tss = 42;
  std::thread t(callable);
  C c;
  std::thread t2(c);

  t.join();
  t2.join();
  test_assert(tss == 42);  
}
