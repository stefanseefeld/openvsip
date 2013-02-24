#include <vsip/initfin.hpp>
#include <vsip/core/thread.hpp>
#include <vsip_csl/test.hpp>
#include <iostream>

using namespace vsip;
using namespace vsip_csl;
using namespace vsip::impl;

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
  thread t(callable);
  C c;
  thread t2(c);

  t.join();
  t2.join();
  test_assert(tss == 42);  
}
