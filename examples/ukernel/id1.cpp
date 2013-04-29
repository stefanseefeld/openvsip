/* Copyright (c) 2008 by CodeSourcery.  All rights reserved. */

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/signal.hpp>
#include <vsip_csl/ukernel/host/ukernel.hpp>
#include <vsip_csl/test.hpp>
#include <vsip_csl/output.hpp>
#include <kernels/host/id1.hpp>

namespace vsip_csl
{
namespace ukernel
{
template <>
struct Task_map<example::Id1, void(float*,float*)>
{
  static char const *plugin() { return "id1.plg";}
};
} // namespace vsip_csl::ukernel
} // namespace vsip_csl

using namespace vsip;
using namespace vsip_csl;

template <typename Kernel, typename T>
void
run_ukernel(length_type size)
{
  Kernel kernel;

  ukernel::Ukernel<Kernel> uk(kernel);

  Vector<T> in(size);
  Vector<T> out(size);

  in = ramp<T>(0, 0.1, size);

  uk(in, out);

  for (index_type i=0; i<size; ++i)
  {
    if (!equal(out.get(i), in.get(i) + T(i)))
    {
      std::cout << i << ": " << out.get(i) << " != "
		<< in.get(i) << " + " << T(i)
		<< std::endl;
    }
    test_assert(equal(out.get(i), in.get(i) + T(i)));
  }
}

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);
  run_ukernel<example::Id1, float>(16384);
}
