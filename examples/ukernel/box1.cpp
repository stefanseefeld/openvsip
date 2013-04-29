/* Copyright (c) 2008 by CodeSourcery.  All rights reserved. */

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/signal.hpp>
#include <vsip_csl/ukernel/host/ukernel.hpp>
#include <vsip_csl/test.hpp>
#include <vsip_csl/output.hpp>
#include <kernels/host/box1.hpp>

namespace vsip_csl
{
namespace ukernel
{
template <>
struct Task_map<example::Box1, void(float*,float*)>
{
  static char const *plugin() { return "box1.plg";}
};
} // namespace vsip_csl::ukernel
} // namespace vsip_csl

using namespace vsip;

template <typename Kernel, typename T>
void
run_ukernel(length_type size, length_type overlap)
{
  using namespace vsip_csl;
  
  Kernel kernel(overlap, 0); // min-support

  ukernel::Ukernel<Kernel> uk(kernel);

  Vector<T> in(size);
  Vector<T> out(size);

  in = ramp<T>(0, 0.1, size);

  uk(in, out);

  int misco = 0;
  for (index_type i=0; i<size; ++i)
  {
    T exp = in.get(i);
    if (i > 0) exp += in.get(i-1);
    if (i < size-1) exp += in.get(i+1);

    if (!equal(out.get(i), exp))
    {
      std::cout << i << ": " << out.get(i) << " != "
		<< exp
		<< std::endl;
      misco++;
    }
  }
  std::cout << "box1: size " << size
	    << "  overlap " << overlap
	    << "  misco " << misco << std::endl;
}

template <typename Kernel, typename T>
void
run_ukernel_subset(length_type size, length_type overlap, length_type offset)
{
  using namespace vsip_csl;

  Kernel kernel(overlap, 1); // same-support

  ukernel::Ukernel<Kernel> uk(kernel);

  Vector<T> in(size+offset+overlap);
  Vector<T> out(size);

  Domain<1> dom(offset, 1, size);

  in = ramp<T>(0, 0.1, size + offset + 1);

  uk(in(dom), out);

  int misco = 0;
  for (index_type i=0; i<size; ++i)
  {
    T exp = in.get(offset + i-1) + in.get(offset + i) + in.get(offset + i+1);

    if (!equal(out.get(i), exp))
    {
      std::cout << i << ": " << out.get(i) << " != "
		<< exp
		<< std::endl;
      misco++;
    }
  }
  std::cout << "box1-subset: size " << size
	    << "  overlap " << overlap
	    << "  offset " << offset
	    << "  misco " << misco << std::endl;
}

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  run_ukernel<example::Box1, float>(256, 1);
  run_ukernel_subset<example::Box1, float>(256, 1, 1);
}
