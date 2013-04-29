/* Copyright (c) 2008 by CodeSourcery.  All rights reserved. */

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/signal.hpp>
#include <vsip/random.hpp>
#include <vsip_csl/ukernel/host/ukernel.hpp>
#include <vsip_csl/test.hpp>
#include <vsip_csl/output.hpp>
#include <kernels/host/id2.hpp>

namespace vsip_csl
{
namespace ukernel
{
template <>
struct Task_map<example::Id2, void(float*,float*)>
{
  static char const *plugin() { return "id2.plg";}
};
} // namespace vsip_csl::ukernel
} // namespace vsip_csl

using namespace vsip;
using namespace vsip_csl;

template <typename Kernel, typename T>
void
run_ukernel(int shape, length_type rows, length_type cols)
{
  Kernel kernel(shape, rows, cols);

  ukernel::Ukernel<Kernel> uk(kernel);

  Matrix<T> in(rows, cols);
  Matrix<T> out(rows, cols);

  Rand<T> rnd(0);

  in = T(0);
  // in = rnd.randu(rows, cols);

  uk(in, out);

  int misco = 0;
  for (index_type i=0; i<rows; ++i)
    for (index_type j=0; j<cols; ++j)
    {
      if (!equal(out.get(i, j), in.get(i, j) + T(i * cols + j)))
      {
	misco++;
#if VERBOSE >= 2
	std::cout << i << ": " << out.get(i, j) << " != "
		  << in.get(i, j) << " + " << T(i*cols + j)
		  << std::endl;
#endif
      }
  }
  std::cout << "id2: size " << rows << " x " << cols 
	    << "  shape " << shape
	    << "  misco " << misco << std::endl;
}

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  for (length_type size=32; size<=1024; size*=2)
  {
    run_ukernel<example::Id2, float>(1, size, size);
    run_ukernel<example::Id2, float>(2, size, size);
    run_ukernel<example::Id2, float>(3, size, size);
    run_ukernel<example::Id2, float>(4, size, size);
    run_ukernel<example::Id2, float>(5, size, size);
  }
  run_ukernel<example::Id2, float>(0, 1024, 1024);
}
