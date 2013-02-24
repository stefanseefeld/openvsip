/* Copyright (c) 2008 by CodeSourcery.  All rights reserved. */

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/signal.hpp>
#include <vsip/random.hpp>
#include <vsip_csl/ukernel/host/ukernel.hpp>
#include <vsip_csl/test.hpp>
#include <vsip_csl/output.hpp>
#include <kernels/host/vmmul.hpp>

namespace vsip_csl
{
namespace ukernel
{
template <>
struct Task_map<example::Vmmul_proxy,
  void(complex<float>*, complex<float>*, complex<float>*)>
{
  static char const *plugin() { return "cvmmul.plg";}
};
template <>
struct Task_map<example::Vmmul_proxy,
  void(std::pair<float*,float*>, std::pair<float*,float*>, std::pair<float*,float*>)>
{
  static char const *plugin() { return "zvmmul.plg";}
};
} // namespace vsip_csl::ukernel
} // namespace vsip_csl

using namespace vsip;

template <typename Kernel, typename T>
void
run_ukernel(length_type rows, length_type cols)
{
  Kernel kernel(cols);

  vsip_csl::ukernel::Ukernel<Kernel> uk(kernel);

  Vector<T> in0(cols);
  Matrix<T> in1(rows, cols);
  Matrix<T> out(rows, cols);

  Rand<T> gen(0, 0);

  in0 = gen.randu(cols);
  in1 = gen.randu(rows, cols);

  in0(Domain<1>(16)) = ramp<T>(0, 1, 16);
  in1.row(0)(Domain<1>(16)) = ramp<T>(0, 0.1, 16);

  uk(in0, in1, out);

  for (index_type r=0; r<rows; ++r)
    for (index_type c=0; c<cols; ++c)
    {
      if (!vsip_csl::equal(in0.get(c) * in1.get(r, c), out.get(r, c)))
      {
	std::cout << "index " << r << ", " << c << ": "
		  << in0.get(c) << " * "
		  << in1.get(r, c) << " = "
		  << in0.get(c) * in1.get(r, c) << "  vs  "
		  << out.get(r, c)
		  << std::endl;
      }
    }
}

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);
  run_ukernel<example::Vmmul_proxy, complex<float> >(128, 1024);
}
