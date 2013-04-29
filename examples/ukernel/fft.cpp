/* Copyright (c) 2008 by CodeSourcery.  All rights reserved. */

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/signal.hpp>
#include <vsip_csl/ukernel/host/ukernel.hpp>
#include <vsip_csl/test.hpp>
#include <vsip_csl/output.hpp>
#include <kernels/host/fft.hpp>

namespace vsip_csl
{
namespace ukernel
{
template <>
struct Task_map<example::Fft_proxy, void(complex<float>*, complex<float>*)>
{
  static char const *plugin() { return "cfft.plg";}
};
template <>
struct Task_map<example::Fft_proxy, void(std::pair<float*,float*>, std::pair<float*,float*>)>
{
  static char const *plugin() { return "zfft.plg";}
};
} // namespace vsip_csl::ukernel
} // namespace vsip_csl

using namespace vsip;

template <typename Kernel, typename T>
void
run_ukernel(length_type rows, length_type cols)
{
  using namespace vsip_csl;

  Kernel kernel(cols, -1, 1.f);

  ukernel::Ukernel<Kernel> uk(kernel);

  Matrix<T> in(rows, cols);
  Matrix<T> out(rows, cols, T(-100));

  for (index_type r=0; r<rows; ++r)
    in.row(r) = T(r);

  uk(in, out);

  for (index_type r=0; r<rows; ++r)
  {
    if (out.get(r, 0) != T(r * cols))
      std::cout << "error row: " << r
		<< " got: " << out.get(r, 0)
		<< " exp: " << T(r * cols)
		<< std::endl;
  }
}

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  run_ukernel<example::Fft_proxy, complex<float> >(32, 2048);
}
