/* Copyright (c) 2008 by CodeSourcery.  All rights reserved. */

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/signal.hpp>
#include <vsip/random.hpp>
#include <vsip_csl/ukernel/host/ukernel.hpp>
#include <vsip_csl/test.hpp>
#include <vsip_csl/error_db.hpp>
#include <vsip_csl/output.hpp>
#include <kernels/host/fconv.hpp>

namespace vsip_csl
{
namespace ukernel
{
template <>
struct Task_map<example::Fconv_proxy,
  void(complex<float>*, complex<float>*, complex<float>*)>
{
  static char const *plugin() { return "cfconv.plg";}
};
template <>
struct Task_map<example::Fconv_proxy,
  void(std::pair<float*,float*>, std::pair<float*,float*>, std::pair<float*,float*>)>
{
  static char const *plugin() { return "zfconv.plg";}
};
} // namespace vsip_csl::ukernel
} // namespace vsip_csl


using namespace vsip;

template <typename Kernel, typename T>
void
run_ukernel(length_type rows, length_type cols, float scale, int tc)
{
  Kernel kernel(cols);

  vsip_csl::ukernel::Ukernel<Kernel> uk(kernel);

  Vector<T> in0(cols);
  Matrix<T> in1(rows, cols);
  Matrix<T> out(rows, cols, T(-100));

  Rand<T> gen(0, 0);

  in0 = T(scale);

  switch(tc)
  {
  case 0:
    in1 = gen.randu(rows, cols);
    in1.row(0)(Domain<1>(16)) = ramp<T>(0, 0.1, 16);
    break;
  case 1:
    for (index_type r=0; r<rows; ++r)
      in1.row(r) = ramp<T>(r, 0.1, cols);
    break;
  }

  uk(in0, in1, out);

  for (index_type r=0; r<rows; ++r)
  {
    float e = vsip_csl::error_db(scale * in1.row(r), out.row(r));
    if (e > -100)
      std::cout << "row " << r << ":  error " << e << std::endl;
    test_assert(e <= -100);
  }
}

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);
  run_ukernel<example::Fconv_proxy, complex<float> >(4, 64, 0.5, 0);
}
