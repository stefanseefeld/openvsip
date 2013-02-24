/* Copyright (c) 2008 by CodeSourcery.  All rights reserved. */


#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/signal.hpp>
#include <vsip/random.hpp>
#include <vsip_csl/ukernel/host/ukernel.hpp>
#include <vsip_csl/test.hpp>
#include <vsip_csl/output.hpp>
#include <kernels/host/madd.hpp>

namespace vsip_csl
{
namespace ukernel
{
template <>
struct Task_map<example::Madd,
  void(float*, float*, float*, float*)>
{
  static char const *plugin() { return "madd.plg";}
};

template <>
struct Task_map<example::Madd,
  void(std::complex<float>*,
       std::complex<float>*,
       std::complex<float>*,
       std::complex<float>*)>
{
  static char const *plugin() { return "cmadd";}
};

template <>
struct Task_map<example::Madd,
  void(float*,
       std::complex<float>*,
       std::complex<float>*,
       std::complex<float>*)>
{
  static char const *plugin() { return "scmadd";}
};
} // namespace vsip_csl::ukernel
} // namespace vsip_csl

using namespace vsip;
using namespace vsip_csl;

// Performs an elementwise multiply-add for the expression
//   Z = A * B + C
// where T1 is the type of A and T2 is the type for B, C and D
//
template <typename Kernel, typename T1, typename T2>
void
run_ukernel(length_type rows, length_type cols)
{
  Kernel kernel;

  ukernel::Ukernel<Kernel> madd_uk(kernel);

  Matrix<T1> in0(rows, cols);
  Matrix<T2> in1(rows, cols);
  Matrix<T2> in2(rows, cols);
  Matrix<T2> out(rows, cols);

  Rand<T1> gen1(0, 0);
  in0 = gen1.randu(rows, cols);

  Rand<T2> gen2(1, 0);
  in1 = gen2.randu(rows, cols);
  in2 = gen2.randu(rows, cols);

  madd_uk(in0, in1, in2, out);


  for (index_type i=0; i < rows; ++i)
    for (index_type j=0; j < cols; ++j)
    {
      T2 madd = in0.get(i, j) * in1.get(i, j) + in2.get(i, j);
      if (!equal(madd, out.get(i, j)))
      {
        std::cout << "index " << i << ", " << j << " : "
                  << in0.get(i, j) << " * "
                  << in1.get(i, j) << " + "
                  << in2.get(i, j) << " = "
                  << in0.get(i, j) * in1.get(i, j) + in2.get(i, j) << "  vs  "
                  << out.get(i, j)
                  << std::endl;
      }
    }
}

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  // Parameters are rows then cols
  run_ukernel<example::Madd, float, float>(64, 1024);

// This kernel is presently only implemented for interleaved complex
#if !VSIP_IMPL_PREFER_SPLIT_COMPLEX
  run_ukernel<example::Madd, float, complex<float> >(64, 1024);
  run_ukernel<example::Madd, complex<float>, complex<float> >(64, 1024);
#endif
}
