/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/** @file    ukernel/cblock.cpp
    @author  Jules Bergmann, Stefan Seefeld
    @date    2008-12-17
    @brief   VSIPL++ Library: demonstrate standalone cblock ukernel.
*/

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/dda.hpp>
#include <vsip/matrix.hpp>
#include <vsip/signal.hpp>
#include <vsip/random.hpp>
#include <vsip_csl/ukernel/host/ukernel.hpp>
#include <vsip_csl/test.hpp>
#include <vsip_csl/output.hpp>
#include <kernels/host/cblock.hpp>

namespace vsip_csl
{
namespace ukernel
{
template <>
struct Task_map<example::Cblock, void()>
{
  static char const *plugin() { return "cblock.plg";}
};
} // namespace vsip_csl::ukernel
} // namespace vsip_csl

using namespace vsip;
using namespace vsip_csl;


// Applies cblock ukernel to perform multiply add:
//   out = in0 * in1 + in2;
//
// Uses standalone ukernel (Cblock_kernel)
template <typename View0,
	  typename View1,
	  typename View2,
	  typename View3>
void
apply_ukernel(View0 in0,
	      View1 in1,
	      View2 in2,
	      View3 out)
{
  dda::Data<typename View0::block_type, dda::in> data0(in0.block());
  dda::Data<typename View1::block_type, dda::in> data1(in1.block());
  dda::Data<typename View2::block_type, dda::in> data2(in2.block());
  dda::Data<typename View3::block_type, dda::out> data3(out.block());

  assert(data0.stride(1) == 1);
  assert(data1.stride(1) == 1);
  assert(data2.stride(1) == 1);
  assert(data3.stride(1) == 1);

  example::Cblock kernel((uintptr_t)data0.ptr(), data0.stride(0),
			 (uintptr_t)data1.ptr(), data1.stride(0),
			 (uintptr_t)data2.ptr(), data2.stride(0),
			 (uintptr_t)data3.ptr(), data3.stride(0),
			 out.size(0), out.size(1));
  
  vsip_csl::ukernel::Ukernel<example::Cblock> uk(kernel);

  uk();
}

template <typename T>
void
run_ukernel(length_type rows, length_type cols)
{
  Matrix<T> in0(rows, cols);
  Matrix<T> in1(rows, cols);
  Matrix<T> in2(rows, cols);
  Matrix<T> out(rows, cols);

  Rand<T> gen1(0, 0);
  in0 = gen1.randu(rows, cols);

  Rand<T> gen2(1, 0);
  in1 = gen2.randu(rows, cols);
  in2 = gen2.randu(rows, cols);

  apply_ukernel(in0, in1, in2, out);
  for (index_type i=0; i < rows; ++i)
    for (index_type j=0; j < cols; ++j)
    {
      T madd = in0.get(i, j) * in1.get(i, j) + in2.get(i, j);
      if (!equal(madd, out.get(i, j)))
      {
	std::cerr << "Error:" << std::endl;
        std::cerr << "index " << i << ", " << j << " : "
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
  run_ukernel<float>(63, 1024);
  run_ukernel<float>(64, 1024);

  return 0;
}
