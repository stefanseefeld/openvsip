//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include <vsip/initfin.hpp>
#include <vsip/selgen.hpp>
#include <ovxx/opencl.hpp>
#include <ovxx/opencl/dda.hpp>
#include <test.hpp>
#include <string>

using namespace ovxx;
namespace ocl = ovxx::opencl;

template <typename T, typename B1, typename B2>
void copy(const_Matrix<T, B1> m1, Matrix<T, B2> m2)
{
  ocl::context context = ocl::default_context();
  ocl::command_queue queue = ocl::default_queue();
  std::vector<ocl::device> devices = context.devices();
  std::string src = 
    "__kernel void copy(__global float const *in, __global float *out)\n"
    "{                                                                \n"
    "  int num = get_global_id(0);                                    \n"
    "  out[num] = in[num];                                            \n"
    "}                                                                \n";
  ocl::Data<B1, vsip::dda::in> data_in(m1.block());
  ocl::Data<B2, vsip::dda::out> data_out(m2.block());
  ocl::program program = ocl::program::create_with_source(context, src);
  program.build(devices);
  ocl::kernel kernel = program.create_kernel("copy");
  kernel.exec(queue, m1.size(), data_in.ptr(), data_out.ptr());
}

int main(int argc, char **argv)
{
  vsipl library(argc, argv);
  Matrix<float> m(8,8);
  for (index_type i = 0; i != m.size(0); ++i)
    m.row(i) = ramp<float>(i*m.size(1), 1, m.size(1));
  Matrix<float> m2(8, 8);
  copy(m, m2);
  test_assert(equal(m, m2));
}
