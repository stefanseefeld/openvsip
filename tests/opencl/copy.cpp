//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#include <vsip/initfin.hpp>
#include <ovxx/opencl.hpp>
#include <test.hpp>
#include <string>


namespace ocl = ovxx::opencl;

void copy(std::string const &in, std::string &out)
{
  ocl::context context = ocl::default_context();
  std::vector<ocl::device> devices = context.devices();
  ocl::command_queue queue = ocl::default_queue();

  std::string src = 
    "__kernel void copy(__global char const *in, __global char *out)\n"
    "{                                                              \n"
    "  int num = get_global_id(0);                                  \n"
    "  out[num] = in[num];                                          \n"
    "}                                                              \n";
  ocl::program program = ocl::program::create_with_source(context, src);
  program.build(devices);
  char *output = new char[in.size() + 1];
  
  ocl::buffer inbuf(context, in.size() + 1, ocl::buffer::read);
  ocl::buffer outbuf(context, in.size() + 1, ocl::buffer::write);
  ocl::kernel kernel = program.create_kernel("copy");
  queue.write(in.data(), inbuf, in.size());
  kernel.exec(queue, in.size(), inbuf, outbuf);
  queue.read(outbuf, output, in.size());
  output[in.size()] = '\0';
  out = output;
  delete [] output;
}

int main(int argc, char **argv)
{
  vsip::vsipl library(argc, argv);
  std::string text = "hello world !";
  std::string result;
  copy(text, result);
  test_assert(text == result);
}
