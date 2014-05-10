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
  std::vector<ocl::platform> platforms = ocl::platform::platforms();
  intptr_t props[] = { CL_CONTEXT_PLATFORM, (intptr_t)(cl_platform_id)platforms[1], 0};
  ocl::context context(props, ocl::device::all);
  std::vector<ocl::device> devices = context.devices();
  ocl::command_queue *queue = context.create_queue(devices[0]);

  std::string src = 
    "__kernel void copy(__global char const *in, __global char *out)\n"
    "{                                                              \n"
    "  int num = get_global_id(0);                                  \n"
    "  out[num] = in[num];                                          \n"
    "}                                                              \n";
  ocl::program *program = context.create_program(src);
  program->build(devices);
  char *output = new char[in.size() + 1];
  
  ocl::buffer *input_buffer = context.create_buffer
    (CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, const_cast<char *>(in.data()), in.size() + 1);
  ocl::buffer *output_buffer = context.create_buffer<char>
    (CL_MEM_WRITE_ONLY, 0, in.size() + 1);
  ocl::kernel *kernel = program->create_kernel("copy");
  kernel->set_arg(0, *input_buffer);
  kernel->set_arg(1, *output_buffer);
	
  queue->push_back(*kernel, in.size());
  queue->push_back(*output_buffer, output, in.size());
  output[in.size()] = '\0';
  out = output;
  delete [] output;
  delete kernel;
  delete program;
  delete input_buffer;
  delete output_buffer;
  delete queue;
}

int main(int argc, char **argv)
{
  vsip::vsipl library(argc, argv);
  std::string text = "hello world !";
  std::string result;
  copy(text, result);
  test_assert(text == result);
}
