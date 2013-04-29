/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#include <vsip/math.hpp>
#include <vsip/opt/cbe/fconv_params.h>
#include <vsip/opt/cbe/ppu/fastconv.hpp>
#include <vsip/opt/cbe/ppu/task_manager.hpp>
#include <vsip/opt/cbe/ppu/util.hpp>
extern "C"
{
#include <libspe2.h>
}

namespace
{
using namespace vsip;
using namespace vsip::impl;
using namespace vsip::impl::cbe;

struct Worker
{
  Worker(char const *code_ea,
	 int code_size,
	 std::pair<float const *, float const *> const &in,
	 std::pair<float const *, float const *> const &kernel,
	 std::pair<float*, float*> const &out,
	 length_type num,
	 length_type length,
	 bool transform_kernel,
	 bool multi)
  {
    Task_manager *mgr = Task_manager::instance();
    // For dim==2 we also send the kernel over the buffer.
    dimension_type dim = multi ? 2 : 1;
    size_t buffer_size = length * dim * 2*sizeof(float);
    size_t num_buffers = length > 4096 ? 1 : 2;
    task = mgr->reserve_lwp_task(buffer_size, num_buffers,
				 (uintptr_t)code_ea, code_size);

    Fastconv_split_params params;
    params.instance_id        = ++instance_id;
    params.elements           = length;
    params.transform_kernel   = transform_kernel;
    params.ea_kernel_re       = ea_from_ptr(kernel.first);
    params.ea_kernel_im       = ea_from_ptr(kernel.second);
    params.ea_input_re        = ea_from_ptr(in.first);
    params.ea_input_im        = ea_from_ptr(in.second);
    params.ea_output_re       = ea_from_ptr(out.first);
    params.ea_output_im       = ea_from_ptr(out.second);
    params.kernel_stride      = length;
    params.input_stride       = length;
    params.output_stride      = length;

    length_type spes         = mgr->num_spes();
    length_type num_per_spe = num / spes;

    for (index_type i = 0; i < spes && i < num; ++i)
    {
      // If 'num' don't divide evenly, give the first SPEs one extra.
      length_type my_num = (i < num % spes) ? num_per_spe + 1 : num_per_spe;
      length_type stride = sizeof(float) * my_num * length;
      lwp::Workblock block = task->create_workblock(my_num);
      block.set_parameters(params);
      block.enqueue();
      params.ea_kernel_re += dim == 1 ? 0 : stride;
      params.ea_kernel_im += dim == 1 ? 0 : stride;
      params.ea_input_re  += stride;
      params.ea_input_im  += stride;
      params.ea_output_re += stride;
      params.ea_output_im += stride;
    }
  }

  // Valid template arguments are float and complex<float>
  template <typename T>
  Worker(char const *code_ea,
	 int code_size,
	 T const *in,
	 T const *kernel,
	 T *out,
	 length_type num,
	 length_type length,
	 bool transform_kernel,
	 bool multi)
  {
    Task_manager *mgr = Task_manager::instance();
    // For dim==2 we also send the kernel over the buffer.
    dimension_type dim = multi ? 2 : 1;
    size_t buffer_size = length * dim * 2*sizeof(float);
    size_t num_buffers = length > 4096 ? 1 : 2;
    task = mgr->reserve_lwp_task(buffer_size, num_buffers,
				 (uintptr_t)code_ea, code_size);

    Fastconv_params params;
    params.instance_id       = ++instance_id;
    params.elements          = length;
    params.transform_kernel  = transform_kernel;
    params.ea_kernel         = ea_from_ptr(kernel);
    params.ea_input          = ea_from_ptr(in);
    params.ea_output         = ea_from_ptr(out);
    params.kernel_stride     = length;
    params.input_stride      = length;
    params.output_stride     = length;

    length_type spes         = mgr->num_spes();
    length_type num_per_spe  = num / spes;

    for (index_type i = 0; i < spes && i < num; ++i)
    {
      // If 'num' don't divide evenly, give the first SPEs one extra.
      length_type my_num = (i < num % spes) ? num_per_spe + 1 : num_per_spe;
      length_type stride = sizeof(T) * my_num * length;
      lwp::Workblock block = task->create_workblock(my_num);
      block.set_parameters(params);
      block.enqueue();
      params.ea_kernel += dim == 1 ? 0 : stride;
      params.ea_input  += stride;
      params.ea_output += stride;
    }
  }

  ~Worker() { task->sync();}

  static unsigned int instance_id;
  std::auto_ptr<lwp::Task> task;
};

unsigned int Worker::instance_id = 0; 

}

namespace vsip
{
namespace impl
{
namespace cbe
{

void fconv(complex<float> const *in,
	   complex<float> const *kernel,
	   complex<float> *out,
	   length_type rows,
	   length_type cols,
	   bool transform_kernels)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/cfconv_f.plg");
  Worker worker(code, size, in, kernel, out, rows, cols, transform_kernels, false);
}

void fconvm(complex<float> const *in,
	    complex<float> const *kernel,
	    complex<float> *out,
	    length_type rows,
	    length_type cols,
	    bool transform_kernels)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/cfconvm_f.plg");
  Worker worker(code, size, in, kernel, out, rows, cols, transform_kernels, true);
}

void fconv(std::pair<float const *,float const *> in,
	   std::pair<float const *,float const *> kernel,
	   std::pair<float*,float*> out,
	   length_type rows,
	   length_type cols,
	   bool transform_kernels)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/zfconv_f.plg");
  Worker worker(code, size, in, kernel, out, rows, cols, transform_kernels, false);
}

void fconvm(std::pair<float const *,float const *> in,
	    std::pair<float const *,float const *> kernel,
	    std::pair<float*,float*> out,
	    length_type rows,
	    length_type cols,
	    bool transform_kernels)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/zfconvm_f.plg");
  Worker worker(code, size, in, kernel, out, rows, cols, transform_kernels, true);
}
          
} // namespace vsip::impl::cbe
} // namespace vsip::impl
} // namespace vsip
