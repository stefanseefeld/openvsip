/* Copyright (c) 2008, 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#include "alf.hpp"
#include <iostream>

namespace vsip
{
namespace impl
{
namespace cbe
{
extern "C" ALF_ERR_POLICY_T
error_handler(void */*context*/, ALF_ERR_TYPE_T type, int code, char *msg)
{
  // For now, we raise unconditionally raise an exception.
  // This may be refined, for example by chosing different exception
  // types depending on 'type' above.
  switch (type)
  {
    default:
      VSIP_IMPL_THROW(runtime_error(code, msg));
  }
  // Unreachable
  return ALF_ERR_POLICY_ABORT;
}

ALF::ALF(unsigned int num_accelerators)
{
  csl_alf_init(num_accelerators);
  alf_error_handler_register(csl_alf_handle(), error_handler, 0);
  num_accelerators_ = csl_alf_num_spes();
}
  
ALF::~ALF()
{
  cml_fini();
}

Task::Task(char const* library,
	   char const* image, length_type ssize, length_type psize,
	   length_type isize, length_type osize, length_type iosize,
	   length_type tsize, unsigned int spes)
  : image_(image),
    ssize_(ssize),
    psize_(psize),
    isize_(isize),
    osize_(osize),
    iosize_(iosize),
    tsize_(tsize)
{
  (void)spes;
  csl_task_desc desc;

  csl_alf_task_desc_init(&desc);
  desc.wb_parm_ctx_buf_size   = psize;
  desc.wb_in_buf_size         = isize;
  desc.wb_out_buf_size        = osize;
  desc.wb_inout_buf_size      = iosize;
  desc.num_dtl_entries        = tsize;
  desc.max_stack_size         = ssize;
  desc.accel_library_ref_l    = library;
  desc.accel_image_ref_l      = image_;
  desc.accel_kernel_ref_l     = "kernel";
  desc.accel_input_dtl_ref_l  = "input";
  desc.accel_output_dtl_ref_l = "output";
  desc.tsk_ctx_data_type      = ALF_DATA_BYTE;

  int status ATTRIBUTE_UNUSED = csl_alf_task_create(&desc, &task_);
  assert(status >= 0);
}

void Task::destroy()
{
  if (task_)
    csl_alf_task_destroy(task_);
}

} // namespace vsip::impl::cbe
} // namespace vsip::impl
} // namespace vsip
