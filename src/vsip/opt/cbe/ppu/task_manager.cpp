/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cbe/ppu/task_manager.cpp
    @author  Stefan Seefeld
    @date    2007-01-31
    @brief   VSIPL++ Library: Wrappers and traits to bridge with IBMs ALF.
*/

#include <vsip/opt/cbe/ppu/task_manager.hpp>
#include <cstring>

namespace vsip
{
namespace impl
{
namespace cbe
{

Task_manager *Task_manager::instance_ = 0;

length_type Task_manager::default_num_spes_ = VSIP_IMPL_CBE_NUM_SPES;

Task_manager::Task_manager(unsigned int num_spes)
  : alf_     (new ALF(num_spes))
{
  num_spes_ = alf_->num_accelerators();
}

Task_manager::~Task_manager()
{
  delete alf_;
}

void Task_manager::initialize(int& argc, char**&argv)
{
  unsigned int num_spes = default_num_spes_;
  for (int i=1; i < argc; ++i)
  {
    if (!strcmp(argv[i], "--vsip-num-spes"))
    {
      num_spes = atoi(argv[i+1]);
      shift_argv(argc, argv, i, 2);
      i -= 1;
    }
  }
  
  instance_ = new Task_manager(num_spes);
}

void Task_manager::set_num_spes(int num_spes, bool verify)
{
  if (num_spes == -1) num_spes = 0;

  default_num_spes_ = num_spes;
  
  if (instance_)
  {
    instance_->set_my_num_spes(num_spes, verify);
  }
}

void Task_manager::set_my_num_spes(int num_spes, bool verify)
{
  delete alf_;
  alf_ = new ALF(num_spes);
  num_spes_ = alf_->num_accelerators();

  if (verify && ((int)num_spes_ != num_spes))
    VSIP_THROW(std::bad_alloc());
}

} // namespace vsip::impl::cbe
} // namespace vsip::impl
} // namespace vsip
