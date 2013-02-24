/* Copyright (c) 2007, 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#ifndef VSIP_OPT_CBE_PPU_TASK_MANAGER_HPP
#define VSIP_OPT_CBE_PPU_TASK_MANAGER_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <memory>

#include <vsip/core/config.hpp>
#include <vsip/core/expr/operations.hpp>
#include <vsip/core/argv_utils.hpp>
#include <vsip/opt/cbe/ppu/alf.hpp>
extern "C"
{
#include <libspe2.h>
}

namespace vsip
{
namespace impl
{

struct Pwarp_tag;

namespace cbe
{

template <typename O, typename S> struct Task_map;

class Task_manager
{
public:
  static Task_manager *instance() { return instance_;}

  static void initialize(int& argc, char**&argv);
  static void finalize() { delete instance_; instance_ = 0;}
  static void set_num_spes(int num_spes, bool verify = false);

  // Return a task for operation O (with signature S).
  // An additional hint parameter may be passed, eventually,
  // indicating how many SPEs to use, etc.
  template <typename O, typename S>
  Task reserve(length_type ssize, // max stack size
	       length_type psize, // parameter buffer size
	       length_type isize, // input buffer size
	       length_type osize, // output buffer size
	       length_type tsize) // number of DMA transfers
  {
    return alf_->create_task("svpp_kernels.so", Task_map<O, S>::image(),
			     ssize, psize, isize, osize, 0, tsize);
  }

  std::auto_ptr<lwp::Task>
  reserve_lwp_task(int buf_size, int num_bufs,
		   uintptr_t code_ea, int code_size, int cmd = 0)
  {
    return std::auto_ptr<lwp::Task>(new lwp::Task(alf_->num_accelerators(),
						  buf_size, num_bufs,
						  code_ea, code_size,
						  cmd));
  }

  length_type num_spes() { return num_spes_; }
  
  ALF* alf_handle() { return alf_; }

private:
  Task_manager(unsigned int num_spes);
  Task_manager(Task_manager const &);
  ~Task_manager();
  Task_manager &operator= (Task_manager const &);

  static Task_manager *instance_;

  static length_type default_num_spes_;
  void set_my_num_spes(int num_spes, bool verify);

  ALF*        alf_;
  length_type num_spes_;
};

} // namespace vsip::impl::cbe
} // namespace vsip::impl
} // namespace vsip

// O: operation tag
// S: operation signature
// K: SPE kernel image name
# define DEFINE_TASK(O, S, I)			   \
namespace vsip { namespace impl { namespace cbe {          \
template <>                                                \
struct Task_map<O, S>                                      \
{                                                          \
  static char const *image() { return "alf_" # I "_spu";}  \
};                                                         \
}}}

DEFINE_TASK(Pwarp_tag, void(unsigned char, unsigned char), pwarp_ub)

#undef DEFINE_TASK

#endif // VSIP_OPT_CBE_PPU_TASK_MANAGER_HPP
