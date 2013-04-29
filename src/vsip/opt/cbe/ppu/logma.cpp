/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
#include <vsip/core/config.hpp>
#include <vsip/math.hpp>
#include <vsip/opt/cbe/ppu/logma.hpp>
#include <vsip/opt/cbe/ppu/task_manager.hpp>
#include <vsip/opt/cbe/ppu/util.hpp>
#include <vsip/opt/cbe/logma_params.h>

namespace
{
using namespace vsip;
using namespace vsip::impl;
using namespace vsip::impl::cbe;

struct Worker
{
  Worker(char const *code_ea,
	 int code_size,
	 float const *A,
	 float const b,
	 float const c,
	 float *R,
	 length_type len,
	 int cmd)
  {
    length_type const subvector_size = 1024;

    assert(is_dma_addr_ok(A));
    assert(is_dma_addr_ok(R));

    Task_manager *mgr = Task_manager::instance();

    task = mgr->reserve_lwp_task(VSIP_IMPL_LOGMA_BUFFER_SIZE, 2,
				 (uintptr_t)code_ea, code_size, cmd);

    assert(sizeof(float)*4*subvector_size <= VSIP_IMPL_LOGMA_BUFFER_SIZE);
    assert(8 < VSIP_IMPL_LOGMA_DTL_SIZE);

    Logma_params params;
    params.cmd = cmd;
    params.length = subvector_size;
    params.b_value = b;
    params.c_value = c;

    length_type subvectors = len / subvector_size;
    length_type spes   = mgr->num_spes();
    length_type subvectors_per_spe = subvectors / spes;
    assert(subvectors_per_spe * spes <= subvectors);
    
    float const* a_ptr = A;
    float*       r_ptr = R;

    for (index_type i = 0; i < spes && i < subvectors; ++i)
    {
      // If subvectors don't divide evenly, give the first SPEs one extra.
      length_type subvects = (i < subvectors % spes) ? subvectors_per_spe + 1
                                                     : subvectors_per_spe;
      
      lwp::Workblock block = task->create_workblock(subvects);
      params.a_ptr = (uintptr_t)a_ptr;
      params.r_ptr = (uintptr_t)r_ptr;
      block.set_parameters(params);
      block.enqueue();

      a_ptr += subvects * subvector_size;
      r_ptr += subvects * subvector_size;
      len -= subvects * subvector_size;
    }

    // Cleanup leftover data that doesn't fit into a full subvector.

    // First, handle data that can be DMA'd to the SPEs.  DMA granularity
    // is 16 bytes for sizes 16 bytes or larger.

    length_type const granularity = VSIP_IMPL_CBE_DMA_GRANULARITY / sizeof(float);
    
    if (len >= granularity)
    {
      params.length = (len / granularity) * granularity;
      assert(is_dma_size_ok(params.length*sizeof(float)));
      lwp::Workblock block = task->create_workblock(1);
      params.a_ptr = (uintptr_t)a_ptr;
      params.r_ptr = (uintptr_t)r_ptr;
      block.set_parameters(params);
      block.enqueue();
      len -= params.length;
    }

    leftover = len;
  }

  ~Worker() { task->sync();}

  std::auto_ptr<lwp::Task> task;
  length_type              leftover;
};

}


namespace vsip
{
namespace impl
{
namespace cbe
{

void vlma(
  float const *A, float const b, float const c, float *R, length_type len)
{
  static char *code = 0;
  static int size;

  if (code == 0)
    lwp::load_plugin(code, size, "plugins/logma_f.plg");

  Worker worker(code, size, A, b, c, R, len, LMA);

  if (worker.leftover)
    for (index_type i = len - worker.leftover; i < len; ++i)
      R[i] = log10(A[i]) * b + c;
}


} // namespace vsip::impl::cbe
} // namespace vsip::impl
} // namespace vsip
