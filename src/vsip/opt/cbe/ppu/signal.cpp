/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#include <vsip/core/config.hpp>
#include <vsip/math.hpp>
#include <vsip/opt/cbe/hist_params.h>
#include <vsip/opt/cbe/ppu/signal.hpp>
#include <vsip/opt/cbe/ppu/task_manager.hpp>
#include <vsip/opt/cbe/ppu/util.hpp>
#include <memory>

namespace
{
using namespace vsip;
using namespace vsip::impl;
using namespace vsip::impl::cbe;

void partial_hist(char const *code_ea, int code_size,
		  float min, float max, int *hist, length_type bins,
		  float const *data, length_type len, length_type &leftover)
{
  length_type const chunk_size = 1024;
  // DMA granularity is 16 bytes for sizes 16 bytes or larger.
  length_type const granularity = VSIP_IMPL_CBE_DMA_GRANULARITY / sizeof(float);

  assert(is_dma_addr_ok(hist));
  assert(is_dma_addr_ok(data));

  Task_manager *mgr = Task_manager::instance();

  std::auto_ptr<lwp::Task> task = 
    mgr->reserve_lwp_task(VSIP_IMPL_HIST_BUFFER_SIZE, 2,
			  (uintptr_t)code_ea, code_size);

  assert(sizeof(float)*2*chunk_size <= VSIP_IMPL_HIST_BUFFER_SIZE);

  Hist_params params;
  params.cmd         = HIST;
  params.length      = chunk_size;
  assert(bins < VSIP_IMPL_HIST_MAX_BINS);
  params.num_bin     = bins;
  params.min         = min;
  params.max         = max;
  length_type chunks = len / chunk_size;
  length_type spes   = mgr->num_spes();
  length_type chunks_per_spe = chunks / spes;
  assert(chunks_per_spe * spes <= chunks);
  // Each participating SPE will provide one (partial) histogram,
  // which we will accumulate at the end.
  length_type num_partial_results = std::min(chunks, spes) + (len % chunk_size ? 1 : 0);
  // Due to DMA constraints we may need to add padding to each chunk.
  aligned_array<int> R(num_partial_results *
		       // float and int have the same width, so this macro works
		       // fine even though bins is an int array.
		       VSIP_IMPL_INCREASE_TO_DMA_SIZE(bins, float));
    
  float const *data_ptr = data;
  int *bin_ptr = R.get();

  for (index_type i = 0; i < std::min(chunks, spes); ++i)
  {
    // If chunks don't divide evenly, give the first SPEs one extra.
    length_type my_chunks = (i < chunks % spes) ? chunks_per_spe + 1
      : chunks_per_spe;
      
    lwp::Workblock block = task->create_workblock(my_chunks);
    params.val_ptr = (uintptr_t)data_ptr;
    params.bin_ptr = (uintptr_t)bin_ptr;
    block.set_parameters(params);
    block.enqueue();

    data_ptr += my_chunks*chunk_size;
    len -= my_chunks * chunk_size;
    bin_ptr += VSIP_IMPL_INCREASE_TO_DMA_SIZE(bins, float);
  }

  // Cleanup leftover data that doesn't fit into a full chunk.

  // First, handle data that can be DMA'd to the SPEs.
    
  if (len >= granularity)
  {
    params.length = (len / granularity) * granularity;
    assert(is_dma_size_ok(params.length*sizeof(float)));
    lwp::Workblock block = task->create_workblock(1);
    params.val_ptr = (uintptr_t)data_ptr;
    params.bin_ptr = (uintptr_t)bin_ptr;
    block.set_parameters(params);
    block.enqueue();
    len -= params.length;
  }
  // Wait for all partial sums...
  task->sync();
  // ...and accumulate them.
  bin_ptr = R.get();
  for (unsigned int h = 0; h != num_partial_results;
       ++h, bin_ptr += VSIP_IMPL_INCREASE_TO_DMA_SIZE(bins, float))
    for (unsigned int i = 0; i != bins; ++i)
      hist[i] += bin_ptr[i];
  leftover = len;
}

// PPU-side variant, for leftovers.
void partial_hist(float min, float max, int *hist, length_type bins,
		  float const *data, length_type len)
{
  float delta = (max - min) / (bins - 2);
  for (index_type i = 0; i < len; ++i)
  {
    index_type bin = 0;
    if (data[i] >= max)
      bin = bins - 1;
    else if (data[i] >= min)
      bin = index_type(((data[i] - min) / delta) + 1);
    ++hist[bin];
  }
}
}

namespace vsip
{
namespace impl
{
namespace cbe
{
void
hist(float min, float max, int *hist, size_t bins, float const *data, size_t len)
{
  static char *code = 0;
  static int   size;
  if (code == 0) lwp::load_plugin(code, size, "plugins/hist_f.plg");
  index_type leftover;
  partial_hist(code, size, min, max, hist, bins, data, len, leftover);
  if (leftover)
    partial_hist(min, max, hist, bins, data + len - leftover, leftover);
}

} // namespace vsip::impl::cbe
} // namespace vsip::impl
} // namespace vsip
