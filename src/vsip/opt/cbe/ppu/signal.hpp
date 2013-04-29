/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
#ifndef VSIP_OPT_CBE_PPU_SIGNAL_HPP
#define VSIP_OPT_CBE_PPU_SIGNAL_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/opt/cbe/ppu/task_manager.hpp>
#include <vsip/opt/cbe/ppu/util.hpp>
#include <vsip/opt/cbe/dma.h>
#include <vsip/dda.hpp>
#include <vsip/core/adjust_layout.hpp>

namespace vsip
{
namespace impl
{
namespace cbe
{
void
hist(float min, float max, int *hist, size_t bins, float const *data, size_t size);


} // namespace vsip::impl::cbe
} // namespace vsip::impl
} // namespace vsip

namespace vsip_csl
{
namespace dispatcher
{
template <typename HBlock, typename DBlock>
struct Evaluator<op::hist, be::cbe_sdk, void(float, float, HBlock &, DBlock const &)>
{
  // Attempt to handle all blocks as 1D dense blocks.
  typedef typename impl::adjust_layout_dim<
      1, typename get_block_layout<HBlock>::type>::type
  hblock_layout;

  typedef typename impl::adjust_layout_dim<
      1, typename get_block_layout<DBlock>::type>::type
  dblock_layout;

  static bool const ct_valid = 
    is_same<typename DBlock::value_type, float>::value &&
     // check that direct access is supported
    dda::Data<HBlock, dda::out>::ct_cost == 0 &&
    dda::Data<DBlock, dda::in>::ct_cost == 0;
    
  static bool rt_valid(float, float, HBlock &hb, DBlock const &db)
  {
    // check if all data is unit stride
    dda::Data<HBlock, dda::out, hblock_layout> data_hb(hb);
    dda::Data<DBlock, dda::in, dblock_layout> data_db(db);
    return 
      data_hb.stride(0) == 1 &&
      data_db.stride(0) == 1 &&
      impl::cbe::is_dma_addr_ok(data_hb.ptr()) &&
      impl::cbe::is_dma_addr_ok(data_db.ptr()) &&
      // obtained from benchmark `histogram -1`
      data_db.size(0) > 512 &&
      impl::cbe::Task_manager::instance()->num_spes() > 0;
  }

  static void exec(float min, float max, HBlock &hist, DBlock const &input)
  {
    profile::event<profile::dispatch>("cbe::hist");
    dda::Data<HBlock, dda::out, hblock_layout> data_hb(hist);
    dda::Data<DBlock, dda::in, dblock_layout> data_db(input);
    impl::cbe::hist(min, max, data_hb.ptr(), data_hb.size(0),
		    data_db.ptr(), data_db.size(0));
  }
};
} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif
