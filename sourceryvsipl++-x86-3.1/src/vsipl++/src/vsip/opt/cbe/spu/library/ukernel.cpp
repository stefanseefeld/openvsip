/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
#include <vsip/opt/ukernel/cbe_accel/ukernel.hpp>
#include <vsip/opt/cbe/dma.h>

namespace vsip_csl
{
namespace ukernel
{
namespace impl
{

void
stream_buffer_size(Uk_stream const &stream,
		   unsigned int iter, 
		   unsigned int /*iter_count*/,
		   unsigned int &num_lines,
		   unsigned int &line_size,
		   int &offset,
		   char /*ptype*/)
{
  unsigned int chunk_idx;
  unsigned int chunk_idx0;
  unsigned int chunk_idx1;

  if (stream.dim == 3)
  {
    // Currently there is a restriction on the third dimension of a view
    // being limited to a whole distribution.  To handle this, the third
    // dimension is folded into the second (similar to the way a dense
    // 2-D view can be recast as a 1-D view)

    assert(stream.num_chunks2 == 1);

    chunk_idx  = stream.chunk_offset + iter;

    chunk_idx0 = chunk_idx / stream.num_chunks1;
    chunk_idx1 = chunk_idx % stream.num_chunks1;
    offset     = 
        chunk_idx0 * stream.chunk_size0 * stream.stride0 * sizeof(float)
      + chunk_idx1 * stream.chunk_size1 * stream.stride1 * sizeof(float);

    num_lines = stream.chunk_size0;
    line_size = stream.chunk_size1 * stream.chunk_size2;
  }
  else
  {
    chunk_idx  = stream.chunk_offset + iter;

    chunk_idx0 = chunk_idx / stream.num_chunks1;
    chunk_idx1 = chunk_idx % stream.num_chunks1;
    offset     = 
        chunk_idx0 * stream.chunk_size0 * stream.stride0 * sizeof(float)
      + chunk_idx1 * stream.chunk_size1 * stream.stride1 * sizeof(float);

    num_lines = stream.chunk_size0;
    line_size = stream.chunk_size1;
  }

  // Handle last chunk in row/column (if odd sized)
  if (chunk_idx0 == stream.num_chunks0-1 && stream.chunk_size0_extra)
    num_lines = stream.chunk_size0_extra;
  if (chunk_idx1 == stream.num_chunks1-1 && stream.chunk_size1_extra)
    line_size = stream.chunk_size1_extra;

  offset    -= stream.align_shift * sizeof(float);
  line_size += stream.align_shift;

  // Handle overlap.
  if (stream.leading_overlap0 && 
      (chunk_idx0 != 0 || !stream.skip_first_overlap0))
  {
    num_lines += stream.leading_overlap0;
    offset    -= stream.leading_overlap0 * stream.stride0 * sizeof(float);
  }
  if (stream.trailing_overlap0 &&
      (chunk_idx0 != stream.num_chunks0-1 || !stream.skip_last_overlap0))
  {
    num_lines += stream.trailing_overlap0;
  }

  if (stream.leading_overlap1 &&
      (chunk_idx1 != 0 || !stream.skip_first_overlap1))
  {
    unsigned int leading1 =
      VSIP_IMPL_INCREASE_TO_DMA_SIZE(stream.leading_overlap1, float);
    line_size += leading1;
    offset    -= leading1 * sizeof(float);
  }
  if (stream.trailing_overlap1 &&
      (chunk_idx1 != stream.num_chunks1-1 || !stream.skip_last_overlap1))
  {
    unsigned int trailing1 =
      VSIP_IMPL_INCREASE_TO_DMA_SIZE(stream.trailing_overlap1, float);
    line_size += trailing1;
  }

  line_size = VSIP_IMPL_INCREASE_TO_DMA_SIZE(line_size, float);

#if DEBUG_ALF_BASE
  printf("add_stream: type: %c  chunk: %d (%d/%d, %d/%d)  size: %d/%d x %d/%d  stride: %d, %d\n",
    ptype, chunk_idx,
    chunk_idx0, stream.num_chunks0,
    chunk_idx1, stream.num_chunks1, 
    stream.chunk_size0, num_lines, stream.chunk_size1, line_size,
    stream.stride0, stream.stride1);
#endif

}

void
set_chunk_info(Uk_stream const &stream,
	       Pinfo &pinfo,
	       int iter)
{
#if 0
  {                                                                     
    register volatile vector unsigned int get_r1 asm("1");
    unsigned int stack_pointer   = spu_extract(get_r1, 0);
    unsigned int p_stack_pointer = *(unsigned int*)stack_pointer;
    unsigned int return_addr     = *(unsigned int*)(p_stack_pointer+16);
    printf("STACK(set_chunk_info): cur %05x  pre %05x  ra %05x\n", 
           stack_pointer, p_stack_pointer, return_addr);
  }
#endif

  unsigned int chunk_idx  = stream.chunk_offset + iter;

  pinfo.dim = stream.dim;
  if (stream.dim == 1)
  {
    pinfo.dim         = 1;
    pinfo.g_offset[0] = chunk_idx * stream.chunk_size1;

    if (chunk_idx == stream.num_chunks1-1 && stream.chunk_size1_extra)
      pinfo.l_size[0] = stream.chunk_size1_extra;
    else
      pinfo.l_size[0] = stream.chunk_size1;

    pinfo.l_total_size = pinfo.l_size[0];

    pinfo.l_stride[0] = 1;

    pinfo.l_offset[0] = stream.align_shift;

    if (stream.leading_overlap1 &&
	(chunk_idx != 0 || !stream.skip_first_overlap1))
    {
      pinfo.o_leading[0] = stream.leading_overlap1;
      pinfo.l_offset[0] +=
	VSIP_IMPL_INCREASE_TO_DMA_SIZE(stream.leading_overlap1, float);
    }
    else
    {
      pinfo.o_leading[0] = 0;
    }

    if (stream.trailing_overlap1 &&
	(chunk_idx != stream.num_chunks1-1 || !stream.skip_last_overlap1))
      pinfo.o_trailing[0] = stream.trailing_overlap1;
    else
      pinfo.o_trailing[0] = 0;

  }
  else if (stream.dim == 2)
  {
    unsigned int chunk_idx0 = chunk_idx / stream.num_chunks1;
    unsigned int chunk_idx1 = chunk_idx % stream.num_chunks1;
    pinfo.g_offset[0]   = chunk_idx0 * stream.chunk_size0;
    pinfo.g_offset[1]   = chunk_idx1 * stream.chunk_size1;

    if (chunk_idx0 == stream.num_chunks0-1 && stream.chunk_size0_extra)
      pinfo.l_size[0] = stream.chunk_size0_extra;
    else
      pinfo.l_size[0] = stream.chunk_size0;

    if (chunk_idx1 == stream.num_chunks1-1 && stream.chunk_size1_extra)
      pinfo.l_size[1] = stream.chunk_size1_extra;
    else
      pinfo.l_size[1] = stream.chunk_size1;

    pinfo.l_total_size = pinfo.l_size[0] * pinfo.l_size[1];

    pinfo.l_stride[0]   = pinfo.l_size[1];
    pinfo.l_stride[1]   = 1;
    pinfo.o_leading[0]  = stream.leading_overlap0;
    pinfo.o_leading[1]  = stream.leading_overlap1;
    pinfo.o_trailing[0] = stream.trailing_overlap0;
    pinfo.o_trailing[1] = stream.trailing_overlap1;

    if (stream.leading_overlap0 &&
	(chunk_idx0 != 0 || !stream.skip_first_overlap0))
    {
      pinfo.o_leading[0] = stream.leading_overlap0;
      pinfo.l_offset[0]  = stream.leading_overlap0;
    }
    else
    {
      pinfo.o_leading[0] = 0;
      pinfo.l_offset[0]  = 0;
    }

    if (stream.trailing_overlap0 &&
	(chunk_idx0 != stream.num_chunks0-1 || !stream.skip_last_overlap0))
      pinfo.o_trailing[0] = stream.trailing_overlap0;
    else
      pinfo.o_trailing[0] = 0;

    if (stream.leading_overlap1 &&
	(chunk_idx1 != 0 || !stream.skip_first_overlap1))
    {
      pinfo.o_leading[1] = stream.leading_overlap1;
      pinfo.l_offset[1] =
	stream.align_shift + 
	VSIP_IMPL_INCREASE_TO_DMA_SIZE(stream.leading_overlap1, float);
      pinfo.l_stride[0] += pinfo.l_offset[1];
    }
    else
    {
      pinfo.o_leading[1] = 0;
      pinfo.l_offset[1] = stream.align_shift;
    }

    if (stream.trailing_overlap1 &&
	(chunk_idx1 != stream.num_chunks1-1 || !stream.skip_last_overlap1))
    {
      pinfo.o_trailing[1] = stream.trailing_overlap1;
      pinfo.l_stride[0]  +=
	VSIP_IMPL_INCREASE_TO_DMA_SIZE(stream.trailing_overlap1, float);
    }
    else
      pinfo.o_trailing[1] = 0;
  }
  else if (stream.dim == 3)
  {
    unsigned int chunk_idx0 = chunk_idx / stream.num_chunks1;
    unsigned int chunk_idx1 = chunk_idx % stream.num_chunks1;
    pinfo.g_offset[0]   = chunk_idx0 * stream.chunk_size0;
    pinfo.g_offset[1]   = chunk_idx1 * stream.chunk_size1;
    pinfo.g_offset[2]   = 0;

    if (chunk_idx0 == stream.num_chunks0-1 && stream.chunk_size0_extra)
      pinfo.l_size[0] = stream.chunk_size0_extra;
    else
      pinfo.l_size[0] = stream.chunk_size0;

    if (chunk_idx1 == stream.num_chunks1-1 && stream.chunk_size1_extra)
      pinfo.l_size[1] = stream.chunk_size1_extra;
    else
      pinfo.l_size[1] = stream.chunk_size1;

    assert(stream.num_chunks2 == 1);
    pinfo.l_size[2] = stream.chunk_size2;

    pinfo.l_total_size = pinfo.l_size[0] * pinfo.l_size[1] * pinfo.l_size[2];

    pinfo.l_stride[0]   = pinfo.l_size[1] * pinfo.l_size[2];
    pinfo.l_stride[1]   = pinfo.l_size[2];
    pinfo.l_stride[2]   = 1;
    pinfo.o_leading[0]  = stream.leading_overlap0;
    pinfo.o_leading[1]  = stream.leading_overlap1;
    pinfo.o_leading[2]  = stream.leading_overlap2;
    pinfo.o_trailing[0] = stream.trailing_overlap0;
    pinfo.o_trailing[1] = stream.trailing_overlap1;
    pinfo.o_trailing[2] = stream.trailing_overlap2;

    if (stream.leading_overlap0 &&
	(chunk_idx0 != 0 || !stream.skip_first_overlap0))
    {
      pinfo.o_leading[0] = stream.leading_overlap0;
      pinfo.l_offset[0]  = stream.leading_overlap0;
    }
    else
    {
      pinfo.o_leading[0] = 0;
      pinfo.l_offset[0]  = 0;
    }

    if (stream.trailing_overlap0 &&
        (chunk_idx0 != stream.num_chunks0-1 || !stream.skip_last_overlap0))
      pinfo.o_trailing[0] = stream.trailing_overlap0;
    else
      pinfo.o_trailing[0] = 0;

    if (stream.leading_overlap1 &&
	(chunk_idx1 != 0 || !stream.skip_first_overlap1))
    {
      pinfo.o_leading[1] = stream.leading_overlap1;
      pinfo.l_offset[1] =
	stream.align_shift + 
	VSIP_IMPL_INCREASE_TO_DMA_SIZE(stream.leading_overlap1, float);
      pinfo.l_stride[0] += pinfo.l_offset[1];
    }
    else
    {
      pinfo.o_leading[1] = 0;
      pinfo.l_offset[1] = stream.align_shift;
    }

    if (stream.trailing_overlap1 &&
	(chunk_idx1 != stream.num_chunks1-1 || !stream.skip_last_overlap1))
    {
      pinfo.o_trailing[1] = stream.trailing_overlap1;
      pinfo.l_stride[0]  +=
	VSIP_IMPL_INCREASE_TO_DMA_SIZE(stream.trailing_overlap1, float);
    }
    else
      pinfo.o_trailing[1] = 0;

    // overlap for the third dimension is not supported
    assert(stream.leading_overlap2 == 0);
    assert(stream.trailing_overlap2 == 0);
  }
  else
    assert(0);
}

} // namespace vsip_csl::ukernel::impl
} // namespace vsip_csl::ukernel
} // namespace vsip_csl
