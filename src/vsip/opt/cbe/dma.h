/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

#ifndef VSIP_OPT_CBE_DMA_H
#define VSIP_OPT_CBE_DMA_H

#include <stdint.h>

/// DMA addresses need to be aligned to 16 bytes.
#define VSIP_IMPL_DMA_ALIGNMENT 16

/// DMA sizes should be a multiple of 16 bytes.
#define VSIP_IMPL_DMA_SIZE_QUANTUM 16

/// DMA size, measured in single precision floating point values
#define VSIP_IMPL_DMA_SIZE(T) (VSIP_IMPL_DMA_SIZE_QUANTUM / sizeof(T))

/// Max number of floats per DMA transfer
#define VSIP_IMPL_DMA_MAX_TRANSFER_SIZE(T) ((16 * 1024) / sizeof(T))
#define VSIP_IMPL_DMA_ALIGNMENT_OF(A) ((uintptr_t)(A) & (VSIP_IMPL_DMA_ALIGNMENT - 1))
#define VSIP_IMPL_IS_DMA_ALIGNED(A) (VSIP_IMPL_DMA_ALIGNMENT_OF(A) == 0)
#define VSIP_IMPL_HAS_SAME_DMA_ALIGNMENT(A, B) (VSIP_IMPL_DMA_ALIGNMENT_OF(A) == VSIP_IMPL_DMA_ALIGNMENT_OF(B))
#define VSIP_IMPL_IS_DMA_SIZE(S, T) ((S) % VSIP_IMPL_DMA_SIZE(T) == 0)
#define VSIP_IMPL_DMA_SIZE_REMAINDER(S) ((S) % VSIP_IMPL_DMA_SIZE_QUANTUM)
#define VSIP_IMPL_INCREASE_TO_DMA_SIZE(S, T) ((S) + (-(unsigned)(S) % (VSIP_IMPL_DMA_SIZE(T))))

#endif
