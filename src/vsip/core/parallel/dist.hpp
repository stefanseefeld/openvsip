//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_OPT_DIST_HPP
#define VSIP_OPT_DIST_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/core/config.hpp>



/***********************************************************************
  Declarations & Class Definitions
***********************************************************************/

namespace vsip
{

namespace impl
{

/// Compute the size of a block distribution segement.

/// Requires:
///   SIZE is the total number of elements
///   NUM is the number of segments
///   POS is the segment position (0 <= POS < NUM).
/// Returns:
///   The size of segment POS.
///
/// Notes:
///   If SIZE does not distribute evenly over NUM segments, then extra
///   elements are allocated to the first SIZE % NUM segments.

inline length_type
segment_size(length_type size, length_type num, index_type pos)
{
  return (size / num) + (pos < size % num ? 1 : 0);
}



/// Compute the size of a block-cyclic distribution segement.

/// Requires:
///   SIZE is the total number of elements,
///   NUM is the number of segments,
///   CHUNK_SIZE is the multiple which elements are grouped together,
///   POS is the segment position (0 <= POS < NUM).
/// Returns:
///   The size of segment POS.
///
/// Notes:
///   If SIZE does not distribute evenly over NUM segments,
///      then extra chunk are allocated to the first SIZE % NUM segments.
///   If SIZE does not distribute evenly into CHUNK_SIZE,
///      then the SIZE%

inline length_type
segment_size(
  length_type size,
  length_type num,
  length_type chunk_size,
  index_type  pos)
{
  assert(pos < num);

  // Compute the total number of chunks to be distributed.
  index_type  num_chunks = size / chunk_size + (size % chunk_size ? 1 : 0);

  // Compute the number of chunks in segment POS.
  length_type seg_chunks = (num_chunks / num) +
                           (pos < num_chunks % num ? 1 : 0);

  // Compute the number of elements in segment POS.
  length_type seg_size    = seg_chunks * chunk_size;

  // Adjust number of elements if SIZE does not evenly distribute into
  // CHUNK_SIZE.
  if (size % chunk_size != 0 && (pos + 1) % num == (num_chunks % num))
    seg_size = seg_size - chunk_size + size % chunk_size;

  return seg_size;
}



/// Compute the size of a block-cyclic distribution segement in chunks.

/// Requires:
///   SIZE is the total number of elements,
///   NUM is the number of segments,
///   CHUNK_SIZE is the multiple which elements are grouped together,
///   POS is the segment position (0 <= POS < NUM).
/// Returns:
///   The number of chunks in segment POS.
///      (This may not equal the number of elements / CHUNK_SIZE, if
///      SIZE does not evenly distribute into CHUNKS).
///
/// Notes:
///   If SIZE does not distribute evenly over NUM segments,
///      then extra chunk are allocated to the first SIZE % NUM segments.
///   If SIZE does not distribute evenly into CHUNK_SIZE,
///      then the SIZE%

inline length_type
segment_chunks(
  length_type size,
  length_type num,
  length_type chunk_size,
  index_type pos)
{
  assert(pos < num);

  // Compute the total number of chunks to be distributed.
  index_type  num_chunks = size / chunk_size + (size % chunk_size ? 1 : 0);

  // Compute the number of chunks in segment POS.
  length_type seg_chunks = (num_chunks / num) +
                           (pos < num_chunks % num ? 1 : 0);

  return seg_chunks;
}



/// Compute the size of a specific chunk in a block-cyclic
/// distribution segment.

/// Requires:
///   SIZE is the total number of elements,
///   NUM is the number of segments,
///   CHUNK_SIZE is the multiple which elements are grouped together,
///   SPOS is the segment position (0 <= SPOS < NUM).
///   CPOS is the chunk position
///      (0 <= CPOS < segment_chunks(SIZE, NUM, CHUNK_SIZE, POS)).
/// Returns:
///   The size of chunk CPOS in segment SPOS.
///

inline length_type
segment_chunk_size(
  length_type size,
  length_type num,
  length_type chunk_size,
  index_type  spos,
  index_type  cpos)
{
  assert(spos < num);

  // Compute the total number of chunks to be distributed.
  index_type  num_chunks = size / chunk_size + (size % chunk_size ? 1 : 0);

  // Compute the number of chunks in segment POS.
  length_type seg_chunks = (num_chunks / num) +
                           (spos < num_chunks % num ? 1 : 0);

  assert(cpos < seg_chunks);

  if (size % chunk_size != 0 &&
      (spos + 1) % num == (num_chunks % num) &&
      cpos == seg_chunks-1)
    return size % chunk_size;
  else
    return chunk_size;
}



/// Compute the first element in a segment.

/// Requires:
///   SIZE is the total number of elements,
///   NUM is the number of segments,
///   POS is the segment position (0 <= POS < NUM).
/// Returns:
///   The index of the first element in segment POS.

inline length_type
segment_start(
  length_type size,
  length_type num,
  index_type  pos)
{
  return pos * (size / num) + std::min(pos, size % num);
}

} // namespace impl

} // namespace vsip

/***********************************************************************
  Class Definitions - Distributions
***********************************************************************/

namespace vsip
{

/// We define the following implementation specific member functions
/// for distributions:
///
/// * impl_subblock_patches(DOM, SB) 
///      returns the number of patches in subblock SB.
///
/// * impl_subblock_size(DOM, SB) 
///      returns the number of elements in subblock SB.
///
/// * impl_patch_global_dom(DOM, SB, P) 
///      returns the global domain of patch P in subblock SB.
///
/// * impl_patch_global_dom(DOM, SB, P) 
///      returns the local domain of patch P in subblock SB.

// Note: distribution_type defined in support.hpp.



/// Whole Distribution.

class Whole_dist
{
  // Constructors and destructor.
public:
  Whole_dist() VSIP_NOTHROW {}

  // This constructor allows users to construct maps without having
  // explicitly say 'Whole_dist()'.
  Whole_dist(length_type n_sb ATTRIBUTE_UNUSED) VSIP_NOTHROW { assert(n_sb == 1); }

  ~Whole_dist() VSIP_NOTHROW {}
  
  // Default copy constructor and assignment are fine.

  // Accessors
public:
  distribution_type distribution() const VSIP_NOTHROW
    { return whole; }

  index_type num_subblocks() const VSIP_NOTHROW
    { return 1; }

  length_type cyclic_contiguity() const VSIP_NOTHROW
    { return 0; }

  // Implementation specific.
public:
  length_type impl_subblock_patches(Domain<1> const&, index_type sb ATTRIBUTE_UNUSED)
    const VSIP_NOTHROW
  { assert(sb == 0); return 1; }

  length_type impl_subblock_size(Domain<1> const& dom, index_type sb ATTRIBUTE_UNUSED)
    const VSIP_NOTHROW
  { assert(sb == 0); return dom.size(); }

  Domain<1> impl_patch_global_dom(Domain<1> const& dom, index_type sb ATTRIBUTE_UNUSED,
				  index_type p ATTRIBUTE_UNUSED)
    const VSIP_NOTHROW
  { assert(sb == 0 && p == 0); return dom; }

  Domain<1> impl_patch_local_dom(Domain<1> const& dom, index_type sb ATTRIBUTE_UNUSED,
				 index_type p ATTRIBUTE_UNUSED)
    const VSIP_NOTHROW
  { assert(sb == 0 && p == 0); return dom; }

  index_type impl_subblock_from_index(Domain<1> const& dom ATTRIBUTE_UNUSED, index_type i ATTRIBUTE_UNUSED)
    const VSIP_NOTHROW
  { assert(i < dom.size()); return 0; }

  index_type impl_patch_from_index(Domain<1> const& dom ATTRIBUTE_UNUSED, index_type i ATTRIBUTE_UNUSED)
    const VSIP_NOTHROW
  { assert(i < dom.size()); return 0; }

  index_type impl_local_from_global_index(Domain<1> const& dom ATTRIBUTE_UNUSED, index_type i)
    const VSIP_NOTHROW
  { assert(i < dom.size()); return i; }

  index_type impl_global_from_local_index(Domain<1> const& dom ATTRIBUTE_UNUSED,
					  index_type       sb ATTRIBUTE_UNUSED,
					  index_type       i)
    const VSIP_NOTHROW
  { assert(i < dom.size() && sb == 0); return i; }

  // No member data.
};



/// Block Distribution.

class Block_dist
{
  // Constructors and destructor.
public:
  Block_dist(length_type num_subblocks = 1) VSIP_NOTHROW;
  ~Block_dist() VSIP_NOTHROW {}
  
  // Default copy constructor and assignment are fine.

  // Accessors
public:
  distribution_type distribution() const VSIP_NOTHROW
    { return block; }

  index_type num_subblocks() const VSIP_NOTHROW
    { return num_subblocks_; }

  length_type cyclic_contiguity() const VSIP_NOTHROW
    { return 0; }

  // Implementation specific.
public:
  length_type impl_subblock_patches(Domain<1> const& dom, index_type sb)
    const VSIP_NOTHROW;

  length_type impl_subblock_size(Domain<1> const& dom, index_type sb)
    const VSIP_NOTHROW;

  Domain<1> impl_patch_global_dom(Domain<1> const& dom, index_type sb,
				  index_type p)
    const VSIP_NOTHROW;

  Domain<1> impl_patch_local_dom(Domain<1> const& dom, index_type sb,
				 index_type p)
    const VSIP_NOTHROW;

  index_type impl_subblock_from_index(Domain<1> const& dom, index_type i)
    const VSIP_NOTHROW;

  index_type impl_patch_from_index(Domain<1> const& dom ATTRIBUTE_UNUSED, index_type i ATTRIBUTE_UNUSED)
    const VSIP_NOTHROW
  { assert(i < dom.size()); return 0; }

  index_type impl_local_from_global_index(Domain<1> const& dom, index_type i)
    const VSIP_NOTHROW;

  index_type impl_global_from_local_index(Domain<1> const& dom,
					  index_type       sb,
					  index_type       i)
    const VSIP_NOTHROW;

  // Member data.
private:
  length_type num_subblocks_;
};



/// Cyclic Distribution

class Cyclic_dist
{
  // Constructors, assignment, and destructor.
public:
  Cyclic_dist(length_type num_subblocks = 1, length_type contiguity = 1)
    VSIP_NOTHROW;
  ~Cyclic_dist() VSIP_NOTHROW {}

  // Default copy constructor and assignment are fine.

  // Accessors
public:
  distribution_type distribution() const VSIP_NOTHROW
    { return cyclic; }

  index_type num_subblocks() const VSIP_NOTHROW
    { return num_subblocks_; }

  length_type cyclic_contiguity() const VSIP_NOTHROW
    { return contiguity_; }

  // Implementation specific functions.
public:
  length_type impl_subblock_patches(Domain<1> const& dom, index_type sb)
    const VSIP_NOTHROW;

  length_type impl_subblock_size(Domain<1> const& dom, index_type sb)
    const VSIP_NOTHROW;

  Domain<1> impl_patch_global_dom(Domain<1> const& dom, index_type sb,
				  index_type p)
    const VSIP_NOTHROW;

  Domain<1> impl_patch_local_dom(Domain<1> const& dom, index_type sb,
				 index_type p)
    const VSIP_NOTHROW;

  index_type impl_subblock_from_index(Domain<1> const& dom, index_type i)
    const VSIP_NOTHROW;

  index_type impl_patch_from_index(Domain<1> const& dom, index_type i)
    const VSIP_NOTHROW;

  index_type impl_local_from_global_index(Domain<1> const& dom, index_type i)
    const VSIP_NOTHROW;

  index_type impl_global_from_local_index(Domain<1> const& dom,
					  index_type       sb,
					  index_type       i)
    const VSIP_NOTHROW;

  // Members
private:
  length_type num_subblocks_;
  length_type contiguity_;
};



/***********************************************************************
  Definitions
***********************************************************************/

// Block_dist

/// Create a block distribution.

/// Requires:
///   NUM_SUBBLOCKS is number of subblocks in distribution
///      (NUM_SUBBLOCKS > 0).
inline
Block_dist::Block_dist(
  length_type num_subblocks)
  VSIP_NOTHROW
: num_subblocks_(num_subblocks)
{
  assert(num_subblocks_ > 0);
}



inline length_type
Block_dist::impl_subblock_patches(Domain<1> const& /*dom*/, index_type sb ATTRIBUTE_UNUSED)
  const VSIP_NOTHROW
{
  assert(sb < num_subblocks_);
  return 1;
}



inline length_type
Block_dist::impl_subblock_size(
  Domain<1> const& dom,
  index_type       sb)
  const VSIP_NOTHROW
{
  assert(sb < num_subblocks_);
  return impl::segment_size(dom.length(), num_subblocks_, sb);
}



inline Domain<1>
Block_dist::impl_patch_global_dom(
  Domain<1> const& dom,
  index_type       sb,
  index_type       p ATTRIBUTE_UNUSED)
  const VSIP_NOTHROW
{
  assert(sb < num_subblocks_);
  assert(p == 0);
  return Domain<1>(impl::segment_start(dom.length(), num_subblocks_, sb),
		   1,
		   impl::segment_size(dom.length(), num_subblocks_, sb));
}



inline Domain<1>
Block_dist::impl_patch_local_dom(
  Domain<1> const& dom,
  index_type       sb,
  index_type       p ATTRIBUTE_UNUSED)
  const VSIP_NOTHROW
{
  assert(sb < num_subblocks_);
  assert(p == 0);
  return Domain<1>(0,
		   1,
		   impl::segment_size(dom.length(), num_subblocks_, sb));
}



// The nominal_block_size is roughly the total number of elements
// divided by the number of subblocks.  If this divides evenly, it
// is the exact block size, however if the number does not divide
// cleanly, then the remaining elements (called the "spill_over")
// are distributed 1 each to first subblocks.

inline index_type
Block_dist::impl_subblock_from_index(Domain<1> const& dom, index_type i)
  const VSIP_NOTHROW
{
  length_type nominal_block_size = dom.length() / num_subblocks_;
  length_type spill_over         = dom.length() % num_subblocks_;

  if (i < (nominal_block_size+1)*spill_over)
    return i / (nominal_block_size+1);
  else
    return (i - spill_over) / nominal_block_size;
}



inline index_type
Block_dist::impl_local_from_global_index(Domain<1> const& dom, index_type i)
  const VSIP_NOTHROW
{
  length_type nominal_block_size = dom.length() / num_subblocks_;
  length_type spill_over         = dom.length() % num_subblocks_;

  if (i < (nominal_block_size+1)*spill_over)
    return i % (nominal_block_size+1);
  else
    return (i - spill_over) % nominal_block_size;
}



/// Determine the global index corresponding to a local subblock index

/// Requires:
///   DOM is the full global domain,
///   SB is a valid subblock (SB < number of subblocks).
///   I  is an index into the subblock.
/// Returns:

inline index_type
Block_dist::impl_global_from_local_index(
  Domain<1> const& dom,
  index_type       sb,
  index_type       i)
const VSIP_NOTHROW
{
  length_type nominal_block_size = dom.length() / num_subblocks_;
  length_type spill_over         = dom.length() % num_subblocks_;

  return sb * nominal_block_size  +
         std::min(sb, spill_over) +
         i;
}



// -------------------------------------------------------------------- //
// Cyclic_dist

/// Create a block-cyclic distribution.

/// Requires:
///   NUM_SUBBLOCKS is number of subblocks in distribution
///      (NUM_SUBBLOCKS > 0).
///   CONTIGUITY is cyclic contiguity of the distribution
///      (CONTIGUITY > 0).

inline
Cyclic_dist::Cyclic_dist(
  length_type num_subblocks,
  length_type contiguity)
VSIP_NOTHROW
: num_subblocks_(num_subblocks),
  contiguity_   (contiguity)
{
  assert(num_subblocks_ > 0 && contiguity_ > 0);
}



/// Get the number of patches in a subblock.

/// Requires:
///   DOM is the full global domain,
///   SB is a valid subblock (SB < number of subblocks).
/// Returns:
///   The number of patches in subblock SB.

inline length_type
Cyclic_dist::impl_subblock_patches(
  Domain<1> const& dom,
  index_type       sb)
  const VSIP_NOTHROW
{
  assert(sb < num_subblocks_);
  return impl::segment_chunks(dom.length(), num_subblocks_, contiguity_, sb);
}



/// Get the number of elements in a subblock.

/// Requires:
///   DOM is the full global domain,
///   SB is a valid subblock (SB < number of subblocks).
/// Returns:
///   The number of elements in subblock SB.

inline length_type
Cyclic_dist::impl_subblock_size(
  Domain<1> const& dom,
  index_type       sb)
  const VSIP_NOTHROW
{
  assert(sb < num_subblocks_);
  return impl::segment_size(dom.length(), num_subblocks_, contiguity_, sb);
}



/// Get the global domain of a subblock/patch.

/// Requires:
///   DOM is the full global domain,
///   SB is a valid subblock (SB < number of subblocks).
///   P  is a valid patch in subblock SB.
/// Returns:
///   The global domain of patch P in subblock SB.

inline Domain<1>
Cyclic_dist::impl_patch_global_dom(
  Domain<1> const& dom,
  index_type       sb,
  index_type       p)
  const VSIP_NOTHROW
{
  assert(sb < num_subblocks_);
  assert(p < impl_subblock_patches(dom, sb));

  length_type patch_size = impl::segment_chunk_size(
    dom.length(), num_subblocks_, contiguity_, sb, p);

  return Domain<1>((sb + p*num_subblocks_)*contiguity_,
		   1,
		   patch_size);
}



/// Get the local domain of a subblock/patch.

/// Requires:
///   DOM is the full global domain,
///   SB is a valid subblock (SB < number of subblocks).
///   P  is a valid patch in subblock SB.
/// Returns:
///   The local domain of patch P in subblock SB.

inline Domain<1>
Cyclic_dist::impl_patch_local_dom(
  Domain<1> const& dom,
  index_type       sb,
  index_type       p)
  const VSIP_NOTHROW
{
  assert(sb < num_subblocks_);
  assert(p < impl_subblock_patches(dom, sb));

  length_type patch_size = impl::segment_chunk_size(
    dom.length(), num_subblocks_, contiguity_, sb, p);

  return Domain<1>(p*contiguity_,
		   1,
		   patch_size);
}



inline index_type
Cyclic_dist::impl_subblock_from_index(
  Domain<1> const& /*dom*/,
  index_type       i)
  const VSIP_NOTHROW
{
  // Determine global_patch containing index i.
  index_type p_g      = i / contiguity_;

  // Determine subblock holding this patch.
  index_type sb  = p_g % num_subblocks_;

  return sb;
}



inline index_type
Cyclic_dist::impl_patch_from_index(
  Domain<1> const& /*dom*/,
  index_type       i)
  const VSIP_NOTHROW
{
  // Determine global_patch containing index i.
  index_type p_g      = i / contiguity_;

  // Determine subblock holding this patch.
  index_type sb  = p_g % num_subblocks_;
  index_type p_l = (p_g - sb) / num_subblocks_;

  return p_l;
}



inline index_type
Cyclic_dist::impl_local_from_global_index(
  Domain<1> const& /*dom*/,
  index_type       i)
  const VSIP_NOTHROW
{
  // Determine global_patch containing index i at offset.
  index_type p_g      = i / contiguity_;
  index_type p_offset = i % contiguity_;

  // Convert this global patch to a subblock-patch.
  index_type sb  = p_g % num_subblocks_;
  index_type p_l = (p_g - sb) / num_subblocks_;

  return p_l * contiguity_ + p_offset;
}



/// Determine the global index corresponding to a local subblock index

/// Requires:
///   DOM is the full global domain,
///   SB is a valid subblock (SB < number of subblocks).
///   I  is an index into the subblock.
/// Returns:
///   The global index corresponding to subblock SB index I.

inline index_type
Cyclic_dist::impl_global_from_local_index(
  Domain<1> const& /*dom*/,
  index_type       sb,
  index_type       i)
const VSIP_NOTHROW
{
  assert(sb < num_subblocks_);

  index_type p        = i / contiguity_;
  index_type p_offset = i % contiguity_;

  return (p * num_subblocks_ + sb) * contiguity_ + p_offset;
}

} // namespace vsip

#endif // VSIP_DIST_HPP

