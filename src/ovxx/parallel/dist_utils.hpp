//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_parallel_dist_utils_hpp_
#define ovxx_parallel_dist_utils_hpp_

#include <vsip/impl/map_fwd.hpp>

namespace ovxx
{
namespace parallel
{
/// Compute the size of a block distribution segement.
///
/// Arguments:
///   size: the total number of elements
///   num:  the number of segments
///   pos:  the segment position (0 <= 'pos' < 'num').
/// Returns:
///   The size of segment 'pos'.
///
/// Notes:
///   If 'size' does not distribute evenly over 'num' segments, then extra
///   elements are allocated to the first 'size' % 'num' segments.
inline length_type
segment_size(length_type size, length_type num, index_type pos)
{
  return (size / num) + (pos < size % num ? 1 : 0);
}

/// Compute the size of a block-cyclic distribution segement.
///
/// Requires:
///   size:       the total number of elements,
///   num:        the number of segments,
///   chunk_size: the multiple which elements are grouped together
///   pos:        is the segment position (0 <= 'pos' < 'num').
/// Returns:
///   The size of segment POS.
///
/// Notes:
///   If SIZE does not distribute evenly over NUM segments,
///      then extra chunk are allocated to the first SIZE % NUM segments.
///   If SIZE does not distribute evenly into CHUNK_SIZE,
///      then the SIZE%
inline length_type
segment_size(length_type size,
	     length_type num,
	     length_type chunk_size,
	     index_type  pos)
{
  OVXX_PRECONDITION(pos < num);

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
///
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
segment_chunks(length_type size,
	       length_type num,
	       length_type chunk_size,
	       index_type pos)
{
  OVXX_PRECONDITION(pos < num);

  // Compute the total number of chunks to be distributed.
  index_type  num_chunks = size / chunk_size + (size % chunk_size ? 1 : 0);

  // Compute the number of chunks in segment POS.
  length_type seg_chunks = (num_chunks / num) +
                           (pos < num_chunks % num ? 1 : 0);

  return seg_chunks;
}

/// Compute the size of a specific chunk in a block-cyclic
/// distribution segment.
///
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
segment_chunk_size(length_type size,
		   length_type num,
		   length_type chunk_size,
		   index_type  spos,
		   index_type  cpos)
{
  OVXX_PRECONDITION(spos < num);

  // Compute the total number of chunks to be distributed.
  index_type  num_chunks = size / chunk_size + (size % chunk_size ? 1 : 0);

  // Compute the number of chunks in segment POS.
  length_type seg_chunks = (num_chunks / num) +
                           (spos < num_chunks % num ? 1 : 0);

  OVXX_PRECONDITION(cpos < seg_chunks);

  if (size % chunk_size != 0 &&
      (spos + 1) % num == (num_chunks % num) &&
      cpos == seg_chunks-1)
    return size % chunk_size;
  else
    return chunk_size;
}

/// Compute the first element in a segment.
///
/// Requires:
///   SIZE is the total number of elements,
///   NUM is the number of segments,
///   POS is the segment position (0 <= POS < NUM).
/// Returns:
///   The index of the first element in segment POS.
inline length_type
segment_start(length_type size,
	      length_type num,
	      index_type  pos)
{
  return pos * (size / num) + std::min(pos, size % num);
}

} // namespace ovxx::parallel
} // namespace ovxx


#endif
