//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef vsip_impl_dist_hpp_
#define vsip_impl_dist_hpp_

#include <vsip/support.hpp>
#include <ovxx/config.hpp>
#include <ovxx/parallel/dist_utils.hpp>

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

class Whole_dist
{
public:
  Whole_dist() VSIP_NOTHROW {}

  // This constructor allows users to construct maps without having
  // explicitly say 'Whole_dist()'.
  Whole_dist(length_type n_sb OVXX_UNUSED) VSIP_NOTHROW { OVXX_PRECONDITION(n_sb == 1);}
  ~Whole_dist() VSIP_NOTHROW {}
  
  distribution_type distribution() const VSIP_NOTHROW { return whole;}
  index_type num_subblocks() const VSIP_NOTHROW { return 1;}
  length_type cyclic_contiguity() const VSIP_NOTHROW { return 0;}

  length_type impl_subblock_patches(Domain<1> const&, index_type sb OVXX_UNUSED)
    const VSIP_NOTHROW
  { OVXX_PRECONDITION(sb == 0); return 1;}

  length_type impl_subblock_size(Domain<1> const& dom, index_type sb OVXX_UNUSED)
    const VSIP_NOTHROW
  { OVXX_PRECONDITION(sb == 0); return dom.size();}

  Domain<1> impl_patch_global_dom(Domain<1> const& dom, index_type sb OVXX_UNUSED,
				  index_type p OVXX_UNUSED)
    const VSIP_NOTHROW
  { OVXX_PRECONDITION(sb == 0 && p == 0); return dom;}

  Domain<1> impl_patch_local_dom(Domain<1> const& dom, index_type sb OVXX_UNUSED,
				 index_type p OVXX_UNUSED)
    const VSIP_NOTHROW
  { OVXX_PRECONDITION(sb == 0 && p == 0); return dom;}

  index_type impl_subblock_from_index(Domain<1> const& dom OVXX_UNUSED, index_type i OVXX_UNUSED)
    const VSIP_NOTHROW
  { OVXX_PRECONDITION(i < dom.size()); return 0;}

  index_type impl_patch_from_index(Domain<1> const& dom OVXX_UNUSED, index_type i OVXX_UNUSED)
    const VSIP_NOTHROW
  { OVXX_PRECONDITION(i < dom.size()); return 0;}

  index_type impl_local_from_global_index(Domain<1> const& dom OVXX_UNUSED, index_type i)
    const VSIP_NOTHROW
  { OVXX_PRECONDITION(i < dom.size()); return i;}

  index_type impl_global_from_local_index(Domain<1> const& dom OVXX_UNUSED,
					  index_type       sb OVXX_UNUSED,
					  index_type       i)
    const VSIP_NOTHROW
  { OVXX_PRECONDITION(i < dom.size() && sb == 0); return i;}
};

class Block_dist
{
public:
  Block_dist(length_type num_subblocks = 1) VSIP_NOTHROW
    : num_subblocks_(num_subblocks)
  {
    OVXX_PRECONDITION(num_subblocks_ > 0);
  }
  ~Block_dist() VSIP_NOTHROW {}
  
  distribution_type distribution() const VSIP_NOTHROW { return block;}
  index_type num_subblocks() const VSIP_NOTHROW { return num_subblocks_;}
  length_type cyclic_contiguity() const VSIP_NOTHROW { return 0;}

  length_type
  impl_subblock_patches(Domain<1> const &, index_type sb OVXX_UNUSED)
    const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(sb < num_subblocks_);
    return 1;
  }

  length_type impl_subblock_size(Domain<1> const& dom, index_type sb)
    const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(sb < num_subblocks_);
    using namespace ovxx::parallel;
    return segment_size(dom.length(), num_subblocks_, sb);
  }

  Domain<1>
  impl_patch_global_dom(Domain<1> const &dom,
			index_type sb,
			index_type p OVXX_UNUSED)
    const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(sb < num_subblocks_);
    OVXX_PRECONDITION(p == 0);
    using namespace ovxx::parallel;
    return Domain<1>(segment_start(dom.length(), num_subblocks_, sb),
		     1,
		     segment_size(dom.length(), num_subblocks_, sb));
  }

  Domain<1>
  impl_patch_local_dom(Domain<1> const &dom,
		       index_type sb,
		       index_type p OVXX_UNUSED)
    const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(sb < num_subblocks_);
    OVXX_PRECONDITION(p == 0);
    using namespace ovxx::parallel;
    return Domain<1>(0, 1, segment_size(dom.length(), num_subblocks_, sb));
  }

  // The nominal_block_size is roughly the total number of elements
  // divided by the number of subblocks.  If this divides evenly, it
  // is the exact block size, however if the number does not divide
  // cleanly, then the remaining elements (called the "spill_over")
  // are distributed 1 each to first subblocks.
  index_type
  impl_subblock_from_index(Domain<1> const &dom, index_type i)
    const VSIP_NOTHROW
  {
    length_type nominal_block_size = dom.length() / num_subblocks_;
    length_type spill_over         = dom.length() % num_subblocks_;

    if (i < (nominal_block_size+1)*spill_over)
      return i / (nominal_block_size+1);
    else
      return (i - spill_over) / nominal_block_size;
  }

  index_type impl_patch_from_index(Domain<1> const& dom OVXX_UNUSED, index_type i OVXX_UNUSED)
    const VSIP_NOTHROW
  { OVXX_PRECONDITION(i < dom.size()); return 0;}

  index_type impl_local_from_global_index(Domain<1> const& dom, index_type i)
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
  ///
  /// Requires:
  ///   DOM is the full global domain,
  ///   SB is a valid subblock (SB < number of subblocks).
  ///   I  is an index into the subblock.
  /// Returns:
  index_type
  impl_global_from_local_index(Domain<1> const &dom,
			       index_type sb,
			       index_type i)
    const VSIP_NOTHROW
  {
    length_type nominal_block_size = dom.length() / num_subblocks_;
    length_type spill_over         = dom.length() % num_subblocks_;

    return sb * nominal_block_size + std::min(sb, spill_over) + i;
  }

private:
  length_type num_subblocks_;
};

class Cyclic_dist
{
public:
  /// Create a block-cyclic distribution.
  ///
  /// Arguments:
  ///   num_subblocks: number of subblocks in distribution (>0)
  ///   contiguity:    cyclic contiguity of the distribution (>0)
  Cyclic_dist(length_type num_subblocks = 1, length_type contiguity = 1)
    VSIP_NOTHROW
  : num_subblocks_(num_subblocks),
    contiguity_(contiguity)
  {
    OVXX_PRECONDITION(num_subblocks_ > 0 && contiguity_ > 0);
  }
  ~Cyclic_dist() VSIP_NOTHROW {}

  distribution_type distribution() const VSIP_NOTHROW { return cyclic;}
  index_type num_subblocks() const VSIP_NOTHROW { return num_subblocks_;}
  length_type cyclic_contiguity() const VSIP_NOTHROW { return contiguity_;}

  /// Get the number of patches in a subblock.
  ///
  /// Arguments:
  ///   dom: the full global domain,
  ///   sb:  a valid subblock (sb < number of subblocks).
  /// Returns:
  ///   The number of patches in subblock sb.
  length_type
  impl_subblock_patches(Domain<1> const &dom,
			index_type sb)
    const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(sb < num_subblocks_);
    using namespace ovxx::parallel;
    return segment_chunks(dom.length(), num_subblocks_, contiguity_, sb);
  }

  /// Get the number of elements in a subblock.
  ///
  /// Arguments:
  ///   dom: the full global domain,
  ///   sb:  a valid subblock (sb < number of subblocks).
  /// Returns:
  ///   The number of elements in subblock sb.
  length_type impl_subblock_size(Domain<1> const& dom, index_type sb)
    const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(sb < num_subblocks_);
    using namespace ovxx::parallel;
    return segment_size(dom.length(), num_subblocks_, contiguity_, sb);
  }

  /// Get the global domain of a subblock/patch.
  ///
  /// Requires:
  ///   DOM is the full global domain,
  ///   SB is a valid subblock (SB < number of subblocks).
  ///   P  is a valid patch in subblock SB.
  /// Returns:
  ///   The global domain of patch P in subblock SB.
  Domain<1> impl_patch_global_dom(Domain<1> const& dom, index_type sb,
				  index_type p)
    const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(sb < num_subblocks_);
    OVXX_PRECONDITION(p < impl_subblock_patches(dom, sb));
    using namespace ovxx::parallel;
    length_type patch_size = 
      segment_chunk_size(dom.length(), num_subblocks_, contiguity_, sb, p);

    return Domain<1>((sb + p*num_subblocks_)*contiguity_,
		     1, patch_size);
  }

  /// Get the local domain of a subblock/patch.
  ///
  /// Requires:
  ///   DOM is the full global domain,
  ///   SB is a valid subblock (SB < number of subblocks).
  ///   P  is a valid patch in subblock SB.
  /// Returns:
  ///   The local domain of patch P in subblock SB.
  Domain<1> impl_patch_local_dom(Domain<1> const& dom, index_type sb,
				 index_type p)
    const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(sb < num_subblocks_);
    OVXX_PRECONDITION(p < impl_subblock_patches(dom, sb));
    using namespace ovxx::parallel;
    length_type patch_size = 
      segment_chunk_size(dom.length(), num_subblocks_, contiguity_, sb, p);
    return Domain<1>(p*contiguity_, 1, patch_size);
  }

  index_type
  impl_subblock_from_index(Domain<1> const &, index_type i) const VSIP_NOTHROW
  {
    // Determine global_patch containing index i.
    index_type p_g      = i / contiguity_;
    // Determine subblock holding this patch.
    index_type sb  = p_g % num_subblocks_;
    return sb;
  }

  index_type
  impl_patch_from_index(Domain<1> const &, index_type i)
    const VSIP_NOTHROW
  {
    // Determine global_patch containing index i.
    index_type p_g      = i / contiguity_;
    // Determine subblock holding this patch.
    index_type sb  = p_g % num_subblocks_;
    index_type p_l = (p_g - sb) / num_subblocks_;
    return p_l;
  }

  index_type
  impl_local_from_global_index(Domain<1> const &, index_type i)
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
  ///
  /// Requires:
  ///   DOM is the full global domain,
  ///   SB is a valid subblock (SB < number of subblocks).
  ///   I  is an index into the subblock.
  /// Returns:
  ///   The global index corresponding to subblock SB index I.
  index_type
  impl_global_from_local_index(Domain<1> const &, index_type sb, index_type i)
    const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(sb < num_subblocks_);
    index_type p        = i / contiguity_;
    index_type p_offset = i % contiguity_;
    return (p * num_subblocks_ + sb) * contiguity_ + p_offset;
  }

private:
  length_type num_subblocks_;
  length_type contiguity_;
};

} // namespace vsip

namespace ovxx
{
namespace parallel
{
namespace detail
{
template <typename D, typename M> struct make_dist;

template <typename M>
struct make_dist<Whole_dist, M>
{
  static Whole_dist copy(M const &, dimension_type)
  { return Whole_dist();}
};

template <typename M>
struct make_dist<Block_dist, M>
{
  static Block_dist copy(M const &map, dimension_type dim)
  { return Block_dist(map.num_subblocks(dim));}
};

template <typename M>
struct make_dist<Cyclic_dist, M>
{
  static Cyclic_dist copy(M const &map, dimension_type dim)
  {
    return Cyclic_dist(map.num_subblocks(dim),
		       map.cyclic_contiguity(dim));
  }
};

} // namespace ovxx::parallel::detail

template <typename D, typename M>
D copy_dist(M const &map, dimension_type dim)
{
  return detail::make_dist<D, M>::copy(map, dim);
}

} // namespace ovxx::parallel
} // namespace ovxx

#endif

