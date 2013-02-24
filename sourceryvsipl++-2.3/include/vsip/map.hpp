/* Copyright (c) 2005, 2006, 2008 by CodeSourcery.  All rights reserved. */

/** @file    vsip/map.hpp
    @author  Jules Bergmann
    @date    2005-02-16
    @brief   VSIPL++ Library: Map class.

*/

#ifndef VSIP_MAP_HPP
#define VSIP_MAP_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vector>
#include <algorithm>

#include <vsip/support.hpp>
#include <vsip/core/vector.hpp>
#include <vsip/core/refcount.hpp>
#include <vsip/core/value_iterator.hpp>
#include <vsip/core/parallel/services.hpp>
#include <vsip/core/map_fwd.hpp>
#include <vsip/core/parallel/dist.hpp>
#include <vsip/core/domain_utils.hpp>
#include <vsip/core/block_traits.hpp>
#include <vsip/core/length.hpp>

#include <vsip/core/parallel/global_map.hpp>
#include <vsip/core/parallel/replicated_map.hpp>
#include <vsip/core/parallel/subset_map.hpp>
#include <vsip/core/parallel/block.hpp>



/***********************************************************************
  Declarations & Class Definitions
***********************************************************************/

namespace vsip
{
namespace impl
{


inline 
void
split_tuple(
  index_type        value,
  dimension_type    dim,
  index_type const* size,
  index_type*       pos)
{
  index_type orig = value;

  (void)orig;

  // We default to row-major because that is the natural way in C/C++.
#if 0
  // Column-major mapping of processors to subblocks.
  for (dimension_type i=0; i<dim; ++i)
  {
    pos[i] = value % size[i];
    value  = (value - pos[i]) / size[i];
  }

  assert(
    (dim == 1 && pos[0]  == orig) ||
    (dim == 2 && pos[0] + pos[1]*size[0] == orig) ||
    (dim == 3 && pos[0] + pos[1]*size[0] + pos[2]*size[0]*size[1] == orig));
#else
  // Row-major mapping of processors to subblocks.
  for (dimension_type i=dim; i-->0; )
  {
    pos[i] = value % size[i];
    value  = (value - pos[i]) / size[i];
  }

  assert(
    (dim == 1 && pos[0]  == orig) ||
    (dim == 2 && pos[0]*size[1] + pos[1] == orig) ||
    (dim == 3 && pos[0]*size[1]*size[2] + pos[1]*size[2] + pos[2] == orig));
#endif
}

} // namespace impl



// Forward declaration.
template <typename Dist0,
	  typename Dist1,
	  typename Dist2>
bool
operator==(Map<Dist0, Dist1, Dist2> const& map1,
	   Map<Dist0, Dist1, Dist2> const& map2) VSIP_NOTHROW;

// Forward declaration.
template <dimension_type Dim,
	  typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
bool
map_equiv(Map<Dist0, Dist1, Dist2> const& map1,
	  Map<Dist0, Dist1, Dist2> const& map2) VSIP_NOTHROW;



template <typename Dist0,
	  typename Dist1,
	  typename Dist2>
struct Map_data
  : public impl::Ref_count<Map_data<Dist0, Dist1, Dist2> >,
    impl::Non_copyable
{
  typedef std::vector<processor_type>      impl_pvec_type;

  // Constructors.
public:
  Map_data(
    Dist0 const&            dist0,
    Dist1 const&            dist1,
    Dist2 const&            dist2)
  VSIP_NOTHROW
  : dist0_ (dist0),
    dist1_ (dist1),
    dist2_ (dist2),
    comm_  (impl::default_communicator()),
    pvec_  (),
    num_subblocks_(dist0.num_subblocks() *
		   dist1.num_subblocks() *
		   dist2.num_subblocks()),
    num_procs_ (num_subblocks_)
  {
    assert(num_subblocks_ <= comm_.pvec().size());

    for (index_type i=0; i<num_subblocks_; ++i)
      pvec_.push_back(comm_.pvec()[i]);

    subblocks_[0] = dist0_.num_subblocks();
    subblocks_[1] = dist1_.num_subblocks();
    subblocks_[2] = dist2_.num_subblocks();

    // It is necessary that the number of subblocks be less than the
    // number of processors.
    assert(num_subblocks_ <= num_procs_);
  }

  template <typename BlockT>
  Map_data(
    const_Vector<processor_type, BlockT> pvec,
    Dist0 const&                         dist0,
    Dist1 const&                         dist1,
    Dist2 const&                         dist2)
  VSIP_NOTHROW
  : dist0_ (dist0),
    dist1_ (dist1),
    dist2_ (dist2),
    comm_  (impl::default_communicator()),
    pvec_  (),
    num_subblocks_(dist0.num_subblocks() *
		   dist1.num_subblocks() *
		   dist2.num_subblocks()),
    num_procs_ (num_subblocks_)
  {
    assert(num_subblocks_ <= pvec.size());

    for (index_type i=0; i<num_subblocks_; ++i)
      pvec_.push_back(pvec.get(i));

    subblocks_[0] = dist0_.num_subblocks();
    subblocks_[1] = dist1_.num_subblocks();
    subblocks_[2] = dist2_.num_subblocks();
  }

   ~Map_data()
  {
  }

  // Member data.
public:
  Dist0               dist0_;
  Dist1               dist1_;
  Dist2               dist2_;

  impl::Communicator& comm_;
  impl_pvec_type      pvec_;		  // Grid function.

  length_type	      num_subblocks_;	  // Total number of subblocks.
  length_type	      num_procs_;	  // Total number of processors.

  index_type	      subblocks_[VSIP_MAX_DIMENSION];
					  // Number of subblocks in each
					  // dimension.
};



// Map class.

template <typename Dist0,
	  typename Dist1,
	  typename Dist2>
class Map
{
  // Compile-time typedefs.
public:
  typedef impl::Value_iterator<processor_type, unsigned> processor_iterator;

  typedef typename Map_data<Dist0, Dist1, Dist2>::impl_pvec_type
    impl_pvec_type;

  static bool const impl_local_only  = false;
  static bool const impl_global_only = true;

  // Constructors and destructor.
public:
  Map(Dist0 const& = Dist0(), Dist1 const& = Dist1(), Dist2 const& = Dist2())
    VSIP_NOTHROW;

  template <typename BlockT>
  Map(const_Vector<processor_type, BlockT>,
      Dist0 const& = Dist0(),
      Dist1 const& = Dist1(),
      Dist2 const& = Dist2())
    VSIP_NOTHROW;

  Map(Map const&) VSIP_NOTHROW;

  Map& operator=(Map const&) VSIP_NOTHROW;

  ~Map() VSIP_NOTHROW
  {
    if (this->impl_is_applied())
      impl::destroy_ll_pset(applied_pset_);
  }


  // Accessors.
public:
  // Information on individual distributions.
  distribution_type distribution     (dimension_type d) const VSIP_NOTHROW;
  length_type       num_subblocks    (dimension_type d) const VSIP_NOTHROW;
  length_type       cyclic_contiguity(dimension_type d) const VSIP_NOTHROW;

  length_type num_subblocks()  const VSIP_NOTHROW
  { return data_->num_subblocks_; }

  length_type num_processors() const VSIP_NOTHROW
  { return data_->num_procs_; }

  index_type subblock(processor_type pr) const VSIP_NOTHROW;
  index_type subblock() const VSIP_NOTHROW;

  processor_iterator processor_begin(index_type sb) const VSIP_NOTHROW;
  processor_iterator processor_end  (index_type sb) const VSIP_NOTHROW;

  const_Vector<processor_type> processor_set() const;


  // Applied map functions.
  length_type impl_num_patches     (index_type sb) const VSIP_NOTHROW;

  template <dimension_type Dim>
  void impl_apply(Domain<Dim> const& dom) VSIP_NOTHROW;

  template <dimension_type Dim>
  Domain<Dim> impl_subblock_domain(index_type sb) const VSIP_NOTHROW;

  template <dimension_type Dim>
  impl::Length<Dim> impl_subblock_extent(index_type sb) const VSIP_NOTHROW;

  template <dimension_type Dim>
  Domain<Dim> impl_global_domain(index_type sb, index_type patch)
    const VSIP_NOTHROW;

  template <dimension_type Dim>
  Domain<Dim> impl_local_domain (index_type sb, index_type patch)
    const VSIP_NOTHROW;

  template <dimension_type Dim>
  Domain<Dim> applied_domain () const VSIP_NOTHROW;

  // Implementation functions.
  impl::par_ll_pset_type impl_ll_pset() const VSIP_NOTHROW
    { assert(this->impl_is_applied()); return applied_pset_; }
  impl_pvec_type const& impl_pvec() const { return data_->pvec_; }
  impl::Communicator&   impl_comm() const { return data_->comm_; }
  bool                  impl_is_applied() const { return dim_ != 0; }

  length_type         impl_working_size() const
    { return std::min(this->num_subblocks(), this->data_->pvec_.size()); }


  // Implementation functions.
public:
  length_type impl_subblock_patches(dimension_type d, index_type sb)
    const VSIP_NOTHROW;
  length_type impl_subblock_size(dimension_type d, index_type sb)
    const VSIP_NOTHROW;
  Domain<1> impl_patch_global_dom(dimension_type d, index_type sb,
				  index_type p)
    const VSIP_NOTHROW;
  Domain<1> impl_patch_local_dom(dimension_type d, index_type sb,
				 index_type p)
    const VSIP_NOTHROW;

  index_type impl_dim_subblock_from_index(dimension_type d, index_type idx)
    const VSIP_NOTHROW;
  index_type impl_dim_patch_from_index(dimension_type d, index_type idx)
    const VSIP_NOTHROW;
  index_type impl_local_from_global_index(dimension_type d, index_type idx)
    const VSIP_NOTHROW;

  template <dimension_type Dim>
  index_type impl_subblock_from_global_index(Index<Dim> const& idx)
    const VSIP_NOTHROW;

  template <dimension_type Dim>
  index_type impl_patch_from_global_index(Index<Dim> const& idx)
    const VSIP_NOTHROW;

  index_type impl_global_from_local_index(dimension_type d, index_type sb,
					  index_type idx)
    const VSIP_NOTHROW;

  template <dimension_type Dim>
  Domain<Dim> impl_local_from_global_domain(index_type sb,
					    Domain<Dim> const& dom)
    const VSIP_NOTHROW;

  // Implementation compile-time types, and friends.
public:
  typedef Dist0 impl_dim0_type;
  typedef Dist1 impl_dim1_type;
  typedef Dist2 impl_dim2_type;

  friend bool operator==<>(Map const&, Map const&) VSIP_NOTHROW;
  friend bool map_equiv<1>(Map const&, Map const&) VSIP_NOTHROW;
  friend bool map_equiv<2>(Map const&, Map const&) VSIP_NOTHROW;
  friend bool map_equiv<3>(Map const&, Map const&) VSIP_NOTHROW;
  friend struct impl::Map_equal<1, Map, Map>;
  friend struct impl::Map_equal<2, Map, Map>;
  friend struct impl::Map_equal<3, Map, Map>;

public:
  index_type     impl_rank_from_proc(processor_type pr) const;
  processor_type impl_proc_from_rank(index_type idx) const
    { return data_->pvec_[idx]; }

  // Members.
private:
  impl::Ref_counted_ptr<Map_data<Dist0, Dist1, Dist2> > data_;

  Domain<3>	         dom_;		  // Applied domain.
  dimension_type         dim_;		  // Dimension of applied domain.
  impl::par_ll_pset_type applied_pset_;
};



/***********************************************************************
  Definitions
***********************************************************************/

template <typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
inline
Map<Dist0, Dist1, Dist2>::Map(
  Dist0 const&            dist0,
  Dist1 const&            dist1,
  Dist2 const&            dist2)
VSIP_NOTHROW
: data_      (new Map_data<Dist0, Dist1, Dist2>(dist0, dist1, dist2),
	      impl::noincrement),
  dim_       (0)
{
  // It is necessary that the number of subblocks be less than the
  // number of processors.
  assert(data_->num_subblocks_ <= data_->num_procs_);
}



template <typename Dist0,
	  typename Dist1,
	  typename Dist2>
template <typename BlockT>
inline
Map<Dist0, Dist1, Dist2>::Map(
  const_Vector<processor_type, BlockT> pvec,
  Dist0 const&                         dist0,
  Dist1 const&                         dist1,
  Dist2 const&                         dist2)
VSIP_NOTHROW
: data_     (new Map_data<Dist0, Dist1, Dist2>(pvec, dist0, dist1, dist2),
	     impl::noincrement),
  dim_      (0)
{
  // It is necessary that the number of subblocks be less than the
  // number of processors.
  assert(data_->num_subblocks_ <= data_->num_procs_);
}



template <typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
inline
Map<Dist0, Dist1, Dist2>::Map(Map const& rhs) VSIP_NOTHROW
: data_      (rhs.data_),
  dom_       (rhs.dom_),
  dim_       (0)
{
}



template <typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
inline Map<Dist0, Dist1, Dist2>&
Map<Dist0, Dist1, Dist2>::operator=(Map const& rhs) VSIP_NOTHROW
{
  data_ = rhs.data_;

  dom_  = rhs.dom_;
  dim_  = 0;

  return *this;
}



// Apply a map to a domain.

// Notes:
// [1] Do not allow maps to partition dimensions beyond the applied domain.
//     This creates empty subblocks outside of the map's dimension,
//     which confuses the routines which convert a map subblock index
//     into individual dimension subblock indices (split_tuple).

template <typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
template <dimension_type Dim>
inline void
Map<Dist0, Dist1, Dist2>::impl_apply(Domain<Dim> const& dom)
  VSIP_NOTHROW
{
  Domain<1> arr[VSIP_MAX_DIMENSION];

  for (dimension_type d=0; d<Dim; ++d)
    arr[d] = dom[d];
  for (dimension_type d=Dim; d<VSIP_MAX_DIMENSION; ++d)
  {
    arr[d] = Domain<1>(1);
    assert(this->num_subblocks(d) == 1); // note [1]
  }

  dim_ = Dim;
  dom_ = impl::construct_domain<VSIP_MAX_DIMENSION>(arr);

  impl_pvec_type const& pvec = this->impl_pvec();
  if (VSIP_IMPL_USE_PAS_SEGMENT_SIZE)
  {
    // Create the applied pset, which excludes processors with empty subblocks.
    impl_pvec_type real_pvec;
    real_pvec.reserve(pvec.size() + 1);

    for (index_type i=0; i<pvec.size(); ++i)
    {
      processor_type pr = pvec[i];
      index_type     sb = this->subblock(pr);
      if (this->template impl_subblock_domain<Dim>(sb).size() > 0)
	real_pvec.push_back(pr);
    }

    impl::create_ll_pset(real_pvec, applied_pset_);
  }
  else
    impl::create_ll_pset(pvec, applied_pset_);
}



template <typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
inline distribution_type
Map<Dist0, Dist1, Dist2>::distribution(dimension_type d)
  const VSIP_NOTHROW
{
  assert(d < VSIP_MAX_DIMENSION);

  switch (d)
  {
  default: assert(false);
  case 0: return data_->dist0_.distribution();
  case 1: return data_->dist1_.distribution();
  case 2: return data_->dist2_.distribution();
  }
}



template <typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
inline length_type
Map<Dist0, Dist1, Dist2>::num_subblocks(dimension_type d)
  const VSIP_NOTHROW
{
  assert(d < VSIP_MAX_DIMENSION);

  switch (d)
  {
  default: assert(false);
  case 0: return data_->dist0_.num_subblocks();
  case 1: return data_->dist1_.num_subblocks();
  case 2: return data_->dist2_.num_subblocks();
  }
}



template <typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
inline length_type
Map<Dist0, Dist1, Dist2>::cyclic_contiguity(dimension_type d)
  const VSIP_NOTHROW
{
  assert(d < VSIP_MAX_DIMENSION);

  switch (d)
  {
  default: assert(false);
  case 0: return data_->dist0_.cyclic_contiguity();
  case 1: return data_->dist1_.cyclic_contiguity();
  case 2: return data_->dist2_.cyclic_contiguity();
  }
}



// Get number of patches in a dimension D's subblock SB.

template <typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
inline length_type
Map<Dist0, Dist1, Dist2>::impl_subblock_patches(
  dimension_type d,
  index_type  sb
  )
  const VSIP_NOTHROW
{
  assert(d < VSIP_MAX_DIMENSION);

  switch (d)
  {
  default: assert(false);
  case 0: return data_->dist0_.impl_subblock_patches(dom_[0], sb);
  case 1: return data_->dist1_.impl_subblock_patches(dom_[1], sb);
  case 2: return data_->dist2_.impl_subblock_patches(dom_[2], sb);
  }
}



// Get size of dimension D's subblock SB.

template <typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
inline length_type
Map<Dist0, Dist1, Dist2>::impl_subblock_size(
  dimension_type d,
  index_type  sb
  )
  const VSIP_NOTHROW
{
  assert(d < VSIP_MAX_DIMENSION);

  switch (d)
  {
  default: assert(false);
  case 0: return data_->dist0_.impl_subblock_size(dom_[0], sb);
  case 1: return data_->dist1_.impl_subblock_size(dom_[1], sb);
  case 2: return data_->dist2_.impl_subblock_size(dom_[2], sb);
  }
}



// Get global domain for a dimension/subblock/patch

template <typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
inline Domain<1>
Map<Dist0, Dist1, Dist2>::impl_patch_global_dom(
  dimension_type d,
  index_type  sb,
  index_type     p
  )
  const VSIP_NOTHROW
{
  assert(d < VSIP_MAX_DIMENSION);

  switch (d)
  {
  default: assert(false);
  case 0: return data_->dist0_.impl_patch_global_dom(dom_[0], sb, p);
  case 1: return data_->dist1_.impl_patch_global_dom(dom_[1], sb, p);
  case 2: return data_->dist2_.impl_patch_global_dom(dom_[2], sb, p);
  }
}



// Get local domain for a dimension's subblock-patch.

template <typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
inline Domain<1>
Map<Dist0, Dist1, Dist2>::impl_patch_local_dom(
  dimension_type d,
  index_type     sb,
  index_type     p
  )
  const VSIP_NOTHROW
{
  assert(d < VSIP_MAX_DIMENSION);

  switch (d)
  {
  default: assert(false);
  case 0: return data_->dist0_.impl_patch_local_dom(dom_[0], sb, p);
  case 1: return data_->dist1_.impl_patch_local_dom(dom_[1], sb, p);
  case 2: return data_->dist2_.impl_patch_local_dom(dom_[2], sb, p);
  }
}



// Get local subblock for a given index.

template <typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
inline index_type
Map<Dist0, Dist1, Dist2>::impl_dim_subblock_from_index(
  dimension_type d,
  index_type     idx
  )
  const VSIP_NOTHROW
{
  assert(d < VSIP_MAX_DIMENSION);
  assert(d < dim_);

  switch (d)
  {
  default: assert(false);
  case 0: return data_->dist0_.impl_subblock_from_index(dom_[0], idx);
  case 1: return data_->dist1_.impl_subblock_from_index(dom_[1], idx);
  case 2: return data_->dist2_.impl_subblock_from_index(dom_[2], idx);
  }
}



// Get local patch for a given index.

template <typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
inline index_type
Map<Dist0, Dist1, Dist2>::impl_dim_patch_from_index(
  dimension_type d,
  index_type     idx
  )
  const VSIP_NOTHROW
{
  assert(d < VSIP_MAX_DIMENSION);
  assert(d < dim_);

  switch (d)
  {
  default: assert(false);
  case 0: return data_->dist0_.impl_patch_from_index(dom_[0], idx);
  case 1: return data_->dist1_.impl_patch_from_index(dom_[1], idx);
  case 2: return data_->dist2_.impl_patch_from_index(dom_[2], idx);
  }
}



template <typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
inline index_type
Map<Dist0, Dist1, Dist2>::impl_local_from_global_index(
  dimension_type d,
  index_type     idx
  )
  const VSIP_NOTHROW
{
  assert(d < VSIP_MAX_DIMENSION);
  assert(d < dim_);

  switch (d)
  {
  default: assert(false);
  case 0: return data_->dist0_.impl_local_from_global_index(dom_[0], idx);
  case 1: return data_->dist1_.impl_local_from_global_index(dom_[1], idx);
  case 2: return data_->dist2_.impl_local_from_global_index(dom_[2], idx);
  }
}



/// Determine subblock holding a global index.

template <typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
template <dimension_type Dim>
inline index_type
Map<Dist0, Dist1, Dist2>::impl_subblock_from_global_index(
  Index<Dim> const& idx
  )
  const VSIP_NOTHROW
{
  index_type sb = 0;
  assert(dim_ != 0 && dim_ == Dim);

  for (dimension_type d=0; d<Dim; ++d)
  {
    assert(idx[d] < dom_[d].size());
    if (d != 0)
      sb *= data_->subblocks_[d];
    sb += impl_dim_subblock_from_index(d, idx[d]);
  }

  assert(sb < data_->num_subblocks_);
  index_type dim_sb[VSIP_MAX_DIMENSION];
  impl::split_tuple(sb, dim_, data_->subblocks_, dim_sb);
  for (dimension_type d=0; d<Dim; ++d)
  {
    assert(dim_sb[d] == impl_dim_subblock_from_index(d, idx[d]));
  }

  return sb;
}



/// Determine subblock/patch holding a global index.

template <typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
template <dimension_type Dim>
inline index_type
Map<Dist0, Dist1, Dist2>::impl_patch_from_global_index(
  Index<Dim> const& idx
  )
  const VSIP_NOTHROW
{
  index_type p = 0;
  index_type dim_sb[VSIP_MAX_DIMENSION];

  assert(dim_ != 0 && dim_ == Dim);

  index_type sb = this->impl_subblock_from_global_index(idx);
  impl::split_tuple(sb, dim_, data_->subblocks_, dim_sb);

  for (dimension_type d=0; d<Dim; ++d)
  {
    assert(idx[d] < dom_[d].size());
    if (d != 0)
      p *= impl_subblock_patches(d, dim_sb[d]);
    p += impl_dim_patch_from_index(d, idx[d]);
  }

  assert(p < this->impl_num_patches(sb));
  return p;
}



/// Determine global index from local index for a single dimension

/// Requires:
///   D to be a dimension for map.
///   SB to be a valid subblock for map.
///   IDX to be an local index within subblock SB.
/// Returns:
///   global index corresponding to local index IDX.

template <typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
inline index_type
Map<Dist0, Dist1, Dist2>::impl_global_from_local_index(
  dimension_type d,
  index_type     sb,
  index_type     idx
  )
  const VSIP_NOTHROW
{
  assert(d < VSIP_MAX_DIMENSION);
  assert(dim_ != 0 && d < dim_);

  index_type dim_sb[VSIP_MAX_DIMENSION];
  impl::split_tuple(sb, dim_, data_->subblocks_, dim_sb);

  switch (d)
  {
  default: assert(false);
  case 0: return data_->dist0_.impl_global_from_local_index(dom_[0], dim_sb[0], idx);
  case 1: return data_->dist1_.impl_global_from_local_index(dom_[1], dim_sb[1], idx);
  case 2: return data_->dist2_.impl_global_from_local_index(dom_[2], dim_sb[2], idx);
  }
}



template <typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
template <dimension_type Dim>
inline Domain<Dim>
Map<Dist0, Dist1, Dist2>::impl_local_from_global_domain(
  index_type         sb,
  Domain<Dim> const& g_dom
  )
  const VSIP_NOTHROW
{
  assert(sb < data_->num_subblocks_);
  assert(dim_ == Dim);

  if (this->impl_num_patches(sb) != 1)
      VSIP_IMPL_THROW(
	impl::unimplemented(
	  "Map<>::impl_local_from_global_domain: Subviews have a single patch"));

  Domain<Dim> sb_g_dom = this->template impl_global_domain<Dim>(sb, 0);
  Domain<Dim> sb_l_dom = this->template impl_local_domain<Dim>(sb, 0);
  Domain<Dim> intr;

  if (impl::intersect(sb_g_dom, g_dom, intr))
  {
    Domain<Dim> l_dom = impl::apply_intr(sb_l_dom, sb_g_dom, intr);
    return l_dom;
  }
  else
    return impl::empty_domain<Dim>();
}



/// Lookup the index of a processor in the map's processor set.

/// Requires:
///   PR is a processor in the map's processor set (its' grid function).
///
/// Effects:
///   Returns the index of processor PR in the set.
///   If PR is not in the set, an exception is thrown and/or an
///      assertion fails.

template <typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
inline index_type
Map<Dist0, Dist1, Dist2>::impl_rank_from_proc(processor_type pr)
  const
{
  for (index_type i=0; i<data_->pvec_.size(); ++i)
    if (data_->pvec_[i] == pr)
      return i;

  return no_rank;
}



/// Subblock held by processor

/// Requires:
///   PR is a processor in the map's processor set.
/// Returns:
///   The subblock held by processor PR, or NO_SUBBLOCK if there is none.

template <typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
inline
index_type
Map<Dist0, Dist1, Dist2>::subblock(processor_type pr)
  const VSIP_NOTHROW
{
  index_type pi = impl_rank_from_proc(pr);

  if (pi != no_rank && pi < data_->num_subblocks_)
    return pi;
  else
    return no_subblock;
}



/// Subblock held by local processor

/// Returns:
///   The subblock held by processor PR, or NO_SUBBLOCK if there is none.

template <typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
inline
index_type
Map<Dist0, Dist1, Dist2>::subblock()
  const VSIP_NOTHROW
{
  processor_type pr = local_processor();
  index_type     pi = impl_rank_from_proc(pr);

  if (pi != no_rank && pi < data_->num_subblocks_)
    return pi;
  else
    return no_subblock;
}



/// Beginning of range for processors holding a subblock.

/// Requires:
///   SB is a subblock of the map.
///
/// Returns:
///   A PROCESSOR_ITERATOR referring to the first processor holding
///      subblock SB.
///
/// Notes:
///  - For a Map, each subblock is only mapped to one processor.

template <typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
inline
typename Map<Dist0, Dist1, Dist2>::processor_iterator
Map<Dist0, Dist1, Dist2>::processor_begin(index_type sb)
  const VSIP_NOTHROW
{
  assert(sb < data_->num_subblocks_);

  return processor_iterator(data_->pvec_[sb % data_->num_procs_], 1);
}



/// End of range for processors holding a subblock.

/// Requires:
///   SB is a subblock of the map.
///
/// Returns:
///   A PROCESSOR_ITERATOR referring to the one past the last
///      processor holding subblock SB.
///
/// Notes:
///  - For a Map, each subblock is only mapped to one processor.

template <typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
inline
typename Map<Dist0, Dist1, Dist2>::processor_iterator
Map<Dist0, Dist1, Dist2>::processor_end(index_type sb)
  const VSIP_NOTHROW
{
  assert(sb < data_->num_subblocks_);

  return processor_iterator(data_->pvec_[sb % data_->num_procs_]+1, 1);
}



/// Map's processor set.

/// Returns:
///   A vector containing the map's processor set.

template <typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
inline
const_Vector<processor_type>
Map<Dist0, Dist1, Dist2>::processor_set()
  const
{
  Vector<processor_type> pset(this->data_->num_procs_);

  for (index_type i=0; i<this->data_->num_procs_; ++i)
    pset.put(i, this->data_->pvec_[i]);

  return pset;
}



/// Get the number of patches in a subblock.

/// Requires:
///   SB is a valid subblock of THIS, or NO_SUBBLOCK.

template <typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
inline
length_type
Map<Dist0, Dist1, Dist2>::impl_num_patches(index_type sb)
  const VSIP_NOTHROW
{
  assert(sb < data_->num_subblocks_ || sb == no_subblock);
  assert(dim_ != 0);

  if (sb == no_subblock)
    return 0;
  else
  {
    index_type dim_sb[VSIP_MAX_DIMENSION];

    impl::split_tuple(sb, dim_, data_->subblocks_, dim_sb);

    length_type patches = 1;
    for (dimension_type d=0; d<dim_; ++d)
      patches *= impl_subblock_patches(d, dim_sb[d]);

    return patches;
  }
}



/// Get the size of a subblock (represented by a domain).

/// Requires:
///   SB is a valid subblock of THIS, or NO_SUBBLOCK.

template <typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
template <dimension_type Dim>
inline
Domain<Dim>
Map<Dist0, Dist1, Dist2>::impl_subblock_domain(index_type sb)
  const VSIP_NOTHROW
{
  assert(sb < data_->num_subblocks_ || sb == no_subblock);
  assert(dim_ == Dim);

  if (sb == no_subblock)
    return impl::empty_domain<Dim>();
  else
  {
    index_type dim_sb[VSIP_MAX_DIMENSION];
    Domain<1>  dom[VSIP_MAX_DIMENSION];

    impl::split_tuple(sb, dim_, data_->subblocks_, dim_sb);

    for (dimension_type d=0; d<dim_; ++d)
      dom[d] = Domain<1>(impl_subblock_size(d, dim_sb[d]));

    return impl::construct_domain<Dim>(dom);
  }
}



/// Get the size of a subblock (represented by a domain).

/// Requires:
///   SB is a valid subblock of THIS, or NO_SUBBLOCK.

template <typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
template <dimension_type Dim>
inline
impl::Length<Dim>
Map<Dist0, Dist1, Dist2>::impl_subblock_extent(index_type sb)
  const VSIP_NOTHROW
{
  assert(sb < data_->num_subblocks_ || sb == no_subblock);
  assert(dim_ == Dim);

  impl::Length<Dim> size;

  if (sb == no_subblock)
  {
    for (dimension_type d=0; d<dim_; ++d)
      size[d] = 0;
  }
  else
  {
    index_type dim_sb[VSIP_MAX_DIMENSION];

    impl::split_tuple(sb, dim_, data_->subblocks_, dim_sb);

    for (dimension_type d=0; d<dim_; ++d)
      size[d] = impl_subblock_size(d, dim_sb[d]);
  }

  return size;
}



/// Return the global domain of a subblock's patch.

/// Requires:
///   SB to be a valid subblock of Map,
///   P to be a valid patch of subblock SB.

template <typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
template <dimension_type Dim>
inline
Domain<Dim>
Map<Dist0, Dist1, Dist2>::impl_global_domain(
  index_type    sb,
  index_type    p)
const VSIP_NOTHROW
{
  index_type dim_sb[VSIP_MAX_DIMENSION];
  index_type p_size[VSIP_MAX_DIMENSION];
  index_type dim_p[VSIP_MAX_DIMENSION];
  Domain<1>  dom[VSIP_MAX_DIMENSION];

  // Handle special-case of no_subblock before checking validity of
  // other arguments.
  if (sb == no_subblock)
  {
    for (dimension_type d=0; d<Dim; ++d)
      dom[d] = Domain<1>(0);
    return impl::construct_domain<Dim>(dom);
  }

  assert(sb < data_->num_subblocks_);
  assert(p  < this->impl_num_patches(sb));
  assert(dim_ == Dim);

  impl::split_tuple(sb, Dim, data_->subblocks_, dim_sb);

  for (dimension_type d=0; d<Dim; ++d)
    p_size[d] = impl_subblock_patches(d, dim_sb[d]);

  impl::split_tuple(p, Dim, p_size, dim_p);

  for (dimension_type d=0; d<Dim; ++d)
    dom[d] = impl_patch_global_dom(d, dim_sb[d], dim_p[d]);

  return impl::construct_domain<Dim>(dom);
}



/// Return the local domain of a subblock's patch.

/// Requires:
///   SB to be a valid subblock of Map,
///   P to be a valid patch of subblock SB.

template <typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
template <dimension_type Dim>
inline
Domain<Dim>
Map<Dist0, Dist1, Dist2>::impl_local_domain(
  index_type    sb,
  index_type    p
  )
  const VSIP_NOTHROW
{
  index_type dim_sb[VSIP_MAX_DIMENSION];
  index_type p_size[VSIP_MAX_DIMENSION];
  index_type dim_p[VSIP_MAX_DIMENSION];
  Domain<1>  dom[VSIP_MAX_DIMENSION];

  // Handle special-case of no_subblock before checking validity of
  // other arguments.
  if (sb == no_subblock)
  {
    for (dimension_type d=0; d<Dim; ++d)
      dom[d] = Domain<1>(0);
    return impl::construct_domain<Dim>(dom);
  }

  assert(sb < data_->num_subblocks_);
  assert(p  < this->impl_num_patches(sb));
  assert(dim_ == Dim);


  impl::split_tuple(sb, Dim, data_->subblocks_, dim_sb);

  for (dimension_type d=0; d<Dim; ++d)
    p_size[d] = impl_subblock_patches(d, dim_sb[d]);

  impl::split_tuple(p, Dim, p_size, dim_p);

  for (dimension_type d=0; d<Dim; ++d)
    dom[d] = impl_patch_local_dom(d, dim_sb[d], dim_p[d]);

  return impl::construct_domain<Dim>(dom);
}



/// Return the applied domain of a map

/// Requires:

template <typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
template <dimension_type Dim>
inline
Domain<Dim>
Map<Dist0, Dist1, Dist2>::applied_domain()
  const VSIP_NOTHROW
{
  assert(dim_ == Dim);

  Domain<1>  dom[VSIP_MAX_DIMENSION];

  for (dimension_type d=0; d<Dim; ++d)
    dom[d] = this->dom_[d];

  return impl::construct_domain<Dim>(dom);
}



template <typename Dim0,
	  typename Dim1,
	  typename Dim2>
bool
operator==(Map<Dim0, Dim1, Dim2> const& map1,
	   Map<Dim0, Dim1, Dim2> const& map2) VSIP_NOTHROW
{
  for (dimension_type d=0; d<VSIP_MAX_DIMENSION; ++d)
  {
    if (map1.distribution(d)      != map2.distribution(d) ||
	map1.num_subblocks(d)     != map2.num_subblocks(d) ||
	map1.cyclic_contiguity(d) != map2.cyclic_contiguity(d))
      return false;
  }

  if (map1.data_->comm_ != map2.data_->comm_)
    return false;

  if (map1.data_->pvec_.size() != map2.data_->pvec_.size())
    return false;

  for (index_type i=0; i<map1.data_->pvec_.size(); ++i)
    if (map1.data_->pvec_[i] != map2.data_->pvec_[i])
      return false;

  return true;
}



template <dimension_type Dim,
	  typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
bool
map_equiv(Map<Dist0, Dist1, Dist2> const& map1,
	  Map<Dist0, Dist1, Dist2> const& map2) VSIP_NOTHROW
{
  if (Dim == 1 &&
         (map1.data_->dist0_.num_subblocks()     !=
          map2.data_->dist0_.num_subblocks()
       || map1.data_->dist0_.cyclic_contiguity() !=
          map2.data_->dist0_.cyclic_contiguity()))
    return false;
  else if (Dim == 2 &&
         (map1.data_->dist0_.num_subblocks()     !=
          map2.data_->dist0_.num_subblocks()
       || map1.data_->dist0_.cyclic_contiguity() !=
          map2.data_->dist0_.cyclic_contiguity()
       || map1.data_->dist1_.num_subblocks()     !=
          map2.data_->dist1_.num_subblocks()
       || map1.data_->dist1_.cyclic_contiguity() !=
          map2.data_->dist1_.cyclic_contiguity()))
    return false;
  else if (Dim == 3 &&
      (   map1.data_->dist0_.num_subblocks()     !=
          map2.data_->dist0_.num_subblocks()
       || map1.data_->dist0_.cyclic_contiguity() !=
          map2.data_->dist0_.cyclic_contiguity()
       || map1.data_->dist1_.num_subblocks()     !=
          map2.data_->dist1_.num_subblocks()
       || map1.data_->dist1_.cyclic_contiguity() !=
          map2.data_->dist1_.cyclic_contiguity()
       || map1.data_->dist2_.num_subblocks()     !=
          map2.data_->dist2_.num_subblocks()
       || map1.data_->dist2_.cyclic_contiguity() !=
          map2.data_->dist2_.cyclic_contiguity()))
    return false;


  // implied by checks on distX_.num_subblocks()
  assert(map1.num_subblocks() == map1.num_subblocks());

  if (map1.data_->comm_ != map2.data_->comm_)
    return false;

  assert(map1.data_->pvec_.size() >= map1.num_subblocks());
  assert(map2.data_->pvec_.size() >= map2.num_subblocks());

  for (index_type i=0; i<map1.num_subblocks(); ++i)
    if (map1.data_->pvec_[i] != map2.data_->pvec_[i])
      return false;

  return true;
}



template <typename DimA0,
	  typename DimA1,
	  typename DimA2,
	  typename DimB0,
	  typename DimB1,
	  typename DimB2>
bool
operator==(Map<DimA0, DimA1, DimA2> const&,
	   Map<DimB0, DimB1, DimB2> const&) VSIP_NOTHROW
{
  return false;
}



namespace impl
{

template <typename Dist0,
	  typename Dist1,
	  typename Dist2>
struct Is_global_map<Map<Dist0, Dist1, Dist2> >
{ static bool const value = true; };

template <dimension_type Dim,
          typename       Map>
struct Select_dist;

template <typename Dist0,
	  typename Dist1,
	  typename Dist2>
struct Select_dist<0, Map<Dist0, Dist1, Dist2> >
{ typedef Dist0 type; };

template <typename Dist0,
	  typename Dist1,
	  typename Dist2>
struct Select_dist<1, Map<Dist0, Dist1, Dist2> >
{ typedef Dist1 type; };

template <typename Dist0,
	  typename Dist1,
	  typename Dist2>
struct Select_dist<2, Map<Dist0, Dist1, Dist2> >
{ typedef Dist2 type; };

template <dimension_type Dim,
	  typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
struct Is_block_dist<Dim, Map<Dist0, Dist1, Dist2> >
{
private:
  typedef typename Select_dist<Dim, Map<Dist0, Dist1, Dist2> >::type dist_type;
public:
  static bool const value = Type_equal<dist_type, Block_dist>::value ||
                            Type_equal<dist_type, Whole_dist>::value;
};


template <dimension_type Dim,
	  typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
struct Map_equal<Dim, Map<Dist0, Dist1, Dist2>, Map<Dist0, Dist1, Dist2> >
{
  static bool value(Map<Dist0, Dist1, Dist2> const& map1,
		    Map<Dist0, Dist1, Dist2> const& map2)
  { return (map1.data_.get() == map2.data_.get()) ||
           map_equiv<Dim>(map1, map2); }
};



/***********************************************************************
  Clone a distribution.
***********************************************************************/

template <typename DistT, typename MapT> struct Dist_factory;

template <typename MapT>
struct Dist_factory<Whole_dist, MapT>
{
  static Whole_dist copy(MapT const&, dimension_type)
  { return Whole_dist(); }
};

template <typename MapT>
struct Dist_factory<Block_dist, MapT>
{
  static Block_dist copy(MapT const& map, dimension_type dim)
  { return Block_dist(map.num_subblocks(dim));}
};

template <typename MapT>
struct Dist_factory<Cyclic_dist, MapT>
{
  static Cyclic_dist copy(MapT const& map, dimension_type dim)
  {
    return Cyclic_dist(map.num_subblocks(dim),
		       map.cyclic_contiguity(dim));
  }
};

template <typename DistT, typename MapT>
DistT copy_dist(MapT const& map, dimension_type dim)
{
  return Dist_factory<DistT, MapT>::copy(map, dim);
}



/***********************************************************************
  Project a Map, removing 1 dimension.
***********************************************************************/

template <typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
struct Map_project_1<0, Map<Dist0, Dist1, Dist2> >
{
  typedef Map<Dist1, Dist2> type;

  // Return the parent subblock corresponding to a child subblock.
  static index_type parent_subblock(
    Map<Dist0, Dist1, Dist2> const& map,
    index_type                      idx,
    index_type                      sb)
  {
    // length_type num_sb_0 = map.num_subblocks(0);
    length_type num_sb_1 = map.num_subblocks(1);
    length_type num_sb_2 = map.num_subblocks(2);

    index_type fix_sb_0 = map.impl_dim_subblock_from_index(0, idx);

    return fix_sb_0*num_sb_1*num_sb_2+ sb;
  }

  static type project(Map<Dist0, Dist1, Dist2> const& map, index_type idx)
  {
    // length_type num_sb_0 = map.num_subblocks(0);
    length_type num_sb_1 = map.num_subblocks(1);
    length_type num_sb_2 = map.num_subblocks(2);

    index_type fix_sb_0 = map.impl_dim_subblock_from_index(0, idx);

    Vector<processor_type> pvec(num_sb_1*num_sb_2);

    for (index_type pi=0; pi<num_sb_1*num_sb_2; ++pi)
      pvec(pi) = map.impl_proc_from_rank(fix_sb_0*num_sb_1*num_sb_2+pi);

    return type(pvec, copy_dist<Dist1>(map, 1), copy_dist<Dist2>(map, 2));
  }
};



template <typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
struct Map_project_1<1, Map<Dist0, Dist1, Dist2> >
{
  typedef Map<Dist0, Dist2> type;

  // Return the parent subblock corresponding to a child subblock.
  static index_type parent_subblock(
    Map<Dist0, Dist1, Dist2> const& map,
    index_type                      idx,
    index_type                      sb)
  {
    length_type num_sb_0 = map.num_subblocks(0);
    length_type num_sb_1 = map.num_subblocks(1);
    length_type num_sb_2 = map.num_subblocks(2);

    index_type fix_sb_1 = map.impl_dim_subblock_from_index(1, idx);

    index_type sb_0 = sb / num_sb_2;
    index_type sb_2 = sb % num_sb_2;

    return sb_0*num_sb_1*num_sb_2 + fix_sb_1*num_sb_2      + sb_2;
  }

  static type project(Map<Dist0, Dist1, Dist2> const& map, index_type idx)
  {
    length_type num_sb_0 = map.num_subblocks(0);
    length_type num_sb_1 = map.num_subblocks(1);
    length_type num_sb_2 = map.num_subblocks(2);

    index_type fix_sb_1 = map.impl_dim_subblock_from_index(1, idx);

    Vector<processor_type> pvec(num_sb_0*num_sb_2);

    for (index_type pi=0; pi<num_sb_0*num_sb_2; ++pi)
    {
      index_type sb_0 = pi / num_sb_2;
      index_type sb_2 = pi % num_sb_2;
      pvec(pi) = map.impl_proc_from_rank(sb_0*num_sb_1*num_sb_2 +
					 fix_sb_1*num_sb_2      + sb_2);
    }

    return type(pvec, copy_dist<Dist1>(map, 0), copy_dist<Dist2>(map, 2));
  }
};



template <typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
struct Map_project_1<2, Map<Dist0, Dist1, Dist2> >
{
  typedef Map<Dist0, Dist1> type;

  // Return the parent subblock corresponding to a child subblock.
  static index_type parent_subblock(
    Map<Dist0, Dist1, Dist2> const& map,
    index_type                      idx,
    index_type                      sb)
  {
    length_type num_sb_0 = map.num_subblocks(0);
    length_type num_sb_1 = map.num_subblocks(1);
    length_type num_sb_2 = map.num_subblocks(2);

    index_type fix_sb_2 = map.impl_dim_subblock_from_index(2, idx);

    index_type sb_0 = sb / num_sb_1;
    index_type sb_1 = sb % num_sb_1;
    return sb_0*num_sb_1*num_sb_2 + sb_1*num_sb_2          + fix_sb_2;
  }

  static type project(Map<Dist0, Dist1, Dist2> const& map, index_type idx)
  {
    length_type num_sb_0 = map.num_subblocks(0);
    length_type num_sb_1 = map.num_subblocks(1);
    length_type num_sb_2 = map.num_subblocks(2);

    index_type fix_sb_2 = map.impl_dim_subblock_from_index(2, idx);

    Vector<processor_type> pvec(num_sb_0*num_sb_1);

    for (index_type pi=0; pi<num_sb_0*num_sb_1; ++pi)
    {
      index_type sb_0 = pi / num_sb_1;
      index_type sb_1 = pi % num_sb_1;
      pvec(pi) = map.impl_proc_from_rank(sb_0*num_sb_1*num_sb_2 +
					 sb_1*num_sb_2          +
					 fix_sb_2);
    }

    return type(pvec, copy_dist<Dist1>(map, 0), copy_dist<Dist2>(map, 1));
  }
};



/***********************************************************************
  Project a Map, removing 2 dimensions.
***********************************************************************/

template <typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
struct Map_project_2<0, 1, Map<Dist0, Dist1, Dist2> >
{
  typedef Map<Dist2> type;

  static type project(
    Map<Dist0, Dist1, Dist2> const& map,
    index_type                      idx0,
    index_type                      idx1)
  {
    // length_type num_sb_0 = map.num_subblocks(0);
    length_type num_sb_1 = map.num_subblocks(1);
    length_type num_sb_2 = map.num_subblocks(2);

    index_type fix_sb_0 = map.impl_dim_subblock_from_index(0, idx0);
    index_type fix_sb_1 = map.impl_dim_subblock_from_index(1, idx1);

    Vector<processor_type> pvec(num_sb_2);

    for (index_type pi=0; pi<num_sb_2; ++pi)
      pvec(pi) = map.impl_proc_from_rank(fix_sb_0*num_sb_1*num_sb_2 +
					 fix_sb_1*num_sb_2 + pi);

    return type(pvec, copy_dist<Dist2>(map, 2));
  }

  // Return the parent subblock corresponding to a child subblock.
  static index_type parent_subblock(
    Map<Dist0, Dist1, Dist2> const& map,
    index_type                      idx0,
    index_type                      idx1,
    index_type                      sb)
  {
    // length_type num_sb_0 = map.num_subblocks(0);
    length_type num_sb_1 = map.num_subblocks(1);
    length_type num_sb_2 = map.num_subblocks(2);

    index_type fix_sb_0 = map.impl_dim_subblock_from_index(0, idx0);
    index_type fix_sb_1 = map.impl_dim_subblock_from_index(1, idx1);

    return fix_sb_0*num_sb_1*num_sb_2 + fix_sb_1*num_sb_2 + sb;
  }
};

template <typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
struct Map_project_2<0, 2, Map<Dist0, Dist1, Dist2> >
{
  typedef Map<Dist2> type;

  static type project(
    Map<Dist0, Dist1, Dist2> const& map,
    index_type                      idx0,
    index_type                      idx2)
  {
    // length_type num_sb_0 = map.num_subblocks(0);
    length_type num_sb_1 = map.num_subblocks(1);
    length_type num_sb_2 = map.num_subblocks(2);

    index_type fix_sb_0 = map.impl_dim_subblock_from_index(0, idx0);
    index_type fix_sb_2 = map.impl_dim_subblock_from_index(2, idx2);

    Vector<processor_type> pvec(num_sb_1);

    for (index_type pi=0; pi<num_sb_1; ++pi)
      pvec(pi) = map.impl_proc_from_rank(fix_sb_0*num_sb_1*num_sb_2 +
					 pi*num_sb_2 + fix_sb_2);

    return type(pvec, copy_dist<Dist1>(map, 1));
  }

  // Return the parent subblock corresponding to a child subblock.
  static index_type parent_subblock(
    Map<Dist0, Dist1, Dist2> const& map,
    index_type                      idx0,
    index_type                      idx2,
    index_type                      sb)
  {
    // length_type num_sb_0 = map.num_subblocks(0);
    length_type num_sb_1 = map.num_subblocks(1);
    length_type num_sb_2 = map.num_subblocks(2);

    index_type fix_sb_0 = map.impl_dim_subblock_from_index(0, idx0);
    index_type fix_sb_2 = map.impl_dim_subblock_from_index(2, idx2);

    return fix_sb_0*num_sb_1*num_sb_2 + sb*num_sb_2 + fix_sb_2;
  }
};

template <typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
struct Map_project_2<1, 2, Map<Dist0, Dist1, Dist2> >
{
  typedef Map<Dist2> type;

  static type project(
    Map<Dist0, Dist1, Dist2> const& map,
    index_type                      idx1,
    index_type                      idx2)
  {
    length_type num_sb_0 = map.num_subblocks(0);
    length_type num_sb_1 = map.num_subblocks(1);
    length_type num_sb_2 = map.num_subblocks(2);

    index_type fix_sb_1 = map.impl_dim_subblock_from_index(1, idx1);
    index_type fix_sb_2 = map.impl_dim_subblock_from_index(2, idx2);

    Vector<processor_type> pvec(num_sb_0);

    for (index_type pi=0; pi<num_sb_0; ++pi)
      pvec(pi) = map.impl_proc_from_rank(pi*num_sb_1*num_sb_2 +
					 fix_sb_1*num_sb_2 + fix_sb_2);

    return type(pvec, copy_dist<Dist0>(map, 0));
  }

  // Return the parent subblock corresponding to a child subblock.
  static index_type parent_subblock(
    Map<Dist0, Dist1, Dist2> const& map,
    index_type                      idx1,
    index_type                      idx2,
    index_type                      sb)
  {
    length_type num_sb_0 = map.num_subblocks(0);
    length_type num_sb_1 = map.num_subblocks(1);
    length_type num_sb_2 = map.num_subblocks(2);

    index_type fix_sb_1 = map.impl_dim_subblock_from_index(1, idx1);
    index_type fix_sb_2 = map.impl_dim_subblock_from_index(2, idx2);

    return sb*num_sb_1*num_sb_2 + fix_sb_1*num_sb_2 + fix_sb_2;
  }
};



/***********************************************************************
  Map subdomain
***********************************************************************/

#if 0
// This functionality is now provided by Subset_map.  Remove after
// performance is characterized.
template <dimension_type Dim,
	  typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
struct Map_subdomain<Dim, Map<Dist0, Dist1, Dist2> >
{
  typedef Map<Dist0, Dist1, Dist2> type;

  static type project(
    Map<Dist0, Dist1, Dist2> const& map,
    Domain<Dim> const&              dom)
  {
    // Check each dimension
    for (dimension_type d=0; d<Dim; ++d)
    {
      if (map.num_subblocks(d) == 1)
	; /* OK */
      // TODO: Handle single index
      else
      {
	// If this dimension is distributed, then subdomain must be full
	if (dom[d].first() != 0 || dom[d].stride() != 1 ||
	    dom[d].size() != map.template applied_domain<Dim>()[d].size())
	{
	  VSIP_IMPL_THROW(
	    impl::unimplemented(
	      "Map_subdomain: Subviews must not break up distributed dimensions"));
	}
      }
    }

    Vector<processor_type> pvec(map.num_subblocks());

    for (index_type pi=0; pi<map.num_subblocks(); ++pi)
      pvec(pi) = map.impl_proc_from_rank(pi);
	
    return type(pvec,
		copy_dist<Dist0>(map, 0),
		copy_dist<Dist1>(map, 1),
		copy_dist<Dist2>(map, 2));
  }

  // Return the parent subblock corresponding to a child subblock.
  static index_type parent_subblock(
    Map<Dist0, Dist1, Dist2> const& map,
    Domain<Dim> const&              /*dom*/,
    index_type                      sb)
  {
    return sb;
  }
};
#endif


} // namespace vsip::impl

} // namespace vsip

#endif
