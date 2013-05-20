//
// Copyright (c) 2005, 2006, 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef vsip_map_hpp_
#define vsip_map_hpp_

#include <vsip/support.hpp>
#include <vsip/impl/vector.hpp>
#include <vsip/impl/map_fwd.hpp>
#include <vsip/impl/replicated_map.hpp>
#include <vsip/impl/dist.hpp>
#include <ovxx/value_iterator.hpp>
#include <ovxx/parallel/service.hpp>
#include <ovxx/domain_utils.hpp>
#include <ovxx/block_traits.hpp>
#include <ovxx/length.hpp>
#include <vector>
#include <algorithm>

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

  OVXX_PRECONDITION(
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

  OVXX_PRECONDITION(
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
template <dimension_type D,
	  typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
bool
map_equiv(Map<Dist0, Dist1, Dist2> const& map1,
	  Map<Dist0, Dist1, Dist2> const& map2) VSIP_NOTHROW;

template <typename Dist0,
	  typename Dist1,
	  typename Dist2>
class Map
{
  struct Data : ovxx::detail::noncopyable
  {
    typedef std::vector<processor_type> pvec_type;

    Data(Dist0 const &d0, Dist1 const &d1, Dist2 const &d2)
    VSIP_NOTHROW
    : dist0(d0), dist1(d1), dist2(d2),
      comm(ovxx::parallel::default_communicator()),
      pvec(),
      num_subblocks(d0.num_subblocks() *
		    d1.num_subblocks() *
		    d2.num_subblocks()),
      num_procs(num_subblocks)
    {
      OVXX_PRECONDITION(num_subblocks <= comm.pvec().size());
      for (index_type i=0; i<num_subblocks; ++i)
	pvec.push_back(comm.pvec()[i]);
      subblocks[0] = dist0.num_subblocks();
      subblocks[1] = dist1.num_subblocks();
      subblocks[2] = dist2.num_subblocks();
    }

    template <typename B>
    Data(const_Vector<processor_type, B> p,
	 Dist0 const &d0, Dist1 const &d1, Dist2 const &d2)
    VSIP_NOTHROW
    : dist0(d0), dist1(d1), dist2(d2),
      comm(ovxx::parallel::default_communicator()),
      pvec(),
      num_subblocks(d0.num_subblocks() *
		    d1.num_subblocks() *
		    d2.num_subblocks()),
      num_procs(num_subblocks)
    {
      OVXX_PRECONDITION(num_subblocks <= p.size());
      for (index_type i=0; i<num_subblocks; ++i)
	pvec.push_back(p.get(i));
      subblocks[0] = dist0.num_subblocks();
      subblocks[1] = dist1.num_subblocks();
      subblocks[2] = dist2.num_subblocks();
    }

    Dist0 dist0;
    Dist1 dist1;
    Dist2 dist2;
    ovxx::parallel::Communicator &comm;
    pvec_type pvec;
    length_type num_subblocks;
    length_type num_procs;
    index_type subblocks[VSIP_MAX_DIMENSION];
  };

public:
  typedef ovxx::value_iterator<processor_type, unsigned> processor_iterator;
  typedef typename Data::pvec_type impl_pvec_type;

  static bool const impl_local_only  = false;
  static bool const impl_global_only = true;

  Map(Dist0 const &d0 = Dist0(), Dist1 const &d1 = Dist1(), Dist2 const &d2 = Dist2())
    VSIP_NOTHROW
    : data_(new Data(d0, d1, d2)), dim_(0) {}

  template <typename B>
  Map(const_Vector<processor_type, B> pvec,
      Dist0 const &d0 = Dist0(), Dist1 const &d1 = Dist1(), Dist2 const &d2 = Dist2())
    VSIP_NOTHROW
    : data_(new Data(pvec, d0, d1, d2)), dim_(0) {}

  Map(Map const &other) VSIP_NOTHROW
    : data_(other.data_), dom_(other.dom_), dim_(0) {}

  Map& operator=(Map const &other) VSIP_NOTHROW
  {
    data_ = other.data_;
    dom_  = other.dom_;
    dim_  = 0;
    return *this;
  }

  ~Map() VSIP_NOTHROW
  {
    if (this->impl_is_applied())
      ovxx::parallel::destroy_ll_pset(applied_pset_);
  }

  // Information on individual distributions.
  distribution_type distribution(dimension_type d) const VSIP_NOTHROW;
  length_type num_subblocks(dimension_type d) const VSIP_NOTHROW;
  length_type cyclic_contiguity(dimension_type d) const VSIP_NOTHROW;

  length_type num_subblocks()  const VSIP_NOTHROW
  { return data_->num_subblocks;}

  length_type num_processors() const VSIP_NOTHROW
  { return data_->num_procs;}

  index_type subblock(processor_type pr) const VSIP_NOTHROW;
  index_type subblock() const VSIP_NOTHROW;

  processor_iterator processor_begin(index_type sb) const VSIP_NOTHROW;
  processor_iterator processor_end(index_type sb) const VSIP_NOTHROW;

  const_Vector<processor_type> processor_set() const;


  // Applied map functions.
  length_type impl_num_patches(index_type sb) const VSIP_NOTHROW;

  template <dimension_type D>
  void impl_apply(Domain<D> const& dom) VSIP_NOTHROW;

  template <dimension_type D>
  Domain<D> impl_subblock_domain(index_type sb) const VSIP_NOTHROW;

  template <dimension_type D>
  ovxx::Length<D> impl_subblock_extent(index_type sb) const VSIP_NOTHROW;

  template <dimension_type D>
  Domain<D> impl_global_domain(index_type sb, index_type patch)
    const VSIP_NOTHROW;

  template <dimension_type D>
  Domain<D> impl_local_domain(index_type sb, index_type patch)
    const VSIP_NOTHROW;

  template <dimension_type D>
  Domain<D> applied_domain() const VSIP_NOTHROW;

  // Implementation functions.
  ovxx::parallel::par_ll_pset_type impl_ll_pset() const VSIP_NOTHROW
  { OVXX_PRECONDITION(this->impl_is_applied()); return applied_pset_;}
  impl_pvec_type const& impl_pvec() const { return data_->pvec;}
  ovxx::parallel::Communicator &impl_comm() const { return data_->comm;}
  bool impl_is_applied() const { return dim_ != 0;}

  length_type impl_working_size() const
  { return std::min(this->num_subblocks(), this->data_->pvec.size());}

  length_type impl_subblock_patches(dimension_type d, index_type sb)
    const VSIP_NOTHROW;
  length_type impl_subblock_size(dimension_type d, index_type sb)
    const VSIP_NOTHROW;
  Domain<1> impl_patch_global_dom(dimension_type d, index_type sb, index_type p)
    const VSIP_NOTHROW;
  Domain<1> impl_patch_local_dom(dimension_type d, index_type sb, index_type p)
    const VSIP_NOTHROW;

  index_type impl_dim_subblock_from_index(dimension_type d, index_type idx)
    const VSIP_NOTHROW;
  index_type impl_dim_patch_from_index(dimension_type d, index_type idx)
    const VSIP_NOTHROW;
  index_type impl_local_from_global_index(dimension_type d, index_type idx)
    const VSIP_NOTHROW;

  template <dimension_type D>
  index_type impl_subblock_from_global_index(Index<D> const &idx)
    const VSIP_NOTHROW;

  template <dimension_type D>
  index_type impl_patch_from_global_index(Index<D> const &idx)
    const VSIP_NOTHROW;

  index_type impl_global_from_local_index(dimension_type d, index_type sb, index_type idx)
    const VSIP_NOTHROW;

  template <dimension_type D>
  Domain<D> impl_local_from_global_domain(index_type sb, Domain<D> const &dom)
    const VSIP_NOTHROW;

  typedef Dist0 impl_dim0_type;
  typedef Dist1 impl_dim1_type;
  typedef Dist2 impl_dim2_type;

  friend bool operator==<>(Map const&, Map const&) VSIP_NOTHROW;
  friend bool map_equiv<1>(Map const&, Map const&) VSIP_NOTHROW;
  friend bool map_equiv<2>(Map const&, Map const&) VSIP_NOTHROW;
  friend bool map_equiv<3>(Map const&, Map const&) VSIP_NOTHROW;
  friend struct ovxx::parallel::map_equal<1, Map, Map>;
  friend struct ovxx::parallel::map_equal<2, Map, Map>;
  friend struct ovxx::parallel::map_equal<3, Map, Map>;

  index_type     impl_rank_from_proc(processor_type pr) const;
  processor_type impl_proc_from_rank(index_type idx) const
  { return data_->pvec[idx];}

private:
  ovxx::shared_ptr<Data> data_;
  Domain<3>	         dom_;		  // Applied domain.
  dimension_type         dim_;		  // Dimension of applied domain.
  ovxx::parallel::par_ll_pset_type applied_pset_;
};

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
    OVXX_PRECONDITION(this->num_subblocks(d) == 1); // note [1]
  }

  dim_ = Dim;
  dom_ = ovxx::construct_domain<VSIP_MAX_DIMENSION>(arr);

  impl_pvec_type const& pvec = this->impl_pvec();
  ovxx::parallel::create_ll_pset(pvec, applied_pset_);
}



template <typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
inline distribution_type
Map<Dist0, Dist1, Dist2>::distribution(dimension_type d)
  const VSIP_NOTHROW
{
  OVXX_PRECONDITION(d < VSIP_MAX_DIMENSION);

  switch (d)
  {
  default: assert(false);
  case 0: return data_->dist0.distribution();
  case 1: return data_->dist1.distribution();
  case 2: return data_->dist2.distribution();
  }
}



template <typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
inline length_type
Map<Dist0, Dist1, Dist2>::num_subblocks(dimension_type d)
  const VSIP_NOTHROW
{
  OVXX_PRECONDITION(d < VSIP_MAX_DIMENSION);

  switch (d)
  {
  default: assert(false);
  case 0: return data_->dist0.num_subblocks();
  case 1: return data_->dist1.num_subblocks();
  case 2: return data_->dist2.num_subblocks();
  }
}



template <typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
inline length_type
Map<Dist0, Dist1, Dist2>::cyclic_contiguity(dimension_type d)
  const VSIP_NOTHROW
{
  OVXX_PRECONDITION(d < VSIP_MAX_DIMENSION);

  switch (d)
  {
  default: assert(false);
  case 0: return data_->dist0.cyclic_contiguity();
  case 1: return data_->dist1.cyclic_contiguity();
  case 2: return data_->dist2.cyclic_contiguity();
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
  OVXX_PRECONDITION(d < VSIP_MAX_DIMENSION);

  switch (d)
  {
  default: assert(false);
  case 0: return data_->dist0.impl_subblock_patches(dom_[0], sb);
  case 1: return data_->dist1.impl_subblock_patches(dom_[1], sb);
  case 2: return data_->dist2.impl_subblock_patches(dom_[2], sb);
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
  OVXX_PRECONDITION(d < VSIP_MAX_DIMENSION);

  switch (d)
  {
  default: assert(false);
  case 0: return data_->dist0.impl_subblock_size(dom_[0], sb);
  case 1: return data_->dist1.impl_subblock_size(dom_[1], sb);
  case 2: return data_->dist2.impl_subblock_size(dom_[2], sb);
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
  OVXX_PRECONDITION(d < VSIP_MAX_DIMENSION);

  switch (d)
  {
  default: assert(false);
  case 0: return data_->dist0.impl_patch_global_dom(dom_[0], sb, p);
  case 1: return data_->dist1.impl_patch_global_dom(dom_[1], sb, p);
  case 2: return data_->dist2.impl_patch_global_dom(dom_[2], sb, p);
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
  OVXX_PRECONDITION(d < VSIP_MAX_DIMENSION);

  switch (d)
  {
  default: assert(false);
  case 0: return data_->dist0.impl_patch_local_dom(dom_[0], sb, p);
  case 1: return data_->dist1.impl_patch_local_dom(dom_[1], sb, p);
  case 2: return data_->dist2.impl_patch_local_dom(dom_[2], sb, p);
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
  OVXX_PRECONDITION(d < VSIP_MAX_DIMENSION);
  OVXX_PRECONDITION(d < dim_);

  switch (d)
  {
  default: assert(false);
  case 0: return data_->dist0.impl_subblock_from_index(dom_[0], idx);
  case 1: return data_->dist1.impl_subblock_from_index(dom_[1], idx);
  case 2: return data_->dist2.impl_subblock_from_index(dom_[2], idx);
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
  OVXX_PRECONDITION(d < VSIP_MAX_DIMENSION);
  OVXX_PRECONDITION(d < dim_);

  switch (d)
  {
  default: assert(false);
  case 0: return data_->dist0.impl_patch_from_index(dom_[0], idx);
  case 1: return data_->dist1.impl_patch_from_index(dom_[1], idx);
  case 2: return data_->dist2.impl_patch_from_index(dom_[2], idx);
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
  OVXX_PRECONDITION(d < VSIP_MAX_DIMENSION);
  OVXX_PRECONDITION(d < dim_);

  switch (d)
  {
  default: assert(false);
  case 0: return data_->dist0.impl_local_from_global_index(dom_[0], idx);
  case 1: return data_->dist1.impl_local_from_global_index(dom_[1], idx);
  case 2: return data_->dist2.impl_local_from_global_index(dom_[2], idx);
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
  OVXX_PRECONDITION(dim_ != 0 && dim_ == Dim);

  for (dimension_type d=0; d<Dim; ++d)
  {
    OVXX_PRECONDITION(idx[d] < dom_[d].size());
    if (d != 0)
      sb *= data_->subblocks[d];
    sb += impl_dim_subblock_from_index(d, idx[d]);
  }

  OVXX_PRECONDITION(sb < data_->num_subblocks);
  index_type dim_sb[VSIP_MAX_DIMENSION];
  impl::split_tuple(sb, dim_, data_->subblocks, dim_sb);
  for (dimension_type d=0; d<Dim; ++d)
  {
    OVXX_PRECONDITION(dim_sb[d] == impl_dim_subblock_from_index(d, idx[d]));
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

  OVXX_PRECONDITION(dim_ != 0 && dim_ == Dim);

  index_type sb = this->impl_subblock_from_global_index(idx);
  impl::split_tuple(sb, dim_, data_->subblocks, dim_sb);

  for (dimension_type d=0; d<Dim; ++d)
  {
    OVXX_PRECONDITION(idx[d] < dom_[d].size());
    if (d != 0)
      p *= impl_subblock_patches(d, dim_sb[d]);
    p += impl_dim_patch_from_index(d, idx[d]);
  }

  OVXX_PRECONDITION(p < this->impl_num_patches(sb));
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
  OVXX_PRECONDITION(d < VSIP_MAX_DIMENSION);
  OVXX_PRECONDITION(dim_ != 0 && d < dim_);

  index_type dim_sb[VSIP_MAX_DIMENSION];
  impl::split_tuple(sb, dim_, data_->subblocks, dim_sb);

  switch (d)
  {
  default: assert(false);
  case 0: return data_->dist0.impl_global_from_local_index(dom_[0], dim_sb[0], idx);
  case 1: return data_->dist1.impl_global_from_local_index(dom_[1], dim_sb[1], idx);
  case 2: return data_->dist2.impl_global_from_local_index(dom_[2], dim_sb[2], idx);
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
  OVXX_PRECONDITION(sb < data_->num_subblocks);
  OVXX_PRECONDITION(dim_ == Dim);

  if (this->impl_num_patches(sb) != 1)
      OVXX_DO_THROW(
	ovxx::unimplemented(
	  "Map<>::impl_local_from_global_domain: Subviews have a single patch"));

  Domain<Dim> sb_g_dom = this->template impl_global_domain<Dim>(sb, 0);
  Domain<Dim> sb_l_dom = this->template impl_local_domain<Dim>(sb, 0);
  Domain<Dim> intr;

  if (ovxx::intersect(sb_g_dom, g_dom, intr))
  {
    Domain<Dim> l_dom = ovxx::apply_intr(sb_l_dom, sb_g_dom, intr);
    return l_dom;
  }
  else
    return ovxx::empty_domain<Dim>();
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
  for (index_type i=0; i<data_->pvec.size(); ++i)
    if (data_->pvec[i] == pr)
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

  if (pi != no_rank && pi < data_->num_subblocks)
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

  if (pi != no_rank && pi < data_->num_subblocks)
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
  OVXX_PRECONDITION(sb < data_->num_subblocks);

  return processor_iterator(data_->pvec[sb % data_->num_procs], 1);
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
  OVXX_PRECONDITION(sb < data_->num_subblocks);

  return processor_iterator(data_->pvec[sb % data_->num_procs]+1, 1);
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
  Vector<processor_type> pset(this->data_->num_procs);

  for (index_type i=0; i<this->data_->num_procs; ++i)
    pset.put(i, this->data_->pvec[i]);

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
  OVXX_PRECONDITION(sb < data_->num_subblocks || sb == no_subblock);
  OVXX_PRECONDITION(dim_ != 0);

  if (sb == no_subblock)
    return 0;
  else
  {
    index_type dim_sb[VSIP_MAX_DIMENSION];

    impl::split_tuple(sb, dim_, data_->subblocks, dim_sb);

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
  OVXX_PRECONDITION(sb < data_->num_subblocks || sb == no_subblock);
  OVXX_PRECONDITION(dim_ == Dim);

  if (sb == no_subblock)
    return ovxx::empty_domain<Dim>();
  else
  {
    index_type dim_sb[VSIP_MAX_DIMENSION];
    Domain<1>  dom[VSIP_MAX_DIMENSION];

    impl::split_tuple(sb, dim_, data_->subblocks, dim_sb);

    for (dimension_type d=0; d<dim_; ++d)
      dom[d] = Domain<1>(impl_subblock_size(d, dim_sb[d]));

    return ovxx::construct_domain<Dim>(dom);
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
ovxx::Length<Dim>
Map<Dist0, Dist1, Dist2>::impl_subblock_extent(index_type sb)
  const VSIP_NOTHROW
{
  OVXX_PRECONDITION(sb < data_->num_subblocks || sb == no_subblock);
  OVXX_PRECONDITION(dim_ == Dim);

  ovxx::Length<Dim> size;

  if (sb == no_subblock)
  {
    for (dimension_type d=0; d<dim_; ++d)
      size[d] = 0;
  }
  else
  {
    index_type dim_sb[VSIP_MAX_DIMENSION];

    impl::split_tuple(sb, dim_, data_->subblocks, dim_sb);

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
    return ovxx::construct_domain<Dim>(dom);
  }

  OVXX_PRECONDITION(sb < data_->num_subblocks);
  OVXX_PRECONDITION(p  < this->impl_num_patches(sb));
  OVXX_PRECONDITION(dim_ == Dim);

  impl::split_tuple(sb, Dim, data_->subblocks, dim_sb);

  for (dimension_type d=0; d<Dim; ++d)
    p_size[d] = impl_subblock_patches(d, dim_sb[d]);

  impl::split_tuple(p, Dim, p_size, dim_p);

  for (dimension_type d=0; d<Dim; ++d)
    dom[d] = impl_patch_global_dom(d, dim_sb[d], dim_p[d]);

  return ovxx::construct_domain<Dim>(dom);
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
    return ovxx::construct_domain<Dim>(dom);
  }

  OVXX_PRECONDITION(sb < data_->num_subblocks);
  OVXX_PRECONDITION(p  < this->impl_num_patches(sb));
  OVXX_PRECONDITION(dim_ == Dim);


  impl::split_tuple(sb, Dim, data_->subblocks, dim_sb);

  for (dimension_type d=0; d<Dim; ++d)
    p_size[d] = impl_subblock_patches(d, dim_sb[d]);

  impl::split_tuple(p, Dim, p_size, dim_p);

  for (dimension_type d=0; d<Dim; ++d)
    dom[d] = impl_patch_local_dom(d, dim_sb[d], dim_p[d]);

  return ovxx::construct_domain<Dim>(dom);
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
  OVXX_PRECONDITION(dim_ == Dim);

  Domain<1>  dom[VSIP_MAX_DIMENSION];

  for (dimension_type d=0; d<Dim; ++d)
    dom[d] = this->dom_[d];

  return ovxx::construct_domain<Dim>(dom);
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

  if (map1.data_->comm != map2.data_->comm)
    return false;

  if (map1.data_->pvec.size() != map2.data_->pvec.size())
    return false;

  for (index_type i=0; i<map1.data_->pvec.size(); ++i)
    if (map1.data_->pvec[i] != map2.data_->pvec[i])
      return false;

  return true;
}

template <dimension_type D,
	  typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
bool
map_equiv(Map<Dist0, Dist1, Dist2> const& map1,
	  Map<Dist0, Dist1, Dist2> const& map2) VSIP_NOTHROW
{
  if (D == 1 &&
      (map1.data_->dist0.num_subblocks() != map2.data_->dist0.num_subblocks() ||
       map1.data_->dist0.cyclic_contiguity() != map2.data_->dist0.cyclic_contiguity()))
    return false;
  else if (D == 2 &&
	   (map1.data_->dist0.num_subblocks() != map2.data_->dist0.num_subblocks() ||
	    map1.data_->dist0.cyclic_contiguity() != map2.data_->dist0.cyclic_contiguity() ||
	    map1.data_->dist1.num_subblocks() != map2.data_->dist1.num_subblocks() ||
	    map1.data_->dist1.cyclic_contiguity() != map2.data_->dist1.cyclic_contiguity()))
    return false;
  else if (D == 3 &&
	   (map1.data_->dist0.num_subblocks() != map2.data_->dist0.num_subblocks() ||
	    map1.data_->dist0.cyclic_contiguity() != map2.data_->dist0.cyclic_contiguity() ||
	    map1.data_->dist1.num_subblocks() != map2.data_->dist1.num_subblocks() ||
	    map1.data_->dist1.cyclic_contiguity() != map2.data_->dist1.cyclic_contiguity() ||
	    map1.data_->dist2.num_subblocks() != map2.data_->dist2.num_subblocks() ||
	    map1.data_->dist2.cyclic_contiguity() != map2.data_->dist2.cyclic_contiguity()))
    return false;

  // implied by checks on distX_.num_subblocks()
  OVXX_PRECONDITION(map1.num_subblocks() == map1.num_subblocks());

  if (map1.data_->comm != map2.data_->comm)
    return false;

  OVXX_PRECONDITION(map1.data_->pvec.size() >= map1.num_subblocks());
  OVXX_PRECONDITION(map2.data_->pvec.size() >= map2.num_subblocks());

  for (index_type i=0; i<map1.num_subblocks(); ++i)
    if (map1.data_->pvec[i] != map2.data_->pvec[i])
      return false;

  return true;
}

template <typename D0A, typename D1A, typename D2A,
	  typename D0B, typename D1B, typename D2B>
bool
operator==(Map<D0A, D1A, D2A> const&, Map<D0B, D1B, D2B> const&) VSIP_NOTHROW
{
  return false;
}

} // namespace vsip

namespace ovxx
{
namespace parallel
{
template <dimension_type D, typename D0, typename D1, typename D2>
struct map_equal<D, Map<D0, D1, D2>, Map<D0, D1, D2> >
{
  static bool value(Map<D0, D1, D2> const &map1, Map<D0, D1, D2> const &map2)
  { return (map1.data_.get() == map2.data_.get()) || map_equiv<D>(map1, map2);}
};

template <dimension_type D, typename M>
struct select_dist;

template <typename D0, typename D1, typename D2>
struct select_dist<0, vsip::Map<D0, D1, D2> >
{ typedef D0 type;};

template <typename D0, typename D1, typename D2>
struct select_dist<1, vsip::Map<D0, D1, D2> >
{ typedef D1 type;};

template <typename D0, typename D1, typename D2>
struct select_dist<2, vsip::Map<D0, D1, D2> >
{ typedef D2 type;};

template <dimension_type D, typename D0, typename D1, typename D2>
struct is_block_dist<D, vsip::Map<D0, D1, D2> >
{
private:
  typedef typename parallel::select_dist<D, vsip::Map<D0, D1, D2> >::type dist_type;
public:
  static bool const value = 
    is_same<dist_type, vsip::Block_dist>::value ||
    is_same<dist_type, vsip::Whole_dist>::value;
};

template <typename D0, typename D1, typename D2>
struct is_global_map<Map<D0, D1, D2> >
{ static bool const value = true;};

} // namespace ovxx::parallel
} // namespace ovxx

#endif
