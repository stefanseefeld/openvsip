/* Copyright (c) 2008 by CodeSourcery.  All rights reserved. */

/// Description
///   Map class for transposes.

#ifndef VSIP_CORE_PARALLEL_TRANSPOSE_MAP_DECL_HPP
#define VSIP_CORE_PARALLEL_TRANSPOSE_MAP_DECL_HPP

#include <vsip/core/view_fwd.hpp>

namespace vsip
{
namespace impl
{

template <typename MapT>
class Transpose_map
{
  // Translate tranpose dimension to parent map dimension.
  dimension_type parent_dim(dimension_type dim)
  {
    return (dim == 0) ? 1 : (dim == 1) ? 0 : dim;
  }
  // Compile-time typedefs.
public:
  static dimension_type const Dim = 2;
  typedef typename MapT::processor_iterator processor_iterator;
  typedef typename MapT::impl_pvec_type     impl_pvec_type;

  // Constructor.
public:
  Transpose_map(MapT const& map)
    : map_(map)
  {}

  ~Transpose_map()
    {}


  // Accessors.
public:
  // Information on individual distributions.
  distribution_type distribution     (dimension_type dim) const VSIP_NOTHROW
    { return map_.distribution(parent_dim(dim)); }

  length_type       num_subblocks    (dimension_type dim) const VSIP_NOTHROW
    { return map_.num_subblock(parent_dim(dim)); }

  length_type       cyclic_contiguity(dimension_type dim) const VSIP_NOTHROW
    { return map_.cyclic_contiguity(parent_dim(dim)); }

  length_type num_subblocks()  const VSIP_NOTHROW
    { return map_.num_subblocks(); }

  length_type num_processors() const VSIP_NOTHROW
    { return map_.num_processors(); }

  index_type subblock(processor_type pr) const VSIP_NOTHROW
    { return map_.subblock(pr); }

  index_type subblock() const VSIP_NOTHROW
    { return map_.subblock(); }

  processor_iterator processor_begin(index_type sb) const VSIP_NOTHROW
    { return map_.processor_begin(sb); }

  processor_iterator processor_end  (index_type sb) const VSIP_NOTHROW
    { return map_.processor_end(sb); }

  const_Vector<processor_type> processor_set() const;

  // Applied map functions.
public:
  length_type impl_num_patches(index_type sb) const VSIP_NOTHROW
    { return map_.impl_num_patches(sb); }

  void impl_apply(Domain<Dim> const& /*dom*/) VSIP_NOTHROW
  {
    // TODO assert(extent(dom_) == transpose(extent(dom)));
  }

  template <dimension_type Dim2>
  Domain<Dim2> impl_subblock_domain(index_type sb) const VSIP_NOTHROW
  {
    assert(sb < this->num_subblocks());
    Domain<Dim2> p_dom = map_.template impl_subblock_domain<Dim2>(sb);
    return Domain<Dim2>(p_dom[1], p_dom[0]);
  }

  template <dimension_type Dim2>
  impl::Length<Dim2> impl_subblock_extent(index_type sb) const VSIP_NOTHROW
  {
    assert(sb < this->num_subblocks());
    impl::Length<Dim2> p_ext = map_.template impl_subblock_extent<Dim2>(sb);
    return impl::Length<Dim2>(p_ext[1], p_ext[0]);
  }

  template <dimension_type Dim2>
  Domain<Dim2> impl_global_domain(index_type sb, index_type p)
    const VSIP_NOTHROW
  {
    assert(sb < this->num_subblocks());
    assert(p  < this->impl_num_patches(sb));
    Domain<Dim2> p_dom = map_.template impl_global_domain<Dim2>(sb, p);
    return Domain<Dim2>(p_dom[1], p_dom[0]);
  }

  template <dimension_type Dim2>
  Domain<Dim2> impl_local_domain (index_type sb, index_type p)
    const VSIP_NOTHROW
  {
    assert(sb < this->num_subblocks());
    assert(p  < this->impl_num_patches(sb));
    Domain<Dim2> p_dom = map_.template impl_local_domain<Dim2>(sb, p);
    return Domain<Dim2>(p_dom[1], p_dom[0]);
  }

  template <dimension_type Dim2>
  Domain<Dim2> impl_parent_local_domain(index_type sb) const VSIP_NOTHROW
  {
    assert(sb < this->num_subblocks());
    Domain<Dim2> p_dom = map_.template impl_parent_local_domain<Dim2>(sb);
    return Domain<Dim2>(p_dom[1], p_dom[0]);
  }

  index_type impl_global_from_local_index(dimension_type /*d*/, index_type sb,
				     index_type idx)
    const VSIP_NOTHROW
  {
    VSIP_IMPL_THROW(impl::unimplemented(
	      "Transpose_map::impl_global_from_local_index not implemented."));
    return 0;
  }

  index_type impl_local_from_global_index(dimension_type dim, index_type idx)
    const VSIP_NOTHROW
  { return map_.impl_local_from_global_index(parent_dim(dim), idx); }

  template <dimension_type Dim2>
  index_type impl_subblock_from_global_index(Index<Dim2> const& /*idx*/)
    const VSIP_NOTHROW
  {
    VSIP_IMPL_THROW(impl::unimplemented(
	      "Transpose_map::impl_subblock_from_global_index not implemented."));
    return 0;
  }

  template <dimension_type Dim2>
  Domain<Dim> impl_local_from_global_domain(index_type sb,
					    Domain<Dim2> const& dom)
    const VSIP_NOTHROW
  {
    VSIP_IMPL_STATIC_ASSERT(Dim == Dim2);
    Domain<Dim> p_gdom(dom[1], dom[0]);
    Domain<Dim> p_ldom = map_.template impl_local_from_global_domain<Dim2>
                                        (sb, p_gdom);
    return Domain<Dim>(p_ldom[1], p_ldom[0]);
  }

  // Extensions.
public:
  impl::par_ll_pset_type impl_ll_pset() const VSIP_NOTHROW
    { return map_.impl_ll_pset(); }

  impl::Communicator&    impl_comm() const
    { return map_.impl_comm(); }

  impl_pvec_type const&  impl_pvec() const
    { return map_.impl_pvec(); }

  length_type            impl_working_size() const
    { return map_.impl_working_size(); }

  processor_type         impl_proc_from_rank(index_type idx) const
    { return map_.impl_proc_from_rank(idx); }

  index_type impl_rank_from_proc(processor_type pr) const
    { return map_.impl_rank_from_proc(pr); }

  // Determine parent map subblock corresponding to this map's subblock
  index_type impl_parent_subblock(index_type sb) const
    { return map_.impl_parent_subblock(sb); }

public:
  typedef std::vector<Domain<Dim> >   p_vector_type;
  typedef std::vector<p_vector_type>  sb_vector_type;


  // Member data.
private:
  MapT const& map_;
};

template <typename MapT>
struct is_global_map<Transpose_map<MapT> >
{ static bool const value = true; };

/// Functor to transpose a map.
///
/// Handles both the type conversion ('type') and the runtime
/// conversion ('project').
template <dimension_type Dim,
	  typename       MapT>
struct Transpose_map_of
{
  typedef Transpose_map<MapT> type;

  static type project(MapT const& map)
    { return type(map); }
};

template <dimension_type Dim>
struct Transpose_map_of<Dim, Local_map>
{
  typedef Local_map type;

  static type project(Local_map const& map)
    { return map; }
};



} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_CORE_PARALLEL_TRANSPOSE_MAP_DECL_HPP
