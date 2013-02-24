/* Copyright (c) 2006, 2007, 2008 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/parallel/subset_map_decl.hpp
    @author  Jules Bergmann
    @date    2006-12-10
    @brief   VSIPL++ Library: Map class for distributed subsets.

*/

#ifndef VSIP_CORE_PARALLEL_SUBSET_MAP_DECL_HPP
#define VSIP_CORE_PARALLEL_SUBSET_MAP_DECL_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/view_fwd.hpp>



/***********************************************************************
  Declarations & Class Definitions
***********************************************************************/

namespace vsip
{
namespace impl
{

template <dimension_type Dim>
class Subset_map
{
  // Compile-time typedefs.
public:
  typedef impl::Value_iterator<processor_type, unsigned> processor_iterator;
  typedef std::vector<processor_type> impl_pvec_type;

  // Constructor.
public:
  template <typename MapT>
  Subset_map(MapT const&, Domain<Dim> const&);

  ~Subset_map()
    {}


  // Accessors.
public:
  // Information on individual distributions.
  distribution_type distribution     (dimension_type) const VSIP_NOTHROW
    { return other; }
  length_type       num_subblocks    (dimension_type) const VSIP_NOTHROW
    { return 1; }
  length_type       cyclic_contiguity(dimension_type) const VSIP_NOTHROW
    { return 0; }

  length_type num_subblocks()  const VSIP_NOTHROW
    { return sb_patch_gd_.size(); }

  length_type num_processors() const VSIP_NOTHROW
    { return pvec_.size(); }

  index_type subblock(processor_type pr) const VSIP_NOTHROW
  {
    index_type pi = impl_rank_from_proc(pr);

    if (pi != no_rank && pi < this->num_subblocks())
      return pi;
    else
      return no_subblock;
  }

  index_type subblock() const VSIP_NOTHROW
  {
    processor_type pr = local_processor();
    return this->subblock(pr);
  }

  processor_iterator processor_begin(index_type sb) const VSIP_NOTHROW
  {
    assert(sb < this->num_subblocks());
    return processor_iterator(this->pvec_[sb], 1);
  }

  processor_iterator processor_end  (index_type sb) const VSIP_NOTHROW
  {
    assert(sb < this->num_subblocks());
    return processor_iterator(this->pvec_[sb]+1, 1);
  }

  const_Vector<processor_type> processor_set() const;

  // Applied map functions.
public:
  length_type impl_num_patches(index_type sb) const VSIP_NOTHROW
  {
    assert(sb < this->num_subblocks());
    return sb_patch_gd_[sb].size();
  }

  void impl_apply(Domain<Dim> const& dom) VSIP_NOTHROW
  {
    assert(extent(dom_) == extent(dom));
    impl::create_ll_pset(pvec_, ll_pset_);
  }

  template <dimension_type Dim2>
  Domain<Dim2> impl_subblock_domain(index_type sb) const VSIP_NOTHROW
    { assert(sb < this->num_subblocks()); return domain(sb_ext_[sb]); }

  template <dimension_type Dim2>
  impl::Length<Dim2> impl_subblock_extent(index_type sb) const VSIP_NOTHROW
    { assert(sb < this->num_subblocks()); return sb_ext_[sb]; }

  template <dimension_type Dim2>
  Domain<Dim2> impl_global_domain(index_type sb, index_type p)
    const VSIP_NOTHROW
  {
    assert(sb < this->num_subblocks());
    assert(p  < this->impl_num_patches(sb));
    return sb_patch_gd_[sb][p];
  }

  template <dimension_type Dim2>
  Domain<Dim2> impl_local_domain (index_type sb, index_type p)
    const VSIP_NOTHROW
  {
    assert(sb < this->num_subblocks());
    assert(p  < this->impl_num_patches(sb));
    return sb_patch_ld_[sb][p];
  }

  template <dimension_type Dim2>
  Domain<Dim2> impl_parent_local_domain(index_type sb) const VSIP_NOTHROW
  { return parent_sdom_[sb]; }

  index_type impl_global_from_local_index(dimension_type /*d*/, index_type sb,
				     index_type idx)
    const VSIP_NOTHROW
  {
    VSIP_IMPL_THROW(impl::unimplemented(
	      "Subset_map::impl_global_from_local_index not implemented."));
    return 0;
  }

  index_type impl_local_from_global_index(dimension_type /*d*/, index_type idx)
    const VSIP_NOTHROW
  {
    VSIP_IMPL_THROW(impl::unimplemented(
	      "Subset_map::impl_local_from_global_index not implemented."));
    return 0;
  }

  template <dimension_type Dim2>
  index_type impl_subblock_from_global_index(Index<Dim2> const& /*idx*/)
    const VSIP_NOTHROW
  {
    VSIP_IMPL_THROW(impl::unimplemented(
	      "Subset_map::impl_subblock_from_global_index not implemented."));
    return 0;
  }

  template <dimension_type Dim2>
  Domain<Dim> impl_local_from_global_domain(index_type /*sb*/,
					    Domain<Dim2> const& dom)
    const VSIP_NOTHROW
  {
    VSIP_IMPL_STATIC_ASSERT(Dim == Dim2);
    VSIP_IMPL_THROW(impl::unimplemented(
	      "Subset_map::impl_local_from_global_domain not implemented."));
    return Domain<Dim>();
  }

  // Extensions.
public:
  impl::par_ll_pset_type impl_ll_pset() const VSIP_NOTHROW
    { return ll_pset_; }
  impl::Communicator&    impl_comm() const
    { return impl::default_communicator(); }
  impl_pvec_type const&  impl_pvec() const
    { return pvec_; }

  length_type            impl_working_size() const
    { return this->num_processors(); }

  processor_type         impl_proc_from_rank(index_type idx) const
    { return this->impl_pvec()[idx]; }

  index_type impl_rank_from_proc(processor_type pr) const
  {
    for (index_type i=0; i<this->num_processors(); ++i)
      if (this->impl_pvec()[i] == pr) return i;
    return no_rank;
  }

  // Determine parent map subblock corresponding to this map's subblock
  index_type impl_parent_subblock(index_type sb) const
    { return parent_sb_[sb]; }

public:
  typedef std::vector<Domain<Dim> >   p_vector_type;
  typedef std::vector<p_vector_type>  sb_vector_type;


  // Member data.
private:
  std::vector<index_type>   parent_sb_; // map: local sb -> parent sb
  std::vector<Length<Dim> > sb_ext_;
  std::vector<Domain<Dim> > parent_sdom_;	// parent subblock dom.

  sb_vector_type            sb_patch_gd_;	// sb-patch global dom
  sb_vector_type            sb_patch_ld_;	// sb-patch local dom
  sb_vector_type            sb_patch_pld_;	// sb-patch parent local dom

  impl_pvec_type            pvec_;		// Grid function.
  Domain<Dim>               dom_;		// Applied domain.
  impl::par_ll_pset_type    ll_pset_;
};



template <dimension_type Dim>
template <typename MapT>
Subset_map<Dim>::Subset_map(
  MapT const&        map,
  Domain<Dim> const& dom)
  : dom_(dom)
{
  // Check that map is only block distributed
  for (dimension_type d = 0; d<Dim; ++d)
  {
    if (map.distribution(d) == cyclic)
      VSIP_IMPL_THROW(impl::unimplemented(
	      "Subset_map: Subviews of cyclic maps not supported."));
  }

  for (index_type sb=0; sb<map.num_subblocks(); ++sb)
  {
    p_vector_type g_vec;
    p_vector_type l_vec;
    p_vector_type pl_vec;
    
    for (index_type p=0; p<map.impl_num_patches(sb); ++p)
    {
      // parent global/local subdomains for sb-p.
      Domain<Dim> pg_dom = map.template impl_global_domain<Dim>(sb, p);
      Domain<Dim> pl_dom = map.template impl_local_domain<Dim>(sb, p);

      Domain<Dim> intr;
      if (intersect(pg_dom, dom, intr))
      {
	// Global domain represented by intersection.
	Domain<Dim> mg_dom = subset_from_intr(dom, intr);

	// Local domain.
	Domain<Dim> ml_dom = normalize(intr);

	// Subset of parent local domain represented by intersection
	Domain<Dim> pl_dom_intr = apply_intr(pl_dom, pg_dom, intr);

	g_vec.push_back (mg_dom);
	l_vec.push_back (ml_dom);
	pl_vec.push_back(pl_dom_intr);
      }
    }

    if (g_vec.size() > 0)
    {
      if (g_vec.size() > 1)
      {
	VSIP_IMPL_THROW(impl::unimplemented(
	    "Subset_map: Subviews creating multiple patches not supported."));
      }
      sb_patch_gd_.push_back(g_vec);
      sb_patch_ld_.push_back(l_vec);
      sb_patch_pld_.push_back(pl_vec);
      pvec_.push_back(map.impl_proc_from_rank(sb));

      Length<Dim> par_sb_ext = map.template impl_subblock_extent<Dim>(sb);
      Length<Dim> sb_ext     = par_sb_ext;
      parent_sdom_.push_back(domain(par_sb_ext));
      sb_ext_.push_back(sb_ext);
      parent_sb_.push_back(sb);
    }
  }
}



/// Specialize global traits for Global_map.

template <dimension_type Dim>
struct Is_global_map<Subset_map<Dim> >
{ static bool const value = true; };



/***********************************************************************
  Map subdomain
***********************************************************************/

// Map_subdomain is a map functor.  It creates a new map for a
// subdomain of an existing map.

// General case where subdomain map is identical to parent map.  This
// applies to Local_map, Global_map, and Local_or_global_map.

template <dimension_type Dim,
	  typename       MapT>
struct Map_subdomain
{
  typedef MapT type;

  static type project(
    MapT const&        map,
    Domain<Dim> const& /*dom*/)
  {
    return map;
  }

  // Return the parent subblock corresponding to a child subblock.
  static index_type parent_subblock(
    MapT const&        /*map*/,
    Domain<Dim> const& /*dom*/,
    index_type         sb)
  {
    assert(0);
    return sb;
  }
};



// Special case for block-cyclic Maps.

template <dimension_type Dim,
	  typename       Dist0,
	  typename       Dist1,
	  typename       Dist2>
struct Map_subdomain<Dim, Map<Dist0, Dist1, Dist2> >
{
  typedef Subset_map<Dim> type;

  static type project(
    Map<Dist0, Dist1, Dist2> const& map,
    Domain<Dim> const&              dom)
  {
    return type(map, dom);
  }

  // Return the parent subblock corresponding to a child subblock.
  static index_type parent_subblock(
    Map<Dist0, Dist1, Dist2> const& /*map*/,
    Domain<Dim> const&              /*dom*/,
    index_type                      sb)
  {
    assert(0);
    return sb;
  }
};



// Special case for Subset_maps

template <dimension_type Dim>
struct Map_subdomain<Dim, Subset_map<Dim> >
{
  typedef Subset_map<Dim> type;

  static type project(
    Subset_map<Dim> const& map,
    Domain<Dim> const&     dom)
  {
    return type(map, dom);
  }

  // Return the parent subblock corresponding to a child subblock.
  static index_type parent_subblock(
    Subset_map<Dim> const& map,
    Domain<Dim> const&     /*dom*/,
    index_type             sb)
  {
    assert(0);
    return sb;
  }
};


} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_CORE_PARALLEL_SUBSET_MAP_DECL_HPP
