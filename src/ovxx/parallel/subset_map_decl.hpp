//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_parallel_subset_map_decl_hpp_
#define ovxx_parallel_subset_map_decl_hpp_

#include <vsip/impl/view_fwd.hpp>
#include <vsip/impl/map_fwd.hpp>
#include <ovxx/length.hpp>
#include <ovxx/ct_assert.hpp>
#include <ovxx/parallel/service.hpp>
#include <ovxx/parallel/map_traits.hpp>
#include <ovxx/value_iterator.hpp>
#include <ovxx/domain_utils.hpp>

namespace ovxx
{
namespace parallel
{
template <dimension_type D>
class subset_map
{
public:
  typedef value_iterator<processor_type, unsigned> processor_iterator;
  typedef std::vector<processor_type> impl_pvec_type;

  template <typename MapT>
  subset_map(MapT const&, Domain<D> const&);

  ~subset_map(){}

  // Information on individual distributions.
  distribution_type distribution(dimension_type) const VSIP_NOTHROW
  { return other;}
  length_type num_subblocks (dimension_type) const VSIP_NOTHROW
  { return 1;}
  length_type cyclic_contiguity(dimension_type) const VSIP_NOTHROW
  { return 0;}

  length_type num_subblocks() const VSIP_NOTHROW
  { return sb_patch_gd_.size();}

  length_type num_processors() const VSIP_NOTHROW
  { return pvec_.size();}

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
    OVXX_PRECONDITION(sb < this->num_subblocks());
    return processor_iterator(this->pvec_[sb], 1);
  }

  processor_iterator processor_end(index_type sb) const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(sb < this->num_subblocks());
    return processor_iterator(this->pvec_[sb]+1, 1);
  }

  const_Vector<processor_type> processor_set() const;

  length_type impl_num_patches(index_type sb) const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(sb < this->num_subblocks());
    return sb_patch_gd_[sb].size();
  }

  void impl_apply(Domain<D> const& dom) VSIP_NOTHROW
  {
    OVXX_PRECONDITION(extent(dom_) == extent(dom));
    create_ll_pset(pvec_, ll_pset_);
  }

  template <dimension_type D1>
  Domain<D1> impl_subblock_domain(index_type sb) const VSIP_NOTHROW
  { OVXX_PRECONDITION(sb < this->num_subblocks()); return domain(sb_ext_[sb]);}

  template <dimension_type D1>
  Length<D1> impl_subblock_extent(index_type sb) const VSIP_NOTHROW
  { OVXX_PRECONDITION(sb < this->num_subblocks()); return sb_ext_[sb];}

  template <dimension_type D1>
  Domain<D1> impl_global_domain(index_type sb, index_type p)
    const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(sb < this->num_subblocks());
    OVXX_PRECONDITION(p < this->impl_num_patches(sb));
    return sb_patch_gd_[sb][p];
  }

  template <dimension_type D1>
  Domain<D1> impl_local_domain (index_type sb, index_type p)
    const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(sb < this->num_subblocks());
    OVXX_PRECONDITION(p < this->impl_num_patches(sb));
    return sb_patch_ld_[sb][p];
  }

  template <dimension_type D1>
  Domain<D1> impl_parent_local_domain(index_type sb) const VSIP_NOTHROW
  { return parent_sdom_[sb];}

  index_type impl_global_from_local_index(dimension_type /*d*/, index_type sb,
				     index_type idx)
    const VSIP_NOTHROW
  {
    OVXX_DO_THROW(unimplemented(
	      "subset_map::impl_global_from_local_index not implemented."));
    return 0;
  }

  index_type impl_local_from_global_index(dimension_type /*d*/, index_type idx)
    const VSIP_NOTHROW
  {
    OVXX_DO_THROW(unimplemented(
	      "subset_map::impl_local_from_global_index not implemented."));
    return 0;
  }

  template <dimension_type D1>
  index_type impl_subblock_from_global_index(Index<D1> const& /*idx*/)
    const VSIP_NOTHROW
  {
    OVXX_DO_THROW(unimplemented(
	      "subset_map::impl_subblock_from_global_index not implemented."));
    return 0;
  }

  template <dimension_type D1>
  Domain<D> impl_local_from_global_domain(index_type /*sb*/,
					  Domain<D1> const& dom)
    const VSIP_NOTHROW
  {
    OVXX_CT_ASSERT(D == D1);
    OVXX_DO_THROW(unimplemented(
	      "subset_map::impl_local_from_global_domain not implemented."));
    return Domain<D>();
  }

  ll_pset_type impl_ll_pset() const VSIP_NOTHROW
  { return ll_pset_;}
  Communicator &impl_comm() const
  { return default_communicator();}
  impl_pvec_type const&  impl_pvec() const
  { return pvec_;}

  length_type impl_working_size() const
  { return this->num_processors();}

  processor_type impl_proc_from_rank(index_type idx) const
  { return this->impl_pvec()[idx];}

  index_type impl_rank_from_proc(processor_type pr) const
  {
    for (index_type i=0; i<this->num_processors(); ++i)
      if (this->impl_pvec()[i] == pr) return i;
    return no_rank;
  }

  // Determine parent map subblock corresponding to this map's subblock
  index_type impl_parent_subblock(index_type sb) const
  { return parent_sb_[sb];}

  typedef std::vector<Domain<D> >   p_vector_type;
  typedef std::vector<p_vector_type>  sb_vector_type;


private:
  std::vector<index_type> parent_sb_; // map: local sb -> parent sb
  std::vector<Length<D> > sb_ext_;
  std::vector<Domain<D> > parent_sdom_;	// parent subblock dom.

  sb_vector_type          sb_patch_gd_;	// sb-patch global dom
  sb_vector_type          sb_patch_ld_;	// sb-patch local dom
  sb_vector_type          sb_patch_pld_;// sb-patch parent local dom

  impl_pvec_type          pvec_;	// Grid function.
  Domain<D>               dom_;		// Applied domain.
  ll_pset_type        ll_pset_;
};

template <dimension_type D>
template <typename M>
subset_map<D>::subset_map(M const &map, Domain<D> const &dom)
  : dom_(dom)
{
  // Check that map is only block distributed
  for (dimension_type d = 0; d < D; ++d)
  {
    if (map.distribution(d) == cyclic)
      OVXX_DO_THROW(unimplemented(
	      "subset_map: Subviews of cyclic maps not supported."));
  }

  for (index_type sb=0; sb<map.num_subblocks(); ++sb)
  {
    p_vector_type g_vec;
    p_vector_type l_vec;
    p_vector_type pl_vec;
    
    for (index_type p=0; p<map.impl_num_patches(sb); ++p)
    {
      // parent global/local subdomains for sb-p.
      Domain<D> pg_dom = map.template impl_global_domain<D>(sb, p);
      Domain<D> pl_dom = map.template impl_local_domain<D>(sb, p);

      Domain<D> intr;
      if (intersect(pg_dom, dom, intr))
      {
	// Global domain represented by intersection.
	Domain<D> mg_dom = subset_from_intr(dom, intr);

	// Local domain.
	Domain<D> ml_dom = normalize(intr);

	// Subset of parent local domain represented by intersection
	Domain<D> pl_dom_intr = apply_intr(pl_dom, pg_dom, intr);

	g_vec.push_back (mg_dom);
	l_vec.push_back (ml_dom);
	pl_vec.push_back(pl_dom_intr);
      }
    }

    if (g_vec.size() > 0)
    {
      if (g_vec.size() > 1)
      {
	OVXX_DO_THROW(unimplemented(
	    "subset_map: Subviews creating multiple patches not supported."));
      }
      sb_patch_gd_.push_back(g_vec);
      sb_patch_ld_.push_back(l_vec);
      sb_patch_pld_.push_back(pl_vec);
      pvec_.push_back(map.impl_proc_from_rank(sb));

      Length<D> par_sb_ext = map.template impl_subblock_extent<D>(sb);
      Length<D> sb_ext     = par_sb_ext;
      parent_sdom_.push_back(domain(par_sb_ext));
      sb_ext_.push_back(sb_ext);
      parent_sb_.push_back(sb);
    }
  }
}

// map_subdomain is a map functor.  It creates a new map for a
// subdomain of an existing map.

// General case where subdomain map is identical to parent map.  This
// applies to Local_map, Replicated_map, and Local_or_global_map.
template <dimension_type D, typename M>
struct map_subdomain
{
  typedef M type;

  static type project(M const &map, vsip::Domain<D> const &)
  {
    return map;
  }

  // Return the parent subblock corresponding to a child subblock.
  static index_type parent_subblock(M const &,
				    Domain<D> const &,
				    index_type sb)
  {
    assert(0);
    return sb;
  }
};

template <dimension_type D, typename D0, typename D1, typename D2>
struct map_subdomain<D, Map<D0, D1, D2> >
{
  typedef subset_map<D> type;

  static type project(Map<D0, D1, D2> const &map, Domain<D> const &dom)
  {
    return type(map, dom);
  }

  // Return the parent subblock corresponding to a child subblock.
  static index_type parent_subblock(Map<D0, D1, D2> const &,
				    Domain<D> const &,
				    index_type sb)
  {
    assert(0);
    return sb;
  }
};

template <dimension_type D>
struct map_subdomain<D, subset_map<D> >
{
  typedef subset_map<D> type;

  static type project(subset_map<D> const &map, Domain<D> const &dom)
  {
    return type(map, dom);
  }

  // Return the parent subblock corresponding to a child subblock.
  static index_type parent_subblock(subset_map<D> const &map,
				    Domain<D> const &,
				    index_type sb)
  {
    assert(0);
    return sb;
  }
};

template <dimension_type D>
struct is_global_map<subset_map<D> >
{ static bool const value = true;};

} // namespace ovxx::parallel
} // namespace ovxx

#endif
