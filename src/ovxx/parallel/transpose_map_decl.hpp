//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_parallel_transpose_map_decl_hpp_
#define ovxx_parallel_transpose_map_decl_hpp_

#include <vsip/impl/view_fwd.hpp>
#include <ovxx/parallel/service.hpp>
#include <vector>

namespace ovxx
{
namespace parallel
{

template <typename M>
class transpose_map
{
  // Translate tranpose dimension to parent map dimension.
  dimension_type parent_dim(dimension_type dim) const
  {
    return (dim == 0) ? 1 : (dim == 1) ? 0 : dim;
  }
public:
  static dimension_type const dim = 2;

  typedef std::vector<Domain<dim> >   p_vector_type;
  typedef std::vector<p_vector_type>  sb_vector_type;

  typedef typename M::processor_iterator processor_iterator;
  typedef typename M::impl_pvec_type     impl_pvec_type;

  transpose_map(M const& map) : map_(map) {}
  ~transpose_map() {}

  // Information on individual distributions.
  distribution_type distribution(dimension_type dim) const VSIP_NOTHROW
  { return map_.distribution(parent_dim(dim));}

  length_type num_subblocks(dimension_type dim) const VSIP_NOTHROW
  { return map_.num_subblock(parent_dim(dim));}

  length_type cyclic_contiguity(dimension_type dim) const VSIP_NOTHROW
  { return map_.cyclic_contiguity(parent_dim(dim));}

  length_type num_subblocks()  const VSIP_NOTHROW
  { return map_.num_subblocks();}

  length_type num_processors() const VSIP_NOTHROW
  { return map_.num_processors();}

  index_type subblock(processor_type pr) const VSIP_NOTHROW
  { return map_.subblock(pr);}

  index_type subblock() const VSIP_NOTHROW
  { return map_.subblock();}

  processor_iterator processor_begin(index_type sb) const VSIP_NOTHROW
  { return map_.processor_begin(sb);}

  processor_iterator processor_end  (index_type sb) const VSIP_NOTHROW
  { return map_.processor_end(sb);}

  const_Vector<processor_type> processor_set() const;

  length_type impl_num_patches(index_type sb) const VSIP_NOTHROW
  { return map_.impl_num_patches(sb);}

  void impl_apply(Domain<dim> const& /*dom*/) VSIP_NOTHROW
  {
    // TODO assert(extent(dom_) == transpose(extent(dom)));
  }

  template <dimension_type D>
  Domain<D> impl_subblock_domain(index_type sb) const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(sb < this->num_subblocks());
    Domain<D> p_dom = map_.template impl_subblock_domain<D>(sb);
    return Domain<D>(p_dom[1], p_dom[0]);
  }

  template <dimension_type D>
  Length<D> impl_subblock_extent(index_type sb) const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(sb < this->num_subblocks());
    Length<D> p_ext = map_.template impl_subblock_extent<D>(sb);
    return Length<D>(p_ext[1], p_ext[0]);
  }

  template <dimension_type D>
  Domain<D> impl_global_domain(index_type sb, index_type p)
    const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(sb < this->num_subblocks());
    OVXX_PRECONDITION(p  < this->impl_num_patches(sb));
    Domain<D> p_dom = map_.template impl_global_domain<D>(sb, p);
    return Domain<D>(p_dom[1], p_dom[0]);
  }

  template <dimension_type D>
  Domain<D> impl_local_domain (index_type sb, index_type p)
    const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(sb < this->num_subblocks());
    OVXX_PRECONDITION(p  < this->impl_num_patches(sb));
    Domain<D> p_dom = map_.template impl_local_domain<D>(sb, p);
    return Domain<D>(p_dom[1], p_dom[0]);
  }

  template <dimension_type D>
  Domain<D> impl_parent_local_domain(index_type sb) const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(sb < this->num_subblocks());
    Domain<D> p_dom = map_.template impl_parent_local_domain<D>(sb);
    return Domain<D>(p_dom[1], p_dom[0]);
  }

  index_type impl_global_from_local_index(dimension_type /*d*/, index_type sb,
				     index_type idx)
    const VSIP_NOTHROW
  {
    OVXX_DO_THROW(unimplemented(
	      "transpose_map::impl_global_from_local_index not implemented."));
    return 0;
  }

  index_type impl_local_from_global_index(dimension_type dim, index_type idx)
    const VSIP_NOTHROW
  { return map_.impl_local_from_global_index(parent_dim(dim), idx); }

  template <dimension_type D>
  index_type impl_subblock_from_global_index(Index<D> const& /*idx*/)
    const VSIP_NOTHROW
  {
    OVXX_DO_THROW(unimplemented(
	      "transpose_map::impl_subblock_from_global_index not implemented."));
    return 0;
  }

  template <dimension_type D>
  Domain<dim> impl_local_from_global_domain(index_type sb,
					    Domain<D> const& dom)
    const VSIP_NOTHROW
  {
    OVXX_CT_ASSERT(dim == D);
    Domain<dim> p_gdom(dom[1], dom[0]);
    Domain<dim> p_ldom = map_.template impl_local_from_global_domain<D>
                                        (sb, p_gdom);
    return Domain<dim>(p_ldom[1], p_ldom[0]);
  }

#ifdef OVXX_PARALLEL

  ovxx::parallel::ll_pset_type impl_ll_pset() const VSIP_NOTHROW
  { return map_.impl_ll_pset();}

  ovxx::parallel::Communicator &impl_comm() const
  { return map_.impl_comm();}

#endif

  impl_pvec_type const & impl_pvec() const
  { return map_.impl_pvec();}

  length_type impl_working_size() const
  { return map_.impl_working_size();}

  processor_type impl_proc_from_rank(index_type idx) const
  { return map_.impl_proc_from_rank(idx);}

  index_type impl_rank_from_proc(processor_type pr) const
  { return map_.impl_rank_from_proc(pr);}

  // Determine parent map subblock corresponding to this map's subblock
  index_type impl_parent_subblock(index_type sb) const
  { return map_.impl_parent_subblock(sb);}

private:
  M const &map_;
};

template <typename M>
struct is_global_map<parallel::transpose_map<M> >
{ static bool const value = true;};

/// Functor to transpose a map.
///
/// Handles both the type conversion ('type') and the runtime
/// conversion ('project').
template <dimension_type D, typename M>
struct transpose_map_of
{
  typedef parallel::transpose_map<M> type;

  static type project(M const &map) { return type(map);}
};

template <dimension_type D>
struct transpose_map_of<D, Local_map>
{
  typedef Local_map type;

  static type project(Local_map const &map) { return map;}
};

} // namespace ovxx::parallel
} // namespace ovxx

#endif
