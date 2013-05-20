//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef vsip_impl_replicated_map_hpp_
#define vsip_impl_replicated_map_hpp_

#include <ovxx/vector_iterator.hpp>
#include <ovxx/parallel/service.hpp>
#include <ovxx/parallel/map_traits.hpp>
#include <ovxx/parallel/scalar_map.hpp>
#include <ovxx/block_traits.hpp>
#include <ovxx/c++11.hpp>

namespace vsip
{
namespace impl
{
class pset_block : public ovxx::refcounted<pset_block>
{
public:
  static dimension_type const dim = 1;

  typedef processor_type value_type;
  typedef Local_map map_type;

  pset_block(Domain<1> const &dom, map_type const & = map_type())
    VSIP_THROW((std::bad_alloc))
    : vector_(dom.length())
  {}
  pset_block(Domain<1> const &dom, value_type v, map_type const& = map_type())
    VSIP_THROW((std::bad_alloc))
    : vector_(dom.length())
  {
    for (index_type i = 0; i != dom.length(); ++i)
      this->put(i, v);
  }
  ~pset_block() VSIP_NOTHROW {}

  value_type get(index_type i) const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(i < size());
    return vector_[i];
  }

  void put(index_type i, value_type v) VSIP_NOTHROW
  {
    OVXX_PRECONDITION(i < size());
    vector_[i] = v;
  }

  length_type size() const VSIP_NOTHROW { return vector_.size();}
  length_type size(dimension_type D, dimension_type d) const VSIP_NOTHROW
  { OVXX_PRECONDITION(D == 1 && d == 0); return vector_.size();}

  map_type const &map() const VSIP_NOTHROW { return map_;}
  std::vector<value_type> const &vector() { return vector_;}

private:
  std::vector<value_type> vector_;
  map_type    map_;
};
}


template <dimension_type D>
class Replicated_map
{
  struct Data : ovxx::detail::noncopyable
  {
    Data(length_type np) : pset(np) {}
    ~Data() { ovxx::parallel::destroy_ll_pset(ll_pset);}
    void init_ll_pset() { ovxx::parallel::create_ll_pset(pset.vector(), ll_pset);}

    impl::pset_block pset;
    ovxx::parallel::par_ll_pset_type ll_pset;
  };

public:
  typedef ovxx::vector_iterator<Vector<processor_type> > processor_iterator;
  typedef std::vector<processor_type> impl_pvec_type;

  Replicated_map() VSIP_THROW((std::bad_alloc))
    : data_(new Data(vsip::num_processors()))
  {
    for (index_type i=0; i<vsip::num_processors(); ++i)
      data_->pset.put(i, vsip::processor_set().get(i));
    data_->init_ll_pset();
  }

  template <typename Block>
  Replicated_map(const_Vector<processor_type, Block> pset) VSIP_THROW((std::bad_alloc))
    : data_(new Data(pset.size()))
  {
    OVXX_PRECONDITION(pset.size() > 0);
    for (index_type i=0; i<pset.size(); ++i)
      data_->pset.put(i, pset.get(i));
    data_->init_ll_pset();
  }

  ~Replicated_map() {}

  // Information on individual distributions.
  distribution_type distribution (dimension_type) const VSIP_NOTHROW
  { return whole;}
  length_type num_subblocks (dimension_type) const VSIP_NOTHROW
  { return 1;}
  length_type cyclic_contiguity(dimension_type) const VSIP_NOTHROW
  { return 0;}
  length_type num_subblocks() const VSIP_NOTHROW { return 1;}
  index_type subblock(processor_type pr) const VSIP_NOTHROW
  {
    if (this->impl_rank_from_proc(pr) != no_rank)
      return 0;
    else
      return no_subblock;
  }
  index_type subblock() const VSIP_NOTHROW
  { return this->subblock(local_processor());}

  length_type num_processors() const VSIP_NOTHROW
  { return this->data_->pset.size();}

  processor_iterator processor_begin(index_type sb) const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(sb == 0);
    return processor_iterator(this->processor_set(), 0);
  }

  processor_iterator processor_end  (index_type sb) const VSIP_NOTHROW
  {
    OVXX_PRECONDITION(sb == 0);
    return processor_iterator(this->processor_set(), this->num_processors());
  }

  const_Vector<processor_type, impl::pset_block> processor_set() const
  {
    return const_Vector<processor_type, impl::pset_block>(data_->pset);
  }

  length_type impl_num_patches(index_type sb) const VSIP_NOTHROW
  { OVXX_PRECONDITION(sb == 0); return 1;}

  void impl_apply(Domain<D> const& dom) VSIP_NOTHROW
  { dom_ = dom;}

  template <dimension_type D1>
  Domain<D1> impl_subblock_domain(index_type sb) const VSIP_NOTHROW
  { OVXX_PRECONDITION(sb == 0); return dom_;}

  template <dimension_type D1>
  ovxx::Length<D1> impl_subblock_extent(index_type sb) const VSIP_NOTHROW
  { OVXX_PRECONDITION(sb == 0); return extent(dom_);}

  template <dimension_type D1>
  Domain<D1> impl_global_domain(index_type sb, index_type patch)
    const VSIP_NOTHROW
  { OVXX_PRECONDITION(sb == 0 && patch == 0); return dom_;}

  template <dimension_type D1>
  Domain<D1> impl_local_domain (index_type sb, index_type patch)
    const VSIP_NOTHROW
  { OVXX_PRECONDITION(sb == 0 && patch == 0); return dom_;}

  index_type impl_global_from_local_index(dimension_type /*d*/, index_type sb,
					  index_type idx)
    const VSIP_NOTHROW
  { OVXX_PRECONDITION(sb == 0); return idx;}

  index_type impl_local_from_global_index(dimension_type /*d*/, index_type idx)
    const VSIP_NOTHROW
  { return idx;}

  template <dimension_type D1>
  index_type impl_subblock_from_global_index(Index<D1> const& /*idx*/)
    const VSIP_NOTHROW
  { return 0;}

  template <dimension_type D1>
  Domain<D> impl_local_from_global_domain(index_type /*sb*/,
					  Domain<D1> const& dom)
    const VSIP_NOTHROW
  {
    OVXX_CT_ASSERT(D == D1);
    return dom;
  }

  ovxx::parallel::par_ll_pset_type impl_ll_pset() const VSIP_NOTHROW
  { return data_->ll_pset;}
  ovxx::parallel::Communicator &impl_comm() const
  { return ovxx::parallel::default_communicator();}
  impl_pvec_type const &impl_pvec() const { return data_->pset.vector();}
  length_type impl_working_size() const { return this->num_processors();}

  processor_type impl_proc_from_rank(index_type rank) const
  {
    OVXX_PRECONDITION(rank < this->num_processors());
    return this->impl_pvec()[rank];
  }

  index_type impl_rank_from_proc(processor_type pr) const
  {
    for (index_type i=0; i<this->num_processors(); ++i)
      if (data_->pset.get(i) == pr) return i;
    return no_rank;
  }

private:
  Domain<D> dom_;
  ovxx::shared_ptr<Data> data_;
};

} // namespace vsip

namespace ovxx
{
namespace parallel
{
template <dimension_type D>
class local_or_global_map : public Replicated_map<D>
{
public:
  static bool const impl_local_only  = false;
  static bool const impl_global_only = false;

  local_or_global_map() {}
};

template <dimension_type D>
struct is_global_map<Replicated_map<D> >
{ static bool const value = true;};

template <dimension_type D>
struct is_local_map<local_or_global_map<D> >
{ static bool const value = true;};

template <dimension_type D>
struct is_global_map<local_or_global_map<D> >
{ static bool const value = true;};

template <dimension_type D>
struct map_equal<D, Replicated_map<D>, Replicated_map<D> >
{
  static bool value(Replicated_map<D> const &a,
		    Replicated_map<D> const &b)
  {
    if (a.num_processors() != b.num_processors())
      return false;
    for (index_type i=0; i<a.num_processors(); ++i)
      if (a.impl_proc_from_rank(i) != b.impl_proc_from_rank(i))
	return false;
    return true;
  }
};

} // namespace ovxx::parallel
} // namespace ovxx

#endif
