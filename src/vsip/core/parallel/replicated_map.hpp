/* Copyright (c) 2005, 2006 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/replicated_map.hpp
    @author  Jules Bergmann
    @date    2005-06-08
    @brief   VSIPL++ Library: Replicated_map class.

*/

#ifndef VSIP_IMPL_REPLICATED_MAP_HPP
#define VSIP_IMPL_REPLICATED_MAP_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/vector_iterator.hpp>
#include <vsip/core/parallel/services.hpp>
#include <vsip/core/parallel/map_traits.hpp>
#include <vsip/core/parallel/util.hpp>
#include <vsip/core/block_traits.hpp>
#include <vsip/core/sv_block.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

template <dimension_type Dim>
class Replicated_map
{
  // Compile-time typedefs.
public:
  typedef impl::Vector_iterator<Vector<processor_type> > processor_iterator;
  typedef std::vector<processor_type> impl_pvec_type;
  typedef impl::Sv_local_block<processor_type> pset_block_type;

  // Replicated_map_data
private:
  struct Data : public impl::Ref_count<Data>, impl::Non_copyable
  {

    Data(length_type np)
      : pset_(np)
    {}

    ~Data()
    {
      impl::destroy_ll_pset(ll_pset_);
    }

    void init_ll_pset()
    {
      impl::create_ll_pset(pset_.impl_vector(), ll_pset_);
    }

    // Member data.
  public:
    pset_block_type        pset_;
    impl::par_ll_pset_type ll_pset_;
  };



  // Constructor.
public:
  Replicated_map()
    VSIP_THROW((std::bad_alloc));

  template <typename Block>
  Replicated_map(const_Vector<processor_type, Block> pset)
    VSIP_THROW((std::bad_alloc));

  ~Replicated_map()
  {
  }

  // Default copy constructor and assignment operator are correct.

  // Accessors.
public:
  // Information on individual distributions.
  distribution_type distribution     (dimension_type) const VSIP_NOTHROW
    { return whole; }
  length_type       num_subblocks    (dimension_type) const VSIP_NOTHROW
    { return 1; }
  length_type       cyclic_contiguity(dimension_type) const VSIP_NOTHROW
    { return 0; }

  length_type num_subblocks() const VSIP_NOTHROW { return 1; }

  index_type subblock(processor_type pr) const VSIP_NOTHROW
  {
    if (this->impl_rank_from_proc(pr) != no_rank)
      return 0;
    else
      return no_subblock;
  }
  index_type subblock() const VSIP_NOTHROW
  { return this->subblock(local_processor()); }

  length_type num_processors() const VSIP_NOTHROW
    { return this->data_->pset_.size(); }

  processor_iterator processor_begin(index_type sb) const VSIP_NOTHROW
  {
    assert(sb == 0);
    return processor_iterator(this->processor_set(), 0);
  }

  processor_iterator processor_end  (index_type sb) const VSIP_NOTHROW
  {
    assert(sb == 0);
    return processor_iterator(this->processor_set(), this->num_processors());
  }

  const_Vector<processor_type, pset_block_type> processor_set() const
  {
    return const_Vector<processor_type, pset_block_type>(data_->pset_);
  }

  // Applied map functions.
public:
  length_type impl_num_patches(index_type sb) const VSIP_NOTHROW
    { assert(sb == 0); return 1; }

  void impl_apply(Domain<Dim> const& dom) VSIP_NOTHROW
    { dom_ = dom; }

  template <dimension_type Dim2>
  Domain<Dim2> impl_subblock_domain(index_type sb) const VSIP_NOTHROW
    { assert(sb == 0); return dom_; }

  template <dimension_type Dim2>
  impl::Length<Dim2> impl_subblock_extent(index_type sb) const VSIP_NOTHROW
    { assert(sb == 0); return extent(dom_); }

  template <dimension_type Dim2>
  Domain<Dim2> impl_global_domain(index_type sb, index_type patch)
    const VSIP_NOTHROW
    { assert(sb == 0 && patch == 0); return dom_; }

  template <dimension_type Dim2>
  Domain<Dim2> impl_local_domain (index_type sb, index_type patch)
    const VSIP_NOTHROW
    { assert(sb == 0 && patch == 0); return dom_; }

  index_type impl_global_from_local_index(dimension_type /*d*/, index_type sb,
					  index_type idx)
    const VSIP_NOTHROW
  { assert(sb == 0); return idx; }

  index_type impl_local_from_global_index(dimension_type /*d*/, index_type idx)
    const VSIP_NOTHROW
  { return idx; }

  template <dimension_type Dim2>
  index_type impl_subblock_from_global_index(Index<Dim2> const& /*idx*/)
    const VSIP_NOTHROW
  { return 0; }

  template <dimension_type Dim2>
  Domain<Dim> impl_local_from_global_domain(index_type /*sb*/,
					    Domain<Dim2> const& dom)
    const VSIP_NOTHROW
  {
    VSIP_IMPL_STATIC_ASSERT(Dim == Dim2);
    return dom;
  }

  // Extensions.
public:
  impl::par_ll_pset_type impl_ll_pset() const VSIP_NOTHROW
    { return data_->ll_pset_; }
  impl::Communicator&    impl_comm() const
    { return impl::default_communicator();}
  impl_pvec_type const&  impl_pvec() const
    { return data_->pset_.impl_vector(); }

  length_type        impl_working_size() const
    { return this->num_processors(); }

  processor_type impl_proc_from_rank(index_type rank) const
  {
    assert(rank < this->num_processors());
    return this->impl_pvec()[rank];
  }

  index_type impl_rank_from_proc(processor_type pr) const
  {
    for (index_type i=0; i<this->num_processors(); ++i)
      if (data_->pset_.get(i) == pr) return i;
    return no_rank;
  }


  // Member data.
private:
  Domain<Dim>                 dom_;		// Applied domain.
  impl::Ref_counted_ptr<Data> data_;
};



template <dimension_type Dim>
Replicated_map<Dim>::Replicated_map()
  VSIP_THROW((std::bad_alloc))
    : data_(new Data(vsip::num_processors()))
{
  for (index_type i=0; i<vsip::num_processors(); ++i)
    data_->pset_.put(i, vsip::processor_set().get(i));

  data_->init_ll_pset();
}



// Create a replicated_map with a given processor_set
//
// Requires
//   PSET to be a non-empty set of valid processors.
template <dimension_type Dim>
template <typename Block>
Replicated_map<Dim>::Replicated_map(
  const_Vector<processor_type, Block> pset)
  VSIP_THROW((std::bad_alloc))
    : data_(new Data(pset.size()))
{
  assert(pset.size() > 0);
  for (index_type i=0; i<pset.size(); ++i)
    data_->pset_.put(i, pset.get(i));

  data_->init_ll_pset();
}



namespace impl
{

template <dimension_type Dim>
struct is_global_map<Replicated_map<Dim> >
{ static bool const value = true; };

template <dimension_type Dim>
struct Map_equal<Dim, Replicated_map<Dim>, Replicated_map<Dim> >
{
  static bool value(Replicated_map<Dim> const& a,
		    Replicated_map<Dim> const& b)
  {
    if (a.num_processors() != b.num_processors())
      return false;
    for (index_type i=0; i<a.num_processors(); ++i)
      if (a.impl_proc_from_rank(i) != b.impl_proc_from_rank(i))
	return false;
    return true;
  }
};

} // namespace vsip::impl

} // namespace vsip

#endif // VSIP_IMPL_REPLICATED_MAP_HPP
