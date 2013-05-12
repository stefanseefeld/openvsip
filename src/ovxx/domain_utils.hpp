//
// Copyright (c) 2005 - 2010 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_domain_utils_hpp_
#define ovxx_domain_utils_hpp_

#include <vsip/domain.hpp>
#include <ovxx/length.hpp>
#include <ovxx/ct_assert.hpp>

namespace ovxx
{
namespace detail
{

/// Helper class to help project a domain: create a lower-dimensional
/// domain from an existing domain.
template <dimension_type D>
struct Project_domain;

/// Specialization for projecting to a 1-dimensional domain.
template <>
struct Project_domain<1>
{
  template <dimension_type D>
  static Domain<1> project(Domain<D> const& dom)
  {
    OVXX_PRECONDITION(1 <= D);
    return Domain<1>(dom[0]);
  }
};

/// Specialization for projecting to a 2-dimensional domain.
template <>
struct Project_domain<2>
{
  template <dimension_type D>
  static Domain<2> project(Domain<D> const& dom)
  {
    OVXX_PRECONDITION(2 <= D);
    return Domain<2>(dom[0], dom[1]);
  }
};

/// Specialization for projecting to a 3-dimensional domain.
template <>
struct Project_domain<3>
{
  template <dimension_type D>
  static Domain<3> project(Domain<D> const& dom)
  {
    OVXX_PRECONDITION(3 <= D);
    return dom;
  }
};

template <dimension_type D, typename BlockT>
struct Block_domain_class;

template <typename BlockT>
struct Block_domain_class<1, BlockT>
{
  static Domain<1> func(BlockT const& block)
  { return Domain<1>(block.size(1, 0));}
};

template <typename BlockT>
struct Block_domain_class<2, BlockT>
{
  static Domain<2> func(BlockT const& block)
  { return Domain<2>(block.size(2, 0), block.size(2, 1));}
};

template <typename BlockT>
struct Block_domain_class<3, BlockT>
{
  static Domain<3> func(BlockT const& block)
  { return Domain<3>(block.size(3, 0), block.size(3, 1), block.size(3, 2)); }
};

} // namespace ovxx::detail

/// Project an ORIGDIM-dimension domain to PROJDIM dimensions.
/// (where PROJDIM <= ORIGDIM)
template <dimension_type ProjDim, dimension_type OrigDim>
Domain<ProjDim>
project(Domain<OrigDim> const& dom)
{
  return detail::Project_domain<ProjDim>::project(dom);
}

/// Construct a Domain<D> from an array of Dim Domain<1>s.
template <dimension_type D>
Domain<D>
construct_domain(Domain<1> const *dom);

/// Specialization to constuct a Domain<1>.
template <>
inline Domain<1>
construct_domain(Domain<1> const *dom) { return dom[0];}

/// Specialization to constuct a Domain<2>.
template <>
inline Domain<2>
construct_domain(Domain<1> const *dom) { return Domain<2>(dom[0], dom[1]);}

/// Specialization to constuct a Domain<3>.
template <>
inline Domain<3>
construct_domain(Domain<1> const *dom)
{ return Domain<3>(dom[0], dom[1], dom[2]);}

/// Compute interstection of two 1-D domains.
/// Arguments:
///
///   :dom1: is a Domain<1> with stride 1,
///   :dom2: is a Domain<1>
///   :res:  is a Domain<1>.
///
/// Effects:
///
///   If dom1 and dom2 intersect, Interstection is placed in res and
///   true is returned. Otherwise, res is unchanged and false is returned.
inline bool
intersect(Domain<1> const &dom1,
	  Domain<1> const &dom2,
	  Domain<1> &res)
{
  OVXX_PRECONDITION(dom1.stride() == 1);

  if (dom2.stride() == 1)
  {
    index_type first   = std::max(dom1.first(),     dom2.first());
    index_type last    = std::min(dom1.impl_last(), dom2.impl_last());

    if (first <= last) 
    {
      res = Domain<1>(first, 1, last-first+1);
      return true;
    }
    else return false;
  }
  else
  {
    index_type first1  = dom1.first();
    index_type first2  = dom2.first();
    index_type last1   = dom1.impl_last();
    index_type last2   = dom2.impl_last();
    if (first2 < first1)
    {
      index_type diff = first1 - first2;
      first2 += diff;
      if (diff % dom2.stride())
	first2 += (dom2.stride() - (diff % dom2.stride()));
    }
    if (last2 > last1)
    {
      index_type diff = last2 - last1;
      if (diff % dom2.stride())
	last2 -= (dom2.stride() - (diff % dom2.stride()));
    }

    index_type first   = std::max(first1, first2);
    index_type last    = std::min(last1,  last2);

    if (first <= last) 
    {
      res = Domain<1>(first, dom2.stride(), (last-first)/dom2.stride()+1);
      return true;
    }
    else return false;
  }
}

/// Compute interstection of two 2-D domains.
inline bool
intersect(Domain<2> const &dom1,
	  Domain<2> const &dom2,
	  Domain<2> &res)
{
  Domain<1> res_dim[2];

  if (intersect(dom1[0], dom2[0], res_dim[0]) &&
      intersect(dom1[1], dom2[1], res_dim[1]))
  {
    res = Domain<2>(res_dim[0], res_dim[1]);
    return true;
  }
  else return false;
}

/// Compute interstection of two 3-D domains.
inline bool
intersect(Domain<3> const &dom1,
	  Domain<3> const &dom2,
	  Domain<3> &res)
{
  Domain<1> res_dim[3];

  if (intersect(dom1[0], dom2[0], res_dim[0]) &&
      intersect(dom1[1], dom2[1], res_dim[1]) &&
      intersect(dom1[2], dom2[2], res_dim[2]))
  {
    res = Domain<3>(res_dim[0], res_dim[1], res_dim[2]);
    return true;
  }
  else return false;
}

#if 0
// General intersection would be possible if Domains were modifiable.
// (Dim == 1 case would need to be specialized, of course).

template <dimension_type D>
bool
intersect(Domain<D> const &dom1,
	  Domain<D> const &dom2,
	  Domain<D> &res)
{
  for (dimension_type d=0; d<D; ++d)
    if (intersect(dom1[d], dom2[d], res[d]) == false)
      return false;
  return true;
}
#endif

/// Apply offset implied by intersection to another domain.
inline Domain<1>
apply_intr(Domain<1> const &x,
	   Domain<1> const &y,
	   Domain<1> const &intr)
{
  return Domain<1>(x.first() + (intr.first() - y.first()) * x.stride(),
		   x.stride() * intr.stride(),
		   intr.size());
}

inline Domain<2>
apply_intr(Domain<2> const &x,
	   Domain<2> const &y,
	   Domain<2> const &intr)
{
  return Domain<2>(apply_intr(x[0], y[0], intr[0]),
		   apply_intr(x[1], y[1], intr[1]));
}

inline Domain<3>
apply_intr(Domain<3> const &x,
	   Domain<3> const &y,
	   Domain<3> const &intr)
{
  return Domain<3>(apply_intr(x[0], y[0], intr[0]),
		   apply_intr(x[1], y[1], intr[1]),
		   apply_intr(x[2], y[2], intr[2]));
}



/// Convert an intersection into a subset
///
/// Subdomain's
///
///   :offset: is adjusted relative to parent domain
///   :stride: is 1 (since parent encodes stride)
///   :size: is unchanged.
inline Domain<1>
subset_from_intr(Domain<1> const &dom,
		 Domain<1> const &intr)
{
  return Domain<1>((intr.first() - dom.first()) / dom.stride(),
		   1,
		   intr.size());
}

inline Domain<2>
subset_from_intr(Domain<2> const &dom,
		 Domain<2> const &intr)
{
  return Domain<2>(subset_from_intr(dom[0], intr[0]),
		   subset_from_intr(dom[1], intr[1]));
}

inline Domain<3>
subset_from_intr(Domain<3> const &dom,
		 Domain<3> const &intr)
{
  return Domain<3>(subset_from_intr(dom[0], intr[0]),
		   subset_from_intr(dom[1], intr[1]),
		   subset_from_intr(dom[2], intr[2]));
}



/// Return the total size of a domain.
template <dimension_type D>
length_type
size(Domain<D> const &dom)
{
  length_type s = 1;
  for (dimension_type d=0; d<D; ++d)
    s *= dom[d].length();
  return s;
}



/// Return the domain of a block.
template <dimension_type D, typename BlockT>
inline Domain<D>
block_domain(BlockT const &block)
{
  return detail::Block_domain_class<D, BlockT>::func(block);
}

/// Return the empty domain.
template <dimension_type D>
Domain<D>
empty_domain()
{
  Domain<1> dom[D];
  for (dimension_type d=0; d<D; ++d)
    dom[d] = Domain<1>(0);
  return construct_domain<D>(dom);
}

/// Normalize a domain -- return a new domain with the same length
/// in each dimension, but with offset = 0 and stride = 1.
inline Domain<1>
normalize(Domain<1> const &dom) 
{ return Domain<1>(dom.size());}

inline Domain<2>
normalize(Domain<2> const &dom)
{ return Domain<2>(dom[0].size(), dom[1].size());}

inline Domain<3>
normalize(Domain<3> const &dom)
{ return Domain<3>(dom[0].size(), dom[1].size(), dom[2].size());}

/// Get the nth index in a domain.
template <dimension_type D>
Index<D>
domain_nth(Domain<D> const &dom, Index<D> const &idx)
{
  Index<D> res;
  for (dimension_type d = 0; d < D; ++d)
    res[d] = dom[d].impl_nth(idx[d]);
  return res;
}

/// Get the extent of a domain as a Length.
template <dimension_type D>
ovxx::Length<D>
extent(Domain<D> const &dom)
{
  ovxx::Length<D> res;
  for (dimension_type d = 0; d < D; ++d)
    res[d] = dom[d].length();
  return res;
}


/// Get the first index of a domain.
template <dimension_type D>
Index<D>
first(Domain<D> const& dom)
{
  Index<D> res;
  for (dimension_type d = 0; d < D; ++d)
    res[d] = dom[d].first();
  return res;
}

/// Construct a 1-dim domain with a size (implicit offset of 0 and
/// stride of 1)
inline Domain<1>
domain(ovxx::Length<1> const &size) { return Domain<1>(size[0]);}

/// Construct a 2-dim domain with a size (implicit offset of 0 and
/// stride of 1)
inline Domain<2>
domain(ovxx::Length<2> const &size) { return Domain<2>(size[0], size[1]);}

/// Construct a 3-dim domain with a size (implicit offset of 0 and
/// stride of 1)
inline Domain<3>
domain(ovxx::Length<3> const &size) { return Domain<3>(size[0], size[1], size[2]);}

/// Construct a 1-dim domain with an offset and a size (implicit
/// stride of 1)
inline Domain<1>
domain(Index<1> const &first, ovxx::Length<1> const &size)
{ return Domain<1>(first[0], 1, size[0]);}

/// Construct a 2-dim domain with an offset and a size (implicit
/// stride of 1)
inline Domain<2>
domain(Index<2> const &first, ovxx::Length<2> const &size)
{
  return Domain<2>(Domain<1>(first[0], 1, size[0]),
		   Domain<1>(first[1], 1, size[1]));
}

/// Construct a 3-dim domain with an offset and a size (implicit
/// stride of 1)
inline Domain<3>
domain(Index<3> const &first, ovxx::Length<3> const &size)
{
  return Domain<3>(Domain<1>(first[0], 1, size[0]),
		   Domain<1>(first[1], 1, size[1]),
		   Domain<1>(first[2], 1, size[2]));
}

/// Return the next Index from 'idx' using OrderT to determine the
/// dimension-order of the traversal.
template <typename OrderT>
inline Index<1>&
next(ovxx::Length<1> const &/*extent*/, Index<1> &idx)
{
  OVXX_CT_ASSERT(OrderT::impl_dim0 == 0);
  ++idx[OrderT::impl_dim0];
  return idx;
}

template <typename OrderT>
inline Index<2>&
next(ovxx::Length<2> const &extent, Index<2> &idx)
{
  OVXX_CT_ASSERT(OrderT::impl_dim0 < 2);
  OVXX_CT_ASSERT(OrderT::impl_dim1 < 2);
  if (++idx[OrderT::impl_dim1] == extent[OrderT::impl_dim1])
  {
    if (++idx[OrderT::impl_dim0] != extent[OrderT::impl_dim0])
      idx[OrderT::impl_dim1] = 0;
  }
  return idx;
}

template <typename OrderT>
inline Index<3>&
next(ovxx::Length<3> const &extent, Index<3> &idx)
{
  if (++idx[OrderT::impl_dim2] == extent[OrderT::impl_dim2])
  {
    if (++idx[OrderT::impl_dim1] == extent[OrderT::impl_dim1])
    {
      if (++idx[OrderT::impl_dim0] == extent[OrderT::impl_dim0])
	return idx;
      idx[OrderT::impl_dim1] = 0;
    }
    idx[OrderT::impl_dim2] = 0;
  }
  return idx;
}

/// Overload of next that performs row-major traversal.
template <dimension_type D>
inline Index<D> &
next(ovxx::Length<D> const &extent, Index<D> &idx)
{
  return next<typename row_major<D>::type>(extent, idx);
}

/// This function checks if the index is valid given a certain length. This
/// function works for multiple dimension spaces.
template <dimension_type D>
inline bool
valid(ovxx::Length<D> const &extent, Index<D> const &idx)
{
  for(dimension_type d = 0; d < D; ++d)
    if(idx[d] >= extent[d])
      return false;
  return true;
}

} // namespace ovxx

#endif
