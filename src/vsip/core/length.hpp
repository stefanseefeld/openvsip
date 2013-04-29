/* Copyright (c) 2005 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/length.hpp
    @author  Stefan Seefeld
    @date    2005-08-02
    @brief   VSIPL++ Library: Length template.

*/

#ifndef VSIP_CORE_LENGTH_HPP
#define VSIP_CORE_LENGTH_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/core/vertex.hpp>

namespace vsip
{
namespace impl
{

template <dimension_type D> struct Length;

template <> class Length<1> : public Vertex<length_type, 1>
{
public:
  Length() VSIP_NOTHROW {}
  Length(length_type x) VSIP_NOTHROW : Vertex<length_type, 1>(x) {}
};

template <> class Length<2> : public Vertex<length_type, 2>
{
public:
  Length() VSIP_NOTHROW {}
  Length(length_type x, length_type y) VSIP_NOTHROW 
  : Vertex<length_type, 2>(x, y) {}
};

template <> class Length<3> : public Vertex<length_type, 3>
{
public:
  Length() VSIP_NOTHROW {}
  Length(length_type x, length_type y, length_type z) VSIP_NOTHROW 
  : Vertex<length_type, 3>(x, y, z) {}
};

template <dimension_type Dim,
	  typename       B>
inline Length<Dim>
extent(B const& block)
{
  Length<Dim> length;
  for (dimension_type d = 0; d != Dim; ++d)
    length[d] = block.size(Dim, d);
  return length;
}


template <dimension_type D>
inline length_type
total_size(Length<D> const& len)
{
  length_type size = len[0];
  for (dimension_type d=1; d<D; ++d)
    size *= len[d];
  return size;
}



// Return size of dimension.

// Generic function.  Overloaded for structures that can encode
// extents (Domain, Length)

template <dimension_type D>
inline length_type
size_of_dim(Length<D> const& len, dimension_type d)
{
  return len[d];
}


} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_CORE_LENGTH_HPP
