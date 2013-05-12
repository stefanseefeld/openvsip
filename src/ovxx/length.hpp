//
// Copyright (c) 2005 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_length_hpp_
#define ovxx_length_hpp_

#include <ovxx/support.hpp>
#include <ovxx/vertex.hpp>

namespace ovxx
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

template <dimension_type D, typename B>
inline Length<D>
extent(B const &block)
{
  Length<D> length;
  for (dimension_type d = 0; d != D; ++d)
    length[d] = block.size(D, d);
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

} // namespace ovxx

#endif
