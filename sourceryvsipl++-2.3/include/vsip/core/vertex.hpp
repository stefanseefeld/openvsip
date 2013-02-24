/* Copyright (c) 2005 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/vertex.hpp
    @author  Stefan Seefeld
    @date    2005-01-20
    @brief   VSIPL++ Library: Helper type.

    This file declares the a helper template from which other types such as
    Index are instantiated.
*/

#ifndef VSIP_CORE_VERTEX_HPP
#define VSIP_CORE_VERTEX_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>

/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

template <typename T, dimension_type D> struct Vertex;

template <typename T> class Vertex<T, 1>
{
public:
  Vertex() VSIP_NOTHROW : coord_(0) {}
  Vertex(T x) VSIP_NOTHROW : coord_(x) {}

  T operator[] (dimension_type) const VSIP_NOTHROW { return coord_;}
  T &operator[] (dimension_type) VSIP_NOTHROW { return coord_;}

  template <typename T1>
  friend bool operator==(Vertex<T1, 1> const&, Vertex<T1, 1> const&) VSIP_NOTHROW;
private:
  T coord_;
};

template <typename T> class Vertex<T, 2>
{
public:
  Vertex() VSIP_NOTHROW { coords_[0] = coords_[1] = 0;}
  Vertex(T x, T y) VSIP_NOTHROW;

  T operator[] (dimension_type d) const VSIP_NOTHROW;
  T &operator[] (dimension_type d) VSIP_NOTHROW;

  template <typename T1>
  friend bool operator==(Vertex<T1, 2> const&, Vertex<T1, 2> const&) VSIP_NOTHROW;
private:
  T coords_[2];
};

template <typename T> class Vertex<T, 3>
{
public:
  Vertex() VSIP_NOTHROW { coords_[0] = coords_[1] = coords_[2] = 0;}
  Vertex(T x, T y, T z) VSIP_NOTHROW;

  T operator[] (dimension_type d) const VSIP_NOTHROW;
  T &operator[] (dimension_type d) VSIP_NOTHROW;

  template <typename T1>
  friend bool operator==(Vertex<T1, 3> const&, Vertex<T1, 3> const&) VSIP_NOTHROW;
private:
  T coords_[3];
};

template <typename T>
inline
Vertex<T, 2>::Vertex(T x, T y) VSIP_NOTHROW
{
  coords_[0] = x;
  coords_[1] = y;
}

template <typename T>
inline T
Vertex<T, 2>::operator[] (dimension_type d) const VSIP_NOTHROW 
{
  assert(d < 2);
  return coords_[d];
}

template <typename T>
inline T&
Vertex<T, 2>::operator[] (dimension_type d) VSIP_NOTHROW 
{
  assert(d < 2);
  return coords_[d];
}

template <typename T>
inline 
Vertex<T, 3>::Vertex(T x, T y, T z) VSIP_NOTHROW
{
  coords_[0] = x;
  coords_[1] = y;
  coords_[2] = z;
}

template <typename T>
inline T
Vertex<T, 3>::operator[] (dimension_type d) const VSIP_NOTHROW 
{
  assert(d < 3);
  return coords_[d];
}

template <typename T>
inline T&
Vertex<T, 3>::operator[] (dimension_type d) VSIP_NOTHROW 
{
  assert(d < 3);
  return coords_[d];
}

/***********************************************************************
  Functions
***********************************************************************/

template <typename T>
inline bool 
operator==(Vertex<T, 1> const& i,
	   Vertex<T, 1> const& j) VSIP_NOTHROW
{
  return i.coord_ == j.coord_;
}

template <typename T>
inline bool
operator!=(Vertex<T, 1> const& i,
	   Vertex<T, 1> const& j) VSIP_NOTHROW
{
  return !operator==(i, j);
}

template <typename T>
inline bool 
operator==(Vertex<T, 2> const& i,
	   Vertex<T, 2> const& j) VSIP_NOTHROW
{
  return i.coords_[0] == j.coords_[0] && i.coords_[1] == j.coords_[1];
}

template <typename T>
inline bool 
operator!=(Vertex<T, 2> const& i,
	   Vertex<T, 2> const& j) VSIP_NOTHROW
{
  return !operator==(i, j);
}

template <typename T>
inline bool 
operator==(Vertex<T, 3> const& i,
	   Vertex<T, 3> const& j) VSIP_NOTHROW
{
  return (i.coords_[0] == j.coords_[0] &&
	  i.coords_[1] == j.coords_[1] &&
	  i.coords_[2] == j.coords_[2]);
}

template <typename T>
inline bool 
operator!=(Vertex<T, 3> const& i,
	   Vertex<T, 3> const& j) VSIP_NOTHROW
{
  return !operator==(i, j);
}

} // namespace vsip

#endif // VSIP_CORE_VERTEX_HPP
