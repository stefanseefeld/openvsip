//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_map_traits_hpp_
#define ovxx_map_traits_hpp_

namespace ovxx
{
namespace parallel
{

/// Traits class to determine if a map is serial or not.

template <typename M> struct is_local_map { static bool const value = false;};
template <typename M> struct is_global_map { static bool const value = false;};

template <typename M>
struct is_local_only
{
  static bool const value =
    is_local_map<M>::value && !is_global_map<M>::value; 
};

template <typename M>
struct is_global_only
{
  static bool const value =
    is_global_map<M>::value && !is_local_map<M>::value; 
};

template <dimension_type D, typename M>
struct is_block_dist
{ static bool const value = false;};

template <dimension_type D, typename M1, typename M2>
struct map_equal
{
  static bool value(M1 const&, M2 const&) { return false;}
};

template <dimension_type D, typename M1, typename M2>
inline bool
is_same_map(M1 const &m1, M2 const &m2)
{
  return map_equal<D, M1, M2>::value(m1, m2);
}

/// Determines whether b's map equals 'map'.
/// By default, simply call is_same_map with b.map().
/// However, for certain block types 'B' this can be implemented
/// more efficiently, making this function a useful optimization.
template <dimension_type D, typename M, typename B>
bool has_same_map(M const &map, B const &b)
{
  return is_same_map<D>(map, b.map());
};

template <typename M>
bool
processor_has_block(M const &map, processor_type proc, index_type sb)
{
  typedef typename M::processor_iterator iterator;

  for (iterator i = map.processor_begin(sb); i != map.processor_end(sb); ++i)
  {
    if (*i == proc)
      return true;
  }
  return false;
}

} // namespace ovxx::parallel
} // namespace ovxx

#endif
