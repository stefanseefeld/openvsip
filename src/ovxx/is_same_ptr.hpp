//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_is_same_ptr_hpp_
#define ovxx_is_same_ptr_hpp_


namespace ovxx
{
namespace detail
{

template <typename P1, typename P2>
struct is_same_ptr
{
  static bool compare(P1, P2) { return false;}
};

template <typename P>
struct is_same_ptr<P, P>
{
  static bool compare(P p1, P p2) { return p1 == p2;}
};

} // namespace ovxx::detail

template <typename P1, typename P2>
inline bool is_same_ptr(P1 *p1, P2 *p2)
{
  typedef typename add_const<P1>::type c_p1_type;
  typedef typename add_const<P2>::type c_p2_type;
  return detail::is_same_ptr<c_p1_type*, c_p2_type*>::compare(p1, p2);
}

template <typename P1, typename P2>
inline bool
is_same_ptr(std::pair<P1*, P1*> const &p1,
	    std::pair<P2*, P2*> const &p2)
{
  return 
    is_same_ptr(p1.first, p2.first) &&
    is_same_ptr(p1.second, p2.second);
}

} // namespace ovxx

#endif
