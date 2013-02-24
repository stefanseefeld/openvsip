/* Copyright (c) 2005 by CodeSourcery, LLC.  All rights reserved. */

/** @file    vsip/core/type_list.hpp
    @author  Stefan Seefeld
    @date    2005-08-05
    @brief   VSIPL++ Library: Type list - related templates.

*/

#ifndef VSIP_CORE_TYPE_LIST_HPP
#define VSIP_CORE_TYPE_LIST_HPP

namespace vsip
{
namespace impl
{

struct None_type;

template <typename First, typename Rest>
struct Type_list
{
  typedef First first;
  typedef Rest rest;
};

template <typename T1 = None_type,
	  typename T2 = None_type,
	  typename T3 = None_type,
	  typename T4 = None_type,
	  typename T5 = None_type,
	  typename T6 = None_type,
	  typename T7 = None_type,
	  typename T8 = None_type,
	  typename T9 = None_type,
	  typename T10 = None_type,
	  typename T11 = None_type,
	  typename T12 = None_type,
	  typename T13 = None_type,
	  typename T14 = None_type,
	  typename T15 = None_type,
	  typename T16 = None_type,
	  typename T17 = None_type>
struct Make_type_list
{
private:
  typedef typename 
  Make_type_list<T2, T3, T4, T5, T6, T7, T8, T9, T10, T11, T12, T13, T14, T15, T16, T17>::type Rest;
public:
  typedef Type_list<T1, Rest> type;
};

template<>
struct Make_type_list<>
{
  typedef None_type type;
};

} // namespace vsip::impl
} // namespace vsip

#endif
