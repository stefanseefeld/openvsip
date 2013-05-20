//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_view_utils_hpp_
#define ovxx_view_utils_hpp_

#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
//#include <ovxx/parallel/services.hpp>
#include <ovxx/ct_assert.hpp>
#include <ovxx/assign_local.hpp>

namespace ovxx
{
// Create a new view of type 'V' that has the same dimensions as the
// existing view.
template <typename V, typename T, typename B>
V
clone_view(const_Vector<T, B> view)
{
  V ret(view.size(0));
  return ret;
}

template <typename V, typename T, typename B>
V
clone_view(const_Matrix<T, B> view)
{
  V ret(view.size(0), view.size(1));
  return ret;
}

template <typename V, typename T, typename B>
V
clone_view(const_Tensor<T, B> view)
{
  V ret(view.size(0), view.size(1), view.size(2));
  return ret;
}

template <typename V, typename T, typename B, typename M>
V
clone_view(const_Vector<T, B> view, M const &map)
{
  V ret(view.size(0), map);
  return ret;
}

template <typename V, typename T, typename B, typename M>
V
clone_view(const_Matrix<T, B> view, M const &map)
{
  V ret(view.size(0), view.size(1), map);
  return ret;
}

template <typename V, typename T, typename B, typename M>
V
clone_view(const_Tensor<T, B> view, M const &map)
{
  V ret(view.size(0), view.size(1), view.size(2), map);
  return ret;
}

template <typename V,
	  typename M = typename V::block_type::map_type>
struct as_local_view
{
  static bool const is_copy = true;
  static dimension_type const dim = V::dim;

  typedef typename V::value_type value_type;
  typedef typename V::block_type block_type;
  typedef typename get_block_layout<block_type>::order_type order_type;

  typedef Dense<dim, value_type, order_type, Local_map> r_block_type;

  typedef typename 
    conditional<is_const_view_type<V>::value,
      typename view_of<r_block_type>::type,
      typename view_of<r_block_type>::const_type
      >::type type;

  static type exec(V view)
  {
    // The internal view needs to be non-const, even if the function
    // return type is const.
    typedef typename view_of<r_block_type>::type view_type;

    view_type ret(clone_view<view_type>(view));
    assign_local(ret, view);
    return ret;
  }
};

template <typename V>
struct as_local_view<V, Local_map>
{
  static bool const is_copy = false;
  typedef V type;

  static type exec(V view) { return view;}
};

template <typename V, dimension_type D>
struct as_local_view<V, Replicated_map<D> >
{
  static bool const is_copy = false;
  typedef typename V::local_type type;

  static type exec(V view) { return view.local();}
};

template <template <typename, typename> class V,
	  typename T, typename B>
typename as_local_view<V<T, B> >::type
convert_to_local(V<T, B> view)
{
  return as_local_view<V<T, B> >::exec(view);
}

template <typename V>
V
create_view(Domain<1> const &dom) { return V(dom.size());} 

template <typename V>
V
create_view(Domain<1> const &dom,
	    typename V::block_type::map_type const& map)
{ return V(dom.size(), map);} 

template <typename V>
V
create_view(Domain<1> const &dom,
	    typename V::value_type v,
	    typename V::block_type::map_type const& map)
{ return V(dom.size(), v, map);} 

template <typename V>
V
create_view(Domain<2> const &dom)
{ return V(dom[0].size(), dom[1].size());}

template <typename V>
V
create_view(Domain<2> const &dom,
	    typename V::block_type::map_type const& map)
{ return V(dom[0].size(), dom[1].size(), map);}

template <typename V>
V
create_view(Domain<2> const &dom,
	    typename V::value_type v,
	    typename V::block_type::map_type const& map)
{ return V(dom[0].size(), dom[1].size(), v, map);}

template <typename V>
V
create_view(Domain<3> const &dom)
{ return V(dom[0].size(), dom[1].size(), dom[2].size());}

template <typename V>
V
create_view(Domain<3> const &dom,
	    typename V::block_type::map_type const& map)
{ return V(dom[0].size(), dom[1].size(), dom[2].size(), map);}

template <typename V>
V
create_view(Domain<3> const &dom,
	    typename V::value_type v,
	    typename V::block_type::map_type const& map)
{ return V(dom[0].size(), dom[1].size(), dom[2].size(), v, map);}

} // namespace ovxx

#endif
