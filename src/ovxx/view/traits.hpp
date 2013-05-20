//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_view_traits_hpp_
#define ovxx_view_traits_hpp_

#include <ovxx/support.hpp>
#include <ovxx/block_traits.hpp>
#include <ovxx/length.hpp>
#include <vsip/impl/view_fwd.hpp>
#include <vsip/domain.hpp>

namespace ovxx
{
template <template <typename, typename> class View,
	  typename                            T,
	  typename                            Block>
View<T, typename distributed_local_block<Block>::type>
get_local_view(View<T, Block> v);

/// Return the view extent as a domain.
template <typename T, typename Block>
inline Domain<1> view_domain(const_Vector<T, Block> const &);
template <typename T, typename Block>
inline Domain<2> view_domain(const_Matrix<T, Block> const &);
template <typename T, typename Block>
inline Domain<3> view_domain(const_Tensor<T, Block> const &);

/// Return the view extent as a Length.
template <typename T, typename Block>
inline Length<1> extent(const_Vector<T, Block> const &);
template <typename T, typename Block>
inline Length<2> extent(const_Matrix<T, Block> const &);
template <typename T, typename Block>
inline Length<3> extent(const_Tensor<T, Block> const &);

template <typename T, typename Block>
T get(const_Vector<T, Block>, Index<1> const &);
template <typename T, typename Block>
T get(const_Matrix<T, Block>, Index<2> const &);
template <typename T, typename Block>
T get(const_Tensor<T, Block>, Index<3> const &);

template <typename T, typename Block>
void put(Vector<T, Block>, Index<1> const &, T);
template <typename T, typename Block>
void put(Matrix<T, Block>, Index<2> const &, T);
template <typename T, typename Block>
void put(Tensor<T, Block>, Index<3> const &, T);

/// As per the specification, the view type can be infered from its block type.
/// This meta-function encodes that dependency by mapping the latter to the former.
template <typename B, dimension_type D = B::dim>
struct view_of;

/// Trait that provides typedef 'type' iff VIEW is a valid view.
///
/// Views should provide specializations similar to:
///    template <> struct is_view_type<Matrix>       { typedef void type; };
///    template <> struct is_view_type<const_Matrix> { typedef void type; };
template <typename> struct is_view_type { static bool const value = false;};


/// Trait that provides typedef 'type' iff VIEW is a valid const view.
template <typename> struct is_const_view_type
{ static bool const value = false;};

template <template <typename,typename> class View> struct dim_of_view;
template <> struct dim_of_view<vsip::Vector> { static const dimension_type dim = 1;};
template <> struct dim_of_view<vsip::Matrix> { static const dimension_type dim = 2;};
template <> struct dim_of_view<vsip::Tensor> { static const dimension_type dim = 3;};
template <> struct dim_of_view<vsip::const_Vector> 
{ static const dimension_type dim = 1;};
template <> struct dim_of_view<vsip::const_Matrix> 
{ static const dimension_type dim = 2;};
template <> struct dim_of_view<vsip::const_Tensor> 
{ static const dimension_type dim = 3;};

template <template <typename,typename> class View, typename Block> struct const_of_view;
template <typename B> struct const_of_view<vsip::Vector, B>
{ typedef vsip::const_Vector<typename B::value_type,B> type;};

template <typename B> struct const_of_view<vsip::Matrix, B>
{ typedef vsip::const_Matrix<typename B::value_type, B> type;};

template <typename B> struct const_of_view<vsip::Tensor, B> 
{ typedef vsip::const_Tensor<typename B::value_type, B> type;};

} // namespace ovxx

#endif
