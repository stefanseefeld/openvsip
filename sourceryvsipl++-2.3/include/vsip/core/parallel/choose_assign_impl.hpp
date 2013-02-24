/* Copyright (c) 2006 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/parallel/choose_assign_impl.hpp
    @author  Jules Bergmann
    @date    2006-08-29
    @brief   VSIPL++ Library: Choose Par_assign impl tag.

*/

#ifndef VSIP_CORE_PARALLEL_CHOOSE_ASSIGN_IMPL_HPP
#define VSIP_CORE_PARALLEL_CHOOSE_ASSIGN_IMPL_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/metaprogramming.hpp>
#include <vsip/core/block_traits.hpp>
#include <vsip/core/parallel/map_traits.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

namespace impl
{

// Only valid if Block1 and Block2 are simple distributed blocks.

#if VSIP_IMPL_PAR_SERVICE == 0 || VSIP_IMPL_PAR_SERVICE == 1
// MPI
template <dimension_type Dim,
	  typename       Block1,
	  typename       Block2,
	  bool           EarlyBinding>
struct Choose_par_assign_impl
{
  typedef typename Block1::map_type map1_type;
  typedef typename Block2::map_type map2_type;

  static int const  is_blkvec     = (Dim == 1) &&
                                    Is_block_dist<0, map1_type>::value &&
                                    Is_block_dist<0, map2_type>::value;

  typedef typename
  ITE_Type<is_blkvec, As_type<Blkvec_assign>, As_type<Chained_assign> >
	::type type;
};
#else
// PAS
template <dimension_type Dim,
	  typename       Block1,
	  typename       Block2,
	  bool           EarlyBinding>
struct Choose_par_assign_impl
{
  static int const  is_pas_assign = Is_pas_block<Block1>::value &&
                                    Is_pas_block<Block2>::value;

  typedef typename
  ITE_Type<is_pas_assign, 
	   ITE_Type<EarlyBinding, As_type<Pas_assign_eb>,
		                  As_type<Pas_assign> >,
	   As_type<Direct_pas_assign>
          >::type type;
};
#endif

} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_IMPL_CHOOSE_PAR_ASSIGN_IMPL_HPP
