/* Copyright (c) 2007 by CodeSourcery.  All rights reserved. */

/** @file    vsip/opt/reductions/par_reductions.hpp
    @author  Assem Salama
    @date    2007-03-14
    @brief   VSIPL++ Library: Parallel index reduction functions.
	     [math.fns.reductidx].
*/

#ifndef VSIP_OPT_REDUCTIONS_PAR_REDUCTIONS_HPP
#define VSIP_OPT_REDUCTIONS_PAR_REDUCTIONS_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/tensor.hpp>



/***********************************************************************
  Definitions
***********************************************************************/

namespace vsip_csl
{
namespace dispatcher
{

template <template <typename> class ReduceT>
struct ReduceOp { typedef op::reduce_idx<ReduceT> reduce_op; };

template <>
struct ReduceOp<impl::Min_magsq_value>
{ typedef op::reduce_idx<impl::Min_value> reduce_op; };
template <>
struct ReduceOp<impl::Max_magsq_value>
{ typedef op::reduce_idx<impl::Max_value> reduce_op; };


template<template <typename> class ReduceT>
struct List<ReduceOp<ReduceT> >
{
  typedef Make_type_list<be::cvsip, be::mercury_sal, be::generic>::type type;
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

namespace vsip
{
namespace impl
{

template <typename                  ReduceT,
          typename                  T,
          dimension_type            Dim,
	  typename                  Block>
inline T
reduce_idx_blk(Block const& b, Index<Dim>& idx)
{
  using namespace vsip_csl::dispatcher;
  T r;
  typedef typename get_block_layout<Block>::order_type order_type;
  typedef Make_type_list<be::cvsip,
                         be::mercury_sal,
                         be::generic>::type list_type;

  Dispatcher<ReduceT,
             void(T&, Block const&, Index<Dim>&, order_type),
             list_type>::
    dispatch(r, b, idx, order_type());

  return r;
}


template<template <typename> class ReduceT,
         typename T,
         typename Block, dimension_type Dim>
inline T
generic_par_idx_op(Block& a, Index<Dim>& idx)
{
  using namespace vsip_csl::dispatcher;

  typedef Map<Block_dist>                          map_type;
  typedef Dense<1,T,row1_type,Map<> >              block_type;
  typedef Dense<1,Index<Dim>,row1_type,Map<> >     block_idx_type;
  typedef Dense<1,T,row1_type,Replicated_map<1> >  g_block_type;
  typedef Dense<1,Index<Dim>,row1_type, Replicated_map<1> > g_block_idx_type;
  typedef Vector<T,block_type>                     vect_type;
  typedef Vector<Index<Dim>,block_idx_type>        vect_idx_type;

  Index<Dim>                                       my_res_idx;
  Index<Dim>                                       my_g_res_idx;
  Index<1>                                         global_res_idx;
  T                                                global_res;

  Vector<processor_type> a_proc_set       = a.map().processor_set();

  // We will make two vectors, results, and results_idx. Results will hold
  // the result of the reduction of each processor. The reults_idx will
  // hold the index of the reduction of each processor.
  Map<>                                   map(a_proc_set,a_proc_set.size());
  vect_type                               results(a_proc_set.size(),map);
  vect_idx_type                           results_idx(a_proc_set.size(),map);

  if(a.map().subblock() != no_subblock) 
  {
    typename ReduceT<T>::result_type result = 
      reduce_idx_blk<op::reduce_idx<ReduceT>,typename ReduceT<T>::result_type>
        (get_local_block(a),my_res_idx);
    results.local().put(0,result);
    my_g_res_idx = global_from_local_index_blk(a,my_res_idx);
    results_idx.local().put(0,my_g_res_idx);
  }


  // Ok, now, perform the same reduction on the local results

  // first, make a vector with a global map that contains all the results
  Vector<T,g_block_type>                  global_results(a_proc_set.size());
  Vector<Index<Dim>,g_block_idx_type>     global_results_idx(a_proc_set.size());
  // do broadcast
  global_results     = results;
  global_results_idx = results_idx;

  global_res = reduce_idx_blk<typename ReduceOp<ReduceT>::reduce_op,
                              typename ReduceT<T>::result_type>
      (global_results.block(),global_res_idx);
  idx = global_results_idx.get(global_res_idx[0]);

  return global_res;

}

template <template <typename> class ReduceT,
          dimension_type            Dim,
          typename                  OrderT>
struct Reduction_idx_supported
{ static bool const value = false; };

// All suppored reductions here
template <dimension_type Dim, typename OrderT>
struct Reduction_idx_supported<Max_value, Dim, OrderT>
{ static bool const value = true; };

template <dimension_type Dim, typename OrderT>
struct Reduction_idx_supported<Min_value, Dim, OrderT>
{ static bool const value = true; };

template <dimension_type Dim, typename OrderT>
struct Reduction_idx_supported<Max_mag_value, Dim, OrderT>
{ static bool const value = true; };

template <dimension_type Dim, typename OrderT>
struct Reduction_idx_supported<Min_mag_value, Dim, OrderT>
{ static bool const value = true; };

template <dimension_type Dim, typename OrderT>
struct Reduction_idx_supported<Max_magsq_value, Dim, OrderT>
{ static bool const value = true; };

template <dimension_type Dim, typename OrderT>
struct Reduction_idx_supported<Min_magsq_value, Dim, OrderT>
{ static bool const value = true; };

} // namespace vsip::impl
} // namespace vsip

/**********************************************************************
* Parallel evaluator for index returning reductions
**********************************************************************/

namespace vsip_csl
{
namespace dispatcher
{

template <template <typename> class ReduceT,
          typename                  T,
	  typename                  Block,
	  typename                  OrderT,
	  dimension_type            Dim >
struct Evaluator<op::reduce_idx<ReduceT>, be::parallel,
                 void(T&, Block const&, Index<Dim>&, OrderT)>
{
  static bool const ct_valid = 
    !impl::Is_local_map<typename Block::map_type>::value &&
    impl::Reduction_idx_supported<ReduceT,
				  Dim,
				  OrderT>::value;

  static bool rt_valid(T&, Block const&, Index<Dim>&, OrderT)
  { return true; };

  static void exec(T& r, Block const& a, Index<Dim>& idx, OrderT)
  {
    r = impl::generic_par_idx_op<ReduceT,T>(a, idx);
  }
};

} // namespace vsip_csl::dispatcher
} // namespace vsip_csl

#endif // VSIP_OPT_REDUCTIONS_PAR_REDUCTIONS_HPP
