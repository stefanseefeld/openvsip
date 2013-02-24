/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    benchmarks/maxval.cpp
    @author  Jules Bergmann
    @date    2006-06-01
    @brief   VSIPL++ Library: Benchmark for maxval reductions.

*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/random.hpp>
#include <vsip/selgen.hpp>
#include <vsip/opt/profile.hpp>
#include <vsip/core/metaprogramming.hpp>
#include <vsip/selgen.hpp>

#include <vsip_csl/test.hpp>
#include "loop.hpp"

using namespace vsip;
using vsip_csl::equal;

// Make a structure to run maxval based on tag.
template <template <typename> class ReduceT,
	  typename T,
          typename Block,
	  vsip::dimension_type dim,
          typename Tag>
struct reduction_op_eval
{

  typedef typename vsip::impl::Block_layout<Block>::order_type order_type;
  typedef vsip_csl::dispatcher::Evaluator<
    vsip_csl::dispatcher::op::reduce_idx<ReduceT>, Tag,
    void(typename ReduceT<T>::result_type&,
        Block const&,
        vsip::Index<dim>&,
        order_type)> evaluator;

  static void exec(T& r, Block const& a, Index<dim>& idx)
  {
    evaluator::exec(r,a,idx,order_type());
  }
};

// structure to help us create test vectors
template <typename MapT,
          template <typename,typename> class ViewT,
	  typename T,
	  typename Block>
struct create_test_vector_helper {};

// override for when destination is a local view
template<template <typename,typename> class ViewT,
         typename T,
	 typename Block>
struct create_test_vector_helper<vsip::Local_map,ViewT,T,Block>
{
  typedef ViewT<T,Block> dst_view;

  // because this is a local to local, we do a normal assign
  template <typename ViewT1>
  void assign_view(ViewT1 sv)
  { view=sv; };

  // the constructor is very simple too
  create_test_vector_helper(vsip::length_type size) : view(size) {};

  dst_view view;


};

// override for when destination is a distributed view
template<template <typename,typename> class ViewT,
         typename T,
	 typename Block>
struct create_test_vector_helper<vsip::Map<>,ViewT,T,Block>
{
  static vsip::dimension_type const dim = ViewT<T,Block>::dim;
  typedef vsip::Dense<dim,T,typename vsip::impl::Row_major<dim>::type,
    vsip::Map<> > dst_block;
  typedef ViewT<T,dst_block>                 dst_view;

  template <typename ViewT1>
  void assign_view(ViewT1 sv)
  {
    // processor 0 will distribute data to all other procs
    vsip::Vector<vsip::processor_type> pvec_in(1);pvec_in(0)=(0);
    vsip::Map<>                  root_map(pvec_in);
    dst_block                    root_block(sv.size(),root_map);
    dst_view                     root_view(root_block);

    // Ok, now move the vector to the distributed view
    vsip::impl::assign_local(root_view,sv);
    view = root_view;

  };

  create_test_vector_helper(vsip::length_type size) :
    view(size, vsip::Map<>(vsip::num_processors())) {};

  dst_view view;

};


template <typename T,
          typename MapT = Local_map>
struct t_maxval1
{
  char const* what() { return "t_maxval_vector"; }
  int ops_per_point(length_type)  { return 1; }
  int riob_per_point(length_type) { return sizeof(T); }
  int wiob_per_point(length_type) { return 0; }
  int mem_per_point(length_type)  { return 1*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    using namespace vsip::impl;

    typedef Dense<1,T,row1_type,MapT>                               block_type;

    create_test_vector_helper<MapT,Vector,float,Dense<1,T,row1_type,MapT> >
      ctvh(size);

    T                      val = T();
    Index<1>               idx;

    Rand<T>     gen(0, 0);

    if (init_ == 0)
      ctvh.assign_view(gen.randu(size));
    else if (init_ == 1)
      ctvh.assign_view(ramp(T(0), T(1), size));
    else if (init_ == 2)
      ctvh.assign_view(ramp(T(size-1), T(-1), size));
   
    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      val = maxval(ctvh.view, idx);
    t1.stop();

    if (init_ == 1)
    {
      test_assert(equal(val, T(size-1)));
      test_assert(idx == size-1);
    }
    else if (init_ == 2)
    {
      test_assert(equal(val, T(size-1)));
      test_assert(idx == 0);
    }
    
    time = t1.delta();
  }

  void diag()
  {
  }

  t_maxval1(int init) : init_(init) {}

  int init_;
};

template <typename T,
          typename MapT = Local_map,
	  typename Tag = vsip_csl::dispatcher::be::cvsip>
struct t_maxval2
{
  char const* what() { return "t_maxval_vector"; }
  int ops_per_point(length_type)  { return 1; }
  int riob_per_point(length_type) { return sizeof(T); }
  int wiob_per_point(length_type) { return 0; }
  int mem_per_point(length_type)  { return 1*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    using namespace vsip::impl;

    typedef Dense<1,T,row1_type,MapT> block_type;
    typedef reduction_op_eval<Max_value,T,block_type,1,Tag> eval;

    create_test_vector_helper<MapT,Vector,float,Dense<1,T,row1_type,MapT> >
      ctvh(size);

    T                      val = T();
    Index<1>               idx;

    Rand<T>     gen(0, 0);

    if (init_ == 0)
      ctvh.assign_view(gen.randu(size));
    else if (init_ == 1)
      ctvh.assign_view(ramp(T(0), T(1), size));
    else if (init_ == 2)
      ctvh.assign_view(ramp(T(size-1), T(-1), size));
   
    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      eval::exec(val, ctvh.view.block(), idx);
    t1.stop();

    if (init_ == 1)
    {
      test_assert(equal(val, T(size-1)));
      test_assert(idx == size-1);
    }
    else if (init_ == 2)
    {
      test_assert(equal(val, T(size-1)));
      test_assert(idx == 0);
    }
    
    time = t1.delta();
  }

  void diag()
  {
  }

  t_maxval2(int init) : init_(init) {}

  int init_;
};



void
defaults(Loop1P&)
{
}



int
test(Loop1P& loop, int what)
{
  using namespace vsip_csl::dispatcher;

  switch (what)
  {
  case  1: loop(t_maxval1<float>(0)); break;
  case  2: loop(t_maxval1<float>(1)); break;
  case  3: loop(t_maxval1<float>(2)); break;
  case  4: loop(t_maxval2<float,Map<>,be::parallel>(0)); break;
  case  5: loop(t_maxval2<float,Map<>,be::parallel>(1)); break;
  case  6: loop(t_maxval2<float,Map<>,be::parallel>(2)); break;
  case  7: loop(t_maxval2<float,Map<>,be::generic>(0)); break;
  case  8: loop(t_maxval2<float,Map<>,be::generic>(1)); break;
  case  9: loop(t_maxval2<float,Map<>,be::generic>(2)); break;

  case  0:
    std::cout
      << "maxval -- find maximum value, and its index, in a vector\n"
      << "  -1 -- local vector -- random numbers\n"
      << "  -2 -- local vector -- forward ramp\n"
      << "  -3 -- local vector -- reverse ramp\n"
      << "  -4 -- mapped vector, parallel backend -- random numbers\n"
      << "  -5 -- mapped vector, parallel backend -- forward ramp\n"
      << "  -6 -- mapped vector, parallel backend -- reverse ramp\n"
      << "  -7 -- mapped vector,  generic backend -- random numbers\n"
      << "  -8 -- mapped vector,  generic backend -- forward ramp\n"
      << "  -9 -- mapped vector,  generic backend -- reverse ramp\n"
      ;
  default: return 0;
  }
  return 1;
}
