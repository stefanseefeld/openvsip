//
// Copyright (c) 2007 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Tests for ramp function

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/map.hpp>
#include <vsip/vector.hpp>
#include <vsip/selgen.hpp>
#include <vsip_csl/test.hpp>


using namespace vsip;
using namespace vsip::impl;

#define DEBUG 0

#if DEBUG==1
#include <vsip_csl/output.hpp>
using namespace vsip_csl;
#endif

template <int test_num>
struct do_test;


// declare all tests here

// TEST1: A simple assignment, view = ramp
template <>
struct do_test<1>
{
  template <typename ViewT>
  static void exec(ViewT& view, length_type size) 
    { typedef typename ViewT::value_type T; view = ramp(T(0),T(1),size); }

  template <typename ViewT, typename T>
  static int check(ViewT& view, Index<ViewT::dim> const& idx, T& val)
    { return (get(view,idx) == val++); }
};

// TEST2: A more complex assignment, view += ramp
template <>
struct do_test<2>
{
  template <typename ViewT>
  static void exec(ViewT& view, length_type size) 
    { 
      typedef typename ViewT::value_type T; 
      ViewT vector1(size);
      vector1 = T(1);
      view = vector1;
      view += ramp(T(0),T(1),size); 
    }

  template <typename ViewT, typename T>
  static int check(ViewT& view, Index<ViewT::dim> const& idx, T& val)
    { return (get(view,idx) == T(1)+val++); }
};

// NOTE: These tests require ViewT to be a distributed vector

// TEST3: A ramp assignment to a map on 1 proc
template <>
struct do_test<3>
{
  template <typename ViewT>
  static void exec(ViewT& view, length_type size) 
    { 
      typedef typename ViewT::value_type T;
      Vector<processor_type> proc_set(1); proc_set(0) = 0;
      Map<> map_root(proc_set,1);
      Vector<T, Dense<1,T,row1_type,Map<> > > vector(size,map_root);
      vector = ramp(T(0),T(1),size); 
      view = vector;
    }

  template <typename ViewT, typename T>
  static int check(ViewT& view, Index<ViewT::dim> const& idx, T& val)
    { return (get(view,idx) == val++); }
};

// TEST4: A ramp += to a map on 1 proc
template <>
struct do_test<4>
{
  template <typename ViewT>
  static void exec(ViewT& view, length_type size) 
    { 
      typedef typename ViewT::value_type T;
      Vector<processor_type> proc_set(1); proc_set(0) = 0;
      Map<> map_root(proc_set,1);
      Vector<T, Dense<1,T,row1_type,Map<> > > vector(size,map_root);
      vector = T(1);
      vector += ramp(T(0),T(1),size); 
      view = vector;
    }

  template <typename ViewT, typename T>
  static int check(ViewT& view, Index<ViewT::dim> const& idx, T& val)
    { return (get(view,idx) == T(1)+val++); }
};

// TEST5: A ramp + to a map on all proc assigned to a map on one proc
template <>
struct do_test<5>
{
  template <typename ViewT>
  static void exec(ViewT& view, length_type size) 
    { 
      typedef typename ViewT::value_type T;
      Vector<processor_type> proc_set(1); proc_set(0) = 0;
      Map<> map_root(proc_set,1);
      Vector<T, Dense<1,T,row1_type,Map<> > > vector_root(size,map_root);
      Map<> map_all(num_processors());
      Vector<T, Dense<1,T,row1_type,Map<> > > vector_all(size,map_all);

      vector_all  = T(2);
      vector_root = ramp(T(0),T(1),size) + vector_all;
      view = vector_root;
    }

  template <typename ViewT, typename T>
  static int check(ViewT& view, Index<ViewT::dim> const& idx, T& val)
    { return (get(view,idx) == T(2)+val++); }
};

// TEST6: A ramp + to a map on one proc assigned to a map on all proc
template <>
struct do_test<6>
{
  template <typename ViewT>
  static void exec(ViewT& view, length_type size) 
    { 
      typedef typename ViewT::value_type T;
      Vector<processor_type> proc_set(1); proc_set(0) = 0;
      Map<> map_root(proc_set,1);
      Vector<T, Dense<1,T,row1_type,Map<> > > vector_root(size,map_root);
      Map<> map_all(num_processors());
      Vector<T, Dense<1,T,row1_type,Map<> > > vector_all(size,map_all);

      vector_root  = T(2);
      vector_all = ramp(T(0),T(1),size) + vector_root;
      view = vector_all;
    }

  template <typename ViewT, typename T>
  static int check(ViewT& view, Index<ViewT::dim> const& idx, T& val)
    { return (get(view,idx) == T(2)+val++); }
};


template <typename MapT,
          dimension_type Dim = 1>
struct Create_map;

template <>
struct Create_map<Local_map>
{
  static Local_map exec() { return Local_map(); }
};

template <>
struct Create_map<Map<> >
{
  static Map<>  exec() { return Map<>(num_processors()); }
};

template <dimension_type dim>
struct Create_map<Replicated_map<dim> >
{
  static Replicated_map<dim>  exec() { return Replicated_map<dim>(); }
};


template <int test_num,
          typename ViewT,
	  typename MapT>
int test_vramp(Domain<ViewT::dim> sz)
{

  const dimension_type                                    dim = ViewT::dim;
  typedef typename ViewT::value_type                      T;
  typedef Dense<dim,T,typename Row_major<dim>::type,MapT> block_type;
  typedef typename view_of<block_type>::type  view_type;

  test_assert(dim == 1); // ramp only works for vectors

  // create view
  MapT                map = Create_map<MapT>::exec();
  block_type          block(sz,map);
  view_type           view(block);

  // assign to a ramp
  do_test<test_num>::exec(view, sz.size());

#if DEBUG == 1
  std::cout << "View of test "<<test_num<<"\n";
  std::cout << view;
#endif

  // check results
  {
    Index<dim> idx;
    Length<dim> ext = extent(view);
    T val = T(0);
    for(;valid(ext,idx);next(ext,idx)) {
      test_assert(do_test<test_num>::check(view,idx,val));
    }
  }

  return 0;
}


int main(int argc, char **argv)
{
  length_type size=16;

  vsipl vpp(argc,argv);

  test_vramp<1,Vector<float>, Local_map>     (Domain<1>(size));
  test_vramp<1,Vector<float>, Map<> >        (Domain<1>(size));
  test_vramp<1,Vector<float>, Replicated_map<1> >(Domain<1>(size));
  
  test_vramp<2,Vector<float>, Local_map>     (Domain<1>(size));
  test_vramp<2,Vector<float>, Map<> >        (Domain<1>(size));
  test_vramp<2,Vector<float>, Replicated_map<1> >(Domain<1>(size));

  test_vramp<3,Vector<float>, Map<> >        (Domain<1>(size));
  test_vramp<4,Vector<float>, Map<> >        (Domain<1>(size));
  test_vramp<5,Vector<float>, Map<> >        (Domain<1>(size));
  test_vramp<6,Vector<float>, Map<> >        (Domain<1>(size));

  return 0;
}
