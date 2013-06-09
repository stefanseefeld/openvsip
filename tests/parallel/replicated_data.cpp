//
// Copyright (c) 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Unit tests for using Replicated_map.

#define VERBOSE 0

#include <vsip/support.hpp>
#include <vsip/map.hpp>
#include <vsip/initfin.hpp>
#include <test.hpp>
#include "util.hpp"
#include <stdio.h>
#include <unistd.h>

using namespace ovxx;

// Test comm using replicated view as source (to non-replicated view)

template <typename T>
void
test_src(int modulo)
{
  // rep_view will be replicated over a subset of processors.
  // dst_view will dim 0 distributed so that each processor has a row.
 
  length_type np = num_processors();

  Vector<processor_type> full_pset = processor_set();
  Vector<processor_type> pset(1+(np-1)/modulo);
  for (index_type i=0; i<np; i+=modulo)
    pset(i/modulo) = full_pset(i);

  length_type size = 16;
  length_type rows = np;

  Replicated_map<1> rep_map(pset);
  Map<>             dst_map(np, 1);

  typedef Dense<1, T, row1_type, Replicated_map<1> > rep_block_type;
  typedef Dense<2, T, row2_type, Map<> >             dst_block_type;

  typedef Vector<T, rep_block_type> rep_view_type;
  typedef Matrix<T, dst_block_type> dst_view_type;
  index_type rank = parallel::default_communicator().rank();
  index_type sb = dst_map.subblock(rank);

  rep_view_type rep_view(size, rep_map);
  dst_view_type dst_view(rows, size, dst_map);

  for (index_type i=0; i<size; ++i)
    rep_view.put(i, T(i));

  if (rep_map.subblock() != no_subblock)
  {
    rep_view.local().put(0, T(local_processor()));
  }

  dump_view("dst_view", dst_view);
  for (index_type r=0; r<rows; ++r)
    dst_view.row(r) = rep_view;

  // Check the results
  typename dst_view_type::local_type l_dst_view = dst_view.local();
  test_assert(l_dst_view.size(0) == 1);

  // dump_view("rep_view", rep_view);
  dump_view("dst_view", dst_view);

  for (index_type i=1; i<size; ++i)
    test_assert(l_dst_view.get(0, i) == T(i));

#if VERBOSE
  std::cout << "(" << local_processor() << "): " << l_dst_view.get(0, 0)
	    << std::endl;
#endif
}



// Test comm using replicated view as destination (from non-replicated view)

template <typename T,
	  typename SrcMapT,
	  typename DstMapT>
void
test_msg(length_type    size,
	 SrcMapT const& src_map,
	 DstMapT const& dst_map,
	 bool           mark_first_element = true)
{
  typedef Dense<1, T, row1_type, SrcMapT> src_block_type;
  typedef Dense<1, T, row1_type, DstMapT> dst_block_type;

  typedef Vector<T, src_block_type> src_view_type;
  typedef Vector<T, dst_block_type> dst_view_type;

  src_view_type src_view(size, src_map);
  dst_view_type dst_view(size, dst_map);
  for (index_type i=0; i<size; ++i)
    src_view.put(i, T(i));

  if (mark_first_element && src_map.subblock() != no_subblock)
  {
    src_view.local().put(0, T(local_processor()));
  }

  dst_view = src_view;

  // Check the results
  if (dst_map.subblock() != no_subblock)
  {
    typename dst_view_type::local_type l_dst_view = dst_view.local();

    for (index_type li=1; li<l_dst_view.size(); ++li)
    {
      index_type gi = global_from_local_index(dst_view, 0, li);
      test_assert(l_dst_view.get(li) == T(gi));
    }

#if VERBOSE
    std::cout << "(" << local_processor() << "): " << l_dst_view.get(0)
	      << std::endl;
#endif
  }
}



template <typename T>
void
test_global_to_repl(int modulo)
{
  length_type np = num_processors();

  Vector<processor_type> full_pset = processor_set();
  Vector<processor_type> pset(1+(np-1)/modulo);

  for (index_type i=0; i<np; i+=modulo)
    pset(i/modulo) = full_pset(i);

  length_type size = 16;

  // src has replica on all processors...
  Replicated_map<1> src_map;
  // ...while dst does not.
  Replicated_map<1> dst_map(pset);

  test_msg<T>(size, src_map, dst_map);
}



template <typename T>
void
test_repl_to_global(int modulo)
{
  length_type np = num_processors();

  Vector<processor_type> full_pset = processor_set();
  Vector<processor_type> pset(1+(np-1)/modulo);

  for (index_type i=0; i<np; i+=modulo)
    pset(i/modulo) = full_pset(i);

  length_type size = 16;

  // src has replica only on some processors...
  Replicated_map<1> src_map(pset);
  // ...while dst has replica on all.
  Replicated_map<1> dst_map;

  test_msg<T>(size, src_map, dst_map);
}



template <typename T>
void
test_repl_to_repl(int src_modulo, int dst_modulo)
{
  length_type np = num_processors();

  Vector<processor_type> full_pset = processor_set();
  Vector<processor_type> src_pset(1+(np-1)/src_modulo);
  Vector<processor_type> dst_pset(1+(np-1)/dst_modulo);

  for (index_type i=0; i<np; i+=src_modulo)
    src_pset(i/src_modulo) = full_pset(i);

  for (index_type i=0; i<np; i+=dst_modulo)
    dst_pset(i/dst_modulo) = full_pset(i);

  length_type size = 16;

  Replicated_map<1> src_map(src_pset);
  Replicated_map<1> dst_map(dst_pset);

  test_msg<T>(size, src_map, dst_map);
}



template <typename T>
void
test_even_odd()
{
  length_type np = num_processors();

  Vector<processor_type> full_pset = processor_set();
  Vector<processor_type> src_pset(std::max<length_type>(np/2 + np%2, 1));
  Vector<processor_type> dst_pset(std::max<length_type>(np/2, 1));

  for (index_type i=0; i<np; ++i)
    if (i%2 == 0)
      src_pset(i/2) = full_pset(i);
    else
      dst_pset(i/2) = full_pset(i);

  if (np == 1)
    dst_pset(0) = full_pset(0);

  length_type size = 16;

  Replicated_map<1> src_map(src_pset);
  Replicated_map<1> dst_map(dst_pset);

  test_msg<T>(size, src_map, dst_map);
}



template <typename T>
void
test_block_to_repl(int modulo)
{
  length_type np = num_processors();

  Vector<processor_type> full_pset = processor_set();
  Vector<processor_type> pset(1+(np-1)/modulo);

  for (index_type i=0; i<np; i+=modulo)
    pset(i/modulo) = full_pset(i);

  length_type size = 16;

  Map<>             src_map(np);
  Replicated_map<1> dst_map(pset);

  test_msg<T>(size, src_map, dst_map, false);
}



template <typename T>
void
test_repl_to_block(int modulo)
{
  length_type np = num_processors();

  Vector<processor_type> full_pset = processor_set();
  Vector<processor_type> pset(1+(np-1)/modulo);

  for (index_type i=0; i<np; i+=modulo)
    pset(i/modulo) = full_pset(i);

  length_type size = 16;

  Replicated_map<1> src_map(pset);
  Map<>             dst_map(np);

  test_msg<T>(size, src_map, dst_map, false);
}



int
main(int argc, char** argv)
{
  vsipl vpp(argc, argv);

  // debug_stub();

  test_src<float>(1);
  test_src<float>(2);
  test_src<float>(4);

  test_global_to_repl<float>(1);
  test_global_to_repl<float>(2);
  test_global_to_repl<float>(4);

  test_repl_to_global<float>(1);
  test_repl_to_global<float>(2);
  test_repl_to_global<float>(4);

  test_repl_to_repl<float>(1, 1);
  test_repl_to_repl<float>(1, 2);
  test_repl_to_repl<float>(1, 4);

  test_repl_to_repl<float>(2, 1);
  test_repl_to_repl<float>(4, 1);

  test_repl_to_repl<float>(2, 2);
  test_repl_to_repl<float>(4, 2);
  test_repl_to_repl<float>(2, 4);

  test_even_odd<float>();

  test_block_to_repl<float>(1);
  test_block_to_repl<float>(2);
  test_block_to_repl<float>(4);

  test_repl_to_block<float>(1);
  test_repl_to_block<float>(2);
  test_repl_to_block<float>(4);
}
