//
// Copyright (c) 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <vsip/support.hpp>
#include <vsip/map.hpp>
#include <vsip/initfin.hpp>

#include <vsip_csl/test.hpp>

using namespace std;
using namespace vsip;
using namespace vsip_csl;


/***********************************************************************
  Definitions
***********************************************************************/

template <dimension_type Dim,
	  typename       Block>
void
check_replicated_map(
  Replicated_map<Dim> const&          map,
  const_Vector<processor_type, Block> pset)
{
  typedef Replicated_map<Dim> map_type;
  typedef typename map_type::processor_iterator iterator;

  // Check num_processors()
  test_assert(map.num_processors() == pset.size());

  // Check processor_set()
  Vector<processor_type> map_pset = map.processor_set();

  test_assert(map_pset.size() == pset.size());
  for (index_type i=0; i<map_pset.size(); ++i)
    test_assert(map_pset(i) == pset(i));

  // Check processor_begin(), processor_end()
  iterator begin = map.processor_begin(0);
  iterator end   = map.processor_end(0);

  assert(static_cast<length_type>(end - begin) == pset.size());

  iterator cur = begin;
  while (cur != end)
  {
    index_type i = cur - begin;
    test_assert(*cur == pset(i));
    ++cur;
  }
}



// Check that map can be constructed with a processor set.

template <dimension_type Dim,
	  typename       Block>
void
test_single_pset(const_Vector<processor_type, Block> pset)
{
  typedef Replicated_map<Dim> map_type;

  map_type map(pset);
  check_replicated_map(map, pset);
}



// Check that map can be constructed with default processor set.

template <dimension_type Dim>
void
test_default_pset()
{
  typedef Replicated_map<Dim> map_type;

  map_type map;
  check_replicated_map(map, vsip::processor_set());
}



template <dimension_type Dim>
void
test_pset()
{
  Vector<processor_type> vec1(1);
  Vector<processor_type> vec4(4);

  vec1(0) = 1;

  vec4(0) = 3;
  vec4(1) = 2;
  vec4(2) = 1;
  vec4(3) = 0;

  test_single_pset<Dim>(vec1);
  test_single_pset<Dim>(vec4);

  test_default_pset<Dim>();
}



int
main(int argc, char** argv)
{
  vsipl vpp(argc, argv);

  test_pset<1>();
  test_pset<2>();
  test_pset<3>();
}
