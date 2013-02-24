/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    benchmarks/create_map.hpp
    @author  Jules Bergmann
    @date    2006-07-26
    @brief   VSIPL++ Library: Benchmark utilities for creating maps.

*/

#ifndef VSIP_BENCHMARKS_CREATE_MAP_HPP
#define VSIP_BENCHMARKS_CREATE_MAP_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/map.hpp>
#include <vsip/parallel.hpp>



/***********************************************************************
  Definitions
***********************************************************************/

template <vsip::dimension_type Dim,
	  typename             MapT>
struct Create_map {};

template <vsip::dimension_type Dim>
struct Create_map<Dim, vsip::Local_map>
{
  typedef vsip::Local_map type;
  static type exec(char) { return type(); }
};

template <vsip::dimension_type Dim>
struct Create_map<Dim, vsip::Global_map<Dim> >
{
  typedef vsip::Global_map<Dim> type;
  static type exec(char) { return type(); }
};

template <typename Dist0, typename Dist1, typename Dist2>
struct Create_map<1, vsip::Map<Dist0, Dist1, Dist2> >
{
  typedef vsip::Map<Dist0, Dist1, Dist2> type;
  static type exec(char type)
  {
    vsip::length_type np = vsip::num_processors();
    switch(type)
    {
    default:
    case 'a':
      // 'a' - all processors
      return vsip::Map<>(vsip::num_processors());
    case '1':
      // '1' - first processor
      return vsip::Map<>(1);
    case '2':
    {
      // '2' - last processor
      vsip::Vector<vsip::processor_type> pset(1); pset.put(0, np-1);
      return vsip::Map<>(pset, 1);
    }
    case 'b':
    {
      // 'b' - non-root processors
      vsip::Vector<vsip::processor_type> pset(np-1);
      for (vsip::index_type i=0; i<np; ++i)
	pset.put(i, i+1);
      return vsip::Map<>(pset, np-1);
    }
    }
  }
};

template <vsip::dimension_type Dim,
	  typename             MapT>
MapT
create_map(char type = 'a')
{
  return Create_map<Dim, MapT>::exec(type);
}


// Sync Policy: use barrier.

struct Barrier
{
  Barrier() : comm_(DEFAULT_COMMUNICATOR()) {}

  void sync() { BARRIER(comm_); }

  COMMUNICATOR_TYPE& comm_;
};



// Sync Policy: no barrier.

struct No_barrier
{
  No_barrier() {}

  void sync() {}
};

#endif // VSIP_BENCHMARKS_CREATE_MAP_HPP
