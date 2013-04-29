/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Benchmark for distributed vector multiply.

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/random.hpp>
#include <vsip_csl/assignment.hpp>
#include "benchmarks.hpp"

using namespace vsip;


/***********************************************************************
  Utilities
***********************************************************************/

// Create a map of a given pattern.

inline Map<>
create_map(char type)
{
  length_type np = num_processors();
  switch(type)
  {
  default:
  case 'a': // all processors
    return Map<>(num_processors());
  case '1': // first processor
    return Map<>(1);
  case '2': // last processor
  {
    Vector<processor_type> pset(1); pset.put(0, np-1);
    return Map<>(pset, 1);
  }
  case 'b': // non-root processors
  {
    Vector<processor_type> pset(np-1);
    for (index_type i=0; i<np-1; ++i)
      pset.put(i, i+1);
    return Map<>(pset, np-1);
  }
  case 'w': // worker processrs (non-root and non-last)
  {
    Vector<processor_type> pset(np-2);
    for (index_type i=0; i<np-2; ++i)
      pset.put(i, i+1);
    return Map<>(pset, np-2);
  }
  }
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



/***********************************************************************
  Definitions - distributed vector element-wise multiply
***********************************************************************/

struct Impl_assign;
struct Impl_sa;

template <typename T,
	  typename MapT    = Local_map,
	  typename SP      = No_barrier,
	  typename ImplTag = Impl_assign>
struct t_dist_vmul;



/***********************************************************************
  Assignment
***********************************************************************/

template <typename T,
	  typename MapT,
	  typename SP>
struct t_dist_vmul<T, MapT, SP, Impl_assign> : Benchmark_base
{
  char const* what() { return "t_vmul"; }
  int ops_per_point(length_type)  { return impl::Ops_info<T>::mul; }
  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
    VSIP_IMPL_NOINLINE
  {
    typedef Dense<1, T, row1_type, MapT> block_type;

    T a_freq = 0.15f;
    T b_freq = 0.15f;

    Vector<T, block_type> A(size, map_compute_);
    Vector<T, block_type> B(size, map_compute_);
    Vector<T, block_type> C(size, map_compute_);

    Vector<T, block_type> A_in (size, T(), map_in_);
    Vector<T, block_type> B_in (size, T(), map_in_);
    Vector<T, block_type> C_out(size,      map_out_);

    for (index_type i=0; i<A_in.local().size(); ++i)
    {
      // A_in and B_in have same map.
      index_type g_i = global_from_local_index(A_in, 0, i);
      A_in.local().put(i, cos(3.1415f * a_freq * T(g_i)));
      B_in.local().put(i, cos(3.1415f * b_freq * T(g_i)));
    }

    vsip_csl::profile::Timer t1;
    SP sp;
    
    t1.start();
    sp.sync();
    for (index_type l=0; l<loop; ++l)
    {
      // scatter data
      A = A_in;
      B = B_in;
      // perform operation
      C = A * B;
      // gather result
      C_out = C;
    }
    sp.sync();
    t1.stop();
    
    for (index_type i=0; i<C_out.local().size(); ++i)
    {
      index_type g_i = global_from_local_index(C_out, 0, i);
      T a_val = cos(3.1415f * a_freq * T(g_i));
      T b_val = cos(3.1415f * b_freq * T(g_i));
      test_assert(equal(C_out.local().get(i), a_val * b_val));
    }
    
    time = t1.delta();
  }

  t_dist_vmul(MapT map_compute, MapT map_in, MapT map_out)
    : map_compute_(map_compute),
      map_in_     (map_in),
      map_out_    (map_out)
  {}

  // Member data
  MapT map_compute_;
  MapT map_in_;
  MapT map_out_;
};



/***********************************************************************
  Setup-assign
***********************************************************************/

template <typename T,
	  typename MapT,
	  typename SP>
struct t_dist_vmul<T, MapT, SP, Impl_sa> : Benchmark_base
{
  char const* what() { return "t_vmul"; }
  int ops_per_point(length_type)  { return impl::Ops_info<T>::mul; }
  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
    VSIP_IMPL_NOINLINE
  {
    typedef Dense<1, T, row1_type, MapT> block_type;

    T a_freq = 0.15f;
    T b_freq = 0.15f;

    Vector<T, block_type> A(size, map_compute_);
    Vector<T, block_type> B(size, map_compute_);
    Vector<T, block_type> C(size, map_compute_);

    Vector<T, block_type> A_in (size, T(), map_in_);
    Vector<T, block_type> B_in (size, T(), map_in_);
    Vector<T, block_type> C_out(size,      map_out_);

    for (index_type i=0; i<A_in.local().size(); ++i)
    {
      // A_in and B_in have same map.
      index_type g_i = global_from_local_index(A_in, 0, i);
      A_in.local().put(i, cos(3.1415f * a_freq * T(g_i)));
      B_in.local().put(i, cos(3.1415f * b_freq * T(g_i)));
    }

    vsip_csl::profile::Timer t1;
    SP sp;

    vsip_csl::Assignment scatter_A(A, A_in);
    vsip_csl::Assignment scatter_B(B, B_in);
    vsip_csl::Assignment gather   (C_out, C);
    
    t1.start();
    sp.sync();
    for (index_type l=0; l<loop; ++l)
    {
      // scatter data
      scatter_A();
      scatter_B();

      // perform operation
      C = A * B;

      // gather result
      gather();
    }
    sp.sync();
    t1.stop();
    
    for (index_type i=0; i<C_out.local().size(); ++i)
    {
      index_type g_i = global_from_local_index(C_out, 0, i);
      T a_val = cos(3.1415f * a_freq * T(g_i));
      T b_val = cos(3.1415f * b_freq * T(g_i));
      test_assert(equal(C_out.local().get(i), a_val * b_val));
    }
    
    time = t1.delta();
  }

  t_dist_vmul(MapT map_compute, MapT map_in, MapT map_out)
    : map_compute_(map_compute),
      map_in_     (map_in),
      map_out_    (map_out)
  {}

  // Member data
  MapT map_compute_;
  MapT map_in_;
  MapT map_out_;
};



/***********************************************************************
  Wrapper classes
***********************************************************************/

// Local wrapper.

template <typename T,
	  typename ImplTag,
	  typename SP = No_barrier>
struct t_dist_vmul_local
  : public t_dist_vmul<T, Local_map, SP, ImplTag>
{
  typedef t_dist_vmul<T, Local_map, SP, ImplTag> base_type;

  t_dist_vmul_local()
    : base_type(Local_map(), Local_map(), Local_map())
  {}
};



// Clique parallelism wrapper.

template <typename T,
	  typename ImplTag,
	  typename SP   = No_barrier>
struct t_dist_vmul_par
  : public t_dist_vmul<T, Map<>, SP, ImplTag>
{
  typedef t_dist_vmul<T, Map<>, SP, ImplTag> base_type;

  t_dist_vmul_par()
    : base_type(Map<>(num_processors()),
		Map<>(1),
		Map<>(1))
  {
    if (num_processors() == 1)
    {
      this->map_in_  = this->map_compute_;
      this->map_out_ = this->map_compute_;
    }
  }
};



// Pipeline parallelism wrapper.

template <typename T,
	  typename ImplTag,
	  typename SP   = No_barrier>
struct t_dist_vmul_pipe
  : public t_dist_vmul<T, Map<>, SP, ImplTag>
{
  typedef t_dist_vmul<T, Map<>, SP, ImplTag> base_type;

  t_dist_vmul_pipe()
    : base_type(create_map('w'),
		create_map('1'),
		create_map('2'))
  {}
};



/***********************************************************************
  Benchmark Definitions
***********************************************************************/

void
defaults(Loop1P&)
{
}



int
test(Loop1P& loop, int what)
{
  switch (what)
  {
  case  11: loop(t_dist_vmul_local<float,          Impl_assign>()); break;
  case  12: loop(t_dist_vmul_local<complex<float>, Impl_assign>()); break;
  case  21: loop(t_dist_vmul_par<float,            Impl_assign>()); break;
  case  22: loop(t_dist_vmul_par<complex<float>,   Impl_assign>()); break;
  case  31: loop(t_dist_vmul_pipe<float,           Impl_assign>()); break;
  case  32: loop(t_dist_vmul_pipe<complex<float>,  Impl_assign>()); break;

  case  41: loop(t_dist_vmul_local<float,          Impl_sa>()); break;
  case  42: loop(t_dist_vmul_local<complex<float>, Impl_sa>()); break;
  case  51: loop(t_dist_vmul_par<float,            Impl_sa>()); break;
  case  52: loop(t_dist_vmul_par<complex<float>,   Impl_sa>()); break;
  case  61: loop(t_dist_vmul_pipe<float,           Impl_sa>()); break;
  case  72: loop(t_dist_vmul_pipe<complex<float>,  Impl_sa>()); break;

  case 0:
    std::cout
      << "dist_vmul -- distributed vector multiplication\n"
      << " Using normal assignment\n"
      << "  -11 -- Local vmul (non-parallel) - float\n"
      << "  -12 -- Local vmul (non-parallel) - complex\n"
      << "  -21 -- Clique vmul               - float\n"
      << "  -22 -- Clique vmul               - complex\n"
      << "  -31 -- Pipelined vmul            - float\n"
      << "  -32 -- Pipelined vmul            - complex\n"
      << " Using Assignment object\n"
      << "  -41 -- Local vmul (non-parallel) - float\n"
      << "  -42 -- Local vmul (non-parallel) - complex\n"
      << "  -51 -- Clique vmul               - float\n"
      << "  -52 -- Clique vmul               - complex\n"
      << "  -61 -- Pipelined vmul            - float\n"
      << "  -62 -- Pipelined vmul            - complex\n"
      ;

  default:
    return 0;
  }
  return 1;
}
