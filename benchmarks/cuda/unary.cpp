/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Benchmark for CUDA-based unary functions.

#include <iostream>
#include <ostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/core/expr/fns_elementwise.hpp>
#include <vsip_csl/profile.hpp>
#include <vsip/opt/cuda/kernels.hpp>
#include <vsip/opt/cuda/dda.hpp>
#include <vsip/selgen.hpp>

#include <vsip_csl/test.hpp>
#include "loop.hpp"

using namespace vsip;
using vsip_csl::equal;
namespace expr = vsip_csl::expr;

#define DEBUG 0

// Functors from fns_elementwise.hpp are used to differentiate the
// benchmark types (instead of the struct *_tag; declarations)


/***********************************************************************
  Unary expression test harness
***********************************************************************/

template <typename T1,
          typename T2,
          class F>
struct t_unary_base : Benchmark_base
{
  typedef Dense<1, T1>  src_block_type;
  typedef Dense<1, T2>  dst_block_type;

  typedef void(unary_functor_type)(T1 const*, T2*, length_type);

  t_unary_base(unary_functor_type f)
    : functor_(f)
  {}

  char const* what() 
  { 
    std::ostringstream out;
    out << "CUDA t_unary<..., " << F::name() << ">";
    return out.str().c_str();
  }
  
  void exec(length_type size, length_type loop, float& time)
  {
    Vector<T1, src_block_type>   A = ramp(T1(0), T1(1), size);
    Vector<T2, dst_block_type>   Z(size, T2());

    // Scoping is used to control the lifetime of dda::Data<> objects.  These 
    // must be destroyed before accessing data through the view again.
    {     
      impl::cuda::dda::Data<src_block_type, vsip::dda::in> dev_a(A.block());
      impl::cuda::dda::Data<dst_block_type, vsip::dda::out> dev_z(Z.block());
      T1 const* pA = dev_a.ptr();
      T2* pZ = dev_z.ptr();

      // Benchmark the operation
      vsip_csl::profile::Timer t1;
      t1.start();
      for (index_type l = 0; l < loop; ++l)
        this->functor_(pA, pZ, size);
      t1.stop();
      time = t1.delta();
    }

    // validate results
    for (index_type i = 0; i < size; ++i)
    {
#if DEBUG
      if (!equal(F::apply(A.get(i)), Z.get(i)))
      {
	std::cout << "ERROR: at location " << i << ", " << j << '\n'
		  << "       expected: " << F::apply(A.get(i)) << '\n'
		  << "       got     : " << Z.get(i) << std::endl;
      }
#endif
      test_assert(equal(F::apply(A.get(i)), Z.get(i)));
    }
  }

private:
  unary_functor_type* functor_;
};

template <typename T1, typename T2, class F>
struct t_unary : public t_unary_base<T1, T2, F>
{
  int ops_per_point(length_type)  { return ops_per_element_; }
  int riob_per_point(length_type) { return sizeof(T1); }
  int wiob_per_point(length_type) { return sizeof(T2); }
  int mem_per_point(length_type)  { return sizeof(T1) + sizeof(T2); }

  void operator()(vsip::length_type size, vsip::length_type loop, float& time)
  {
    this->exec(size, loop, time);
  }

  typedef typename t_unary_base<T1, T2, F>::unary_functor_type unary_functor_type;

  t_unary(unary_functor_type func, int ops)
    : t_unary_base<T1, T2, F>(func), ops_per_element_(ops)
  {}

// Member data
  int ops_per_element_;
};


/***********************************************************************
  Benchmark Driver
***********************************************************************/

void
defaults(Loop1P &)
{
}

int
test(Loop1P& loop, int what)
{
  typedef float F;
  typedef complex<float> C;

  switch (what)
  {
    // Template parameters are:  
    //     <Input Type, Output Type, Functor>
    // Constructor parameters are:  
    //     (function handler, size of the fixed-dimension, operation count per element)

    // sweep # columns
  case  1: loop(t_unary<C, F, expr::op::Mag<C> >(impl::cuda::mag, 1)); break;

    // help
  default:
    std::cout << "CUDA unary expressions -- fixed rows\n"
	      << "   -1 -- complex magnitude\n"
      ;
    return 0;
  }
  return 1;
}
