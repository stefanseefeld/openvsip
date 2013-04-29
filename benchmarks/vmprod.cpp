/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   VSIPL++ Library: Vector-Matrix product benchmarks

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/signal.hpp>
#include <vsip_csl/profile.hpp>
#include <vsip_csl/diagnostics.hpp>
#include <vsip_csl/test.hpp>
#include "loop.hpp"
#include "mprod.hpp"


using namespace vsip;


template <typename T>
struct t_vmprod : public t_mprod_base<T>
{
  char const* what() { return "t_vmprod"; }
  float ops_per_point(length_type) { return this->ops_total(1, 1, p_); }
  int riob_per_point(length_type) { return this->riob_total(1, 1, p_); }
  int wiob_per_point(length_type) { return this->wiob_total(1, p_); }
  int mem_per_point(length_type)   { return this->mem_total(1, 1, p_); }

  void operator()(length_type const N, length_type loop, float& time)
  {
    length_type const P = p_;

    Vector<T> A(   N, T(1));  // 1 x N
    Matrix<T> B(N, P, T(1));  // N x P
    Vector<T> Z(   P, T(1));  // 1 x P

    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      Z = prod(A, B);
    t1.stop();
    
    time = t1.delta();
  }

  t_vmprod(length_type size)
    : p_(size)
  {}

  void diag()
  {
    length_type N = 1;
    length_type P = 1;

    Vector<T> A(   N, T(1));  // 1 x N
    Matrix<T> B(N, P, T(1));  // N x P
    Vector<T> Z(   P, T(1));  // 1 x P

    vsip_csl::assign_diagnostics(Z, prod(A, B));
  }

  // data members
  length_type const p_;
};


template <typename ImplTag,
          typename T>
struct t_vmprod_tag : public t_mprod_base<T>
{
  char const* what() { return "t_vmprod_tag"; }
  float ops_per_point(length_type) { return this->ops_total(1, 1, p_); }
  int riob_per_point(length_type) { return this->riob_total(1, 1, p_); }
  int wiob_per_point(length_type) { return this->wiob_total(1, p_); }
  int mem_per_point(length_type)   { return this->mem_total(1, 1, p_); }

  void operator()(length_type const N, length_type loop, float& time)
  {
    using namespace vsip_csl::dispatcher;

    typedef Dense<1, T> vblock_type;
    typedef Dense<2, T> mblock_type;

    typedef Evaluator<op::prod, ImplTag,
      void(vblock_type &, vblock_type const &, mblock_type const &)>
      evaluator_type;

    length_type const P = p_;

    Vector<T, vblock_type> A(   N, T(1));  // 1 x N
    Matrix<T, mblock_type> B(N, P, T(1));  // N x P
    Vector<T, vblock_type> Z(   P, T(1));  // 1 x P

    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      evaluator_type::exec(Z.block(), A.block(), B.block());
    t1.stop();
    
    time = t1.delta();
  }

  t_vmprod_tag(length_type size)
    : p_(size)
  {}

  void diag()
  {
    using namespace vsip_csl::dispatcher;
    std::cout << Backend_name<ImplTag>::name() << std::endl;
  }

  // data members
  length_type const p_;
};

	  
void
defaults(Loop1P& loop)
{
  loop.start_ = 4;
  loop.stop_  = 10;

  loop.param_["size"] = "1024";
}



int
test(Loop1P& loop, int what)
{
  using namespace vsip_csl::dispatcher;

  length_type size  = atoi(loop.param_["size"].c_str());

  switch (what)
  {
  case  1: loop(t_vmprod<float>(size)); break;
  case  2: loop(t_vmprod<complex<float> >(size)); break;

  case  3: loop(t_vmprod_tag<be::generic, float>(size)); break;
  case  4: loop(t_vmprod_tag<be::generic, complex<float> >(size)); break;

#if VSIP_IMPL_HAVE_BLAS
  case  5: loop(t_vmprod_tag<be::blas, float>(size)); break;
    // The BLAS backend doesn't handle split-complex, so don't attempt
    // to instantiate that code if we are using split-complex blocks.
# if !VSIP_IMPL_PREFER_SPLIT_COMPLEX
  case  6: loop(t_vmprod_tag<be::blas, complex<float> >(size)); break;
# endif
#endif

#if VSIP_IMPL_HAVE_CUDA
  case  7: loop(t_vmprod_tag<be::cuda, float>(size)); break;
  case  8: loop(t_vmprod_tag<be::cuda, complex<float> >(size)); break;
#endif

  case  0:
    std::cout
      << "vmprod -- vector-matrix product A x B --> Z (sizes 1xN and NxP --> 1xP)\n"
      << "    -1 --   default implementation, float\n"
      << "    -2 --   default implementation, complex<float>\n"
      << "    -3 --   Generic implementation, float\n"
      << "    -4 --   Generic implementation, complex<float>\n"
      << "    -5 --   BLAS implementation, float\n"
      << "    -6 --   BLAS implementation, complex<float>\n"
      << "    -7 --   CUDA implementation, float\n"
      << "    -8 --   CUDA implementation, complex<float>\n"
      << "\n"
      << " Parameters (for sweeping N, size of A and number of rows in B)\n"
      << "  -p:size P -- fix number of cols in B (default 1024)\n"
      ;
  default: return 0;
  }
  return 1;
}
