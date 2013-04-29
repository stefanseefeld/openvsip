//
// Copyright (c) 2010 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   VSIPL++ Library: Matrix-Vector product benchmarks

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
struct t_mvprod : public t_mprod_base<T>
{
  char const* what() { return "t_mvprod"; }
  float ops_per_point(length_type) { return this->ops_total(m_, 1, 1); }
  int riob_per_point(length_type) { return this->riob_total(m_, 1, 1); }
  int wiob_per_point(length_type) { return this->wiob_total(m_, 1); }
  int mem_per_point(length_type)   { return this->mem_total(m_, 1, 1); }

  void operator()(length_type const N, length_type loop, float& time)
  {
    length_type const M = m_;

    Matrix<T, Dense<2, T, col2_type> > A(M, N, T(1));  // M x N
    Vector<T, Dense<1, T> >            B(N,    T(1));  // N x 1
    Vector<T, Dense<1, T> >            Z(M,    T(1));  // M x 1

    vsip_csl::profile::Timer t1;
    
    // column-major dimension ordering is used, as required by BLAS
    t1.start();
    for (index_type l=0; l<loop; ++l)
      Z = prod(A, B);
    t1.stop();
    
    time = t1.delta();
  }

  t_mvprod(length_type size)
    : m_(size)
  {}

  void diag()
  {
    length_type M = 1;
    length_type N = 1;

    Matrix<T> A(M, N, T(1));  // M x N
    Vector<T> B(N,    T(1));  // N x 1
    Vector<T> Z(M,    T(1));  // M x 1

    vsip_csl::assign_diagnostics(Z, prod(A, B));
  }

  // data members
  length_type const m_;
};


template <typename ImplTag,
          typename T>
struct t_mvprod_tag : public t_mprod_base<T>
{
  char const* what() { return "t_mvprod_tag"; }
  float ops_per_point(length_type) { return this->ops_total(m_, 1, 1); }
  int riob_per_point(length_type) { return this->riob_total(m_, 1, 1); }
  int wiob_per_point(length_type) { return this->wiob_total(m_, 1); }
  int mem_per_point(length_type)   { return this->mem_total(m_, 1, 1); }

  void operator()(length_type const N, length_type loop, float& time)
  {
    using namespace vsip_csl::dispatcher;

    // column-major dimension ordering is used, as required by BLAS
    typedef Dense<1, T> vblock_type;
    typedef Dense<2, T, col2_type> mblock_type;

    typedef Evaluator<op::prod, ImplTag,
      void(vblock_type &, mblock_type const &, vblock_type const &)>
      evaluator_type;

    length_type const M = m_;

    Matrix<T, mblock_type> A(M, N, T(1));  // M x N
    Vector<T, vblock_type> B(N,    T(1));  // N x 1
    Vector<T, vblock_type> Z(M,    T(1));  // M x 1

    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      evaluator_type::exec(Z.block(), A.block(), B.block());
    t1.stop();
    
    time = t1.delta();
  }

  t_mvprod_tag(length_type size)
    : m_(size)
  {}

  void diag()
  {
    using namespace vsip_csl::dispatcher;
    std::cout << Backend_name<ImplTag>::name() << std::endl;
  }

  // data members
  length_type const m_;
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
  case  1: loop(t_mvprod<float>(size)); break;
  case  2: loop(t_mvprod<complex<float> >(size)); break;

  case  3: loop(t_mvprod_tag<be::generic, float>(size)); break;
  case  4: loop(t_mvprod_tag<be::generic, complex<float> >(size)); break;

#if VSIP_IMPL_HAVE_BLAS
  case  5: loop(t_mvprod_tag<be::blas, float>(size)); break;
    // The BLAS backend doesn't handle split-complex, so don't attempt
    // to instantiate that code if we are using split-complex blocks.
# if !VSIP_IMPL_PREFER_SPLIT_COMPLEX
  case  6: loop(t_mvprod_tag<be::blas, complex<float> >(size)); break;
# endif
#endif

#if VSIP_IMPL_HAVE_CUDA
  case  7: loop(t_mvprod_tag<be::cuda, float>(size)); break;
  case  8: loop(t_mvprod_tag<be::cuda, complex<float> >(size)); break;
#endif

  case  0:
    std::cout
      << "mvprod -- matrix-vector product A x B --> Z (sizes MxN and Nx1 --> Mx1)\n"
      << "    -1 --   default implementation, float\n"
      << "    -2 --   default implementation, complex<float>\n"
      << "    -3 --   Generic implementation, float\n"
      << "    -4 --   Generic implementation, complex<float>\n"
      << "    -5 --   BLAS implementation, float\n"
      << "    -6 --   BLAS implementation, complex<float>\n"
      << "    -7 --   CUDA implementation, float\n"
      << "    -8 --   CUDA implementation, complex<float>\n"
      << "\n"
      << " Parameters (for sweeping N, size of B and number of cols in A)\n"
      << "  -p:size M -- fix number of rows in A (default 1024)\n"
      ;
  default: return 0;
  }
  return 1;
}
