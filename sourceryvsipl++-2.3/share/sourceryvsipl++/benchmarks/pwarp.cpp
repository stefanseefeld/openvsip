/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    benchmarks/pwarp.cpp
    @author  Jules Bergmann
    @date    2007-11-08
    @brief   VSIPL++ Library: Benchmark for Perspective Image Warp.

*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/signal.hpp>

#include <vsip_csl/img/perspective_warp.hpp>

#include <vsip/opt/assign_diagnostics.hpp>

#include <vsip_csl/test.hpp>
#include "loop.hpp"

using namespace vsip;
using namespace vsip_csl::img;



/***********************************************************************
  Definitions
***********************************************************************/

template <typename T,
	  typename BlockT>
void
setup_p(
  Matrix<T, BlockT> P,
  int               i)
{
  switch (i) {
  case 0:
    P        = T();
    P.diag() = T(1);

  case 1:
    P(0,0) = T(0.999982);    P(0,1) = T(0.000427585); P(0,2) = T(-0.180836);
    P(1,0) = T(-0.00207906); P(1,1) = T(0.999923);    P(1,2) = T(0.745001);
    P(2,0) = T(1.01958e-07); P(2,1) = T(8.99655e-08); P(2,2) = T(1);
    break;

  case 2:
    P(0,0) = 8.28282751190698e-01; 
    P(0,1) = 2.26355321374407e-02;
    P(0,2) = -1.10504985681804e+01;

    P(1,0) = -2.42950546474237e-01;
    P(1,1) = 8.98035288576380e-01;
    P(1,2) = 1.05162748265872e+02;

    P(2,0) = -1.38973743578922e-04;
    P(2,1) = -9.01955477542629e-05;
    P(2,2) = 1;
    break;
  }
}




template <typename         CoeffT,
	  typename         T,
	  interpolate_type InterpT>
struct t_pwarp_obj : Benchmark_base
{
  char const* what() { return "t_pwarp_obj"; }

  float ops_per_point(length_type)
  {
    return rows_;
  }

  int riob_per_point(length_type) { return -1; }
  int wiob_per_point(length_type) { return -1; }
  int mem_per_point(length_type)  { return 2*sizeof(T); }

  void operator()(length_type cols, length_type loop, float& time)
  {
    Matrix<CoeffT> P(3, 3);
    Matrix<T>      in (rows_, cols, T());
    Matrix<T>      out(rows_, cols);

    setup_p(P, idx_);

    vsip::impl::profile::Timer t1;

    Perspective_warp<CoeffT, T, InterpT, forward>
      warp(P, Domain<2>(rows_, cols));
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      warp(in, out);
    t1.stop();
    
    time = t1.delta();
  }

  t_pwarp_obj(length_type rows, int idx)
    : rows_(rows), idx_(idx)
  {}

  void diag()
  {
  }

  length_type rows_;
  int         idx_;
};



template <typename CoeffT,
	  typename T>
struct t_pwarp_fun : Benchmark_base
{
  char const* what() { return "t_pwarp_fun"; }

  float ops_per_point(length_type)
  {
    return rows_;
  }

  int riob_per_point(length_type) { return -1; }
  int wiob_per_point(length_type) { return -1; }
  int mem_per_point(length_type)  { return 2*sizeof(T); }

  void operator()(length_type cols, length_type loop, float& time)
  {
    Matrix<CoeffT> P(3, 3);
    Matrix<T>      in (rows_, cols, T());
    Matrix<T>      out(rows_, cols);

    setup_p(P, idx_);

    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      perspective_warp(P, in, out);
    t1.stop();
    
    time = t1.delta();
  }

  t_pwarp_fun(length_type rows, int idx)
    : rows_(rows), idx_(idx)
  {}

  void diag()
  {
    std::cout << "perspective_warp function interface\n";
  }

  length_type rows_;
  int         idx_;
};



void
defaults(Loop1P& loop)
{
  loop.loop_start_ = 1;
  loop.start_ = 4;
  loop.stop_  = 13;

  loop.param_["rows"] = "512";
  loop.param_["pi"]   = "0";
}



int
test(Loop1P& loop, int what)
{
  typedef complex<float> cf_type;

  length_type rows = atoi(loop.param_["rows"].c_str());
  length_type pi   = atoi(loop.param_["pi"].c_str());

  interpolate_type const IL = interp_linear;

  switch (what)
  {
  case  1: loop(t_pwarp_obj<float, float, IL> (rows, pi)); break;
  case  2: loop(t_pwarp_obj<float, unsigned char, IL>(rows, pi)); break;

  case 11: loop(t_pwarp_fun<float, float> (rows, pi)); break;
  case 12: loop(t_pwarp_fun<float, unsigned char>(rows, pi)); break;

  case  0:
    std::cout
      << "pwarp -- Perspective_warp\n"
      << " Object:\n"
      << "   -1 -- float\n"
      << "   -2 -- char\n"
      << " Funcion:\n"
      << "  -11 -- float\n"
      << "  -12 -- char\n"
      << "\n"
      << "Parameters:\n"
      << "   -p:rows ROWS -- set image rows (default 16)\n"
      ;
    

  default: return 0;
  }
  return 1;
}
