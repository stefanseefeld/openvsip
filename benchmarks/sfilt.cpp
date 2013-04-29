/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Benchmark for 2D Separable Filter.

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/signal.hpp>

#include <vsip_csl/img/separable_filter.hpp>

#include <vsip_csl/diagnostics.hpp>

#include <vsip_csl/test.hpp>
#include "loop.hpp"

using namespace vsip;
using namespace vsip_csl::img;

template <support_region_type Supp,
	  typename            T>
struct t_sfilt : Benchmark_base
{
  char const* what() { return "t_sfilt"; }

  void output_size(
    length_type  rows,   length_type  cols,
    length_type& o_rows, length_type& o_cols)
  {
    if      (Supp == support_full)
    {
      o_rows = (rows + m_ - 2) + 1;
      o_cols = (cols + n_ - 2) + 1;
    }
    else if (Supp == support_same)
    {
      o_rows = (rows - 1) + 1;
      o_cols = (cols - 1) + 1;
    }
    else if (Supp == support_min_zeropad)
    {
      o_rows = rows;
      o_cols = cols;
    }
    else /* (Supp == support_min) */
    {
      o_rows = (rows-1) - (m_-1) + 1;
      o_cols = (cols-1) - (n_-1) + 1;
    }
  }
  
  float ops_per_point(length_type cols)
  {
    length_type o_rows, o_cols;

    output_size(rows_, cols, o_rows, o_cols);

    float ops = 
      ( (o_rows * o_cols * n_) +
	(o_cols * o_rows * m_) ) *
      (vsip::impl::Ops_info<T>::mul + vsip::impl::Ops_info<T>::add);

    return ops / cols;
  }

  int riob_per_point(length_type) { return -1; }
  int wiob_per_point(length_type) { return -1; }
  int mem_per_point(length_type)  { return 2*sizeof(T); }

  void operator()(length_type cols, length_type loop, float& time)
  {
    length_type o_rows, o_cols;

    output_size(rows_, cols, o_rows, o_cols);

    Matrix<T>   in (rows_, cols, T());
    Matrix<T>   out(o_rows, o_cols);
    Vector<T>   coeff0(m_, T());
    Vector<T>   coeff1(n_, T());

    coeff0 = T(1);
    coeff1 = T(1);

    typedef Separable_filter<T, Supp, edge_zero> filt_type;

    filt_type filt(coeff0, coeff1, Domain<2>(rows_, cols));

    vsip_csl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      filt(in, out);
    t1.stop();
    
    time = t1.delta();
  }

  t_sfilt(length_type rows, length_type m, length_type n)
    : rows_(rows), m_(m), n_(n)
  {}

  void diag()
  {
  }

  length_type rows_;
  length_type m_;
  length_type n_;
};



void
defaults(Loop1P& loop)
{
  loop.loop_start_ = 5000;
  loop.start_ = 4;

  loop.param_["rows"] = "16";
  loop.param_["mn"]   = "0";
  loop.param_["m"]    = "3";
  loop.param_["n"]    = "3";
}



int
test(Loop1P& loop, int what)
{
  typedef complex<float> cf_type;

  length_type rows = atoi(loop.param_["rows"].c_str());
  length_type MN   = atoi(loop.param_["mn"].c_str());
  length_type M    = atoi(loop.param_["m"].c_str());
  length_type N    = atoi(loop.param_["n"].c_str());

  if (MN != 0)
    M = N = MN;

  switch (what)
  {
  case  1: loop(t_sfilt<support_min_zeropad, float> (rows, M, N)); break;

  case  0:
    std::cout
      << "sfilt -- Separable_filter\n"
      << "   -1 -- float (min_zeropad)\n"
      << "\n"
      << "Parameters:\n"
      << "   -p:m M       -- set filter size M (default 3)\n"
      << "   -p:n N       -- set filter size N (default 3)\n"
      << "   -p:mn MN     -- set filter sizes M and N at once\n"
      << "   -p:rows ROWS -- set image rows (default 16)\n"
      ;
    

  default: return 0;
  }
  return 1;
}
