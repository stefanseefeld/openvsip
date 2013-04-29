//
// Copyright (c) 2005, 2006, 2007 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <iostream>

#include <vsip/initfin.hpp>

#include "vmul.hpp"

using namespace vsip;



/***********************************************************************
  Definitions
***********************************************************************/

void
defaults(Loop1P&)
{
}



int
test(Loop1P& loop, int what)
{
  length_type footprint = 1 << loop.stop_;

  switch (what)
  {
  case   1: loop(t_svmul1<float,          float>()); break;
  case   2: loop(t_svmul1<float,          complex<float> >()); break;
  case   3: loop(t_svmul1<complex<float>, complex<float> >()); break;
  case   4: loop(t_svmul1<float, uint32_t, float>()); break;

  case  11: loop(t_svmul_dom<float,          float>()); break;
  case  12: loop(t_svmul_dom<float,          complex<float> >()); break;
  case  13: loop(t_svmul_dom<complex<float>, complex<float> >()); break;

  case  21: loop(t_svmul_cc<float, float>(footprint)); break;

/*
  case  99: loop(t_svmul3<float>()); break;

  // Double-precision

  case 101: loop(t_svmul1<double,          double>()); break;
  case 102: loop(t_svmul1<double,          complex<double> >()); break;
  case 103: loop(t_svmul1<complex<double>, complex<double> >()); break;
*/

  case 0:
    std::cout
      << "svmul -- scalar-vector multiplication\n"
      << "single-precision:\n"
      << " Scalar-Vector:\n"
      << "   -1 --                float   * Vector<        float >\n"
      << "   -2 --                float   * Vector<complex<float>>\n"
      << "   -3 --        complex<float>  * Vector<complex<float>>\n"
      << "   -4 --                float   * Vector<       int32_t>\n"
      << "  -15 -- t_svmul3\n"
      << "  -15 -- t_svmul4\n"
      << "\ndouble-precision:\n"
      << "  (101-113)\n"
      << "  (131-132)\n"
      ;

  default:
    return 0;
  }
  return 1;
}
