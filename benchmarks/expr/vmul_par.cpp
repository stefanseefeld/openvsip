//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/random.hpp>
#include <vsip/selgen.hpp>

#include "vmul.hpp"
#include "../benchmarks.hpp"

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
  switch (what)
  {
  case  92: loop(t_vmul1_nonglobal<complex<float> >()); break;

  case 101: loop(t_vmul1<        float , Map<> >()); break;
  case 102: loop(t_vmul1<complex<float>, Map<> >()); break;

  case 111: loop(t_vmul1<        float , Map<>, Barrier>()); break;
  case 112: loop(t_vmul1<complex<float>, Map<>, Barrier>()); break;

  case 121: loop(t_vmul1_local<        float  >()); break;
  case 122: loop(t_vmul1_local<complex<float> >()); break;

  case 131: loop(t_vmul1_early_local<        float  >()); break;
  case 132: loop(t_vmul1_early_local<complex<float> >()); break;

  case 141: loop(t_vmul1_sa<        float  >()); break;
  case 142: loop(t_vmul1_sa<complex<float> >()); break;

  case 0:
    std::cout
      << "vmul -- vector multiplication\n"

      << "  -91 -- Vector<        float > * Vector<        float > NONGLOBAL\n"
      << "  -92 -- Vector<complex<float>> * Vector<complex<float>> NONGLOBAL\n"

      << " -101 -- Vector<        float > * Vector<        float > PAR\n"
      << " -102 -- Vector<complex<float>> * Vector<complex<float>> PAR\n"
      << " -111 -- Vector<        float > * Vector<        float > PAR sync\n"
      << " -112 -- Vector<complex<float>> * Vector<complex<float>> PAR sync\n"
      << " -121 -- Vector<        float > * Vector<        float > PAR local\n"
      << " -122 -- Vector<complex<float>> * Vector<complex<float>> PAR local\n"
      << " -131 -- Vector<        float > * Vector<        float > PAR early local\n"
      << " -132 -- Vector<complex<float>> * Vector<complex<float>> PAR early local\n"
      << " -141 -- Vector<        float > * Vector<        float > PAR setup assign\n"
      << " -142 -- Vector<complex<float>> * Vector<complex<float>> PAR setup assign\n"
      ;

  default:
    return 0;
  }
  return 1;
}
