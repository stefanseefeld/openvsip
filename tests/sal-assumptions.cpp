//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <cmath>
#include <iostream>
#include <vsip/initfin.hpp>
#include <vsip/core/layout.hpp>
#include <vsip_csl/test.hpp>
#include <vsip_csl/output.hpp>

using namespace std;
using namespace vsip;
using namespace vsip_csl;

#ifdef VSIP_IMPL_HAVE_SAL
#include <sal.h>

// this verifies that std::pair<> and the SAL types for split
// complex share the same layout.  the explicit cast is exactly the
// method used when the SAL library is used for elementwise operations.
void 
check_split_layout()
{
  {
    float real_value[1] = { 1.23 };
    float imag_value[1] = { 4.56 };
    std::pair<float *, float *> p(real_value, imag_value);
    COMPLEX_SPLIT *pcs = (COMPLEX_SPLIT *) &p;
    
    test_assert( pcs->realp == p.first );
    test_assert( pcs->imagp == p.second );
    test_assert( *pcs->realp == *p.first );
    test_assert( *pcs->imagp == *p.second );
  }

  {
    double real_value[1] = { 1.23 };
    double imag_value[1] = { 4.56 };
    std::pair<double *, double *> p(real_value, imag_value);
    DOUBLE_COMPLEX_SPLIT *pcs = (DOUBLE_COMPLEX_SPLIT *) &p;
    
    test_assert( pcs->realp == p.first );
    test_assert( pcs->imagp == p.second );
    test_assert( *pcs->realp == *p.first );
    test_assert( *pcs->imagp == *p.second );
  }
}
#endif

int 
main()
{
#ifdef VSIP_IMPL_HAVE_SAL
  check_split_layout();
#endif

  return 0;
}

