/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/sal-assumptions.cpp
    @author  Don McCoy
    @date    2005-10-13
    @brief   VSIPL++ Library: Check SAL assumptions.
*/


/***********************************************************************
  Included Files
***********************************************************************/

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

