/* Copyright (c) 2005 by CodeSourcery, LLC.  All rights reserved. */

/** @file    example1.cpp
    @author  Mark Mitchell
    @date    05/25/2005
    @brief   VSIPL++ Library: First VSIPL++ program.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/config.hpp>
#include <cmath>
#include <iostream>
#include <vsip/initfin.hpp>
#include <vsip/vector.hpp>
#include <vsip/math.hpp>

/***********************************************************************
  Type Definitions
***********************************************************************/

typedef vsip::Vector<vsip::scalar_f> vector_type;

/***********************************************************************
  Main Program
***********************************************************************/

int 
main(int argc, char **argv)
{
  vsip::vsipl init(argc, argv);

  vector_type v1(10);
  // Initialize all values to PI.
  v1 = VSIP_IMPL_PI;
  // Multiply all values by two.
  v1 *= 2;
  
  // Print the values.
  for (vsip::index_type i = 0; i < v1.size (); ++i)
    std::cout << "v1[" << i << "] = " << v1.get(i) << "\n";
}

