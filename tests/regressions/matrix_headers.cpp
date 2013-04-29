//
// Copyright (c) 2007 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

// Test that matrix.hpp header is sufficient to use a Matrix.
//
//    This is requires that Local_or_global_map be defined.  However,
//    global_map.hpp (and map.hpp) cannot be included until after the
//    definitions for Matrix are made.

#include <vsip/initfin.hpp>
#include <vsip/matrix.hpp>



/***********************************************************************
  Definitions
***********************************************************************/

int
main(int argc, char** argv)
{
   vsip::vsipl init(argc, argv);

   vsip::Matrix<float> foo(5, 7, 3.f);
   vsip::Matrix<float> bar(5, 7, 4.f);

   bar *= foo;
}
