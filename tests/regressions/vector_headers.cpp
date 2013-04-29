//
// Copyright (c) 2007 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

// Test that vector.hpp header is sufficient to use a Vector.
//
//    This is requires that Local_or_global_map be defined.  However,
//    global_map.hpp (and map.hpp) cannot be included until after the
//    definitions for Vector are made.

#include <vsip/initfin.hpp>
#include <vsip/vector.hpp>



/***********************************************************************
  Definitions
***********************************************************************/

int
main(int argc, char** argv)
{
   vsip::vsipl init(argc, argv);

   vsip::Vector<float> foo(10, 3.f);
   vsip::Vector<float> bar(10, 4.f);

   bar *= foo;
}
