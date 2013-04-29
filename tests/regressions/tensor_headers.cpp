/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/** @file    tests/tensor_header.cpp
    @author  Jules Bergmann
    @date    2007-03-26
    @brief   VSIPL++ Library: Test that tensor.hpp header is sufficient
                              to use a Tensor.

    This is requires that Local_or_global_map be defined.  However,
    global_map.hpp (and map.hpp) cannot be included until after the
    definitions for Tensor are made.
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/initfin.hpp>
#include <vsip/tensor.hpp>



/***********************************************************************
  Definitions
***********************************************************************/

int
main(int argc, char** argv)
{
   vsip::vsipl init(argc, argv);

   vsip::Tensor<float> foo(3, 5, 7, 3.f);
   vsip::Tensor<float> bar(3, 5, 7, 4.f);

   bar *= foo;
}
