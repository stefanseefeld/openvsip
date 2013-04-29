/* Copyright (c) 2009 by CodeSourcery.  All rights reserved. */

/** @file    kernels/cblock_params.hpp
    @author  Jules Bergmann, Stefan Seefeld
    @date    2009-06-23
    @brief   Parameters for cblock user defined kernel example.
*/

#ifndef KERNELS_CBLOCK_PARAMS_HPP
#define KERNELS_CBLOCK_PARAMS_HPP

namespace example
{
struct Cblock_params
{
  uint64_t in0;
  int32_t  in0_stride;
  uint64_t in1;
  int32_t  in1_stride;
  uint64_t in2;
  int32_t  in2_stride;
  uint64_t out;
  int32_t  out_stride;
  int32_t  rows, cols;
};
}

#endif
