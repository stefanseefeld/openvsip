/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.
   
   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
#ifndef UTIL_HPP
#define UTIL_HPP

#ifndef __SPU__
#error This header is only for use on the SPU.
#endif

// Shuffle patterns to map interleaved to split.
vector unsigned char const shuffle_0246 =
  {0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27};
vector unsigned char const shuffle_1357 =
  {4,5,6,7,12,13,14,15,20,21,22,23,28,29,30,31};

// Shuffle patterns to map split to interleaved.
vector unsigned char const shuffle_0415 =
  {0,1,2,3,16,17,18,19,4,5,6,7,20,21,22,23};
vector unsigned char const shuffle_2637 =
  {8,9,10,11,24,25,26,27,12,13,14,15,28,29,30,31};

#endif
