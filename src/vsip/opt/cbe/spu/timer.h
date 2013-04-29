/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cbe/spu/timer.h
    @author  Don McCoy
    @date    2006-02-16
    @brief   VSIPL++ Library: Simple timing routines.
*/

#ifndef VSIP_OPT_CBE_SPU_TIMER_H
#define VSIP_OPT_CBE_SPU_TIMER_H

#include <spu_mfcio.h>


/***********************************************************************
  Definitions
***********************************************************************/

// Note: This value should be adjusted based on the system-dependent
// timebase values found in /proc/cpuinfo.
#define TIMEBASE 14318000              // 14318000 for CHRP IBM,0200-150

typedef unsigned int spu_timer_type;

inline
spu_timer_type read_timer()
{ 
  return (spu_timer_type)(~spu_read_decrementer()); 
}


// This must be called prior to using the start() and stop() 
// functions below for the first time.  It should never be called 
// in between though, as it will lead to an inaccurate timing 
// measurement.  Furthermore, once reset, it will wrap when
// it reaches zero so it is limited to measuring 2^31/TIMEBASE
// seconds (about 2.5 minutes at 14.3 MHz).
inline 
void reset_timer() 
{ 
  spu_write_decrementer(0x7fffffff); 
}


typedef struct 
{ 
  spu_timer_type start; 
  spu_timer_type stop; 
  spu_timer_type total; 
  int count;
} acc_timer_type;

inline 
void init_timer(acc_timer_type* t) 
{ 
  t->start = 0;
  t->stop = 0; 
  t->total = 0; 
  t->count = 0;
}

inline
void start_timer(acc_timer_type* t) 
{ 
  t->start = read_timer(); 
}

inline
void stop_timer(acc_timer_type* t) 
{ 
  t->stop = read_timer(); 
  t->total += t->stop - t->start;
  t->count += 1;
}

inline 
double timer_delta(acc_timer_type* t) 
{ 
  return (double)(t->stop - t->start) / TIMEBASE;  
}

inline
double timer_total(acc_timer_type* t) 
{ 
  return (double)t->total / TIMEBASE;  
}

#endif // VSIP_OPT_CBE_SPU_VMUL_SPLIT_H
