/*
 * Copyright (c) 2001 Matteo Frigo
 * Copyright (c) 2001 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "bench.h"

int aligned_main(int argc, char *argv[])
{
#if defined(__GNUC__) && defined(__i386__)
     /*
      * horrible hack to align the stack to a 16-byte boundary.
      *
      * We assume a gcc version >= 2.95 so that
      * -mpreferred-stack-boundary works.  Otherwise, all bets are
      * off.  However, -mpreferred-stack-boundary does not create a
      * stack alignment, but it only preserves it.  Unfortunately,
      * many versions of libc on linux call main() with the wrong
      * initial stack alignment, with the result that the code is now
      * pessimally aligned instead of having a 50% chance of being
      * correct.
      */
     {
	  /*
	   * Use alloca to allocate some memory on the stack.
	   * This alerts gcc that something funny is going
	   * on, so that it does not omit the frame pointer
	   * etc.
	   */
	  (void)__builtin_alloca(16); 

	  /*
	   * Now align the stack pointer
	   */
	  __asm__ __volatile__ ("andl $-16, %esp");

#  ifdef FFTW_DEBUG_ALIGNMENT
	  /* pessimally align the stack, in order to check whether the
	     stack re-alignment hacks in FFTW3 work */
	  __asm__ __volatile__ ("addl $-4, %esp");
#  endif
     }
#endif

#ifdef __ICC /* Intel's compiler for ia32 */
     {
	  /*
	   * Simply calling alloca seems to do the right thing. 
	   * The size of the allocated block seems to be irrelevant.
	   */
	  _alloca(16);
     }
#endif

     return bench_main(argc, argv);
}
