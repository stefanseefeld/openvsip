/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "ifftw.h"
#include "simd.h"

#if HAVE_SSE

# if defined(__x86_64__) || defined(_M_X64) || defined(_M_AMD64)

  int RIGHT_CPU(void)
  {
       return 1;
  }

# else /* !x86_64 */

# include <signal.h>
# include <setjmp.h>
# include "x86-cpuid.h"

  static jmp_buf jb;

  static void sighandler(int x)
  {
       UNUSED(x);
       longjmp(jb, 1);
  }

  static int sse_works(void)
  {
       void (*oldsig)(int);
       oldsig = signal(SIGILL, sighandler);
       if (setjmp(jb)) {
	    signal(SIGILL, oldsig);
	    return 0;
       } else {
#        ifdef _MSC_VER
	    _asm { xorps xmm0,xmm0 }
#        else
	    /* asm volatile ("xorps %xmm0, %xmm0"); */
	    asm volatile (".byte 0x0f; .byte 0x57; .byte 0xc0");
#        endif
	    signal(SIGILL, oldsig);
	    return 1;
       }
  }

  int RIGHT_CPU(void)
  {
       static int init = 0, res;
       extern void X(check_alignment_of_sse_pmpm)(void);

       if (!init) {
	    res =   !is_386() 
		 && has_cpuid()
		 && (cpuid_edx(1) & (1 << 25)) 
		 && sse_works();
	    init = 1;
	    X(check_alignment_of_sse_pmpm)();
       }
       return res;
  }
# endif /* x86_64 */

#endif /* HAVE_SSE */
