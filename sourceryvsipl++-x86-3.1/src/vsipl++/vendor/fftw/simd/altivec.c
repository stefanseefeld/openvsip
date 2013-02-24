/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */


#include "ifftw.h"
#include "simd.h"

#if HAVE_ALTIVEC

#if HAVE_SYS_SYSCTL_H
#include <sys/sysctl.h>
#endif

#if HAVE_SYS_SYSCTL_H && HAVE_SYSCTL && defined(CTL_HW) && defined(HW_VECTORUNIT)
/* code for darwin */
static int really_have_altivec(void)
{
     int mib[2], altivecp;
     size_t len;
     mib[0] = CTL_HW;
     mib[1] = HW_VECTORUNIT;
     len = sizeof(altivecp);
     sysctl(mib, 2, &altivecp, &len, NULL, 0);
     return altivecp;
} 
#else /* HAVE_SYS_SYSCTL_H etc. */

#include <signal.h>
#include <setjmp.h>

static jmp_buf jb;

static void sighandler(int x)
{
     longjmp(jb, 1);
}

static int really_have_altivec(void)
{
     void (*oldsig)(int);
     oldsig = signal(SIGILL, sighandler);
     if (setjmp(jb)) {
	  signal(SIGILL, oldsig);
	  return 0;
     } else {
	  __asm__ __volatile__ (".long 0x10000484"); /* vor 0,0,0 */
	  signal(SIGILL, oldsig);
	  return 1;
     }
     return 0;
}
#endif

int RIGHT_CPU(void)
{
     static int init = 0, res;
     if (!init) {
	  res = really_have_altivec();
	  init = 1;
     }
     return res;
}
#endif
