/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"

/* if F77_FUNC is not defined, then we don't know how to mangle identifiers
   for the Fortran linker, and we must omit the f77 API. */
#if defined(F77_FUNC) || defined(WINDOWS_F77_MANGLING)

#include "x77.h"

#define F77(a, A) F77x(x77(a), X77(A))

#ifndef WINDOWS_F77_MANGLING

#if defined(F77_FUNC)
#  define F77x(a, A) F77_FUNC(a, A)
#  include "f77funcs.h"
#endif

#if defined(F77_FUNC_) && !defined(F77_FUNC_EQUIV)
#  undef F77x
#  define F77x(a, A) F77_FUNC_(a, A)
#  include "f77funcs.h"
#endif

#else /* WINDOWS_F77_MANGLING */

/* Various mangling conventions common (?) under Windows. */

/* g77 */
#  define WINDOWS_F77_FUNC(a, A) a ## __
#  define F77x(a, A) WINDOWS_F77_FUNC(a, A)
#  include "f77funcs.h"

/* Intel, etc. */
#  undef WINDOWS_F77_FUNC
#  define WINDOWS_F77_FUNC(a, A) a ## _
#  include "f77funcs.h"

/* Digital/Compaq/HP Visual Fortran, Intel Fortran.  stdcall attribute
   is apparently required to adjust for calling conventions (callee
   pops stack in stdcall).  See also:
       http://msdn.microsoft.com/library/en-us/vccore98/html/_core_mixed.2d.language_programming.3a_.overview.asp
*/
#  undef WINDOWS_F77_FUNC
#  if defined(__GNUC__)
#    define WINDOWS_F77_FUNC(a, A) __attribute__((stdcall)) A
#  elif defined(_MSC_VER) || defined(_ICC) || defined(_STDCALL_SUPPORTED)
#    define WINDOWS_F77_FUNC(a, A) __stdcall A
#  else
#    define WINDOWS_F77_FUNC(a, A) A /* oh well */
#  endif
#  include "f77funcs.h"

#endif /* WINDOWS_F77_MANGLING */

#endif				/* F77_FUNC */
