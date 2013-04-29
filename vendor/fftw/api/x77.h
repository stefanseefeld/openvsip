/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

/* Fortran-like (e.g. as in BLAS) type prefixes for F77 interface */
#if defined(FFTW_SINGLE)
#  define x77(name) CONCAT(sfftw_, name)
#  define X77(NAME) CONCAT(SFFTW_, NAME)
#elif defined(FFTW_LDOUBLE)
/* FIXME: what is best?  BLAS uses D..._X, apparently.  Ugh. */
#  define x77(name) CONCAT(lfftw_, name)
#  define X77(NAME) CONCAT(LFFTW_, NAME)
#else
#  define x77(name) CONCAT(dfftw_, name)
#  define X77(NAME) CONCAT(DFFTW_, NAME)
#endif

/* If F77_FUNC is not defined and the user didn't explicitly specify
   --disable-fortran, then make our best guess at default wrappers
   (since F77_FUNC_EQUIV should not be defined in this case, we
    will use both double-underscored g77 wrappers and single- or
    non-underscored wrappers).  This saves us from dealing with
    complaints in the cases where the user failed to specify
    an F77 compiler or wrapper detection failed for some reason. */
#if !defined(F77_FUNC) && !defined(DISABLE_FORTRAN)
#  if (defined(_WIN32) || defined(__WIN32__)) && !defined(WINDOWS_F77_MANGLING)
#    define WINDOWS_F77_MANGLING 1
#  endif
#  if defined(_AIX) || defined(__hpux) || defined(hpux)
#    define F77_FUNC(a, A) a
#  elif defined(CRAY) || defined(_CRAY) || defined(_UNICOS)
#    define F77_FUNC(a, A) A
#  else
#    define F77_FUNC(a, A) a ## _
#  endif
#  define F77_FUNC_(a, A) a ## __
#endif

#if defined(WITH_G77_WRAPPERS) && !defined(DISABLE_FORTRAN)
#  undef F77_FUNC_
#  define F77_FUNC_(a, A) a ## __
#  undef F77_FUNC_EQUIV
#endif

/* annoying Windows syntax for shared-library declarations */
#if defined(FFTW_DLL) && (defined(_WIN32) || defined(__WIN32__))
#  define FFTW_VOIDFUNC __declspec(dllexport) void
#else
#  define FFTW_VOIDFUNC void
#endif
