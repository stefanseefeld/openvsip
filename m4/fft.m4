dnl
dnl Copyright (c) 2007 by CodeSourcery
dnl Copyright (c) 2013 Stefan Seefeld
dnl All rights reserved.
dnl
dnl This file is part of OpenVSIP. It is made available under the
dnl license contained in the accompanying LICENSE.BSD file.

AC_DEFUN([OVXX_CHECK_FFT],
[
AC_ARG_ENABLE(fft,
  AS_HELP_STRING([--enable-fft],
                 [Specify list of FFT engines. Available engines are:
                  fftw, ipp, sal, cvsip, cuda, dft, or no_fft [[fftw]].]),,
  [enable_fft=fftw])
  
AC_ARG_WITH(fftw_prefix,
  AS_HELP_STRING([--with-fftw-prefix=PATH],
                 [Specify the installation prefix of the fftw library.
                  Headers must be in PATH/include; libraries in PATH/lib.]))

AC_ARG_ENABLE(fftw_threads,
  AS_HELP_STRING([--enable-fftw-threads],
                 [Specify that the multi-threaded FFTW API is to be used.]),,
  [enable_fftw_threads=false])


fft_backends=`echo "${enable_fft}" | \
                sed -e 's/[[ 	,]][[ 	,]]*/ /g' -e 's/,$//'`

enable_fftw="no"
enable_ipp_fft="no"
enable_sal_fft="no"
enable_cvsip_fft="no"
enable_cuda_fft="no"

for fft_be in ${fft_backends} ; do
  case ${fft_be} in
    sal) enable_sal_fft="yes";;
    ipp) enable_ipp_fft="yes";;
    cvsip) enable_cvsip_fft="yes";;
    fftw) enable_fftw="yes";;
    cuda) 
      enable_cuda_fft="yes"
      if test "$with_cuda" != "yes"; then
        AC_MSG_ERROR([The cuda FFT backend requires --with-cuda.])
      fi
      ;;
    dft)
      AC_SUBST(OVXX_DFT_FFT, 1)
      AC_DEFINE_UNQUOTED(OVXX_DFT_FFT, 1,
        [Define to enable DFT FFT backend.])
      ;;
    no_fft)
      AC_SUBST(OVXX_NO_FFT, 1)
      AC_DEFINE_UNQUOTED(OVXX_NO_FFT, 1,
        [Define to enable dummy FFT backend.])
      ;;
    *) AC_MSG_ERROR([Unknown fft engine ${fft_be}.]);;
  esac
done
if test "x$with_fftw_prefix" != x; then
  enable_fftw="yes"
fi
if test "$enable_fftw" != "no" ; then
  AC_SUBST(OVXX_FFTW, 1)
  AC_DEFINE_UNQUOTED(OVXX_FFTW, 1, [Define to build using FFTW headers.])
fi
if test "$enable_fftw_threads" != false ; then
  AC_SUBST(OVXX_FFTW_THREADS, 1)
  AC_DEFINE_UNQUOTED(OVXX_FFTW_THREADS, 1, [Define to build using multi-threaded FFTW API.])
fi

dnl
dnl fftw needs some special care, so we will do some extra checks here.
dnl
if test "$enable_fftw" != "no"; then

  if test -n "$with_fftw_prefix"; then
    CPPFLAGS="-I$with_fftw_prefix/include $CPPFLAGS"
    LIBS="-L$with_fftw_prefix/lib $LIBS"
  fi
  AC_CHECK_HEADERS([fftw3.h], [],
    [ AC_MSG_ERROR([FFTW enabled but no fftw3.h found.])],
    [// no prerequisites])
  keep_LIBS="$LIBS"
  if test "$enable_fftw_threads" = yes ; then
    LIBS="$LIBS -lfftw3f_threads -lfftw3f"
  else
    LIBS="$LIBS -lfftw3f"
  fi
  syms="const char* fftwf_version; (void)fftwf_version;"

  AC_MSG_CHECKING([if FFTW library supports float])
  AC_LINK_IFELSE(
    [AC_LANG_PROGRAM([#include <fftw3.h>], [$syms])],
    [AC_MSG_RESULT([yes.])
     fftw_has_float=1],
    [AC_MSG_RESULT([no.])
     LIBS=$keep_LIBS])

  keep_LIBS="$LIBS"
  if test "$enable_fftw_threads" = yes ; then
    LIBS="$LIBS -lfftw3_threads -lfftw3"
  else
    LIBS="$LIBS -lfftw3"
  fi
  syms="const char* fftw_version; (void)fftw_version;"

  AC_MSG_CHECKING([if FFTW library supports double])
  AC_LINK_IFELSE(
    [AC_LANG_PROGRAM([#include <fftw3.h>], [$syms])],
    [AC_MSG_RESULT([yes.])
     fftw_has_double=1],
    [AC_MSG_RESULT([no.])
     LIBS=$keep_LIBS])

  if test "$fftw_has_float" = 1; then
    AC_DEFINE_UNQUOTED(OVXX_FFTW_HAVE_FLOAT, $fftw_has_float,
      [Define to 1 if -lfftw3f was found.])
  fi
  if test "$fftw_has_double" = 1; then
    AC_DEFINE_UNQUOTED(OVXX_FFTW_HAVE_DOUBLE, $fftw_has_double, 
      [Define to 1 if -lfftw3d was found.])
  fi
fi
])
