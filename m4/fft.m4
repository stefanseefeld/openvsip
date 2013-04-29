dnl Copyright (c) 2007 by CodeSourcery, Inc.  All rights reserved.
dnl
dnl File:   fft.m4
dnl Author: Stefan Seefeld
dnl Date:   2007-12-20
dnl
dnl Contents: fft configuration for Sourcery VSIPL++
dnl

AC_DEFUN([SVXX_CHECK_FFT],
[
AC_ARG_ENABLE(fft,
  AS_HELP_STRING([--enable-fft],
                 [Specify list of FFT engines. Available engines are:
                  fftw3, ipp, sal, cvsip, cbe_sdk, cuda, dft, or no_fft [[fftw3]].]),,
  [enable_fft=fftw3])
  
AC_ARG_WITH(fftw3_prefix,
  AS_HELP_STRING([--with-fftw3-prefix=PATH],
                 [Specify the installation prefix of the fftw3 library.
                  Headers must be in PATH/include; libraries in PATH/lib.]))

AC_ARG_ENABLE([fft-float],
  AS_HELP_STRING([--disable-fft-float],
                 [Omit support for FFT applied to float elements.]),,
  [enable_fft_float=yes])

AC_ARG_ENABLE([fft-double],
  AS_HELP_STRING([--disable-fft-double],
                 [Omit support for FFT applied to double elements.]),,
  [enable_fft_double=yes])

AC_ARG_ENABLE([fft-long-double],
  AS_HELP_STRING([--disable-fft-long-double],
                 [Omit support for FFT applied to long double elements.]),,
  [enable_fft_long_double=yes])

#
# Find the FFT backends.
# At present, SAL, IPP, and FFTW3 are supported.
#
if test "$enable_fft_float" = yes; then
  vsip_impl_fft_use_float=1
fi
if test "$enable_fft_double" = yes; then
  vsip_impl_fft_use_double=1
fi
if test "$enable_fft_long_double" = yes; then
  vsip_impl_fft_use_long_double=1
fi

if test "$only_ref_impl" = "1"; then
  enable_fft="cvsip"
fi


fft_backends=`echo "${enable_fft}" | \
                sed -e 's/[[ 	,]][[ 	,]]*/ /g' -e 's/,$//'`

enable_fftw3="no"
enable_ipp_fft="no"
enable_sal_fft="no"
enable_cvsip_fft="no"
enable_cbe_sdk_fft="no"
enable_cuda_fft="no"

if test "$enable_fft_float" = yes -o \
        "$enable_fft_double" = yes -o \
        "$enable_fft_long_double" = yes ; then

  for fft_be in ${fft_backends} ; do
    case ${fft_be} in
      sal) enable_sal_fft="yes";;
      ipp) enable_ipp_fft="yes";;
      cvsip) enable_cvsip_fft="yes";;
      fftw3) enable_fftw3="yes";;
      cbe_sdk)
        if test "with_cbe_sdk" == "no" ; then
          AC_MSG_ERROR([The cbe_sdk FFT backend requires --with-cbe-sdk.])
        fi
        AC_SUBST(VSIP_IMPL_CBE_SDK_FFT, 1)
        AC_DEFINE_UNQUOTED(VSIP_IMPL_CBE_SDK_FFT, 1,
          [Define to enable Cell/B.E. SDK FFT backend.])
        ;;
      cuda) 
        enable_cuda_fft="yes"
        if test "$with_cuda" != "yes"; then
	  AC_MSG_ERROR([The cuda FFT backend requires --with-cuda.])
        fi
        ;;
      dft)
        AC_SUBST(VSIP_IMPL_DFT_FFT, 1)
        AC_DEFINE_UNQUOTED(VSIP_IMPL_DFT_FFT, 1,
          [Define to enable DFT FFT backend.])
        ;;
      no_fft)
        AC_SUBST(VSIP_IMPL_NO_FFT, 1)
        AC_DEFINE_UNQUOTED(VSIP_IMPL_NO_FFT, 1,
          [Define to enable dummy FFT backend.])
        ;;
      *) AC_MSG_ERROR([Unknown fft engine ${fft_be}.]);;
    esac
  done
  if test "x$with_fftw3_prefix" != x; then
    enable_fftw3="yes"
  fi
  if test "$enable_fftw3" != "no" ; then
    AC_SUBST(VSIP_IMPL_FFTW3, 1)
    if test "$neutral_acconfig" = 'y'; then
      CPPFLAGS="$CPPFLAGS -DVSIP_IMPL_FFTW3=1"
    else
      AC_DEFINE_UNQUOTED(VSIP_IMPL_FFTW3, 1, [Define to build using FFTW3 headers.])
    fi
  fi
fi

dnl
dnl fftw3 needs some special care, so we will do some extra checks here.
dnl
if test "$enable_fftw3" != "no"; then

  if test -n "$with_fftw3_prefix"; then
    CPPFLAGS="-I$with_fftw3_prefix/include $CPPFLAGS"
    LIBS="-L$with_fftw3_prefix/lib $LIBS"
  fi
  AC_CHECK_HEADERS([fftw3.h], [],
    [ AC_MSG_ERROR([FFTW3 enabled but no fftw3.h found.])],
    [// no prerequisites])

  if test "$enable_fft_float" = yes ; then
    keep_LIBS="$LIBS"
    LIBS="$LIBS -lfftw3f"
    syms="const char* fftwf_version;"

    AC_MSG_CHECKING([if external FFTW3 library supports float])
    AC_LINK_IFELSE(
      [AC_LANG_PROGRAM([#include <fftw3.h>], [$syms])],
      [AC_MSG_RESULT([yes.])
       fftw_has_float=1],
      [AC_MSG_RESULT([no.])
       LIBS=$keep_LIBS])
  fi
  if test "$enable_fft_double" = yes ; then
    keep_LIBS="$LIBS"
    LIBS="$LIBS -lfftw3"
    syms="const char* fftw_version;"

    AC_MSG_CHECKING([if external FFTW3 library supports double])
    AC_LINK_IFELSE(
      [AC_LANG_PROGRAM([#include <fftw3.h>], [$syms])],
      [AC_MSG_RESULT([yes.])
       fftw_has_double=1],
      [AC_MSG_RESULT([no.])
       LIBS=$keep_LIBS])
  fi
  if test "$enable_fft_long_double" = yes; then
    keep_LIBS="$LIBS"
    LIBS="$LIBS -lfftw3l"
    syms="const char* fftwl_version;"

    AC_MSG_CHECKING([if external FFTW3 library supports long double])
    AC_LINK_IFELSE(
      [AC_LANG_PROGRAM([#include <fftw3.h>], [$syms])],
      [AC_MSG_RESULT([yes.])
       fftw_has_long_double=1],
      [AC_MSG_RESULT([no.])
       LIBS=$keep_LIBS])
  fi
fi
if test "x$provide_fft_float" = "x"
then provide_fft_float=$fftw_has_float
fi
if test "x$provide_fft_double" = "x"
then provide_fft_double=$fftw_has_double
fi
if test "x$provide_fft_long_double" = "x"
then provide_fft_long_double=$fftw_has_long_double
fi

if test "$neutral_acconfig" = 'y'; then
  if test "$fftw_has_float" = 1; then
    CPPFLAGS="$CPPFLAGS -DVSIP_IMPL_FFTW3_HAVE_FLOAT"
  fi
  if test "$fftw_has_double" = 1; then
    CPPFLAGS="$CPPFLAGS -DVSIP_IMPL_FFTW3_HAVE_DOUBLE"
  fi
  if test "$fftw_has_long_double" = 1; then
    CPPFLAGS="$CPPFLAGS -DVSIP_IMPL_FFTW3_HAVE_LONG_DOUBLE"
  fi
else
  if test "$fftw_has_float" = 1; then
    AC_DEFINE_UNQUOTED(VSIP_IMPL_FFTW3_HAVE_FLOAT, $fftw_has_float,
      [Define to 1 if -lfftw3f was found.])
  fi
  if test "$fftw_has_double" = 1; then
    AC_DEFINE_UNQUOTED(VSIP_IMPL_FFTW3_HAVE_DOUBLE, $fftw_has_double, 
      [Define to 1 if -lfftw3d was found.])
  fi
  if test "$fftw_has_long_double" = 1; then
    AC_DEFINE_UNQUOTED(VSIP_IMPL_FFTW3_HAVE_LONG_DOUBLE, $fftw_has_long_double,
      [Define to 1 if -lfftw3l was found.])
  fi
fi

])
