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
                  fftw3, ipp, sal, cvsip, cbe_sdk, cuda, builtin, dft, or no_fft [[builtin]].]),,
  [enable_fft=builtin])
  
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


AC_ARG_WITH(fftw3_cflags,
  AS_HELP_STRING([--with-fftw3-cflags=CFLAGS],
                 [Specify CFLAGS to use when building built-in FFTW3.
		  Only used if --enable-fft=builtin.]))

AC_ARG_WITH(fftw3_cfg_opts,
  AS_HELP_STRING([--with-fftw3-cfg-opts=OPTS],
                 [Specify additional options to use when configuring built-in
                  FFTW3. Only used if --enable-fft=builtin.]))

AC_ARG_ENABLE(fftw3_simd,
  AS_HELP_STRING([--disable-fftw3-simd],
                 [Disable use of SIMD instructions by FFTW3.  Useful
		  when cross-compiling for a host that does not have
		  SIMD ISA]),,
  [enable_fftw3_simd=yes])

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
enable_builtin_fft="no"

if test "$enable_fft_float" = yes -o \
        "$enable_fft_double" = yes -o \
        "$enable_fft_long_double" = yes ; then

  for fft_be in ${fft_backends} ; do
    case ${fft_be} in
      sal) enable_sal_fft="yes";;
      ipp) enable_ipp_fft="yes";;
      cvsip) enable_cvsip_fft="yes";;
      fftw3) enable_fftw3="yes";;
      builtin) enable_builtin_fft="yes";;
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
  if test "$enable_fftw3" != "no" -a "$enable_builtin_fft" != "no" ; then
    AC_MSG_ERROR([Cannot use both external as well as builtin fftw3 libraries.])
  fi
  if test "$enable_fftw3" != "no" -o "$enable_builtin_fft" != "no" ; then
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
if test "$enable_builtin_fft" != "no"; then

  AC_MSG_NOTICE([Using built-in FFTW3 support.])

  # Build and use builtin fftw3.
  AC_MSG_CHECKING([for built-in FFTW3 library source])
  libs=
  fftw3_configure="$srcdir/vendor/fftw/configure"
  if test -e "$fftw3_configure"; then
    AC_MSG_RESULT([found])

    # assert(NOT CROSS-COMPILING)

    # Determine whether long double is supported.
    AC_CHECK_SIZEOF(double)
    AC_CHECK_SIZEOF(long double)
    if test "$enable_fft_long_double" = yes; then
      AC_MSG_CHECKING([for long double support])
      if test $ac_cv_sizeof_long_double = 0; then
        AC_MSG_RESULT([not a supported type.])
        AC_MSG_NOTICE([Disabling FFT support (--disable-fft-long-double).])
        enable_fft_long_double=no 
      elif test $ac_cv_sizeof_long_double = $ac_cv_sizeof_double; then
        AC_MSG_RESULT([same size as double.])
        AC_MSG_NOTICE([Disabling FFT support (--disable-fft-long-double).])
        enable_fft_long_double=no 
      else
        AC_MSG_RESULT([supported.])
      fi
    fi

    # if $srcdir is relative, correct for chdir into vendor/fftw3*.
    fftw3_configure="`(cd $srcdir/vendor/fftw; echo \"$PWD\")`"/configure

    fftw3_opts="--disable-dependency-tracking --silent"
    fftw3_opts="$fftw3_opts --disable-fortran"
    if test "x$host" != "x"; then
      fftw3_opts="$fftw3_opts --host=$host"
    fi
    if test "x$build" != "x"; then
      fftw3_opts="$fftw3_opts --build=$build"
    fi
    if test "x$target" != "x"; then
      fftw3_opts="$fftw3_opts --target=$target"
    fi
    if test "x$BUILD_SHARED_LIBS" != "x"; then
      fftw3_opts="$fftw3_opts --enable-shared"
    fi

    fftw3_f_simd=
    fftw3_d_simd=
    fftw3_l_simd=
    if test "$enable_fftw3_simd" = "yes"; then
      case "$host_cpu" in
        ia32|i686|x86_64) fftw3_f_simd="--enable-sse"
	                  fftw3_d_simd="--enable-sse2" 
	                  ;;
        ppc*)             fftw3_f_simd="--enable-altivec" ;;
        powerpc*)         fftw3_f_simd="--enable-altivec" ;;
      esac
    fi
    AC_MSG_NOTICE([fftw3 config options: $fftw3_opts $fftw3_simd.])

    # We don't export CFLAGS to FFTW configure because this overrides its
    # choice of optimization flags (unless the --with-fftw3-cflags options
    # is given).  Because of this, we need to pass -m32/-m64 as part of CC.
    if expr "$CFLAGS" : ".*-m32" > /dev/null; then
      fftw_CC="$CC -m32"
    elif expr "$CFLAGS" : ".*-m64" > /dev/null; then
      fftw_CC="$CC -m64"
    else
      fftw_CC="$CC"
    fi

    # Add to keep abi flags in all modules consistent, regardless of
    # whether they actually contain altivec code or not.
    case "$host_cpu" in
      ppc*)             fftw_CC="$fftw_CC -mabi=altivec" ;;
      powerpc*)         fftw_CC="$fftw_CC -mabi=altivec" ;;
    esac

    keep_CFLAGS="$CFLAGS"

    if test "x$with_fftw3_cflags" != "x"; then
      export CFLAGS="$with_fftw3_cflags"
    else
      unset CFLAGS
    fi

    echo "==============================================================="

    if test "$enable_fft_float" = yes; then
      fftw_has_float=1
      mkdir -p vendor/fftw3f
      AC_MSG_NOTICE([Configuring fftw3f (float).])
      AC_MSG_NOTICE([extra config options: '$fftw3_f_simd'.])
      (cd vendor/fftw3f; $fftw3_configure CC="$fftw_CC" $fftw3_f_simd $fftw3_opts $with_fftw3_cfg_opts --enable-float)
      libs="$libs -lfftw3f"
    fi
    if test "$enable_fft_double" = yes; then
      fftw_has_double=1
      mkdir -p vendor/fftw3
      AC_MSG_NOTICE([Configuring fftw3 (double).])
      AC_MSG_NOTICE([extra config options: '$fftw3_d_simd'.])
      (cd vendor/fftw3; $fftw3_configure CC="$fftw_CC" $fftw3_d_simd $fftw3_opts $with_fftw3_cfg_opts )
      libs="$libs -lfftw3"
    fi
    if test "$enable_fft_long_double" = yes; then
      fftw_has_long_double=1
      # fftw3l config doesn't get SIMD option
      mkdir -p vendor/fftw3l
      AC_MSG_NOTICE([Configuring fftw3l (long double).])
      AC_MSG_NOTICE([extra config options: '$fftw3_l_simd'.])
      (cd vendor/fftw3l; $fftw3_configure CC="$fftw_CC" $fftw3_l_simd $fftw3_opts $with_fftw3_cfg_opts --enable-long-double)
      libs="$libs -lfftw3l"
    fi

    echo "==============================================================="

    export CFLAGS="$keep_CFLAGS"

    # these don't refer to anything yet.
    if test "$enable_fft_float" = yes; then
      AC_SUBST(USE_BUILTIN_FFTW_FLOAT, 1)
    fi
    if test "$enable_fft_double" = yes; then
      AC_SUBST(USE_BUILTIN_FFTW_DOUBLE, 1)
    fi
    if test "$enable_fft_long_double" = yes; then
      AC_SUBST(USE_BUILTIN_FFTW_LONG_DOUBLE, 1)
    fi
    mkdir -p src
    cp $srcdir/vendor/fftw/api/fftw3.h src/fftw3.h
  else
    AC_MSG_RESULT([not found])
  fi

   
  if test \( "$enable_fft_float" != yes -o -f "vendor/fftw3f/Makefile" \) -a \
          \( "$enable_fft_double" != yes -o -f "vendor/fftw3/Makefile" \) -a \
          \( "$enable_fft_long_double" != yes -o -f "vendor/fftw3l/Makefile" \)
  then
    AC_MSG_RESULT([Built-in FFTW3 configures successful.])
  else
    AC_MSG_ERROR([Built-in FFTW3 configures FAILED, see config.log
                  and vendor/fftw3*/config.log.])
  fi

  curdir=`pwd`
  if test "`echo $srcdir | sed -n '/^\//p'`" != ""; then
    my_abs_top_srcdir="$srcdir"
  else
    my_abs_top_srcdir="$curdir/$srcdir"
  fi

  FFTW3_LIBS="$libs"
  AC_MSG_NOTICE([Will link with $FFTW3_LIBS.])

  AC_SUBST(USE_BUILTIN_FFTW, 1)

  # These libraries have not been built yet so we have to wait before
  # adding them to LIBS (otherwise subsequent AC_LINK_IFELSE's will
  # fail).  Instead we add them to LATE_LIBS, which gets added to
  # LIBS just before AC_OUTPUT.

  LATE_LIBS="$FFTW3_LIBS $LATE_LIBS"
  CPPFLAGS="-I$includedir/fftw3 $CPPFLAGS"

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
