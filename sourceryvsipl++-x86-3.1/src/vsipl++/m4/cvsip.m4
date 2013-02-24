dnl Copyright (c) 2007 by CodeSourcery, Inc.  All rights reserved.
dnl
dnl File:   cvsip.m4
dnl Author: Stefan Seefeld
dnl Date:   2007-12-28
dnl
dnl Contents: C-VSIPL configuration for Sourcery VSIPL++
dnl

AC_DEFUN([SVXX_CHECK_CVSIP],
[
#
# Find the C-VSIPL library, if enabled.
#

if test "$with_cvsip" != "no"; then
  if test -n "$with_cvsip_prefix"; then
    CVSIP_CPPFLAGS="-I$with_cvsip_prefix/include"
    CVSIP_LDFLAGS="-L$with_cvsip_prefix/lib"
  fi
  CPPFLAGS="$CPPFLAGS $CVSIP_CPPFLAGS"
  AC_CHECK_HEADER([vsip.h])
  LDFLAGS="$LDFLAGS $CVSIP_LDFLAGS"
  AC_CHECK_LIB(vsip, vsip_vcreate_bl,[ cvsip_have_bool=1])
  AC_CHECK_LIB(vsip, vsip_vcreate_i,[ cvsip_have_int=1])
  AC_CHECK_LIB(vsip, vsip_ccfftop_create_f,[ cvsip_have_float=1])
  AC_CHECK_LIB(vsip, vsip_ccfftop_create_d,[ cvsip_have_double=1])
  if test -n "$cvsip_have_float" -o -n "$cvsip_have_double"; then
    LIBS="-lvsip $LIBS"
    AC_CHECK_FUNCS([vsip_conv1d_create_f vsip_conv1d_create_d\
                    vsip_conv2d_create_f vsip_conv2d_create_d\
                    vsip_corr1d_create_f vsip_corr1d_create_d\
                    vsip_corr2d_create_f vsip_corr2d_create_d],,,
      [#include <vsip.h>])
    AC_CHECK_TYPES([vsip_fir_attr],[],[],[[#include <vsip.h>]])
    AC_SUBST(VSIP_IMPL_CVSIP_HAVE_BOOL, $cvsip_have_bool)
    AC_SUBST(VSIP_IMPL_CVSIP_HAVE_INT, $cvsip_have_int)
    AC_SUBST(VSIP_IMPL_CVSIP_HAVE_FLOAT, $cvsip_have_float)
    AC_SUBST(VSIP_IMPL_CVSIP_HAVE_DOUBLE, $cvsip_have_double)
    AC_SUBST(VSIP_IMPL_HAVE_CVSIP, 1)
    if test "$neutral_acconfig" = 'y'; then
      CPPFLAGS="$CPPFLAGS -DVSIP_IMPL_HAVE_CVSIP=1"
      CPPFLAGS="$CPPFLAGS -DVSIP_IMPL_CVSIP_HAVE_BOOL=$cvsip_have_bool"
      CPPFLAGS="$CPPFLAGS -DVSIP_IMPL_CVSIP_HAVE_INT=$cvsip_have_int"
      CPPFLAGS="$CPPFLAGS -DVSIP_IMPL_CVSIP_HAVE_FLOAT=$cvsip_have_float"
      CPPFLAGS="$CPPFLAGS -DVSIP_IMPL_CVSIP_HAVE_DOUBLE=$cvsip_have_double"
    else
      AC_DEFINE_UNQUOTED(VSIP_IMPL_HAVE_CVSIP, 1,
        [Define to use C-VSIPL library.])
      AC_DEFINE_UNQUOTED(VSIP_IMPL_CVSIP_HAVE_BOOL, $cvsip_have_bool,
        [Define if C-VSIPL supports bool views.])
      AC_DEFINE_UNQUOTED(VSIP_IMPL_CVSIP_HAVE_INT, $cvsip_have_int,
        [Define if C-VSIPL supports int views.])
      AC_DEFINE_UNQUOTED(VSIP_IMPL_CVSIP_HAVE_FLOAT, $cvsip_have_float,
        [Define if C-VSIPL supports float operations.])
      AC_DEFINE_UNQUOTED(VSIP_IMPL_CVSIP_HAVE_DOUBLE, $cvsip_have_double,
        [Define if C-VSIPL supports double operations.])
    fi
  fi
  if test "$enable_cvsip_fft" != "no"; then 
    if test "$cvsip_have_float" = "1"; then
      provide_fft_float=1
    fi
    if test "$cvsip_have_double" = "1"; then
      provide_fft_double=1
    fi
    AC_SUBST(VSIP_IMPL_CVSIP_FFT, 1)
    if test "$neutral_acconfig" = 'y'; then
      CPPFLAGS="$CPPFLAGS -DVSIP_IMPL_CVSIP_FFT=1"
    else
      AC_DEFINE_UNQUOTED(VSIP_IMPL_CVSIP_FFT, 1,
            [Define to use the C-VSIPL library to perform FFTs.])
    fi
  fi
fi

])
