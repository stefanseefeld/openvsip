dnl
dnl Copyright (c) 2007 by CodeSourcery
dnl Copyright (c) 2013 Stefan Seefeld
dnl All rights reserved.
dnl
dnl This file is part of OpenVSIP. It is made available under the
dnl license contained in the accompanying LICENSE.BSD file.

AC_DEFUN([OVXX_CHECK_CVSIP],
[
#
# Find the C-VSIPL library, if enabled.
#

if test "$with_cvsip" != "no"; then
  if test -n "$with_cvsip_prefix"; then
    CVSIP_CPPFLAGS="-I$with_cvsip_prefix/include"
    CVSIP_LDFLAGS="-L$with_cvsip_prefix/lib"
  fi
  if test -n "$with_cvsip_include"; then
    CVSIP_CPPFLAGS="-I$with_cvsip_include"
  fi
  if test -n "$with_cvsip_lib"; then
    CVSIP_LDFLAGS="-L$with_cvsip_lib"
  fi
  CPPFLAGS="$CPPFLAGS $CVSIP_CPPFLAGS"
  AC_CHECK_HEADER([vsip.h])
  LDFLAGS="$LDFLAGS $CVSIP_LDFLAGS"
  AC_CHECK_LIB(vsip, vsip_init,[],[AC_MSG_ERROR([unable to link to -lvsip])])
  AC_CHECK_LIB(vsip, vsip_vcreate_bl,[ cvsip_have_bool=1],[ cvsip_have_bool=0])
  AC_CHECK_LIB(vsip, vsip_vcreate_i,[ cvsip_have_int=1],[ cvsip_have_int=0])
  AC_CHECK_LIB(vsip, vsip_mgetrowlength_bl,[],[ cvsip_have_bool=0])
  AC_CHECK_LIB(vsip, vsip_mgetrowlength_i,[],[ cvsip_have_int=0])
  AC_CHECK_LIB(vsip, vsip_ccfftop_create_f,[ cvsip_have_float=1],[ cvsip_have_float=0])
  AC_CHECK_LIB(vsip, vsip_ccfftop_create_d,[ cvsip_have_double=1],[ cvsip_have_double=0])
  if test "$cvsip_have_float" = '1' -o "$cvsip_have_double" = '1'; then
    LIBS="-lvsip $LIBS"
    AC_CHECK_FUNCS([vsip_conv1d_create_f vsip_conv1d_create_d\
                    vsip_conv2d_create_f vsip_conv2d_create_d\
                    vsip_corr1d_create_f vsip_corr1d_create_d\
                    vsip_corr2d_create_f vsip_corr2d_create_d\
                    vsip_msumval_bl vsip_msumval_i\
                    vsip_msumval_f vsip_msumval_d\
                    vsip_mmaxval_f vsip_mmaxval_d],,,
      [#include <vsip.h>])
    AC_CHECK_TYPES([vsip_fir_attr],[],[],[[#include <vsip.h>]])
    AC_SUBST(OVXX_CVSIP_HAVE_BOOL, $cvsip_have_bool)
    AC_SUBST(OVXX_CVSIP_HAVE_INT, $cvsip_have_int)
    AC_SUBST(OVXX_CVSIP_HAVE_FLOAT, $cvsip_have_float)
    AC_SUBST(OVXX_CVSIP_HAVE_DOUBLE, $cvsip_have_double)
    AC_SUBST(OVXX_HAVE_CVSIP, 1)
    AC_DEFINE_UNQUOTED(OVXX_HAVE_CVSIP, 1, [Define to use C-VSIPL library.])
    AC_DEFINE_UNQUOTED(OVXX_CVSIP_HAVE_BOOL, $cvsip_have_bool,
      [Define if C-VSIPL supports bool views.])
    AC_DEFINE_UNQUOTED(OVXX_CVSIP_HAVE_INT, $cvsip_have_int,
      [Define if C-VSIPL supports int views.])
    AC_DEFINE_UNQUOTED(OVXX_CVSIP_HAVE_FLOAT, $cvsip_have_float,
      [Define if C-VSIPL supports float operations.])
    AC_DEFINE_UNQUOTED(OVXX_CVSIP_HAVE_DOUBLE, $cvsip_have_double,
      [Define if C-VSIPL supports double operations.])
  fi
  if test "$enable_cvsip_fft" != "no"; then 
    if test "$cvsip_have_float" = "1"; then
      provide_fft_float=1
    fi
    if test "$cvsip_have_double" = "1"; then
      provide_fft_double=1
    fi
    AC_SUBST(OVXX_CVSIP_FFT, 1)
    AC_DEFINE_UNQUOTED(OVXX_CVSIP_FFT, 1,
      [Define to use the C-VSIPL library to perform FFTs.])
  fi
fi

])
