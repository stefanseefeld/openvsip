dnl
dnl Copyright (c) 2014 Stefan Seefeld
dnl All rights reserved.
dnl
dnl This file is part of OpenVSIP. It is made available under the
dnl license contained in the accompanying LICENSE.BSD file.

AC_DEFUN([OVXX_CHECK_TRACING],
[
AC_SUBST(enable_tracing)
if test "$enable_tracing" == "lttng"; then

  AC_ARG_WITH(lttng-prefix,
    [  --with-lttng-prefix=PATH      Specify the lttng installation prefix.],
    LTTNG_PREFIX="$with_lttng_prefix",
  )

  if test -n "$with_lttng_prefix"; then
    CPPFLAGS="$CPPFLAGS -I$with_lttng_prefix/include"
    LDFLAGS="$LDFLAGS -L$with_lttng_prefix/lib"
  fi
  LIBS="$LIBS -llttng-ust"
  save_CPPFLAGS=$CPPFLAGS
  CPPFLAGS="$CPPFLAGS $LTTNG_CPPFLAGS"
  AC_CHECK_HEADER([lttng/tracef.h], [], 
    [AC_MSG_ERROR([LTTng could not be found or is too old (version 2.4.1+ required).])])
  AC_DEFINE_UNQUOTED(OVXX_HAVE_LTTNG, 1, [Define if LTTNG is available.])
fi
])
