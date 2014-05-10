dnl
dnl Copyright (c) 2014 Stefan Seefeld
dnl All rights reserved.
dnl
dnl This file is part of OpenVSIP. It is made available under the
dnl license contained in the accompanying LICENSE.BSD file.

AC_DEFUN([OVXX_CHECK_OPENCL],
[
  AC_ARG_WITH(opencl_prefix,
    AS_HELP_STRING([--with-opencl-prefix=PATH],
                   [Specify the installation prefix of the OpenCL libraries.  Headers
                    must be in PATH/include; libraries in PATH/lib64 or PATH/lib.]))

  if test "$with_opencl" != "no"; then

    # Chose reasonable default if no prefix provided.
    if test -z "$with_opencl_prefix"; then
      with_opencl_prefix="/usr/local/opencl"
    fi

    OPENCL_CPPFLAGS="-I$with_opencl_prefix/include"

    save_CPPFLAGS="$CPPFLAGS"
    CPPFLAGS="$CPPFLAGS $OPENCL_CPPFLAGS"
  
    # Find CL/cl.h.
    AC_CHECK_HEADER([CL/cl.h],, [AC_MSG_ERROR([OpenCL enabled, but CL/cl.h not found.])])

    if test "$ac_cv_sizeof_int_p" -eq 8 ; then
      OPENCL_LDFLAGS="-L$with_opencl_prefix/lib/x86_64"
    else
      OPENCL_LDFLAGS="-L$with_opencl_prefix/lib/x86"
    fi

    # Find the library.
    save_LDFLAGS="$LDFLAGS"
    LDFLAGS="$OPENCL_LDFLAGS $LDFLAGS"
    LIBS="$LIBS -lOpenCL"

    # Declare it as found.	
    AC_SUBST(OVXX_HAVE_OPENCL, 1)
    AC_DEFINE_UNQUOTED(OVXX_HAVE_OPENCL, 1,
      [Define to set whether or not to use OpenCL.])
  fi
])
