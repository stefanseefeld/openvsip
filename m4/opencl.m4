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

  AC_ARG_WITH(opencl_libs,
    AS_HELP_STRING([--with-opencl-libs=PATH],
                   [Specify the installation directory of the OpenCL libraries.]))

  if test "$with_opencl" != "no"; then

    AC_ARG_WITH(clmath, AS_HELP_STRING([--with-clmath],[Build with clMath support.]))
    AC_ARG_WITH(clmath_prefix, AS_HELP_STRING([--with-clmath-prefix],[Specify the clMath install prefix.]))


    # Chose reasonable default if no prefix provided.
    if test -z "$with_opencl_prefix"; then
      with_opencl_prefix="/usr/local/opencl"
    fi

    CPPFLAGS="$CPPFLAGS -I$with_opencl_prefix/include"
    # Find CL/cl.h.
    AC_CHECK_HEADER([CL/cl.h],, [AC_MSG_ERROR([OpenCL enabled, but CL/cl.h not found.])])

    if test -n "$with_opencl_libs" ; then
      OPENCL_LDFLAGS="-L$with_opencl_libs"
    elif test "$ac_cv_sizeof_int_p" -eq 8 ; then
      OPENCL_LDFLAGS="-L$with_opencl_prefix/lib64"
    else
      OPENCL_LDFLAGS="-L$with_opencl_prefix/lib"
    fi
    OPENCL_LIBS="-lOpenCL"

    # Find the library.
    LDFLAGS="$LDFLAGS $OPENCL_LDFLAGS"
    LIBS="$LIBS $OPENCL_LIBS"

    AC_CHECK_LIB(OpenCL, clGetPlatformIDs, [], [AC_MSG_ERROR([OpenCL enabled but library not found.])])


    if test -n "$with_clmath" -o -n "$with_clmath_prefix"; then
      if test -n "$with_clmath_prefix"; then
        CPPFLAGS="$CPPFLAGS -I$with_clmath_prefix/include"
        if test "$ac_cv_sizeof_int_p" -eq 8 ; then
          OPENCL_LDFLAGS="$OPENCL_LDFLAGS -L$with_clmath_prefix/lib64"
        else
	  OPENCL_LDFLAGS="$OPENCL_LDFLAGS -L$with_clmath_prefix/lib32"
	fi
      fi
      AC_CHECK_HEADER([clAmdBlas.h],, [AC_MSG_ERROR([clMath enabled, but clAmdBlas.h not found.])])
      have_clmath=1
      OPENCL_LIBS="-lclAmdBlas $OPENCL_LIBS"
    fi

    # Declare it as found.	
    AC_SUBST(OVXX_HAVE_OPENCL, 1)
    AC_DEFINE_UNQUOTED(OVXX_HAVE_OPENCL, 1,
      [Define to set whether or not to use OpenCL.])

    if test "$have_clmath" = "1"; then
      AC_SUBST(OVXX_HAVE_CLMATH, 1)
      AC_DEFINE_UNQUOTED(OVXX_HAVE_CLMATH, 1,
        [Define to set whether or not to use clMath.])
    fi
  fi
])
