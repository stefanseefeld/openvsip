dnl
dnl Copyright (c) 2008 by CodeSourcery
dnl Copyright (c) 2013 Stefan Seefeld
dnl All rights reserved.
dnl
dnl This file is part of OpenVSIP. It is made available under the
dnl license contained in the accompanying LICENSE.BSD file.

AC_DEFUN([OVXX_CHECK_PYTHON],
[
AC_SUBST(enable_python_bindings)
if test "$enable_python_bindings" == "1"; then

  if test "x$enable_shared_libs" != "xyes"; then
    AC_MSG_ERROR([The Python bindings require --enable-shared-libs.])
  fi

  PYTHON_INCLUDE=`$PYTHON -c "from distutils import sysconfig; print(sysconfig.get_python_inc())"`
  PYTHON_EXT=`$PYTHON -c "from distutils import sysconfig; print(sysconfig.get_config_var('SO'))"`
  PYTHON_VERSION=`$PYTHON -c "import sys; print('%d.%d'%(sys.version_info[[0]],sys.version_info[[1]]))"`
  PYTHON_LIBS=`$PYTHON -c "from distutils import sysconfig; print(sysconfig.get_config_var('BLDLIBRARY'))"`
  PYTHON3=`$PYTHON -c "import sys; print('3' if sys.version_info[[0]] == 3 else '')"`

  AC_MSG_CHECKING([for numpy])
  if $PYTHON -c "import numpy" > /dev/null; then
    AC_MSG_RESULT(yes)
  else
    AC_MSG_RESULT(no)
    AC_MSG_ERROR([The OpenVSIP Python interface requires numpy to be installed.])
  fi

  NUMPY_TEST="from imp import find_module"
  NUMPY_TEST="${NUMPY_TEST};from os.path import join"
  NUMPY_TEST="${NUMPY_TEST};file, path, descr = find_module('numpy')"
  NUMPY_TEST="${NUMPY_TEST};print(join(path, 'core', 'include'))"
  NUMPY_INCLUDE=`$PYTHON -c "$NUMPY_TEST"`
  AC_SUBST(PYTHON)
  AC_SUBST(PYTHON_CPPFLAGS, "-I $PYTHON_INCLUDE -I $NUMPY_INCLUDE")
  AC_SUBST(PYTHON_LIBS)
  AC_SUBST(PYTHON_EXT)

  AC_LANG(C++)
  if test -n "$with_boost_prefix"; then
    BOOST_CPPFLAGS="-I$with_boost_prefix/include"
    BOOST_LDFLAGS="-L$with_boost_prefix/lib"
  fi
  BOOST_LIBS="-lboost_python$PYTHON3"
  save_CPPFLAGS=$CPPFLAGS
  CPPFLAGS="$CPPFLAGS $BOOST_CPPFLAGS $PYTHON_CPPFLAGS"
  AC_CHECK_HEADER([boost/python.hpp], [], 
    [AC_MSG_ERROR([boost.python could not be found])])
  CPPFLAGS="$save_CPPFLAGS"

  AC_SUBST(BOOST_CPPFLAGS)
  AC_SUBST(BOOST_LDFLAGS)
  AC_SUBST(BOOST_LIBS)
fi
])
