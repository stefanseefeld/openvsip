dnl
dnl Copyright (c) 2007 by CodeSourcery
dnl Copyright (c) 2013 Stefan Seefeld
dnl All rights reserved.
dnl
dnl This file is part of OpenVSIP. It is made available under the
dnl license contained in the accompanying LICENSE.BSD file.

AC_DEFUN([OVXX_CHECK_SAL],
[
#
# Find the Mercury SAL library, if enabled.
#
if test "$enable_sal_fft" == "yes"; then
  if test "$with_sal" == "no"; then
    AC_MSG_ERROR([SAL FFT requires SAL])
  else
    with_sal="yes"
  fi 
fi

if test "$with_sal" != "no"; then

  if test -n "$with_sal_include"; then
    SAL_CPPFLAGS="-I$with_sal_include"
  fi
  save_CPPFLAGS="$CPPFLAGS"
  CPPFLAGS="$CPPFLAGS $SAL_CPPFLAGS"
  
  # Find sal.h.
  vsipl_sal_h_name="not found"
  AC_CHECK_HEADER([sal.h], [vsipl_sal_h_name='<sal.h>'],, [// no prerequisites])
  if test "$vsipl_sal_h_name" == "not found"; then
    if test "$with_sal" = "yes"
    then AC_MSG_ERROR([SAL enabled, but no sal.h detected])
    else CPPFLAGS="$save_CPPFLAGS"
    fi
  else

    # Find the library.

    save_LDFLAGS="$LDFLAGS"
    if test -n "$with_sal_lib"; then
      LDFLAGS="$LDFLAGS -L$with_sal_lib"
    fi
    AC_SEARCH_LIBS(vaddx, [sal csal],
	           [sal_lib=$ac_lib
		    sal_found="yes"],
 		   [sal_found="no"])

    AC_MSG_CHECKING([for std::complex-compatible SAL-types.])
    AC_COMPILE_IFELSE([AC_LANG_SOURCE([
#include <sal.h>

template <bool V> struct Static_assert;
template <> struct Static_assert<true>
{
  static bool const value = true;
};

int main(int, char **)
{
  bool value;
  value = Static_assert<sizeof(COMPLEX_SPLIT) == 
			2*sizeof(float *)>::value;
  value = Static_assert<sizeof(DOUBLE_COMPLEX_SPLIT) == 
			2*sizeof(double *)>::value;
  (void)value;
}
])],
[AC_MSG_RESULT(yes)],
[AC_MSG_ERROR([std::complex-incompatible SAL-types detected!])])

  fi     

  if test "$sal_found" == "no"; then
    AC_MSG_ERROR([No SAL library found])
    CPPFLAGS=$save_CPPFLAGS
    LDFLAGS=$save_LDFLAGS
  else
    AC_MSG_RESULT([SAL Library found: $sal_lib])

    # General test for float and double support.

    AC_CHECK_LIB($sal_lib, vsmulx,  [sal_have_float=1], [sal_have_float=0])
    AC_CHECK_LIB($sal_lib, vsmuldx, [sal_have_double=1], [sal_have_double=0])

    # Check specific SAL signatures

    AC_MSG_CHECKING([for vconvert_s8_f32x signature.])
    AC_COMPILE_IFELSE([AC_LANG_SOURCE([
#include <sal.h>

int main(int, char **)
{
  signed char input;
  float output;
  vconvert_s8_f32x(&input, 1, &output, 1, 0, 0, 1, 0, 0);
}
])],
[
  vconvert_s8_f32x_is_signed=1
  AC_MSG_RESULT([signed char *])
],
[
  vconvert_s8_f32x_is_signed=0
  AC_MSG_RESULT([char *])
])

    AC_SUBST(OVXX_HAVE_SAL, 1)
    AC_DEFINE_UNQUOTED(OVXX_HAVE_SAL, 1,
      [Define to set whether or not to use Mercury's SAL library.])
    AC_DEFINE_UNQUOTED(OVXX_HAVE_SAL_FLOAT, $sal_have_float,
      [Define if Mercury's SAL library provides float support.])
    AC_DEFINE_UNQUOTED(OVXX_HAVE_SAL_DOUBLE, $sal_have_double,
      [Define if Mercury's SAL library provides double support.])
    AC_DEFINE_UNQUOTED(OVXX_SAL_USES_SIGNED, $vconvert_s8_f32x_is_signed,
      [Define if Mercury's SAL uses signed char *.])

    # Specific function tests.

    AC_CHECK_LIB($sal_lib, vsdivix, [sal_have_vsdivix=1], [sal_have_vsdivix=0])
    AC_CHECK_LIB($sal_lib, vthrx,  [sal_have_vthrx=1], [sal_have_vthrx=0])


    AC_DEFINE_UNQUOTED(OVXX_HAVE_SAL_VSDIVIX, 1,
        [Define if Mercury's SAL library has vsdivix.])
    AC_DEFINE_UNQUOTED(OVXX_HAVE_SAL_VTHRX, $sal_have_vthrx,
        [Define if Mercury's SAL library provides vthrx.])

    if test "$enable_sal_fft" != "no"; then 
      provide_fft_float=1
      provide_fft_double=1
      AC_SUBST(OVXX_SAL_FFT, 1)
      AC_DEFINE_UNQUOTED(OVXX_SAL_FFT, 1,
        [Define to use Mercury's SAL library to perform FFTs.])
    fi

    if test "$enable_multicore_sal" != "no"; then
      if test "$with_sal" == "no"; then
        AC_MSG_ERROR([MultiCore option requires SAL])
      else
        AC_SUBST(SAL_ENABLE_MULTICORE)
        AC_DEFINE_UNQUOTED(SAL_ENABLE_MULTICORE,,
          [Define if Mercury's MultiCore SAL is enabled.])
      fi
    fi
  fi
fi

])
