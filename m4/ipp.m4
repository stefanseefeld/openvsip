dnl
dnl Copyright (c) 2007 by CodeSourcery
dnl Copyright (c) 2013 Stefan Seefeld
dnl All rights reserved.
dnl
dnl This file is part of OpenVSIP. It is made available under the
dnl license contained in the accompanying LICENSE.BSD file.

AC_DEFUN([OVXX_CHECK_IPP],
[
AC_ARG_ENABLE([ipp],,  
  AC_MSG_ERROR([The option --enable-ipp is obsolete; use 
    --with-ipp instead.  (Run 'configure --help' for details)]),)

AC_ARG_WITH([ipp],
  AS_HELP_STRING([--with-ipp],
                 [Use IPP if found (default is to not search for it).]),,
  [with_ipp=no])
AC_ARG_WITH(ipp_prefix,
  AS_HELP_STRING([--with-ipp-prefix=PATH],
                 [Specify the installation prefix of the IPP library.  Headers
                  must be in PATH/include; libraries in PATH/lib.]),
  )
AC_ARG_WITH(ipp_suffix,
  AS_HELP_STRING([--with-ipp-suffix=TARGET],
                 [Specify the optimization target of IPP libraries, such as
		  a6, em64t, i7, m7, mx, px, t7, w7.  E.g. a6 => -lippsa6.
                  TARGET may be the empty string.]),
  )
AC_ARG_WITH([ipp_arch],
  AS_HELP_STRING([--with-ipp-arch=ARCH],
                 [Specify the IPP library architecture directory.  IPP
		  libraries from PATH/lib/ARCH will be used, where
		  PATH is specified with '--with-ipp-prefix' option.
		  E.g. ARCH can be 'intel64' or 'ia32'.
		  (Required only for IPP version 7 and up).]),
  )


# If the user specified an IPP prefix, they definitely want IPP.
# However, we need to avoid overwriting the value of $with_ipp
# if the user set it (i.e. '--enable-ipp=win').

if test -n "$with_ipp_prefix" -o -n "$with_ipp_suffix"; then
  if test $with_ipp != "win"; then
    with_ipp="yes"
  fi
fi


if test "$enable_ipp_fft" == "yes"; then
  if test "$with_ipp" == "no"; then
    AC_MSG_ERROR([IPP FFT requires IPP])
  fi 
fi

# LDFLAGS notes:
# Version 7 uses a common prefix and requires an option to specify
# the architecture (--with-ipp-arch).  The prefix includes everything
# up to ipp/ in the example below:
#
#   {/path/to/}ipp/lib/{intel64|ia32}
#
# Version 5 and 6.1 used unique prefixes and kept shared libs in
# a separate location.  The prefix included the architecture,
# so --with-ipp-arch is not needed. E.g.:
#
#   {/path/to/}ipp/{em64t|ia32}/lib
#   {/path/to/}ipp/{em64t|ia32}/sharedlib
#

# Find the IPP library, if enabled.
#
if test "$with_ipp" = "win"; then
  AC_MSG_RESULT([Using IPP for Windows.])
  if test -n "$with_ipp_prefix"; then
    IPP_CPPFLAGS="-I$with_ipp_prefix/include"
    IPP_LDFLAGS="-L$with_ipp_prefix/sharedlib"
  fi

  # Check for headers ipps.h.
  vsipl_ipps_h_name="not found"
  AC_CHECK_HEADER([ipps.h], [vsipl_ipps_h_name='<ipps.h>'],, [// no prerequisites])
  if test "$vsipl_ipps_h_name" == "not found"; then
    AC_MSG_ERROR([IPP for windows enabled, but no ipps.h detected])
  fi

  LIBS="$LIBS ipps.lib ippi.lib ippm.lib"

  AC_MSG_CHECKING([for ippsMul_32f])
  AC_LINK_IFELSE(
    [AC_LANG_PROGRAM([[#include <ipps.h>]],
		     [[Ipp32f const* A; Ipp32f const* B; Ipp32f* Z; int len;
                       ippsMul_32f(A, B, Z, len);]])],
    [AC_MSG_RESULT(yes)],
    [AC_MSG_ERROR(not found.)] )

  if test "$enable_ipp_fft" != "no"; then 
    provide_fft_float=1
    provide_fft_double=1
    AC_SUBST(OVXX_IPP_FFT, 1)
    AC_DEFINE_UNQUOTED(OVXX_IPP_FFT, 1,
      [Define to use Intel's IPP library to perform FFTs.])
  fi

elif test "$with_ipp" != "no"; then

  if test -n "$with_ipp_prefix"; then
    IPP_CPPFLAGS="-I$with_ipp_prefix/include"
    if test -n "$with_ipp_arch"; then
      IPP_LDFLAGS="-L$with_ipp_prefix/lib/$with_ipp_arch"
      if test "$with_intel_omp_prefix" != ""; then
        IPP_LDFLAGS="$IPP_LDFLAGS -L$with_intel_omp_prefix/lib/$with_ipp_arch"
      fi
    else
      IPP_LDFLAGS="-L$with_ipp_prefix/sharedlib"
    fi
  fi
  save_CPPFLAGS="$CPPFLAGS"
  CPPFLAGS="$CPPFLAGS $IPP_CPPFLAGS"

  # Find ipps.h.
  vsipl_ipps_h_name="not found"
  AC_CHECK_HEADER([ipps.h], [vsipl_ipps_h_name='<ipps.h>'],, [// no prerequisites])
  if test "$vsipl_ipps_h_name" == "not found"; then
    if test "$with_ipp" != "probe" -o "$enable_ipp_fft" == "yes"; then
      AC_MSG_ERROR([IPP enabled, but no ipps.h detected])
    else
      CPPFLAGS="$save_CPPFLAGS"
    fi
  else

    if test "${with_ipp_suffix-unset}" == "unset"; then
      ippcore_search="ippcore ippcoreem64t ippcore64"
      ipps_search="ipps ippsem64t ipps64"
      ippi_search="ippi ippiem64t ippi64"
      ippm_search="ippm ippmem64t ippm64"
    else
      # Use of suffix not consistent:
      #  - for em64t, ipp 5.0 has libippcoreem64t.so
      #  - for ia32,  ipp 5.1 has libippcore.so
      ippcore_search="ippcore ippcore$with_ipp_suffix"
      ipps_search="ipps$with_ipp_suffix"
      ippi_search="ippi$with_ipp_suffix"
      ippm_search="ippm$with_ipp_suffix"
    fi

    # Find the library.
    save_LDFLAGS="$LDFLAGS"
    LDFLAGS="$LDFLAGS $IPP_LDFLAGS"
    LIBS="-liomp5 -lpthread $LIBS"

    AC_SEARCH_LIBS(ippGetLibVersion, [$ippcore_search],,
      [AC_MSG_ERROR([IPP library not found])])
    
    AC_SEARCH_LIBS(ippsMul_32f, [$ipps_search],
      [
        AC_SUBST(OVXX_HAVE_IPP, 1)
        AC_DEFINE_UNQUOTED(OVXX_HAVE_IPP, 1,
          [Define to set whether or not to use Intel's IPP library.])
      ],
      [AC_MSG_ERROR([IPP library not found])])

    AC_MSG_CHECKING([for std::complex-compatible IPP-types.])
    AC_COMPILE_IFELSE([AC_LANG_SOURCE([
#include <ippdefs.h>

template <bool V> struct Static_assert;
template <> struct Static_assert<true>
{
  static bool const value = true;
};

int main(int, char **)
{
  bool value;
  value = Static_assert<sizeof(Ipp32fc) == 2*sizeof(float)>::value;
  value = Static_assert<sizeof(Ipp64fc) == 2*sizeof(double)>::value;
  (void)value;
}
      ])],
      [AC_MSG_RESULT(yes)],
      [AC_MSG_ERROR([std::complex-incompatible IPP-types detected!])])

    save_LDFLAGS="$LDFLAGS"
    LDFLAGS="$LDFLAGS $IPP_FFT_LDFLAGS"

    AC_SEARCH_LIBS(
	[ippiFFTFwd_CToC_32fc_C1R], [$ippi_search],,
	[AC_MSG_ERROR([IPP library not found])])

    AC_SEARCH_LIBS(
	[ippmCopy_ma_32f_SS], [$ippm_search],,
	[AC_MSG_ERROR([IPP library not found])])

    if test "$enable_ipp_fft" != "no"; then 
      provide_fft_float=1
      provide_fft_double=1
      AC_SUBST(OVXX_IPP_FFT, 1)
      AC_DEFINE_UNQUOTED(OVXX_IPP_FFT, 1,
        [Define to use Intel's IPP library to perform FFTs.])
    fi
  fi
fi

])
