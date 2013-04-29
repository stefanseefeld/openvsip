dnl
dnl Copyright (c) 2007 by CodeSourcery
dnl Copyright (c) 2013 Stefan Seefeld
dnl All rights reserved.
dnl
dnl This file is part of OpenVSIP. It is made available under the
dnl license contained in the accompanying LICENSE.BSD file.

AC_DEFUN([SVXX_CHECK_IPP],
[
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
    AC_SUBST(VSIP_IMPL_IPP_FFT, 1)
    if test "$neutral_acconfig" = 'y'; then
      CPPFLAGS="$CPPFLAGS -DVSIP_IMPL_IPP_FFT=1"
    else
      AC_DEFINE_UNQUOTED(VSIP_IMPL_IPP_FFT, 1,
	      [Define to use Intel's IPP library to perform FFTs.])
    fi
  fi

elif test "$with_ipp" != "no"; then

  if test -n "$with_ipp_prefix"; then
    IPP_CPPFLAGS="-I$with_ipp_prefix/include"
    IPP_LDFLAGS="-L$with_ipp_prefix/sharedlib"
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
    LIBS="-lpthread $LIBS"
    AC_SEARCH_LIBS(ippGetLibVersion, [$ippcore_search],,
      [LD_FLAGS="$save_LDFLAGS"])
    
    save_LDFLAGS="$LDFLAGS"
    LDFLAGS="$LDFLAGS $IPP_LDFLAGS"
    AC_SEARCH_LIBS(ippsMul_32f, [$ipps_search],
      [
        AC_SUBST(VSIP_IMPL_HAVE_IPP, 1)
        if test "$neutral_acconfig" = 'y'; then
          CPPFLAGS="$CPPFLAGS -DVSIP_IMPL_HAVE_IPP=1"
        else
          AC_DEFINE_UNQUOTED(VSIP_IMPL_HAVE_IPP, 1,
            [Define to set whether or not to use Intel's IPP library.])
        fi
      ],
      [LD_FLAGS="$save_LDFLAGS"])

    AC_MSG_CHECKING([for std::complex-compatible IPP-types.])
    AC_COMPILE_IFELSE([
#include <ippdefs.h>

template <bool V> struct static_assert;
template <> struct static_assert<true>
{
  static bool const value = true;
};

int main(int, char **)
{
  bool value;
  value = static_assert<sizeof(Ipp32fc) == 2*sizeof(float)>::value;
  value = static_assert<sizeof(Ipp64fc) == 2*sizeof(double)>::value;
}
      ],
      [AC_MSG_RESULT(yes)],
      [AC_MSG_ERROR([std::complex-incompatible IPP-types detected!])])

    save_LDFLAGS="$LDFLAGS"
    LDFLAGS="$LDFLAGS $IPP_FFT_LDFLAGS"

    AC_SEARCH_LIBS(
	[ippiFFTFwd_CToC_32fc_C1R], [$ippi_search],
	[have_ippi="yes"],
	[have_ippi="no"
         LD_FLAGS="$save_LDFLAGS"])

    AC_SEARCH_LIBS(
	[ippmCopy_ma_32f_SS], [$ippm_search],
	[have_ippm="yes"],
	[have_ippm="no"
         LD_FLAGS="$save_LDFLAGS"])

    if test "$enable_ipp_fft" != "no"; then 
      provide_fft_float=1
      provide_fft_double=1
      AC_SUBST(VSIP_IMPL_IPP_FFT, 1)
      if test "$neutral_acconfig" = 'y'; then
        CPPFLAGS="$CPPFLAGS -DVSIP_IMPL_IPP_FFT=1"
      else
        AC_DEFINE_UNQUOTED(VSIP_IMPL_IPP_FFT, 1,
	      [Define to use Intel's IPP library to perform FFTs.])
      fi
    fi
  fi
fi

])
