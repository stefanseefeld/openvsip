dnl Copyright (c) 2010 by CodeSourcery, Inc.  All rights reserved.
dnl
dnl Contents: MKL configuration for Sourcery VSIPL++
dnl

AC_DEFUN([SVXX_CHECK_MKL],
[
AC_ARG_WITH([mkl], AS_HELP_STRING([--with-mkl], [Use MKL]),, [with_mkl=no])

AC_ARG_WITH([mkl_prefix],
  AS_HELP_STRING([--with-mkl-prefix=PATH],
                 [Specify the installation prefix of the MKL library.  Headers
                  must be in PATH/include; libraries in PATH/lib/ARCH (where
		  ARCH is either deduced or set by the --with-mkl-arch option).
	          (Enables LAPACK).]))

AC_ARG_WITH([intel_omp_prefix],
  AS_HELP_STRING([--with-intel-omp-prefix=PATH],
                 [Specify the installation prefix of the Intel OMP library.
                  This may be required for multi-threaded MKL & IPP configurations.]))

AC_ARG_WITH([mkl_arch],
  AS_HELP_STRING([--with-mkl-arch=ARCH],
                 [Specify the MKL library architecture directory.  MKL
		  libraries from PATH/lib/ARCH will be used, where
		  PATH is specified with '--with-mkl-prefix' option.
		  (Default is to probe arch based on host cpu type).]),,
  [with_mkl_arch=probe])

AC_ARG_ENABLE([mkl_omp],
  AS_HELP_STRING([--disable-mkl-omp],
                 [Use a single-threaded MKL backend.]),
  ,
  [enable_mkl_omp="yes"])

if test "$with_mkl" != "no"; then

  if test "$with_mkl_arch" == "probe"; then
    if test "$host_cpu" == "x86_64"; then
      with_mkl_arch="em64t"
    elif test "$host_cpu" == "ia64"; then
      with_mkl_arch="64"
    else
      with_mkl_arch="32"
    fi
  fi

  keep_CPPFLAGS=$CPPFLAGS
  keep_LDFLAGS=$LDFLAGS
  keep_LIBS=$LIBS

  if test "$with_mkl" != yes; then
    mkl_versions="$with_mkl"
  else
    mkl_versions="mkl10 mkl7 mkl5"
  fi

  mkl_found=no
  mkl_vml=no
  # Despite the names, these checks don't actually check MKL versions.
  # Rather, they attempt various settings that are known to work with
  # these particular MKL versions.
  for mkl_version in $mkl_versions; do
    case $mkl_version in
      mkl5)
        CPPFLAGS="$keep_CPPFLAGS -I$with_mkl_prefix/include"
        LDFLAGS="$keep_LDFLAGS -L$with_mkl_prefix/lib/$with_mkl_arch"
        LIBS="$keep_LIBS -lmkl -lpthread"
        AC_CHECK_FUNC([MKLGetVersion], [mkl_found=yes])
      ;;
      mkl7)
        CPPFLAGS="$keep_CPPFLAGS -I$with_mkl_prefix/include -pthread"
        LDFLAGS="$keep_LDFLAGS -L$with_mkl_prefix/lib/$with_mkl_arch"
        LIBS="$keep_LIBS -lmkl -lguide -lpthread"
        AC_CHECK_FUNC([MKLGetVersion], [mkl_found=yes])
      ;;
      mkl10)
        CPPFLAGS="$keep_CPPFLAGS -I$with_mkl_prefix/include -pthread"
        LDFLAGS="$keep_LDFLAGS -L$with_mkl_prefix/lib/$with_mkl_arch"
        if test "$with_mkl_arch" = "em64t"; then
          if test "$enable_mkl_omp" = "yes" -a "$with_intel_omp_prefix" != ""; then
            LDFLAGS="$LDFLAGS -L$with_intel_omp_prefix/lib/intel64"
	    LIBS="$keep_LIBS -lmkl_core -lmkl_intel_lp64"
          fi
        else
          if test "$enable_mkl_omp" = "yes" -a "$with_intel_omp_prefix" != ""; then
            LDFLAGS="$LDFLAGS -L$with_intel_omp_prefix/lib/ia32"
	    LIBS="$keep_LIBS -lmkl_core -lmkl_intel"
          fi
        fi
        if test "$enable_mkl_omp" = "yes"; then
          LIBS="$LIBS -lmkl_intel_thread -liomp5 -lpthread"
        else
          LIBS="$LIBS -lmkl_sequential"
        fi
        # MKL 10 introduces a number of elementwise vector functions.
        # We use this as a version indicator (lacking other means to discover that).
        AC_CHECK_FUNC([vsAdd], 
          [mkl_found=yes
           mkl_vml=yes
          ])
      ;;
      win)
        # This is based on MKL 8, the only version we tried on Windows.
        if test -n "$with_mkl_prefix"; then
          CPPFLAGS="$keep_CPPFLAGS -I$with_mkl_prefix/include"
          LDFLAGS="$keep_LDFLAGS -L$with_mkl_prefix/$with_mkl_arch/lib"
        fi
        LIBS="$keep_LIBS mkl_c.lib libguide.lib"
        AC_CHECK_FUNC([MKLGetVersion], [mkl_found=yes])
      ;;
      win_nocheck)
        if test -n "$with_mkl_prefix"; then
          CPPFLAGS="$keep_CPPFLAGS -I$with_mkl_prefix/include"
          LDFLAGS="$keep_LDFLAGS -L$with_mkl_prefix/$with_mkl_arch/lib"
        fi
        LIBS="$keep_LIBS mkl_c.lib -lguide "
        mkl_found="mkl_nocheck"
      ;;
    esac
    if test "$mkl_found" != no; then
      break
    fi
  done
  if test "$mkl_found" = no; then 
    AC_MSG_ERROR([MKL could not be found.])
  fi

  if test "$mkl_vml" = yes; then
    if test "$neutral_acconfig" = 'y'; then
      CPPFLAGS="$CPPFLAGS -DVSIP_IMPL_HAVE_MKL_VML"
    else
      AC_DEFINE_UNQUOTED(VSIP_IMPL_HAVE_MKL_VML, 1,
        [Define to 1 if MKL vector math library is available.])
    fi
    AC_SUBST(VSIP_IMPL_HAVE_MKL_VML, 1)
  fi
  
fi
])
