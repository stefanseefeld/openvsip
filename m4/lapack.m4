dnl
dnl Copyright (c) 2007 by CodeSourcery
dnl Copyright (c) 2013 Stefan Seefeld
dnl All rights reserved.
dnl
dnl This file is part of OpenVSIP. It is made available under the
dnl license contained in the accompanying LICENSE.BSD file.

AC_DEFUN([SVXX_CHECK_LAPACK],
[
AC_ARG_WITH(f2c-abi,
  AS_HELP_STRING([--with-f2c-abi],
                 [Specify that the F77 F2C calling conventions 
                  should be used when interfacing fortran code.]),
            [use_f2c_abi=$withval])


AC_ARG_ENABLE([lapack],,  
  AC_MSG_ERROR([The option --enable-lapack is not correct; use 
    --with-lapack instead.  (Run 'configure --help' for details)]),)

AC_ARG_WITH([lapack],
  AS_HELP_STRING([--with-lapack\[=PKG\]],
                 [Select one or more LAPACK libraries to search for.
                  The default is to probe for atlas and generic,
	          using the first one found.  Sourcery VSIPL++ understands the
		  following LAPACK library selections: mkl (Intel Math Kernel
		  Library), acml (AMD Core Math Library), atlas (system
		  ATLAS/LAPACK installation), and generic (system generic
		  LAPACK installation). 
		  Specifying 'no' disables the search for a LAPACK library.]),,
  [with_lapack=probe])

AC_ARG_WITH(atlas_prefix,
  AS_HELP_STRING([--with-atlas-prefix=PATH],
                 [Specify the installation prefix of the ATLAS library.
	          Headers must be in PATH/include; libraries in PATH/lib.
	          (Enables LAPACK).]))

AC_ARG_WITH(atlas_libdir,
  AS_HELP_STRING([--with-atlas-libdir=PATH],
                 [Specify the directory containing ATLAS libraries.
	          (Enables LAPACK).]))

AC_ARG_WITH(atlas_include,
  AS_HELP_STRING([--with-atlas-include=PATH],
                 [Specify the directory containing ATLAS header files.
	          (Enables LAPACK).]))

AC_ARG_WITH(acml_prefix,
  AS_HELP_STRING([--with-acml-prefix=PATH],
                 [Specify the installation prefix of the ACML library.  Headers
                  must be in PATH/include; libraries in PATH/lib
	          (Enables LAPACK).]))

AC_ARG_ENABLE([cblas],,  
  AC_MSG_ERROR([The option --disable-cblas is obsolete; use 
    --without-cblas instead.  (Run 'configure --help' for details)]),)

AC_ARG_WITH([cblas],
  AS_HELP_STRING([--without-cblas],
                 [Disable C BLAS API (default is to use it if possible)]),,
  [with_cblas=yes])

# Disable lapack if building ref-impl
if test "$only_ref_impl" = "1"; then
  if test "$with_lapack" == "probe"; then
    with_lapack="no"
  fi
  if test "$with_lapack" != "no"; then
    AC_MSG_ERROR([Cannot use LAPACK with reference implementation.])
  fi
fi
#
# Check to see if any options have implied with_lapack
#
if test "$with_lapack" == "probe"; then
  already_prefix=0
  if test "$with_atlas_prefix" != "" -o "$with_atlas_libdir" != "" -o "$with_atlas_include" != ""; then
    AC_MSG_RESULT([ATLAS prefixes specified, assume --with-lapack=atlas])
    with_lapack="atlas"
    already_prefix=1
  fi
  if test "$with_acml_prefix" != ""; then
    if test "$already_prefix" = "1"; then
      AC_MSG_ERROR([Multiple prefixes given for LAPACk libraries (i.e.
		    MKL, ACML, and/or ATLAS])
    fi
    AC_MSG_RESULT([ACML prefixes specified, assume --with-lapack=acml])
    with_lapack="acml"
    already_prefix=1
  fi
fi

#
# Find the lapack library, if enabled.
#
if test "$with_lapack" != "no"; then
  keep_CPPFLAGS=$CPPFLAGS
  keep_LDFLAGS=$LDFLAGS
  keep_LIBS=$LIBS
  cblas_style="0"

  case $with_lapack in
    mkl)
      if test "$with_mkl" = "no" ; then
        AC_MSG_ERROR([The MKL lapack backend requires --with-mkl.])
      fi
      lapack_packages="mkl"
    ;;
    yes | probe)
      lapack_packages="atlas atlas_blas_v3 generic_wo_blas generic_with_blas generic_v3_wo_blas generic_v3_with_blas"
    ;;
    generic)
      lapack_packages="generic_wo_blas generic_with_blas generic_v3_wo_blas generic_v3_with_blas"
    ;;
    *)
      lapack_packages="$with_lapack"
    ;;
  esac
  AC_MSG_RESULT([Searching for LAPACK packages: $lapack_packages])

  lapack_found="no"
  for trypkg in $lapack_packages; do
    case $trypkg in
      mkl)
        AC_MSG_CHECKING([for LAPACK/MKL library])
        LIBS="-lmkl_lapack $keep_LIBS"
        cblas_style="2"	# use mkl_cblas.h
        lapack_use_ilaenv=0
      ;;
      mkl_win)
        AC_MSG_CHECKING([for LAPACK/MKL 8.x library for Windows])
        LIBS="mkl_lapack.lib $keep_LIBS"
        cblas_style="2"	# use mkl_cblas.h
        lapack_use_ilaenv=0
      ;;
      mkl_win_nocheck)
        AC_MSG_RESULT([Using LAPACK/MKL 8.x library for Windows (without check)])
        LIBS="mkl_lapack.lib $keep_LIBS"
        cblas_style="2"	# use mkl_cblas.h
        lapack_use_ilaenv=0
        lapack_found="mkl_nocheck"
        break
      ;;
      acml)
        AC_MSG_CHECKING([for LAPACK/ACML library])

        dnl We don't use the ACML header files:
        dnl CPPFLAGS="$keep_CPPFLAGS -I$with_acml_prefix/include"
        LDFLAGS="$keep_LDFLAGS -L$with_acml_prefix/lib"
        LIBS="$keep_LIBS -lacml"
        cblas_style="3"	# use acml_cblas.h

        lapack_use_ilaenv=0
      ;;
      atlas)
        AC_MSG_CHECKING([for LAPACK/ATLAS library ($trypkg w/CBLAS)])

        if test "$with_atlas_libdir" != ""; then
	  atlas_libdir=" -L$with_atlas_libdir"
        elif test "$with_atlas_prefix" != ""; then
	  atlas_libdir=" -L$with_atlas_prefix/lib"
        else
	  atlas_libdir=""
        fi

        if test "$with_atlas_include" != ""; then
	  atlas_incdir=" -I$with_atlas_include"
        elif test "$with_atlas_prefix" != ""; then
	  atlas_incdir=" -I$with_atlas_prefix/include"
        else
	  atlas_incdir=""
        fi

        LDFLAGS="$keep_LDFLAGS$atlas_libdir"
        CPPFLAGS="$keep_CPPFLAGS$atlas_incdir"
        LIBS="$keep_LIBS -llapack -lcblas -lf77blas -latlas"
        cblas_style="1"	# use cblas.h

        lapack_use_ilaenv=0
      ;;
      atlas_blas_v3)
	# 080130: This configuration exists on Ubuntu 7.04 (ubuntu) 
        AC_MSG_CHECKING([for LAPACK/ATLAS v3 library ($trypkg w/BLAS)])

        if test "$with_atlas_libdir" != ""; then
	  atlas_libdir=" -L$with_atlas_libdir"
        elif test "$with_atlas_prefix" != ""; then
	  atlas_libdir=" -L$with_atlas_prefix/lib"
        else
	  atlas_libdir=""
        fi

        if test "$with_atlas_include" != ""; then
	  atlas_incdir=" -I$with_atlas_include"
        elif test "$with_atlas_prefix" != ""; then
	  atlas_incdir=" -I$with_atlas_prefix/include"
        else
	  atlas_incdir=""
        fi

        LDFLAGS="$keep_LDFLAGS$atlas_libdir"
        CPPFLAGS="$keep_CPPFLAGS$atlas_incdir"
        LIBS="$keep_LIBS -llapack-3 -lblas-3 -latlas"

        cblas_style="1"	# use cblas.h

        lapack_use_ilaenv=0
      ;;
      atlas_no_cblas)
        AC_MSG_CHECKING([for LAPACK/ATLAS library (w/o CBLAS)])

        if test "$with_atlas_libdir" != ""; then
	  atlas_libdir=" -L$with_atlas_libdir"
        elif test "$with_atlas_prefix" != ""; then
	  atlas_libdir=" -L$with_atlas_prefix/lib"
        else
	  atlas_libdir=""
        fi

        if test "$with_atlas_include" != ""; then
	  atlas_incdir=" -I$with_atlas_include"
        elif test "$with_atlas_prefix" != ""; then
	  atlas_incdir=" -I$with_atlas_prefix/include"
        else
	  atlas_incdir=""
        fi

        LDFLAGS="$keep_LDFLAGS$atlas_libdir"
        CPPFLAGS="$keep_CPPFLAGS$atlas_incdir"
        LIBS="$keep_LIBS -llapack -lf77blas -latlas"
        cblas_style="0"	# no cblas.h

        lapack_use_ilaenv=0
      ;;
      generic_wo_blas)
        AC_MSG_CHECKING([for LAPACK/Generic library (w/o blas)])
        LIBS="$keep_LIBS -llapack"
        cblas_style="0"	# no cblas.h
        lapack_use_ilaenv=0
      ;;
      generic_with_blas)
        AC_MSG_CHECKING([for LAPACK/Generic library (w/blas)])
        LIBS="$keep_LIBS -llapack -lblas"
        cblas_style="0"	# no cblas.h
        lapack_use_ilaenv=0
      ;;
      generic_v3_wo_blas)
        AC_MSG_CHECKING([for LAPACK/Generic v3 library (w/o blas)])
        LIBS="$keep_LIBS -llapack-3"
        cblas_style="0"	# no cblas.h
        lapack_use_ilaenv=0
      ;;
      generic_v3_with_blas)
        # This configuration is found on ubuntu 7.04 (Zelda)

        AC_MSG_CHECKING([for LAPACK/Generic v3 library (w/blas)])
        LIBS="$keep_LIBS -llapack-3 -lblas-3"
        cblas_style="0"	# no cblas.h
        lapack_use_ilaenv=0
      ;;
      *)
        AC_MSG_ERROR([Unknown lapack trypkg: $trypkg])
      ;;
    esac

    AC_LINK_IFELSE(
      [AC_LANG_PROGRAM(
	[[ extern "C" { void sgeqrf_(int*, int*, float*, int*, float*,
	                             float*, int*, int*);
                        void strsm_ (char*, char*, char*, char*,
				     int*, int*, float*, float*, int*,
				     float*, int*); };]],
	[[int    m, n, lda, ldb, lwork, info;
	  float *a, *b, *tau, *work, alpha;
	  sgeqrf_(&m, &n, a, &lda, tau, work, &lwork, &info);
	  char  side, uplo, transa, diag;
	  strsm_(&side, &uplo, &transa, &diag,
	         &m, &n, &alpha, a, &lda, b, &ldb);
        ]]
        )],
      [lapack_found=$trypkg
       AC_MSG_RESULT([found])
       break],
      [lapack_found="no"
       AC_MSG_RESULT([not found]) ])
  done

  if test "$lapack_found" == "no"; then
    if test "$with_lapack" != "probe"; then
      AC_MSG_ERROR([LAPACK enabled but no library found])
    fi
    AC_MSG_RESULT([No LAPACK library found])
    CPPFLAGS=$keep_CPPFLAGS
    LDFLAGS=$keep_LDFLAGS
    LIBS=$keep_LIBS
  else
    AC_MSG_RESULT([Using $lapack_found for LAPACK])
    AC_SUBST(VSIP_IMPL_HAVE_BLAS, 1)
    AC_DEFINE_UNQUOTED(VSIP_IMPL_HAVE_BLAS, 1,
      [Define to set whether or not BLAS is present.])
    AC_SUBST(VSIP_IMPL_HAVE_LAPACK, 1)
    AC_DEFINE_UNQUOTED(VSIP_IMPL_HAVE_LAPACK, 1,
      [Define to set whether or not LAPACK is present.])
    AC_DEFINE_UNQUOTED(VSIP_IMPL_USE_LAPACK_ILAENV, $lapack_use_ilaenv,
      [Use LAPACK ILAENV (0 == do not use, 1 = use).])
    if test $with_cblas == "yes"; then
      with_cblas=$cblas_style
    else
      with_cblas="0"
    fi

    # g77 by default uses the F2C ABI, while gfortran does not.
    if test "$use_f2c_abi" = yes; then
      if test "$neutral_acconfig" = 'y'; then
        CPPFLAGS="$CPPFLAGS -DVSIP_IMPL_USE_F2C_ABI"
      else
        AC_DEFINE_UNQUOTED(VSIP_IMPL_USE_F2C_ABI, 1,
          [Define to 1 if f2c ABI is to be used to interface with Fortran code.])
      fi
    fi
    if test "$neutral_acconfig" = 'y'; then
      CPPFLAGS="$CPPFLAGS -DVSIP_IMPL_USE_CBLAS=$with_cblas"
    else
      AC_DEFINE_UNQUOTED(VSIP_IMPL_USE_CBLAS, $with_cblas,
        [CBLAS style (0 == no CBLAS, 1 = ATLAS CBLAS, 2 = MKL CBLAS).])
    fi
  fi
fi

if test "x$lapack_found" = "x"; then
  lapack_found="no"
fi



])
