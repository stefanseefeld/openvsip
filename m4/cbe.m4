dnl Copyright (c) 2008 by CodeSourcery, Inc.  All rights reserved.
dnl
dnl File:   cbe.m4
dnl Author: Stefan Seefeld
dnl Date:   2008-04-17
dnl
dnl Contents: Cell/B.E. configuration for Sourcery VSIPL++
dnl

AC_DEFUN([SVXX_CHECK_CBE],
[
AC_ARG_ENABLE([cbe_sdk],,  
  AC_MSG_ERROR([The option --enable-cbe-sdk is obsolete; use 
    --with-cbe-sdk instead.  (Run 'configure --help' for details)]),)

AC_ARG_WITH([cbe_sdk],
  AS_HELP_STRING([--with-cbe-sdk],
                 [Use CBE SDK.]),,
  [with_cbe_sdk="no"])
AC_ARG_WITH(cbe_sdk_sysroot,
  AS_HELP_STRING([--with-cbe-sdk-sysroot=PATH],
                 [Specify the installation sysroot of the CBE SDK.]),
  [if test "$with_cbe_sdk" == "no"; then
     with_cbe_sdk="yes"
   fi],  [with_cbe_sdk_sysroot=])
AC_ARG_WITH(cbe_default_num_spes,
  AS_HELP_STRING([--with-cbe-default-num-spes=NUMBER],
  [Specify the default number of SPEs.]),
  [],
  [with_cbe_default_num_spes=8])

AC_ARG_WITH(cml_prefix,
  AS_HELP_STRING([--with-cml-prefix=PATH],
                 [Specify the installation path of CML.  Only valid
		  when using the CBE SDK.]))

AC_ARG_WITH(cml_libdir,
  AS_HELP_STRING([--with-cml-libdir=PATH],
                 [Specify the directory containing CML libraries.
		  Only valid when using the CBE SDK.]))

AC_ARG_WITH(cml_include,
  AS_HELP_STRING([--with-cml-include=PATH],
                 [Specify the directory containing CML header files.
		  Only valid when using the CBE SDK.]))


if test "$with_cbe_sdk" != "no"; then

  cbe_sdk_version=300

  AC_DEFINE_UNQUOTED(VSIP_IMPL_CBE_SDK, 1,
        [Set to 1 to support Cell Broadband Engine (requires CML).])
  AC_DEFINE_UNQUOTED(VSIP_IMPL_HAVE_CML, 1,
        [Set to 1 if CML is available (requires SDK).])
  AC_DEFINE_UNQUOTED(VSIP_IMPL_CBE_NUM_SPES, $with_cbe_default_num_spes,
        [Define default number of SPEs.])
  AC_SUBST(VSIP_IMPL_HAVE_CBE_SDK, 1)

  CPPFLAGS="$CPPFLAGS -I$with_cbe_sdk_sysroot/opt/cell/sdk/usr/include"
  if test -n "`echo $LDFLAGS | sed -n '/-m64/p'`" -o \
          -n "`echo $LDFLAGS | sed -n '/-q64/p'`"; then
    LDFLAGS="$LDFLAGS -L$with_cbe_sdk_sysroot/opt/cell/sdk/usr/lib64"
  else
    LDFLAGS="$LDFLAGS -L$with_cbe_sdk_sysroot/opt/cell/sdk/usr/lib"
  fi
  if test -n "$with_cbe_sdk_sysroot"; then
    CPPFLAGS="$CPPFLAGS --sysroot=$with_cbe_sdk_sysroot"
    LDFLAGS="$LDFLAGS --sysroot=$with_cbe_sdk_sysroot"
  fi

  CPP_FLAGS_SPU="$CPP_FLAGS_SPU -I$extra_include/spu"
  LD_FLAGS_SPU="$LD_FLAGS_SPU -L$extra_libdir/spu"
  LIBS="-lcml -lcsl_alf -lalf -lspe2 -ldl $LIBS"

  if test "$with_cml_include" != ""; then
    cml_incdir="$with_cml_include"
  elif test "$with_cml_prefix" != ""; then
    cml_incdir="$with_cml_prefix/include"
  else
    cml_incdir=""
  fi

  if test "$with_cml_libdir" != ""; then
    cml_libdirs="$with_cml_libdir"
  elif test "$with_cml_prefix" != ""; then
    cml_libdirs="$with_cml_prefix/lib $with_cml_prefix/lib64"
  else
    cml_libdirs=""
  fi

  if test -n "$cml_incdir" -o -n "$cml_libdirs"; then
    CPPFLAGS="-I$cml_incdir -I$cml_incdir/cml/ppu $CPPFLAGS"
    CPP_FLAGS_SPU="-I$cml_incdir/spu -I$cml_incdir/cml/spu $CPP_FLAGS_SPU"

    orig_LDFLAGS=$LDFLAGS
    orig_LD_FLAGS_SPU=$LD_FLAGS_SPU

    cml_libdir_found=no

    for trylibdir in $cml_libdirs; do
      AC_MSG_CHECKING([for CML libdir: $trylibdir])

      LDFLAGS="-L$trylibdir $orig_LDFLAGS"
      LD_FLAGS_SPU="-L$trylibdir/spu $orig_LD_FLAGS_SPU"

      AC_LINK_IFELSE(
        [AC_LANG_PROGRAM(
	  [[#include <cml.h>]],
	  [[cml_init(); cml_fini();]]
          )],
        [cml_libdir_found=$trylibdir
         AC_MSG_RESULT([found])
         break],
        [AC_MSG_RESULT([not found]) ])

    done

    if test "$cml_libdir_found" = "no"; then
      AC_MSG_ERROR([Cannot find CML libdir])
    fi

  fi

  if test "$neutral_acconfig" = 'y'; then
    CPPFLAGS="$CPPFLAGS -DVSIP_CBE_SDK_VERSION=$cbe_sdk_version"
    CPP_FLAGS_SPU="$CPP_FLAGS_SPU -DVSIP_CBE_SDK_VERSION=$cbe_sdk_version"
  else
    AC_DEFINE_UNQUOTED(VSIP_CBE_SDK_VERSION, $cbe_sdk_version,
          [Cell SDK version.])
  fi

  if test "x$CXXFLAGS_SPU" == "x"; then
    CXXFLAGS_SPU="-O3 -fno-threadsafe-statics -fno-rtti -fno-exceptions"
  fi

  if test "x$CFLAGS_SPU" == "x"; then
    CFLAGS_SPU="-O3"
  fi

  AC_SUBST(CPP_FLAGS_SPU, $CPP_FLAGS_SPU)
  AC_SUBST(LD_FLAGS_SPU, $LD_FLAGS_SPU)
else
  AC_SUBST(VSIP_IMPL_HAVE_CBE_SDK, "")
  cbe_sdk_version="none"
fi

AC_SUBST(CXXFLAGS_SPU)
AC_SUBST(CFLAGS_SPU)

AC_SUBST(cbe_sdk_sysroot, $with_cbe_sdk_sysroot)
AC_SUBST(cbe_sdk_version, $cbe_sdk_version)

])
