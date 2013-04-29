dnl Copyright (c) 2009 by CodeSourcery, Inc.  All rights reserved.
dnl
dnl File:   release.m4
dnl Author: Stefan Seefeld
dnl Date:   2009-03-26
dnl
dnl Contents: release-related configuration for Sourcery VSIPL++
dnl

AC_DEFUN([SVXX_RELEASE],
[

AC_ARG_WITH(version-string,
  AS_HELP_STRING([--with-version-string=VERSION],
		 [The version string, with just the version number.]),
  [case "$withval" in
    yes) AC_MSG_ERROR([version-string not specified]) ;;
    no) ;;
    *) version_string="$withval" 
       major_version_string="${withval%-*}"
       ;;
  esac])
AC_SUBST(version_string)
AC_DEFINE_UNQUOTED([VSIP_IMPL_VERSION_STRING],["$version_string"], [Version string.])
AC_SUBST(major_version_string)
AC_DEFINE_UNQUOTED([VSIP_IMPL_MAJOR_VERSION_STRING],["$major_version_string"], [Major version string.])

AC_ARG_WITH(pkgversion,
  AS_HELP_STRING([--with-pkgversion=VERSION],
		 [The version string, including package prefixes.]),
  [case "$withval" in
     yes | no) AC_MSG_ERROR([pkgversion not specified]) ;;
     *) pkgversion="$withval" ;;
   esac],
  [pkgversion="Sourcery VSIPL++"])
AC_SUBST(pkgversion)

])
