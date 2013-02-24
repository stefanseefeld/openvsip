dnl Copyright (c) 2007 by CodeSourcery, Inc.  All rights reserved.
dnl
dnl File:   profile.m4
dnl Author: Stefan Seefeld
dnl Date:   2007-12-28
dnl
dnl Contents: Profile configuration for Sourcery VSIPL++
dnl

AC_DEFUN([SVXX_CHECK_PROFILE],
[
#
# Configure profile timer
#
case $enable_timer in
  none | no)
    AC_DEFINE_UNQUOTED(VSIP_IMPL_PROFILE_TIMER, 0,
      [Profile timer (0: None, 1: Posix, 2: ia32 TSC, 3: x86_64 TSC, 4: MCOE TMR, 5: Power TB).])
  ;;
  posix)
    AC_MSG_CHECKING([if Posix monotonic clock_gettime() available.])
    AC_SEARCH_LIBS([clock_gettime], [rt],
      [AC_MSG_RESULT(yes)],
      [AC_MSG_ERROR(Posix monotonic clock_gettime() not found.)] )

    AC_DEFINE_UNQUOTED(VSIP_IMPL_PROFILE_TIMER, 1,
      [Profile timer (0: None, 1: Posix, 2: ia32 TSC, 3: x86_64 TSC, 4: MCOE TMR, 5: Power TB).])
  ;;
  pentiumtsc)
    AC_MSG_CHECKING([if Pentium ia32 TSC assembly syntax supported.])
    AC_LINK_IFELSE(
      [AC_LANG_PROGRAM([],
	               [[long long time;
                         __asm__ __volatile__("rdtsc": "=A" (time));]])],
      [AC_MSG_RESULT(yes)],
      [AC_MSG_ERROR(GNU in-line assembly for Pentium ia32 rdtsc not supported.)] )

    AC_DEFINE_UNQUOTED(VSIP_IMPL_PROFILE_TIMER, 2,
      [Profile timer (0: None, 1: Posix, 2: ia32 TSC, 3: x86_64 TSC, 4: MCOE TMR, 5: Power TB).])
  ;;
  x86_64_tsc)
    AC_MSG_CHECKING([if x86_64 TSC assembly syntax supported.])
    AC_LINK_IFELSE(
      [AC_LANG_PROGRAM([],
	               [[typedef unsigned long long stamp_type;
		         stamp_type time; unsigned a, d;
                         __asm__ __volatile__("rdtsc": "=a" (a), "=d" (d));
                         time = ((stamp_type)a) | (((stamp_type)d) << 32);]])],
      [AC_MSG_RESULT(yes)],
      [AC_MSG_ERROR(GNU in-line assembly for x86_64 rdtsc not supported.)] )
    AC_DEFINE_UNQUOTED(VSIP_IMPL_PROFILE_TIMER, 3,
      [Profile timer (0: None, 1: Posix, 2: ia32 TSC, 3: x86_64 TSC, 4: MCOE TMR, 5: Power TB).])
  ;;
  mcoe_tmr)
    AC_MSG_CHECKING([if MCOE TMR timer is available.])
    AC_LINK_IFELSE(
      [AC_LANG_PROGRAM([[#include <mcos.h>]],
	 	       [[TMR_ts ts;
                         tmr_timestamp(&ts);]])],
      [AC_MSG_RESULT(yes)],
      [AC_MSG_ERROR(MCOE TMR timer not found.)] )

    AC_DEFINE_UNQUOTED(VSIP_IMPL_PROFILE_TIMER, 4,
      [Profile timer (0: None, 1: Posix, 2: ia32 TSC, 3: x86_64 TSC, 4: MCOE TMR, 5: Power TB).])
  ;;
  power_tb)
    AC_MSG_CHECKING([if PowerPC timebase assembly syntax supported.])
    AC_LINK_IFELSE(
      [AC_LANG_PROGRAM([],
	  	       [[
       unsigned int tbl, tbu0, tbu1;

       do {
	    __asm__ __volatile__ ("mftbu %0" : "=r"(tbu0));
	    __asm__ __volatile__ ("mftb %0" : "=r"(tbl));
	    __asm__ __volatile__ ("mftbu %0" : "=r"(tbu1));
       } while (tbu0 != tbu1);
	 	       ]])],
      [AC_MSG_RESULT(yes)],
      [AC_MSG_ERROR(GNU in-line assembly for PowerPC timebase not supported.)] )
    AC_DEFINE_UNQUOTED(VSIP_IMPL_PROFILE_TIMER, 5,
      [Profile timer (0: None, 1: Posix, 2: ia32 TSC, 3: x86_64 TSC, 4: MCOE TMR, 5: Power TB).])
  ;;
  *)
    AC_MSG_ERROR([Invalid timer choosen --enable-timer=$enable_timer.])
esac

if test "$enable_cpu_mhz" != "none"; then
  AC_DEFINE_UNQUOTED(VSIP_IMPL_PROFILE_HARDCODE_CPU_SPEED, $enable_cpu_mhz,
    [Hardcoded CPU Speed (in MHz).])
fi

])
