dnl
dnl Copyright (c) 2007 by CodeSourcery
dnl Copyright (c) 2013 Stefan Seefeld
dnl All rights reserved.
dnl
dnl This file is part of OpenVSIP. It is made available under the
dnl license contained in the accompanying LICENSE.BSD file.

AC_DEFUN([OVXX_CHECK_TIMER],
[
#
# Configure timer
#
case $enable_timer in
  system)
    AC_MSG_CHECKING([if compiler supports std::chrono (C++11).])
    AC_LINK_IFELSE(
      [AC_LANG_PROGRAM([#include <chrono>],
	               [[using std::chrono::high_resolution_clock;]])],
      [AC_MSG_RESULT(yes)],
      [AC_MSG_ERROR(system timer requires C++11 support.)] )

    AC_DEFINE_UNQUOTED(OVXX_TIMER_SYSTEM, [1], [Use std::chrono::high_resolution_clock])
  ;;
  posix)
    AC_MSG_CHECKING([if Posix monotonic clock_gettime() available.])
    AC_SEARCH_LIBS([clock_gettime], [rt],
      [AC_MSG_RESULT(yes)],
      [AC_MSG_ERROR(Posix monotonic clock_gettime() not found.)] )

    AC_DEFINE_UNQUOTED(OVXX_TIMER_POSIX, [1], [Use clock_gettime()])
  ;;
  ia32_tsc)
    AC_MSG_CHECKING([if Pentium ia32 TSC assembly syntax supported.])
    AC_LINK_IFELSE(
      [AC_LANG_PROGRAM([],
	               [[long long time;
                         __asm__ __volatile__("rdtsc": "=A" (time));]])],
      [AC_MSG_RESULT(yes)],
      [AC_MSG_ERROR(GNU in-line assembly for Pentium ia32 rdtsc not supported.)] )

    AC_DEFINE_UNQUOTED(OVXX_TIMER_IA32_TSC, [1], [Use ia32 time stamp counter])
  ;;
  x64_tsc)
    AC_MSG_CHECKING([if x86_64 TSC assembly syntax supported.])
    AC_LINK_IFELSE(
      [AC_LANG_PROGRAM([],
	               [[typedef unsigned long long stamp_type;
		         stamp_type time; unsigned a, d;
                         __asm__ __volatile__("rdtsc": "=a" (a), "=d" (d));
                         time = ((stamp_type)a) | (((stamp_type)d) << 32);]])],
      [AC_MSG_RESULT(yes)],
      [AC_MSG_ERROR(GNU in-line assembly for x86_64 rdtsc not supported.)] )
    AC_DEFINE_UNQUOTED(OVXX_TIMER_X64_TSC, [1], [Use x64 time stamp counter])
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
    AC_DEFINE_UNQUOTED(OVXX_TIMER_POWER, 1, [Use Power timebase])
  ;;
  *)
    AC_MSG_ERROR([Invalid timer choosen --enable-timer=$enable_timer.])
esac

if test "$enable_cpu_mhz" != "none"; then
  AC_DEFINE_UNQUOTED(OVXX_CPU_SPEED, $enable_cpu_mhz,
    [Hardcoded CPU Speed (in MHz).])
fi

])
