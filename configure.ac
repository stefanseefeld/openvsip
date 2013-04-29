dnl
dnl Copyright (c) 2005-2010 by CodeSourcery
dnl Copyright (c) 2013 Stefan Seefeld
dnl All rights reserved.
dnl
dnl This file is part of OpenVSIP. It is made available under the
dnl license contained in the accompanying LICENSE.BSD file.

dnl ------------------------------------------------------------------
dnl Autoconf initialization
dnl ------------------------------------------------------------------
AC_PREREQ(2.56)
AC_INIT(OpenVSIP, 1.0, devel@openvsip.org, openvsip)

######################################################################
# Configure command line arguments.
######################################################################

### Overall options.
#
# Rationale for option naming:
#
#  The general rule is that --enable is used for features
#  and --with is used for specific libraries.
#
#  In cases where an option controls both a feature and specifies
#  an external library, such as --enable-parallel, --enable should
#  be used.  The primary purpose of the option is turning on
#  the feature, the secondary purpose is specifying which
#  external code to use to enable that feature.

# These may get set from different backend checks.
provide_fft_float=
provide_fft_double=
provide_fft_long_double=

AC_ARG_WITH(variant,
  AS_HELP_STRING([--with-variant=VARIANT],
                 [Specify a build variant id to be used.
                  This allows multiple variants to share the same set of
		  header files, as the generated config headers use 
		  different names.]),
  [variant=$withval],
  [variant=])
if test -n "$variant"; then
  CPPFLAGS="$CPPFLAGS -DOVXX_VARIANT=$variant"
  AC_SUBST(variant)
fi

#
# Extra paths for CPPFLAGS and LDFLAGS to use.
#
AC_ARG_WITH(extra-include,, [extra_include=$withval])
if test -n "$extra_include"; then
  CPPFLAGS="$CPPFLAGS -I$extra_include"
fi
AC_ARG_WITH(extra-libdir,, [extra_libdir=$withval])
if test -n "$extra_libdir"; then
  LDFLAGS="$LDFLAGS -L$extra_libdir"
fi


### Filename extensions. 
AC_ARG_WITH(obj_ext,
  AS_HELP_STRING([--with-obj-ext=EXT],
                 [Specify the file extension to be used for object files.
                  Object files will be named file.$EXT.]),
  [obj_ext=$withval])
AC_ARG_WITH(lib_ext,
  AS_HELP_STRING([--with-lib-ext=EXT],
                 [Specify the file extension to be used for library
                  (archive) files. Library files will be named file.$EXT.]),
  [lib_ext=$withval])
AC_ARG_WITH(exe_ext,
  AS_HELP_STRING([--with-exe-ext=EXT],
                 [Specify the suffix to be used for executable files.
		  Executable files will be named file$EXT.  (Note: unlike
		  --with-obj-ext, no "." is implied.]),
  [exe_ext=$withval])

AC_ARG_ENABLE([exceptions],
  AS_HELP_STRING([--disable-exceptions],
                 [Don't use C++ exceptions.]),,
  [enable_exceptions=probe])

AC_ARG_ENABLE(threading,
  [  --enable-threading         Specify whether or not to make the library thread-safe.],,
  [enable_threading=no])
if test "$enable_threading" = yes; then
  AC_DEFINE_UNQUOTED(OVXX_ENABLE_THREADING, 1, [Set to 1 to enable multi-threading.])
  AC_SUBST(OVXX_ENABLE_THREADING, 1)
  CFLAGS="$CFLAGS -pthread"
  CXXFLAGS="$CXXFLAGS -pthread"
  LDFLAGS="$LDFLAGS -pthread"
fi

AC_ARG_ENABLE(omp,
  [  --enable-omp         Specify whether or not to enable support for OpenMP.],,
  [enable_omp=no])
if test "$enable_omp" = yes; then
  if test "$enable_threading" != "yes"; then
    AC_MSG_ERROR([OpenMP requires --enable-threading.])
  fi
  CXXFLAGS="$CXXFLAGS -fopenmp"
  LDFLAGS="$LDFLAGS -fopenmp"
  AC_DEFINE_UNQUOTED(OVXX_ENABLE_OMP, 1, [Set to 1 to enable OpenMP.])
  AC_SUBST(OVXX_ENABLE_OMP, 1)
fi

### NVidia CUDA
AC_ARG_WITH([cuda],
  AS_HELP_STRING([--with-cuda],
                 [Use NVidia's CUDA.]),,
  [with_cuda=no])

### CULA Tools
AC_ARG_WITH([cula],
  AS_HELP_STRING([--with-cula],
                 [Use CULA Tools.]),,
  [with_cula=no])

if test $with_cula != "no"; then
  if test "$with_cuda" != "yes"; then
    AC_MSG_ERROR([CULA Tools requires --with-cuda.])
  fi
fi

### Mercury Scientific Algorithm (SAL)
AC_ARG_ENABLE([sal],,  
  AC_MSG_ERROR([The option --enable-sal is obsolete; use 
    --with-sal instead.  (Run 'configure --help' for details)]),)

AC_ARG_WITH([sal],
  AS_HELP_STRING([--with-sal],
                 [Use SAL if found (default is to not search for it).]),,
  [with_sal=no])
AC_ARG_WITH(sal_include,
  AS_HELP_STRING([--with-sal-include=PATH],
                 [Specify the path to the SAL include directory.]),
  dnl If the user specified --with-sal-include, they mean to use SAL for sure.
  [with_sal=yes])
AC_ARG_WITH(sal_lib,
  AS_HELP_STRING([--with-sal-lib=PATH],
                 [Specify the installation path of the SAL library.]),
  dnl If the user specified --with-sal-lib, they mean to use SAL for sure.
  [with_sal=yes])

AC_ARG_ENABLE([multicore-sal],
  AS_HELP_STRING([--enable-multicore-sal],
                 [Enable the use of MultiCore SAL]),,
  [enable_multicore_sal=no])


### C-VSIPL
AC_ARG_WITH([cvsip],
  AS_HELP_STRING([--with-cvsip],
                 [Use C-VSIPL (as computational backend) (default is to not use it).]),,
  [with_cvsip=no])

AC_ARG_WITH(cvsip_prefix,
  AS_HELP_STRING([--with-cvsip-prefix=PATH],
                 [Specify the installation prefix of the C-VSIPL library.  Headers
                  must be in PATH/include; libraries in PATH/lib.]))
AC_ARG_WITH(cvsip_include,
  AS_HELP_STRING([--with-cvsip-include=PATH],
                 [Specify the path to the C-VSIPL include directory.]))
AC_ARG_WITH(cvsip_lib,
  AS_HELP_STRING([--with-cvsip-lib=PATH],
                 [Specify the installation path of the C-VSIPL library.]))

# Control default complex storage of library (split vs array)
AC_ARG_WITH([complex],
  AS_HELP_STRING([--with-complex=FORMAT],
                 [Specify FORMAT to use for complex data: either 'split'
		  or 'array'.  Default is to use preferred
		  format for system.]),
  ,
  )

# Control default alignment for memory allocations.
AC_ARG_WITH([alignment],
  AS_HELP_STRING([--with-alignment=ALIGNMENT],
                 [Specify ALIGNMENT to use for allocated data in bytes.
		  Default is to use preferred alignment for system.]),
  ,
  [with_alignment=probe])

AC_ARG_ENABLE([profile_nesting],
  AS_HELP_STRING([--disable-profile-nesting],
                 [Disable nesting of events for profile accumulate mode
                  (enabled by default)]),,
  [enable_profile_nesting=yes])

AC_ARG_ENABLE([timer],
  AS_HELP_STRING([--enable-timer=type],
                 [Set profile timer type.  Choices include system, posix, pentiumtsc, x86_64_tsc, power_tb [[posix]].]),,
  [enable_timer=posix])

AC_ARG_ENABLE([cpu_mhz],
  AS_HELP_STRING([--enable-cpu-mhz=speed],
                 [Set CPU speed in MHz.  Only necessary for TSC and if /proc/cpuinfo does not exist or is wrong.]),,
  [enable_cpu_mhz=none])

AC_ARG_WITH([test_level],
  AS_HELP_STRING([--with-test-level=WHAT],
                 [Set effort level for test-suite.  0 for low-level
		  (avoids long-running and long-compiling tests),
		  1 for regular effort, 2 for high-level (enables
		  long-running tests) [[1]].]),,
  [with_test_level=1])

AC_ARG_ENABLE(all-tests,,
  [case x"$enableval" in
     xyes) enable_all_tests=1 ;;
     *) enable_all_tests=0 
   esac],
  [enable_all_tests=0])
AC_SUBST(enable_all_tests)

AC_ARG_WITH(qmtest,,,[with_qmtest="qmtest"]) 

AC_SUBST(QMTEST, $with_qmtest)

AC_ARG_WITH(qmtest-target,,,[with_qmtest_target=""]) 

if test "x$with_qmtest_target" = "x"; then
  AC_SUBST(QMTEST_TARGET, "local_host.LocalHost")
else
  AC_SUBST(QMTEST_TARGET, "$with_qmtest_target")
fi
 
AC_ARG_ENABLE(python-bindings,
  [  --enable-python-bindings         Specify whether or not to build the python bindings.],,
  [enable_python_bindings=no])
AS_IF([test "x$enable_python_bindings" != xno],
  [enable_python_bindings=1], [enable_python_bindings=])

AC_ARG_WITH(python, 
  [  --with-python=PATH      Specify the Python interpreter.],
  PYTHON="$with_python",
  PYTHON="python"
)

AC_ARG_WITH(boost-prefix,
  [  --with-boost-prefix=PATH      Specify the boost installation prefix.],
  BOOST_PREFIX="$with_boost_prefix",
  BOOST_PREFIX="/usr"
)

AC_ARG_WITH(boost-version,
  [  --with-boost-version=VERSION      Specify the boost version.],
  BOOST_VERSION="$with_boost_version",
  BOOST_VERSION="1.33"
)

AC_ARG_ENABLE([huge_page_pool],
  AS_HELP_STRING([--disable-huge-page-pool],
                 [Disable support for huge page memory allocator pool
                  (enabled by default)]),,
  [enable_huge_page_pool=yes])


AC_ARG_ENABLE(cvsip-bindings,
  [  --enable-cvsip-bindings         Specify whether or not to build the C-VSIPL bindings.],,
  [enable_cvsip_bindings=no])
AC_SUBST(enable_cvsip_bindings)

AC_ARG_ENABLE(shared-libs,
  AS_HELP_STRING([--enable-shared-libs], [Build shared libraries.]),,
  [enable_shared_libs=yes])

AC_C_BIGENDIAN(
  AC_DEFINE(VSIP_BIG_ENDIAN, 1, [machine is bigendian]),
  AC_DEFINE(VSIP_BIG_ENDIAN, 0, [machine is littleendian]),
  AC_MSG_ERROR(unknown endianess),
  AC_MSG_ERROR(universial endianess not supported)
) 

#
# Files to generate.
#

# Find all template files and strip off the $srcdir and .in suffix.
templates=`find $srcdir -name '_darcs' -prune -o	\
	 -name 'rules*.in' -prune -o			\
	 -name 'ovxx*.pc.in' -prune -o		\
	 -name '*-config.hpp.in' -prune -o		\
	 -name '*.in' -print | 
           sed -e "s,^$srcdir/,,g" -e 's,\.in$,,g'`

rules_mk=rules.mk.in
if test "$INTEL_WIN" = 1; then
  rules_mk=rules.intel-win.mk.in
elif test "$with_cl6x" = yes; then
  rules_mk=rules.ti-cl6x.mk.in
fi

ovxx_pc_in=ovxx.pc.in
if test "$with_cuda" = "yes"; then
  ovxx_pc_in=ovxx-cuda.pc.in
elif test "$with_cl6x" = yes; then
  ovxx_pc_in=ovxx-cl6x.pc.in
fi

if test -n "$variant"; then
  acconfig=config-$variant.hpp
  ovxx_pc=ovxx-$variant.pc
  ovxxlib=ovxx-$variant
else
  acconfig=config.hpp
  ovxx_pc=ovxx.pc
  ovxxlib=ovxx
fi
if test "$with_cl6x" = yes; then
  acconfig_in=config-ti-cl6x.hpp.in
else
  acconfig_in=config.hpp.in
fi

AC_SUBST(OVXXLIB, $ovxxlib)

AC_CONFIG_FILES([$templates])
AC_CONFIG_FILES([rules.mk:$rules_mk])
AC_CONFIG_FILES([$ovxx_pc:$ovxx_pc_in])
dnl Unfortunately autoheader is unable to recognize the name of the header
dnl template to be generated, if the name itself is held in a shell variable.
dnl So we need to hardcode the additional 'config.hpp' file below, even if 
dnl we do not need it.
dnl We add a cleanup command via AC_CONFIG_COMMANDS below to remove it in that case.
AC_CONFIG_HEADERS([src/ovxx/detail/config.hpp])
AC_CONFIG_HEADERS([src/ovxx/detail/$acconfig:src/ovxx/detail/$acconfig_in])

#
# QMTest wants to know this.
#
AC_CANONICAL_HOST

#
# Find the compiler.
#
AC_PROG_CXX
AC_PROG_CC
AC_PROG_F77
AC_CHECK_TOOL([STRIP], [strip])
AC_CHECK_TOOL([RANLIB], [ranlib])
AC_CHECK_TOOL([AR], [ar])

if test "x$obj_ext" != "x"; then
  OBJEXT="$obj_ext"
  ac_cv_objext="$obj_ext"
fi
if test "x$exe_ext" != "x"; then
  EXEEXT="$exe_ext"
  ac_exeext="$exe_ext"
  ac_cv_exeext="$exe_ext"
fi
INTEL_WIN=
if test "$CXX" == "cxppc"; then
  CCDEP="$CC -Make"
  CXXDEP="$CXX -Make"
elif test "$CXX" == "icl"; then
  CCDEP="$CXX /QM"
  CXXDEP="$CXX /QM"
  INTEL_WIN=1
  cygwin_mount=`cygpath -w /`
elif test "$CXX" == "ccmc++"; then
  CCDEP="$CXX -M"
  CXXDEP="$CXX -M"
elif expr "$CXX" : ".*xlc" > /dev/null; then
  CCDEP="$CXX -M"
  CXXDEP="$CXX -M"
  LDSHARED="$CXX -qmkshrobj"
else # assume GCC
  CCDEP="$CC -MM -x c"
  CXXDEP="$CXX -MM -x c++"
  LDSHARED="$CXX -shared"
  cygwin_mount=
fi
AC_SUBST(CCDEP)
AC_SUBST(CXXDEP)
AC_SUBST(LDSHARED)
AC_SUBST(INTEL_WIN, $INTEL_WIN)
AC_SUBST(cygwin_mount)


#
# Build shared libraries, if requested
#
if test "x$GCC" = "xyes" -a "x$enable_shared_libs" = "xyes"; then
  # Put -fPIC into CXXFLAGS for building library only (not for
  # user applications).
  CXXFLAGS_LIB="-fPIC"
  CFLAGS_LIB="-fPIC"
  NVCFLAGS_LIB="--compiler-options -fPIC"
  AC_SUBST(CXXFLAGS_LIB)
  AC_SUBST(CFLAGS_LIB)
  AC_SUBST(NVCFLAGS_LIB)
  AC_SUBST(enable_shared_libs)
fi

#
# Determine library extension
#

if test "x$lib_ext" != "x"; then
  LIBEXT="$lib_ext"
else
  case $target in
    *-mingw32)
      LIBEXT="lib"
      ;;
    *)
      # Default is to use .a as the library archive extension.
      LIBEXT="a"

      # However, when cross-compiling for Mercury PowerPC systems,
      # .appc/.appc_le is preferred for big- and little- endian
      # systems.
      if test "$host" = "powerpc-unknown-none"; then
        if test "$OBJEXT" = "oppc"; then
          LIBEXT="appc"
        elif test "$OBJEXT" = "oppc_le"; then
          LIBEXT="appc_le"
        fi
      fi
      ;;
  esac
fi

AC_SUBST(LIBEXT)

AC_LANG(C++)
AC_CHECK_SIZEOF(int *)
AC_CHECK_SIZEOF(bool)
AC_CHECK_SIZEOF(short)
AC_CHECK_SIZEOF(int)
AC_CHECK_SIZEOF(long)
AC_CHECK_SIZEOF(long long)

if test "$enable_exceptions" != "no"; then
  AC_MSG_CHECKING([for exceptions])
  AC_COMPILE_IFELSE([AC_LANG_SOURCE([
    #include <stdexcept>

    void function() throw (std::bad_alloc)
    { throw std::bad_alloc(); }

    int main()
    {
      int i = 0;
      try { function(); } catch(std::bad_alloc e) { i = 1; }
      (void)i;
    }
    ])],
    [AC_MSG_RESULT(yes)
     has_exceptions=1],
    [AC_MSG_RESULT(no)
     has_exceptions=0
     if test "$enable_exceptions" = "yes"; then
       AC_MSG_ERROR([Exceptions enabled (--enable-exceptions), but
                     not supported by the compiler]) ])
     fi
else
  has_exceptions=0
fi

if test "$enable_exceptions" = "probe"; then
  if test "$has_exceptions" = "1"; then
    status_exceptions="probe -- found"
  else
    status_exceptions="probe -- not found"
  fi
else
  status_exceptions=$enable_exceptions
fi

AC_DEFINE_UNQUOTED(OVXX_HAS_EXCEPTIONS, $has_exceptions,
                   [Define to 1 to use C++ exceptions.])

# Weed out buggy compilers and/or C++ runtime libraries.
# This is not an AC_CACHE_CHECK because it's likely to grow, so
# the cache would become invalid.
AC_MSG_CHECKING([for bugs in $CXX and its runtime])
# As of now, we're not checking for any bugs.
vsipl_cxx_buggy="no bugs found"
AC_MSG_RESULT($vsipl_cxx_buggy)

AC_MSG_CHECKING([for compatible std::complex types.])
AC_COMPILE_IFELSE([AC_LANG_SOURCE([
#include <complex>

template <bool V> struct ct_assert;
template <> struct ct_assert<true>
{
  static bool const value = true;
};

int main(int, char **)
{
  bool value;
  value = ct_assert<sizeof(std::complex<float>) == 2*sizeof(float)>::value;
  value = ct_assert<sizeof(std::complex<double>) == 2*sizeof(double)>::value;
  (void)value;
}
])],
[AC_MSG_RESULT(yes)],
[AC_MSG_ERROR([Incompatible std::complex types detected!])])


#
# Check for the std::abs(...) overloads.
#
# GreenHills <cmath> defines ::abs(float), ::abs(double) and
# ::abs(long double), but does not place them into the std namespace when
# targeting mercury (when _MC_EXEC is defined).

# First check if std::abs handles float and double:
AC_MSG_CHECKING([for std::abs(float), std::abs(double), and std::abs(long double).])
have_abs_float="no"
AC_COMPILE_IFELSE([AC_LANG_SOURCE([
#include <cmath>

int main(int, char **)
{
  float f1 = 1.f;
  f1 = std::abs(f1); 
  (void)f1;
  double d1 = 1.0;
  d1 = std::abs(d1); 
  (void)d1;
  long double l1 = 1.0;
  l1 = std::abs(l1);
  (void)l1;
}
])],
[have_abs_float="std"
 AC_MSG_RESULT(yes)],
[AC_MSG_RESULT([missing!])])

if test "$have_abs_float" = "no"; then
  # next check for them in ::
  AC_MSG_CHECKING([for ::abs(float), ::abs(double), and ::abs(long double).])
  AC_COMPILE_IFELSE([AC_LANG_SOURCE([
#include <cmath>

int main(int, char **)
{
  float f1 = 1.f;
  f1 = ::abs(f1); 
  (void)f1;
  double d1 = 1.0;
  d1 = ::abs(d1); 
  (void)d1;
  long double l1 = 1.0;
  l1 = ::abs(l1);
  (void)l1;
}
])],
  [have_abs_float="global"
   AC_MSG_RESULT(yes)],
  [AC_MSG_ERROR([missing!])])
  AC_DEFINE_UNQUOTED(OVXX_FIX_MISSING_ABS, 1,
      [Define to use both ::abs and std::abs for vsip::mag.])
fi

#
# Check for the exp10 function.  
#
# The VSIPL++ specification requires a vsip::exp10 function.  Although
# there is no std::exp10 function in ISO C++, some systems (e.g.,
# those with GLIBC) do provide this function.
#
AC_CHECK_DECLS([exp10, exp10f, exp10l], [], [], 
	       [#include <cmath>])

#
# On GreenHills/Mercury, cmath/math.c provide declaration for acosh(),
# but do not link against a library containing.
# Solaris 2.8 does not declare hypotf, but g++'s runtime
# library provides a replacement.
#
# On GCC 3.4.4/Mercury, hypot is not provided
#
AC_CHECK_FUNCS([acosh hypotf hypot], [], [], [#include <cmath>])
AC_CHECK_DECLS([hypotf, hypot], [], [], [#include <cmath>])

#
# Check for std::isfinite, std::isnan, and std::isnormal
#
# isfinite, isnan, and isnormal are macros provided by C99 <math.h>
# They are not part of C++ <cmath>.
#
# GCC's cmath captures them, removing the macros, and providing
# functions std::isfinite, std::isnan, and std::isnormal.
#

# AC_CHECK_FUNCS will 'find' isfinite and isnan on Windows (ICC),
# even though the declarations are missing. (AC_CHECK_DECLS will fail.)
# However, instead of forward-declaring those functions, thus altering
# the environment as seen by user code, we don't use them at all.

# AC_CHECK_FUNCS doesn't find std::isfinite, even though it is there
# for GCC 4.1 on Linux.  Fall back to AC_COMPILE_IFELSE
#
# AC_CHECK_FUNCS([std::isfinite std::isnan std::isnormal], [], [],
#               [#include <cmath>])

for fcn in std::isfinite std::isnan std::isnormal; do

  AC_MSG_CHECKING([for $fcn])
  AC_LINK_IFELSE([AC_LANG_SOURCE([
    #include <cmath>

    int main(int, char **)
    {
      using $fcn;
      float x = 1.f;
      return $fcn(x) ? 1 : 0;
    }
    ])],
    [AC_MSG_RESULT(yes)
     if test $fcn = "std::isfinite"; then
       AC_DEFINE_UNQUOTED(HAVE_STD_ISFINITE, 1,
		   [Define to 1 if you have the '$fcn' function.])
     elif test $fcn = "std::isnan"; then
       AC_DEFINE_UNQUOTED(HAVE_STD_ISNAN, 1,
		   [Define to 1 if you have the '$fcn' function.])
     elif test $fcn = "std::isnormal"; then
       AC_DEFINE_UNQUOTED(HAVE_STD_ISNORMAL, 1,
		   [Define to 1 if you have the '$fcn' function.])
     fi],
    [AC_MSG_RESULT([no])])
done

AC_CHECK_FUNCS([getenv], [], [], [#include <cstdlib>])

#
# Check for posix_memalign, memalign
#
AC_CHECK_HEADERS([malloc.h], [], [], [// no prerequisites])
AC_CHECK_FUNCS([posix_memalign memalign])
AC_CHECK_DECLS([posix_memalign, memalign], [], [],
[#include <stdlib.h> // assumed to exist
#ifdef HAVE_MALLOC_H
#include <malloc.h>
#endif])

AC_ARG_WITH([png],
            AS_HELP_STRING([--with-png], [Enable PNG I/O using libpng.]),
            [with_png=$withval],            
            [with_png=probe])

if test "$with_png" != no; then
  AC_CHECK_HEADERS([png.h], []
                [AC_CHECK_LIB(png, png_read_info,
                  [have_png=yes
                   AC_SUBST(OVXX_HAVE_PNG, 1)])])
  if test "$with_png" = yes -a "$have_png" != yes
    then AC_MSG_ERROR([--with-png requires libpng to be installed])
  elif test "$with_png" = probe; then
    if test "$have_png" = yes
      then with_png="probe -- found"
    else
      with_png="probe -- not found"
    fi
  fi
fi

OVXX_CHECK_FFT
OVXX_CHECK_MPI

OVXX_CHECK_CUDA
OVXX_CHECK_SAL
OVXX_CHECK_MKL
OVXX_CHECK_IPP

if test "x$with_cvsip_prefix" != x; then
  with_cvsip="yes"
fi
if test "$with_cvsip_fft" == "yes"; then
  if test "$with_cvsip" == "no"; then
    AC_MSG_ERROR([C-VSIPL FFT requires C-VSIPL])
  fi 
fi

OVXX_CHECK_CVSIP
OVXX_CHECK_LAPACK

#
# Configure complex storage
#
if test "$with_complex" == "split"; then
  AC_DEFINE_UNQUOTED(OVXX_DEFAULT_COMPLEX_STORAGE_SPLIT, 1, [Description])
else
  AC_DEFINE_UNQUOTED(OVXX_DEFAULT_COMPLEX_STORAGE_SPLIT, 0, [Description])
fi

#
# Configure alignment
#
if test "$with_alignment" == "probe"; then
  with_alignment=32
fi
AC_DEFINE_UNQUOTED(OVXX_ALLOC_ALIGNMENT, $with_alignment,
                   [Alignment for allocated memory (in bytes)])

OVXX_CHECK_TIMER

#
# Configure huge_page_pool support
#
AC_CHECK_HEADERS([sys/mman.h], [], [ enable_huge_page_pool="no"], [])
if test "$enable_huge_page_pool" = "yes"; then
  AC_DEFINE_UNQUOTED(OVXX_ENABLE_HUGE_PAGE_POOL, 1,
                     [Define to enable huge page pool support.])
  AC_SUBST(OVXX_HAVE_HUGE_PAGE_POOL, 1)
else
  AC_SUBST(OVXX_HAVE_HUGE_PAGE_POOL, "")
fi 

#
# Installation
# 
AC_PROG_INSTALL

# Python interface
OVXX_CHECK_PYTHON

#
# Print summary.
#
AC_MSG_NOTICE(Summary)
AC_MSG_RESULT([Using config variant:                    $variant])
AC_MSG_RESULT([Exceptions enabled:                      $status_exceptions])
AC_MSG_RESULT([With MPI:                                $mpi_backend])
AC_MSG_RESULT([With OMP:                                $enable_omp])
AC_MSG_RESULT([With LAPACK:                             $lapack_found])
AC_MSG_RESULT([With CUDA:                               $with_cuda])
AC_MSG_RESULT([With CULA:                               $with_cula])
AC_MSG_RESULT([With SAL:                                $with_sal])
AC_MSG_RESULT([With IPP:                                $with_ipp])
AC_MSG_RESULT([With C-VSIPL:                            $with_cvsip])
AC_MSG_RESULT([Using FFT backends:                      ${enable_fft}])
if test "$provide_fft_float" == "1"; then
  AC_MSG_RESULT([  Provides float FFTs])
fi
if test "$provide_fft_double" == "1"; then
  AC_MSG_RESULT([  Provides double FFTs])
fi
if test "$provide_fft_long_double" == "1"; then
  AC_MSG_RESULT([  Provides long double FFTs])
fi

if test "$with_complex" == "split"; then
  AC_MSG_RESULT([Complex storage format:                  split])
else
  AC_MSG_RESULT([Complex storage format:                  array])
fi
AC_MSG_RESULT([Timer:                                   ${enable_timer}])
AS_IF([test "$enable_python_bindings" == "1"],
  [AC_MSG_RESULT([With Python bindings:                    yes])],
  [AC_MSG_RESULT([With Python bindings:                    no])])
AC_MSG_RESULT([With C-VSIPL bindings:                   ${enable_cvsip_bindings}])
AC_MSG_RESULT([With PNG support:                        ${with_png}])

#
# Done.
#
mkdir -p bin
mkdir -p lib
if test "$enable_python_bindings" = "1"; then
  mkdir -p lib/python/vsip
fi
mkdir -p benchmarks/expr
mkdir -p benchmarks/mpi
mkdir -p benchmarks/ipp
mkdir -p benchmarks/sal
mkdir -p benchmarks/fftw
mkdir -p benchmarks/lapack
mkdir -p benchmarks/cuda
mkdir -p benchmarks/cvsip

dnl Replace generated config.hpp if we are using a variant-specific header.
dnl (See above for why we are generating one only to overwrite it.)
AC_CONFIG_COMMANDS([cleanup-acconfig], 
  [[if test -n "$variant"; then
      cp $ac_abs_top_srcdir/src/ovxx/detail/config-error.hpp src/ovxx/detail/config.hpp
    fi]], [variant=$variant])

AC_OUTPUT
