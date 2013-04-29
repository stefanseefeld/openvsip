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
AC_INIT(Sourcery VSIPL++, 3.1-4,
  vsipl++@codesourcery.com, sourceryvsipl++)

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

# Determine whether acconfig should be "neutral".  A neutral acconfig
# can be shared by different variants (parallel vs serial, IPP/MKL vs
# builtin) in the same binary package.
AC_ARG_ENABLE(shared-acconfig,
  AS_HELP_STRING([--enable-shared-acconfig],
                 [Attempt to make acconfig.hpp that can be shared
	 	  by different configurations.  Use this if you want
                  to share a common set of header files for multiple
                  Sourcery VSIPL++ configurations.]),
  [case x"$enableval" in
    xyes) neutral_acconfig="y" ;;
    xno)  neutral_acconfig="n" ;;
    *)   AC_MSG_ERROR([Invalid argument to --enable-shared-acconfig.])
   esac],
  [neutral_acconfig="n"])
 
AC_ARG_WITH(suffix,
  AS_HELP_STRING([--with-suffix=SUFFIX],
                 [Specify a suffix to be appended to the library and pc file.
                  This allows multiple configurations of the same library to
                  coexist in the same prefix, while headers are shared.
                  Typical suffixes are '-opt' or '-debug'.]),
  [suffix=$withval])
AC_SUBST(suffix)

AC_ARG_WITH(builtin-libdir,
  AS_HELP_STRING([--with-builtin-libdir=PATH],
                 [Specify a separate path to install builtin libraries,
 	          such as FFTW, that is different from libdir.
		  Defaults to libdir.]),
  [builtin_libdir=$withval],
  [builtin_libdir="\${libdir}"])
AC_SUBST(builtin_libdir)

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

AC_ARG_ENABLE([only-ref-impl],
  AS_HELP_STRING([--enable-only-ref-impl],
                 [Use reference implementation.]),
    [case x"$enableval" in
       xyes) only_ref_impl=1 ;;
       xno)  only_ref_impl=0 ;;
       *)   AC_MSG_ERROR([Invalid argument to --enable-only-ref-impl.])
     esac],
    [only_ref_impl=0]) 

# VSIP_IMPL_REF_IMPL is defined to 1 when building the reference
# implementation.  The reference implementation uses only the core
# of the library and the C-VSIP backends, all of which are available
# under GPL.  The reference implementation does not use the opt
# parts of the library, which are only available under GPL and
# commercial license.

# VSIP_CSL_LIB is an AC_SUBST used by the pkg-config vsipl++.pc and
# QMtest context files to determine if they should link against the
# vsip_csl library.  The vsip_csl library is not built with the
# reference implementation.

if test "$only_ref_impl" = "1"; then
  AC_DEFINE_UNQUOTED(VSIP_IMPL_REF_IMPL, 1,
        [Set to 1 to compile the reference implementation.])
  AC_SUBST(VSIP_IMPL_REF_IMPL, 1)
  AC_SUBST(VSIP_CSL_LIB, "")
else
  AC_SUBST(VSIP_CSL_LIB, "-lvsip_csl")
fi

AC_ARG_ENABLE([exceptions],
  AS_HELP_STRING([--disable-exceptions],
                 [Don't use C++ exceptions.]),,
  [enable_exceptions=probe])

AC_ARG_ENABLE(threading,
  [  --enable-threading         Specify whether or not to make the library thread-safe.],,
  [enable_threading=no])
if test "$enable_threading" = yes; then
  if test "$neutral_acconfig" = 'y'; then
    CPPFLAGS="$CPPFLAGS -DVSIP_IMPL_ENABLE_THREADING=1"
  else
    AC_DEFINE_UNQUOTED(VSIP_IMPL_ENABLE_THREADING, 1,
      [Set to 1 to enable multi-threading.])
  fi
  AC_SUBST(VSIP_IMPL_ENABLE_THREADING, 1)
fi

### NVidia CUDA
AC_ARG_WITH([cuda],
  AS_HELP_STRING([--with-cuda],
                 [Use NVidia's CUDA.]),,
  [with_cuda=no])

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

### Intel Performance Primitives (IPP)
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

### C-VSIPL
AC_ARG_ENABLE([cvsip],,  
  AC_MSG_ERROR([The option --enable-cvsip is obsolete; use 
    either --with-cvsip to use C-VSIPL as a backend or
    --enable-cvsip-bindings to build C-VSIPL bindings.
    (Run 'configure --help' for details)]),)

AC_ARG_WITH([cvsip],
  AS_HELP_STRING([--with-cvsip],
                 [Use C-VSIPL (as computational backend) (default is to not use it).]),,
  [with_cvsip=no])

AC_ARG_WITH(cvsip_prefix,
  AS_HELP_STRING([--with-cvsip-prefix=PATH],
                 [Specify the installation prefix of the C-VSIPL library.  Headers
                  must be in PATH/include; libraries in PATH/lib.]))

# Control default complex storage of library (split vs interleaved)
AC_ARG_WITH([complex],
  AS_HELP_STRING([--with-complex=FORMAT],
                 [Specify FORMAT to use for complex data: either 'split'
		  or 'interleaved'.  Default is to use preferred
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
                 [Set profile timer type.  Choices include none, posix, pentiumtsc, x86_64_tsc, mcoe_tmr, power_tb [[none]].]),,
  [enable_timer=none])

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

AC_ARG_ENABLE(eval-dense-expr,
  AS_HELP_STRING([--enable-eval-dense-expr],
                 [Activate evaluation of dense matrix and tensor expressions
		  as vector expressions when possible.  Enabled by default]),
  [case x"$enableval" in
    xyes) eval_dense_expr=1 ;;
    xno)  eval_dense_expr=0 ;;
    *)   AC_MSG_ERROR([Invalid argument to --enable-eval-dense-expr.])
   esac],
  [eval_dense_expr=1]) 

AC_DEFINE_UNQUOTED(VSIP_IMPL_ENABLE_EVAL_DENSE_EXPR, $eval_dense_expr,
      [Set 1 to enable eval_dense_expr evaluator, set 0 to disable.])

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
  [enable_python_bindings="no"])

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
  AS_HELP_STRING([--enable-shared-libs],
                 [Build VSIPL++ as shared libraries.]),
  [case x"$enableval" in
    xyes) BUILD_SHARED_LIBS=1 ;;
    xno)  BUILD_SHARED_LIBS= ;;
    *)   AC_MSG_ERROR([Invalid argument to --enable-shared-libs.])
   esac],
  [BUILD_SHARED_LIBS=]) 

#
# Files to generate.
#

# Find all template files and strip off the $srcdir and .in suffix.
templates=`find $srcdir -name '_darcs' -prune -o	\
	 -name '*.in' -print | 
           sed -e "s,^$srcdir/,,g" -e 's,\.in$,,g'`

AC_CONFIG_FILES([$templates])
AC_CONFIG_HEADERS([src/vsip/core/acconfig.hpp])

#
# QMTest wants to know this.
#
AC_CANONICAL_HOST

#
# Find the compiler.
#
if test "$with_cbe_sdk" != "no"; then
  if test "`echo $CXXFLAGS | sed -n '/-maltivec/p'`" == ""; then
    CXXFLAGS="-maltivec $CXXFLAGS"
    CFLAGS="-maltivec $CFLAGS"
  fi

  if test "`echo $CXXFLAGS | sed -n '/-mcpu=cell/p'`" == ""; then
    CXXFLAGS="-mcpu=cell $CXXFLAGS"
    CFLAGS="-mcpu=cell $CFLAGS"
  fi

  cxx_compiler_list="ppu-g++ g++ c++"
  c_compiler_list="ppu-gcc gcc cc"
  strip_list="ppu-strip strip"
  AC_CHECK_PROGS(CC_SPU, [spu-gcc])
  AC_CHECK_PROGS(CXX_SPU, [spu-g++])
  AC_CHECK_PROGS(EMBED_SPU, [ppu-embedspu embedspu])
  AC_CHECK_PROGS(OBJDUMP_SPU, [spu-objdump])
  AC_CHECK_PROGS(LD_SPU, [spu-ld])
  AC_CHECK_PROGS(AR_SPU, [spu-ar])

  if test -n "`echo $CFLAGS | sed -n '/-m32/p'`" -o \
          -n "`echo $CFLAGS | sed -n '/-q32/p'`"; then
    CPP_FLAGS_SPU="$CPP_FLAGS_SPU -DPPU_IS_32BIT=1"
    EMBED_SPU="$EMBED_SPU -m32"
  elif test -n "`echo $CFLAGS | sed -n '/-m64/p'`" -o \
            -n "`echo $CFLAGS | sed -n '/-q64/p'`"; then
    EMBED_SPU="$EMBED_SPU -m64"
  fi
else
  # Use autoconf default lists
  cxx_compiler_list="g++ c++ gpp aCC CC cxx cc++ cl.exe FCC KCC RCC xlC_r xlC"
  c_compiler_list="gcc cc cl.exe"
  strip_list="strip"
fi
AC_PROG_CXX($cxx_compiler_list)
AC_PROG_CC($c_compiler_list)

AC_PROG_F77
AC_CHECK_PROGS(STRIP, [$strip_list])

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

if test "x$GCC" = "xyes" -a "x$BUILD_SHARED_LIBS" != "x"; then
  # Put -fPIC into CXXFLAGS for building library only (not for
  # user applications).
  CXXFLAGS_LIB="-fPIC"
  CFLAGS_LIB="-fPIC"
  NVCCFLAGS_LIB="--compiler-options -fPIC"
  AC_SUBST(CXXFLAGS_LIB)
  AC_SUBST(CFLAGS_LIB)
  AC_SUBST(NVCCFLAGS_LIB)
  AC_SUBST(BUILD_SHARED_LIBS)
fi



#
### Enable NUMA support
# This must go after finding the compiler, due to the use of
# AC_CHECK_HEADERS
AC_ARG_ENABLE([numa],,  
  AC_MSG_ERROR([The option --enable-numa is obsolete; use 
    --with-numa instead.  (Run 'configure --help' for details)]),)

AC_ARG_WITH([numa],
  AS_HELP_STRING([--with-numa], [Enable support for libnuma]),,
  [with_numa="no"])
AC_CHECK_HEADERS([numa.h], [ have_numa_h="yes"], [], [])
if test "$with_numa" != "no" -a "$have_numa_h" == "yes"; then
  AC_DEFINE_UNQUOTED(VSIP_IMPL_NUMA, 1, [Set to 1 to support libnuma.])
  AC_SUBST(VSIP_IMPL_HAVE_NUMA, 1)
  LIBS="$LIBS -lnuma"
else
  AC_SUBST(VSIP_IMPL_HAVE_NUMA,"")
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
AC_CHECK_SIZEOF(short)
AC_CHECK_SIZEOF(int)
AC_CHECK_SIZEOF(long)
AC_CHECK_SIZEOF(long long)

#
# Set ar
#
if test "x$AR" == "x"; then
  if test "$with_cbe_sdk" != "no"; then
    AR="ppu-ar"
  else
    AR="ar"
  fi
fi

AC_SUBST(AR)

if test "$enable_exceptions" != "no"; then
  AC_MSG_CHECKING([for exceptions])
  AC_COMPILE_IFELSE([
    #include <stdexcept>

    void function() throw (std::bad_alloc)
    { throw std::bad_alloc(); }

    int main()
    {
      int i = 0;
      try { function(); } catch(std::bad_alloc e) { i = 1; }
    }
    ],
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

AC_DEFINE_UNQUOTED(VSIP_HAS_EXCEPTIONS, $has_exceptions,
                   [Define to 1 to use C++ exceptions.])

# Weed out buggy compilers and/or C++ runtime libraries.
# This is not an AC_CACHE_CHECK because it's likely to grow, so
# the cache would become invalid.
AC_MSG_CHECKING([for bugs in $CXX and its runtime])
# As of now, we're not checking for any bugs.
vsipl_cxx_buggy="no bugs found"
AC_MSG_RESULT($vsipl_cxx_buggy)

AC_MSG_CHECKING([for compatible std::complex types.])
AC_COMPILE_IFELSE([
#include <complex>

template <bool V> struct static_assert;
template <> struct static_assert<true>
{
  static bool const value = true;
};

int main(int, char **)
{
  bool value;
  value = static_assert<sizeof(std::complex<float>) == 2*sizeof(float)>::value;
  value = static_assert<sizeof(std::complex<double>) == 2*sizeof(double)>::value;
}
],
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
AC_COMPILE_IFELSE([
#include <cmath>

int main(int, char **)
{
  float f1 = 1.f;
  f1 = std::abs(f1); 
  double d1 = 1.0;
  d1 = std::abs(d1); 
  long double l1 = 1.0;
  l1 = std::abs(l1);
}
],
[have_abs_float="std"
 AC_MSG_RESULT(yes)],
[AC_MSG_RESULT([missing!])])

if test "$have_abs_float" = "no"; then
  # next check for them in ::
  AC_MSG_CHECKING([for ::abs(float), ::abs(double), and ::abs(long double).])
  AC_COMPILE_IFELSE([
#include <cmath>

int main(int, char **)
{
  float f1 = 1.f;
  f1 = ::abs(f1); 
  double d1 = 1.0;
  d1 = ::abs(d1); 
  long double l1 = 1.0;
  l1 = ::abs(l1);
}
],
  [have_abs_float="global"
   AC_MSG_RESULT(yes)],
  [AC_MSG_ERROR([missing!])])
  AC_DEFINE_UNQUOTED(VSIP_IMPL_FIX_MISSING_ABS, 1,
      [Define to use both ::abs and std::abs for vsip::mag.])
fi

#
# Check if standard library supports complex<long double>
#
# GreenHills std::abs support complex<float> and complex<double>,
# but not complex<long double> (neither does ::abs).
#

AC_MSG_CHECKING([if complex<long double> supported.])
AC_COMPILE_IFELSE([
#include <complex>

int main(int, char **)
{
  std::complex<long double> c1 = std::complex<long double>(1.0, 0.0);
  long double l1 = std::abs(c1);
}
],
[AC_DEFINE_UNQUOTED(VSIP_IMPL_HAVE_COMPLEX_LONG_DOUBLE, 1,
	         [Define if standard library supports complex<long double>.])
 AC_MSG_RESULT(yes)],
[AC_MSG_RESULT([no])])


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
  AC_LINK_IFELSE([
    #include <cmath>

    int main(int, char **)
    {
      using $fcn;
      float x = 1.f;
      return $fcn(x) ? 1 : 0;
    }
    ],
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
                   AC_SUBST(VSIP_IMPL_HAVE_PNG, 1)])])
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

SVXX_CHECK_FFT
SVXX_CHECK_PARALLEL
SVXX_CHECK_CUDA
SVXX_CHECK_SAL
SVXX_CHECK_MKL
SVXX_CHECK_IPP

if test "$only_ref_impl" = "1" -o "x$with_cvsip_prefix" != x; then
  with_cvsip="yes"
fi
if test "$with_cvsip_fft" == "yes"; then
  if test "$with_cvsip" == "no"; then
    AC_MSG_ERROR([C-VSIPL FFT requires C-VSIPL])
  fi 
fi

SVXX_CHECK_CVSIP

# Check if C-VSIPL front-end bindings enabled, and include -lsvsip in
# the package-config libs if so.
if test "$enable_cvsip_bindings" != "no"; then
   SVSIP_LIB="-lsvsip"
else
   SVSIP_LIB=""
fi
AC_SUBST(SVSIP_LIB)


if test "$neutral_acconfig" = 'y'; then
  if test "x$vsip_impl_fft_use_float" != "x"; then
    CPPFLAGS="$CPPFLAGS -DVSIP_IMPL_FFT_USE_FLOAT=$vsip_impl_fft_use_float"
  fi
  if test "x$vsip_impl_fft_use_double" != "x"; then
    CPPFLAGS="$CPPFLAGS -DVSIP_IMPL_FFT_USE_DOUBLE=$vsip_impl_fft_use_double"
  fi
  if test "x$vsip_impl_fft_use_long_double" != "x"; then
    CPPFLAGS="$CPPFLAGS -DVSIP_IMPL_FFT_USE_LONG_DOUBLE=$vsip_impl_fft_use_long_double"
  fi
else
  if test "x$vsip_impl_fft_use_float" != "x"; then
    AC_DEFINE_UNQUOTED(VSIP_IMPL_FFT_USE_FLOAT, $vsip_impl_fft_use_float,
	      [Define to build code with support for FFT on float types.])
  fi
  if test "x$vsip_impl_fft_use_double" != "x"; then
    AC_DEFINE_UNQUOTED(VSIP_IMPL_FFT_USE_DOUBLE, $vsip_impl_fft_use_double,
	      [Define to build code with support for FFT on double types.])
  fi
  if test "x$vsip_impl_fft_use_long_double" != "x"; then
    AC_DEFINE_UNQUOTED(VSIP_IMPL_FFT_USE_LONG_DOUBLE,
	               $vsip_impl_fft_use_long_double,
                       [Define to build code for long-double-precision FFT.])
  fi
fi

if test "$neutral_acconfig" = 'y'; then
  if test "x$provide_fft_float" != "x"; then
    CPPFLAGS="$CPPFLAGS -DVSIP_IMPL_PROVIDE_FFT_FLOAT=1"
  fi
  if test "x$provide_fft_double" != "x"; then
    CPPFLAGS="$CPPFLAGS -DVSIP_IMPL_PROVIDE_FFT_DOUBLE=1"
  fi
  if test "x$provide_fft_long_double" != "x"; then
    CPPFLAGS="$CPPFLAGS -DVSIP_IMPL_PROVIDE_FFT_LONG_DOUBLE=1"
  fi
else
  if test "x$provide_fft_float" != "x"; then
    AC_DEFINE_UNQUOTED([VSIP_IMPL_PROVIDE_FFT_FLOAT],1,
	      [Defined if Sourcery VSIPL++ supports for FFT on float types.])
  fi
  if test "x$provide_fft_double" != "x"; then
    AC_DEFINE_UNQUOTED([VSIP_IMPL_PROVIDE_FFT_DOUBLE],1,
	      [Defined if Sourcery VSIPL++ supports for FFT on double types.])
  fi
  if test "x$provide_fft_long_double" != "x"; then
    AC_DEFINE_UNQUOTED([VSIP_IMPL_PROVIDE_FFT_LONG_DOUBLE],1,
	      [Defined if Sourcery VSIPL++ supports for FFT on long double.])
  fi
fi

SVXX_CHECK_LAPACK

#
# Configure complex storage
#
if test "$with_complex" == "split"; then
  AC_DEFINE_UNQUOTED(VSIP_IMPL_PREFER_SPLIT_COMPLEX, 1, [Description])
else
  AC_DEFINE_UNQUOTED(VSIP_IMPL_PREFER_SPLIT_COMPLEX, 0, [Description])
fi

#
# Configure alignment
#
if test "$with_alignment" == "probe"; then
  with_alignment=32
fi
AC_DEFINE_UNQUOTED(VSIP_IMPL_ALLOC_ALIGNMENT, $with_alignment,
                   [Alignment for allocated memory (in bytes)])

#
# Configure profile mode
#
if test "$enable_profile_nesting" = "yes"; then
  AC_DEFINE_UNQUOTED(VSIP_IMPL_PROFILE_NESTING, 1,
                     [Define to nest events in profile accum mode.])
fi 

SVXX_CHECK_PROFILE

#
# Define VSIP_IMPL_TEST_LEVEL
#
AC_DEFINE_UNQUOTED(VSIP_IMPL_TEST_LEVEL, $with_test_level,
          [Define to set test suite effort level (0, 1, or 2).])


#
# set to allow apps to test which VSIPL++ they are using
#
AC_DEFINE([VSIP_IMPL_SOURCERY_VPP], [], 
          [Define to indicate this is CodeSourcery's VSIPL++.])

#
# Configure huge_page_pool support
#
AC_CHECK_HEADERS([sys/mman.h], [], [ enable_huge_page_pool="no"], [])
if test "$enable_huge_page_pool" = "yes"; then
  AC_DEFINE_UNQUOTED(VSIP_IMPL_ENABLE_HUGE_PAGE_POOL, 1,
                     [Define to enable huge page pool support.])
  AC_SUBST(VSIP_IMPL_HAVE_HUGE_PAGE_POOL, 1)
else
  AC_SUBST(VSIP_IMPL_HAVE_HUGE_PAGE_POOL, "")
fi 

#
# library
#
ARFLAGS="r"
RANLIB="echo"

AC_SUBST(ARFLAGS)
AC_SUBST(RANLIB)

#
# Documentation
#
AC_CHECK_PROGS(SYNOPSIS, synopsis)
AC_CHECK_PROGS(XSLTPROC, xsltproc)
AC_CHECK_PROGS(XEP, xep)

#
# Installation
# 
AC_PROG_INSTALL

# "Late" variables
LIBS="$LATE_LIBS $LIBS"

# Python interface
SVXX_CHECK_PYTHON

#
# Print summary.
#
AC_MSG_NOTICE(Summary)
AC_MSG_RESULT([Using config suffix:                     $suffix])
AC_MSG_RESULT([Exceptions enabled:                      $status_exceptions])
AC_MSG_RESULT([With parallel service enabled:           $par_service])
AC_MSG_RESULT([With LAPACK:                             $lapack_found])
AC_MSG_RESULT([With CUDA:                               $with_cuda])
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
  AC_MSG_RESULT([Complex storage format:                  interleaved])
fi
AC_MSG_RESULT([Using SIMD aligned loop-fusion           ${enable_simd_loop_fusion}])
AC_MSG_RESULT([Using SIMD unaligned loop-fusion         ${enable_simd_unaligned_loop_fusion}])
AC_MSG_RESULT([Timer:                                   ${enable_timer}])
AC_MSG_RESULT([With Python bindings:                    ${enable_python_bindings}])
AC_MSG_RESULT([With C-VSIPL bindings:                   ${enable_cvsip_bindings}])
AC_MSG_RESULT([With PNG support:                        ${with_png}])

#
# Done.
#
mkdir -p bin
mkdir -p lib
if test "$with_cbe_sdk" != "no"; then
  mkdir -p lib/spu
  mkdir -p lib/plugins
  mkdir -p lib/uk_plugin
fi
if test "$enable_scripting" = "yes"; then
  mkdir -p lib/python/vsip
fi
mkdir -p src/vsip/core/cvsip
mkdir -p src/vsip/core/expr
mkdir -p src/vsip/core/fft
mkdir -p src/vsip/core/mpi
mkdir -p src/vsip/core/parallel
mkdir -p src/vsip/core/reductions
mkdir -p src/vsip/core/signal
mkdir -p src/vsip/core/solver
mkdir -p src/vsip/opt/cuda
mkdir -p src/vsip/opt/cuda/kernels
mkdir -p src/vsip/opt/expr
mkdir -p src/vsip/opt/parallel
mkdir -p src/vsip/opt/pas
mkdir -p src/vsip/opt/signal
mkdir -p src/vsip/opt/simd
mkdir -p src/vsip/opt/lapack
mkdir -p src/vsip/opt/sal
mkdir -p src/vsip/opt/fftw3
mkdir -p src/vsip/opt/mkl
mkdir -p src/vsip/opt/ipp
mkdir -p src/vsip/opt/cbe/ppu
mkdir -p src/vsip/opt/cbe/spu
mkdir -p src/vsip/opt/cbe/spu/plugin
mkdir -p src/vsip/opt/cbe/spu/alf
mkdir -p src/vsip_csl/ukernel/kernels/cbe_accel
mkdir -p benchmarks/expr
mkdir -p benchmarks/mpi
mkdir -p benchmarks/ipp
mkdir -p benchmarks/sal
mkdir -p benchmarks/fftw3
mkdir -p benchmarks/lapack
mkdir -p benchmarks/cell
mkdir -p benchmarks/cuda
mkdir -p benchmarks/cvsip

AC_OUTPUT

#
# Tidy up library prefixes in .pc file
#

# if $srcdir is relative, correct by chdir into scripts/*.
fix_pc="`(cd $srcdir/scripts; echo \"$PWD\")`"/fix-pkg-config-prefix.sh

# Tidy up IPP
if test "$with_ipp" == "yes" -a "$with_ipp_prefix" != ""; then
  $fix_pc -p vsipl++.pc -d -k ipp_prefix -v $with_ipp_prefix
fi

# Tidy up MKL
if expr "$lapack_found" : "mkl" > /dev/null; then
  if test "$with_mkl_prefix" != ""; then
    $fix_pc -p vsipl++.pc -k mkl_prefix -v $with_mkl_prefix
  fi
fi

# Tidy up MPI
if test "$enable_parallel" = "yes"; then
  if test "x$with_mpi_prefix" != x; then
    $fix_pc -p vsipl++.pc -k mpi_prefix -v $with_mpi_prefix
  elif test "x$with_mpi_prefix64" != x; then
    $fix_pc -p vsipl++.pc -k mpi_prefix -v $with_mpi_prefix64
  fi
fi
  
