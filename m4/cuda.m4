dnl
dnl Copyright (c) 2007 by CodeSourcery
dnl Copyright (c) 2013 Stefan Seefeld
dnl All rights reserved.
dnl
dnl This file is part of OpenVSIP. It is made available under the
dnl license contained in the accompanying LICENSE.BSD file.

AC_DEFUN([SVXX_CHECK_CUDA],
[
  AC_ARG_WITH(cuda_prefix,
    AS_HELP_STRING([--with-cuda-prefix=PATH],
                   [Specify the installation prefix of the CUDA libraries.  Headers
                    must be in PATH/include; libraries in PATH/lib64 or PATH/lib.]))

  # There are two cuda libraries supported: CUBLAS and CUFFT.  They are
  # configured using --with-cuda and --enable-fft=cuda.  Use of the FFT
  # library requires use of the --with-cuda option.


  # CUBLAS
  if test "$with_cuda" != "no"; then

    # Chose reasonable default if no prefix provided.
    if test -z "$with_cuda_prefix"; then
      with_cuda_prefix="/usr/local/cuda"
    fi

    CUDA_CPPFLAGS="-I$with_cuda_prefix/include"

    save_CPPFLAGS="$CPPFLAGS"
    CPPFLAGS="$CPPFLAGS $CUDA_CPPFLAGS"
  
    # Find cuda.h.
    AC_CHECK_HEADER([cuda.h],, [AC_MSG_ERROR([CUDA enabled, but cuda.h not found.])])

    # Find more CUDA headers.
    AC_CHECK_HEADERS([cublas.h cufft.h],,
      [AC_MSG_ERROR([CUDA enabled, but headers are not found.])])

    # Set the library path depending on whether this is a 64-bit build or
    # not.  The libcuda.so file has various other shared-library dependencies
    # that may or may not be in the linker's search path, so we also set 
    # -Wl,--allow-shlib-undefined to avoid trying to resolve them.
    if test "$ac_cv_sizeof_int_p" -eq 8 ; then
      CUDA_LDFLAGS="-L$with_cuda_prefix/lib64 -Wl,--allow-shlib-undefined"
    else
      CUDA_LDFLAGS="-L$with_cuda_prefix/lib32 -L$with_cuda_prefix/lib -Wl,--allow-shlib-undefined"
    fi

    # Find the library.
    save_LDFLAGS="$LDFLAGS"
    LDFLAGS="$CUDA_LDFLAGS $LDFLAGS"
    LIBS="$LIBS -lcuda -lcudart"
    AC_SEARCH_LIBS(cublasGetError, [cublas],
      [cublas_lib=$ac_lib cuda_found="yes"],
      [AC_MSG_ERROR([CUDA BLAS library not found])])

    # Declare it as found.	
    AC_SUBST(VSIP_IMPL_HAVE_CUDA, 1)
    if test "$neutral_acconfig" = 'y'; then
      CPPFLAGS="$CPPFLAGS -DVSIP_IMPL_HAVE_CUDA=1"
    else
      AC_DEFINE_UNQUOTED(VSIP_IMPL_HAVE_CUDA, 1,
        [Define to set whether or not to use NVIDIA's CUDA libraries.])
    fi

    # CUFFT
    if test $enable_cuda_fft != "no"; then

      # Find the library file
      AC_SEARCH_LIBS(cufftPlan1d, [cufft],
                     [cufft_lib=$ac_lib
  		        cuda_fft_found="yes"],
  		       [cuda_fft_found="no"])

      if test "$cuda_fft_found" = "no"; then
        AC_MSG_ERROR([CUDA FFT library not found])
      else

        # Declare it as found.
  	  provide_fft_float=1
        AC_SUBST(VSIP_IMPL_CUDA_FFT, 1)
        if test "$neutral_acconfig" = 'y'; then
          CPPFLAGS="$CPPFLAGS -DVSIP_IMPL_CUDA_FFT=1"
        else
          AC_DEFINE_UNQUOTED(VSIP_IMPL_CUDA_FFT, 1,
            [Define to set whether or not to use NVIDIA's CUDA FFT library.])
        fi
      fi
    fi

    # Make sure the appropriate flags are passed down to nvcc.
    if test -n "`echo $CFLAGS | sed -n '/-m32/p'`"; then
      NVCCFLAGS="-m32"
    elif test -n "`echo $CFLAGS | sed -n '/-m64/p'`"; then
      NVCCFLAGS="-m64"
    fi
    AC_SUBST(NVCCFLAGS)
  fi
])
