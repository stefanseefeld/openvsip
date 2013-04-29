/* Copyright (c) 2008 by CodeSourcery.  All rights reserved. */

/** @file    vsip/core/check_config_body.hpp
    @author  Jules Bergmann
    @date    2008-01-30
    @brief   VSIPL++ Library: Function body for checking library configuration.

    NOTE: Do not include this file directly.  Instead use check_config.hpp.
*/

  cfg << "Sourcery VSIPL++ Library Configuration\n";

#if VSIP_IMPL_PAR_SERVICE == 0
  cfg << "  VSIP_IMPL_PAR_SERVICE             - 0 (Serial)\n";
#elif VSIP_IMPL_PAR_SERVICE == 1
  cfg << "  VSIP_IMPL_PAR_SERVICE             - 1 (MPI)\n";
#else
  cfg << "  VSIP_IMPL_PAR_SERVICE             - Unknown\n";
#endif

#if VSIP_IMPL_HAVE_IPP
  cfg << "  VSIP_IMPL_IPP                     - 1\n";
#else
  cfg << "  VSIP_IMPL_IPP                     - 0\n";
#endif

#if VSIP_IMPL_HAVE_SAL
  cfg << "  VSIP_IMPL_SAL                     - 1\n";
#else
  cfg << "  VSIP_IMPL_SAL                     - 0\n";
#endif

#if VSIP_IMPL_CBE_SDK
  cfg << "  VSIP_IMPL_CBE_SDK                 - 1\n";
#else
  cfg << "  VSIP_IMPL_CBE_SDK                 - 0\n";
#endif

#if VSIP_IMPL_PREFER_SPLIT_COMPLEX
  cfg << "  VSIP_IMPL_PREFER_SPLIT_COMPLEX    - 1\n";
#else
  cfg << "  VSIP_IMPL_PREFER_SPLIT_COMPLEX    - 0\n";
#endif

#if VSIP_IMPL_HAS_EXCEPTIONS
  cfg << "  VSIP_IMPL_HAS_EXCEPTIONS          - 1\n";
#else
  cfg << "  VSIP_IMPL_HAS_EXCEPTIONS          - 0\n";
#endif

  cfg << "  VSIP_IMPL_ALLOC_ALIGNMENT         - "
      << VSIP_IMPL_ALLOC_ALIGNMENT << "\n";

#if VSIP_IMPL_AVOID_POSIX_MEMALIGN
  cfg << "  VSIP_IMPL_AVOID_POSIX_MEMALIGN    - 1\n";
#else
  cfg << "  VSIP_IMPL_AVOID_POSIX_MEMALIGN    - 0\n";
#endif

#if HAVE_POSIX_MEMALIGN
  cfg << "  HAVE_POSIX_MEMALIGN               - 1\n";
#else
  cfg << "  HAVE_POSIX_MEMALIGN               - 0\n";
#endif

#if HAVE_MEMALIGN
  cfg << "  HAVE_MEMALIGN                     - 1\n";
#else
  cfg << "  HAVE_MEMALIGN                     - 0\n";
#endif

#if __SSE__
  cfg << "  __SSE__                           - 1\n";
#else
  cfg << "  __SSE__                           - 0\n";
#endif

#if __SSE2__
  cfg << "  __SSE2__                          - 1\n";
#else
  cfg << "  __SSE2__                          - 0\n";
#endif

#if __VEC__
  cfg << "  __VEC__                           - 1\n";
#else
  cfg << "  __VEC__                           - 0\n";
#endif

#if _MC_EXEC
  cfg << "  _MC_EXEC                          - 1\n";
#else
  cfg << "  _MC_EXEC                          - 0\n";
#endif


  cfg << "Sourcery VSIPL++ FFT BE Configuration\n";

#if VSIP_IMPL_CBE_SDK_FFT
  cfg << "  VSIP_IMPL_CBE_SDK_FFT             - 1\n";
#else
  cfg << "  VSIP_IMPL_CBE_SDK_FFT             - 0\n";
#endif

#if VSIP_IMPL_CVSIP_FFT
  cfg << "  VSIP_IMPL_CVSIP_FFT               - 1\n";
#else
  cfg << "  VSIP_IMPL_CVSIP_FFT               - 0\n";
#endif

#if VSIP_IMPL_FFTW3
  cfg << "  VSIP_IMPL_FFTW3                   - 1\n";
#else
  cfg << "  VSIP_IMPL_FFTW3                   - 0\n";
#endif

#if VSIP_IMPL_IPP_FFT
  cfg << "  VSIP_IMPL_IPP_FFT                 - 1\n";
#else
  cfg << "  VSIP_IMPL_IPP_FFT                 - 0\n";
#endif

#if VSIP_IMPL_SAL_FFT
  cfg << "  VSIP_IMPL_SAL_FFT                 - 1\n";
#else
  cfg << "  VSIP_IMPL_SAL_FFT                 - 0\n";
#endif

#if VSIP_IMPL_DFT_FFT
  cfg << "  VSIP_IMPL_DFT_FFT                 - 1\n";
#else
  cfg << "  VSIP_IMPL_DFT_FFT                 - 0\n";
#endif

#if VSIP_IMPL_NO_FFT
  cfg << "  VSIP_IMPL_NO_FFT                  - 1\n";
#else
  cfg << "  VSIP_IMPL_NO_FFT                  - 0\n";
#endif

#if VSIP_IMPL_PROVIDE_FFT_FLOAT
  cfg << "  VSIP_IMPL_PROVIDE_FFT_FLOAT       - 1\n";
#else
  cfg << "  VSIP_IMPL_PROVIDE_FFT_FLOAT       - 0\n";
#endif

#if VSIP_IMPL_PROVIDE_FFT_DOUBLE
  cfg << "  VSIP_IMPL_PROVIDE_FFT_DOUBLE      - 1\n";
#else
  cfg << "  VSIP_IMPL_PROVIDE_FFT_DOUBLE      - 0\n";
#endif

#if VSIP_IMPL_PROVIDE_FFT_LONG_DOUBLE
  cfg << "  VSIP_IMPL_PROVIDE_FFT_LONG_DOUBLE - 1\n";
#else
  cfg << "  VSIP_IMPL_PROVIDE_FFT_LONG_DOUBLE - 0\n";
#endif


  cfg << "Sourcery VSIPL++ Compiler Configuration\n";

#if __GNUC__
  cfg << "  __GNUC__                          - " << __GNUC__ << "\n";
#endif

#if __ghs__
  cfg << "  __ghs__                           - " << __ghs__ << "\n";
#endif

#if __ICL
  cfg << "  __ICL                             - " << __ICL << "\n";
#endif
