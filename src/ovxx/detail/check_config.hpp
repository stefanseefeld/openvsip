//
// Copyright (c) 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.


  cfg << "OpenVSIP Library Configuration\n";

#if OVXX_HAVE_IPP
  cfg << "  OVXX_IPP                 - 1\n";
#else
  cfg << "  OVXX_IPP                 - 0\n";
#endif

#if OVXX_HAVE_SAL
  cfg << "  OVXX_SAL                 - 1\n";
#else
  cfg << "  OVXX_SAL                 - 0\n";
#endif

#if OVXX_DEFAULT_COMPLEX_STORAGE_SPLIT
  cfg << "  OVXX_DEFAULT_COMPLEX_STORAGE_SPLIT - 1\n";
#else
  cfg << "  OVXX_DEFAULT_COMPLEX_STORAGE_SPLIT - 0\n";
#endif

#if OVXX_HAS_EXCEPTIONS
  cfg << "  OVXX_HAS_EXCEPTIONS      - 1\n";
#else
  cfg << "  OVXX_HAS_EXCEPTIONS      - 0\n";
#endif

  cfg << "  OVXX_ALLOC_ALIGNMENT          - "
      << OVXX_ALLOC_ALIGNMENT << "\n";

#if HAVE_POSIX_MEMALIGN
  cfg << "  HAVE_POSIX_MEMALIGN           - 1\n";
#else
  cfg << "  HAVE_POSIX_MEMALIGN           - 0\n";
#endif

#if HAVE_MEMALIGN
  cfg << "  HAVE_MEMALIGN                 - 1\n";
#else
  cfg << "  HAVE_MEMALIGN                 - 0\n";
#endif

#if __SSE__
  cfg << "  __SSE__                       - 1\n";
#else
  cfg << "  __SSE__                       - 0\n";
#endif

#if __SSE2__
  cfg << "  __SSE2__                      - 1\n";
#else
  cfg << "  __SSE2__                      - 0\n";
#endif

#if __VEC__
  cfg << "  __VEC__                       - 1\n";
#else
  cfg << "  __VEC__                       - 0\n";
#endif

#if _MC_EXEC
  cfg << "  _MC_EXEC                      - 1\n";
#else
  cfg << "  _MC_EXEC                      - 0\n";
#endif


  cfg << "Open VSIP FFT BE Configuration\n";

#if OVXX_CVSIP_FFT
  cfg << "  OVXX_CVSIP_FFT                - 1\n";
#else
  cfg << "  OVXX_CVSIP_FFT                - 0\n";
#endif

#if OVXX_FFTW
  cfg << "  OVXX_FFTW                     - 1\n";
#else
  cfg << "  OVXX_FFTW                     - 0\n";
#endif

#if OVXX_IPP_FFT
  cfg << "  OVXX_IPP_FFT                  - 1\n";
#else
  cfg << "  OVXX_IPP_FFT                  - 0\n";
#endif

#if OVXX_SAL_FFT
  cfg << "  OVXX_SAL_FFT                  - 1\n";
#else
  cfg << "  OVXX_SAL_FFT                  - 0\n";
#endif

#if OVXX_DFT_FFT
  cfg << "  OVXX_DFT_FFT                  - 1\n";
#else
  cfg << "  OVXX_DFT_FFT                  - 0\n";
#endif

#if OVXX_NO_FFT
  cfg << "  OVXX_NO_FFT                   - 1\n";
#else
  cfg << "  OVXX_NO_FFT                   - 0\n";
#endif

  cfg << "OpenVSIP Compiler Configuration\n";

#if __GNUC__
  cfg << "  __GNUC__                      - " << __GNUC__ << "\n";
#endif

#if __ghs__
  cfg << "  __ghs__                       - " << __ghs__ << "\n";
#endif

#if __ICL
  cfg << "  __ICL                         - " << __ICL << "\n";
#endif
