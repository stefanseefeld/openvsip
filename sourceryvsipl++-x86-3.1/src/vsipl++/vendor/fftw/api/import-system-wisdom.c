/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "api.h"

#if defined(FFTW_SINGLE)
#  define WISDOM_NAME "wisdomf"
#elif defined(FFTW_LDOUBLE)
#  define WISDOM_NAME "wisdoml"
#else
#  define WISDOM_NAME "wisdom"
#endif

/* OS-specific configuration-file directory */
#if defined(__DJGPP__)
#  define WISDOM_DIR "/dev/env/DJDIR/etc/fftw/"
#else
#  define WISDOM_DIR "/etc/fftw/"
#endif

int X(import_system_wisdom)(void)
{
#if defined(__WIN32__) || defined(WIN32) || defined(_WINDOWS)
     return 0; /* TODO? */
#else

     FILE *f;
     f = fopen(WISDOM_DIR WISDOM_NAME, "r");
     if (f) {
          int ret = X(import_wisdom_from_file)(f);
          fclose(f);
          return ret;
     } else
          return 0;
#endif
}
