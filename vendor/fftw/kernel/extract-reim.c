/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "ifftw.h"

/* decompose complex pointer into real and imaginary parts.
   Flip real and imaginary if there the sign does not match
   FFTW's idea of what the sign should be */

void X(extract_reim)(int sign, R *c, R **r, R **i)
{
     if (sign == FFT_SIGN) {
          *r = c + 0;
          *i = c + 1;
     } else {
          *r = c + 1;
          *i = c + 0;
     }
}
