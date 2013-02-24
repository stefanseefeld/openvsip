/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#include "ifftw.h"


/* Given a solver which_dim, a vector sz, and whether or not the
   transform is out-of-place, return the actual dimension index that
   it corresponds to.  The basic idea here is that we return the
   which_dim'th valid dimension, starting from the end if
   which_dim < 0. */
static int really_pickdim(int which_dim, const tensor *sz, int oop, int *dp)
{
     int i;
     int count_ok = 0;
     if (which_dim > 0) {
          for (i = 0; i < sz->rnk; ++i) {
               if (oop || sz->dims[i].is == sz->dims[i].os)
                    if (++count_ok == which_dim) {
                         *dp = i;
                         return 1;
                    }
          }
     }
     else if (which_dim < 0) {
          for (i = sz->rnk - 1; i >= 0; --i) {
               if (oop || sz->dims[i].is == sz->dims[i].os)
                    if (++count_ok == -which_dim) {
                         *dp = i;
                         return 1;
                    }
          }
     }
     else { /* zero: pick the middle, if valid */
	  i = (sz->rnk - 1) / 2;
	  if (i >= 0 && (oop || sz->dims[i].is == sz->dims[i].os)) {
	       *dp = i;
	       return 1;
	  }
     }
     return 0;
}

/* Like really_pickdim, but only returns 1 if no previous "buddy"
   which_dim in the buddies list would give the same dim. */
int X(pickdim)(int which_dim, const int *buddies, int nbuddies,
	       const tensor *sz, int oop, int *dp)
{
     int i, d1;

     if (!really_pickdim(which_dim, sz, oop, dp))
          return 0;

     /* check whether some buddy solver would produce the same dim.
        If so, consider this solver unapplicable and let the buddy
        take care of it.  The smallest-indexed buddy is applicable. */
     for (i = 0; i < nbuddies; ++i) {
          if (buddies[i] == which_dim)
               break;  /* found self */
          if (really_pickdim(buddies[i], sz, oop, &d1) && *dp == d1)
               return 0; /* found equivalent buddy */
     }
     return 1;
}
