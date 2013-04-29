/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
// Header file for read_tbv.c
// Jules Bergmann, 2 Mar 05

#ifndef _SARSIM_READ_TBV_H_
#define _SARSIM_READ_TBV_H_

int
read_tbv(
   FILE		*fp,
   short int	*data,
   short int	*aux,
   float	*r2a,
   int		pol,
   int		ncsamples);

#endif // _SARSIM_READ_TBV_H_
