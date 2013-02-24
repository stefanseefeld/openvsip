/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
// Header file for read_adts.c
// Jules Bergmann, 2 Mar 05


#ifndef _SARSIM_READ_ADTS_H_
#define _SARSIM_READ_ADTS_H_

/**
Function to read one polarization of data.
**/
int
read_adts(
   FILE		*fp,
   short int	*data,
   short int	*aux,
   float	*r2a,
   int		pol,
   int		ncsamples);

int cmp_barker(FILE* fp);
int idcheck(unsigned short hdr);

#endif // _SARSIM_READ_ADTS_H_
