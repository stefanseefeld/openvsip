/*=====================================================================

		Copyright 1994 MIT Lincoln Laboratory

		C.T. Sung, MIT Lincoln Laboratory

	Functions to perform reading and writing in the SAR output
	format.  For more details on the format, consult RASSP
	Benchmark-0.

=====================================================================*/

/**
 * Revision 1.2  1998/06/01  20:21:39  anderson
 * *** empty log message ***
 *
 * Revision 1.1  1998/01/12  22:18:36  anderson
 * Initial revision
 *
 * Revision 1.1  1995/04/11  20:26:26  sung
 * Initial revision
 *
**/

#include <stdio.h>
#include <math.h>
#include "sarx.h"
#include "read_adts.h"
#include "util_io.h"

int
read_frm_hdr(
   FILE		*fp,
   short int	*aux,
   int		*pol,
   float	*sq_ang)
{
	float		rlos=0., heading=0., dx, dy,
				vnms, vems, pnms, pems, trgn, trge;
	int		i, rval, word;
	float		pi = 3.14159265358979323846;
/**
Perform barker code detection.
**/
	if ((rval=cmp_barker(fp)) < 0)
	{   if (rval==-1) fprintf(stderr,"Unable to find barker sequence.\n");
	    if (rval==-2) fprintf(stderr, "Unable to read input file.\n");
	    return(-1);
	}
/**
Read header info.
**/
	if (4 != fread_bo(pol, 1, 4, fp, 4))
	{   fprintf(stderr, "Error reading frame header.\n");
	    return(-1);
	}
/**
Read aux info.
**/
	for (i=0; i<NAUX; i++)
	{   if (4 != fread_bo(&word, 1, 4, fp, 4))
	    {   fprintf(stderr, "Error reading frame header.\n");
		return(-1);
	    }
	    aux[i] = (short int) (word & 0x0000ffff);
	}
/**
Compute squint angle.
**/
		/* Pos. north motion sensing */
	pnms = 2.0*aux[7] + (aux[8] & 0x7fff)/16384.;
		/* Pos. east motion sensing */
	pems = 2.0*aux[9] + (aux[10] & 0x7fff)/16384.;
		/* Vel. north motion sensing */
	vnms = aux[13]/128. + (aux[14] & 0x7fff)/4194300.;
		/* Vel. east motion sensing */
	vems = aux[15]/128. + (aux[16] & 0x7fff)/4194300.;
		 /* Aimpoint north position */
	trgn = 2.0*aux[19] + (aux[20] & 0x7fff)/16384.;
		 /* Aimpoint east position */
	trge = 2.0*aux[21] + (aux[22] & 0x7fff)/16384.;

	heading = pi/2. - atan2(vnms,vems);
	if (heading > 2.*pi) heading -= 2.*pi;
	if (heading < 0.) heading += 2.*pi;
	dy = trgn - pnms;
	dx = trge - pems;
	rlos = pi/2. - atan2(dy,dx);
	if (rlos < 0.) rlos += 2.*pi;
	*sq_ang = rlos - heading;
	if (*sq_ang > pi) *sq_ang -= 2.*pi;
	if (*sq_ang < -pi) *sq_ang += 2.*pi;

	return(0);
}

int
write_frm_hdr(
   FILE		*fp,
   short int	*aux,
   int		pol)
{
	int	bark[20], laux[NAUX], lpol[1], i;
/**
Compose Barker Code.
**/
	for (i=0; i<5; i++) bark[i] = 0x00000000;
	for (i=5; i<8; i++) bark[i] = 0xffffffff;
	bark[8] = 0x00000000;
	bark[9] = 0xffffffff;
	for (i=10; i<12; i++) bark[i] = 0x00000000;
	bark[12] = 0xffffffff;
	for (i=13; i<16; i++) bark[i] = 0x00000000;
	for (i=16; i<18; i++) bark[i] = 0xffffffff;
	for (i=18; i<20; i++) bark[i] = 0x00000000;
/**
Repeat polarization header word.
**/
	if (pol==HH) lpol[0] = 0x03f003f0;
	else if (pol==HV) lpol[0] = 0x43f543f5;
	else if (pol==VH) lpol[0] = 0x83fa83fa;
	else if (pol==VV) lpol[0] = 0xc3ffc3ff;
	else
	{   fprintf(stderr, "Cannot form polarization header word\n");
	    return(-1);
	}
/**
Compose repeated aux data.
**/
	for (i=0; i<NAUX; i++)
	{   laux[i] = (int) ((aux[i] << 16) & 0xffff0000);
	    laux[i] = laux[i] | (aux[i] & 0x0000ffff);
	}
/**
Write this information to file.
**/
	if (fwrite_bo(bark, BCODE_SIZE*sizeof(int), 1, fp, sizeof(int)) != 1)
	{   fprintf(stderr,"Error writing output frame header.\n");
	    return(-1);
	}
	if (fwrite_bo(lpol, sizeof(int), 1, fp, sizeof(int)) != 1)
	{   fprintf(stderr,"Error writing output frame header.\n");
	    return(-1);
	}
	if (fwrite_bo(laux, NAUX*sizeof(int), 1, fp, sizeof(int)) != 1)
	{   fprintf(stderr,"Error writing output frame header.\n");
	    return(-1);
	}

	return(0);
}
