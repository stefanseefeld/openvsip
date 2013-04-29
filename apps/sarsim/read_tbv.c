/*=====================================================================

		Copyright 1994 MIT Lincoln Laboratory

		C.T. Sung, MIT Lincoln Laboratory

	Function to read 32-bit input VHDL testbench format and
	return one polarization.

=====================================================================*/

/**
 * Revision 1.2  1998/06/01  20:23:38  anderson
 * *** empty log message ***
 *
 * Revision 1.1  1998/01/12  22:18:36  anderson
 * Initial revision
 *
 * Revision 1.1  1995/04/11  20:28:30  sung
 * Initial revision
 *
**/

#include <stdio.h>
#include <math.h>
#include "sarx.h"
#include "read_tbv.h"
#include "util_io.h"

void	conv2adts();
int find_fillcount(FILE* fp);


#define LIMIT	15000	/* Max number of words before quitting search
				for Barker Code */

int
read_tbv(
   FILE		*fp,
   short int	*data,
   short int	*aux,
   float	*r2a,
   int		pol,
   int		ncsamples)
{
	unsigned short	aux_mask[16], header;
	short int	odd, even;
	int		rval, k, tmpbuf[BCODE_SIZE], buf[MAXCSAMPLES],
			count,		/* Count of decoded Aux words(NAUX+1) */
			flag=0;		/* Set when desired pol ID is found */
/**
Set auxiliary mask.
**/
	aux_mask[0] = 32768;
	for(k=1; k<16; k++) aux_mask[k] = aux_mask[k-1] >> 1;
/**
Enter main reading loop.  Flag2 is set when the end of the desired
polarization has been read.
**/
	while (flag==0)
	{
/**
Perform barker code detection.
**/
	    if ((rval=find_fillcount(fp)) < 0)
	    { if (rval==-1) fprintf(stderr,"Unable to find fillcount word.\n");
	      if (rval==-2) fprintf(stderr, "Unable to read input file.\n");
	      return(-1);
	    }
/**
Dump the barker code.
**/
	    if (fread_bo(tmpbuf, BCODE_SIZE*sizeof(int), 1, fp, sizeof(int)) != 1)
	    {   fprintf(stderr,"Error reading input data.\n");
		exit(1);
	    }
/**
Get the data.
**/
	    if (fread_bo(buf, ncsamples*sizeof(int), 1, fp, sizeof(int)) != 1)
	    {   fprintf(stderr,"Error reading input data.\n");
		exit(1);
	    }
/**
Convert the data.
**/
	    (void) conv2adts(buf,ncsamples);
/**
Move each pair into the output data array.
**/
	    for(k=0; k<NAUX; k++) aux[k] = 0x0000;
	    for (k=0, count=0, header=0; k<ncsamples; k++)
	    {	odd = (short int) ((buf[k] >> 20) & 0x000003ff);
		if (((buf[k] >> 30) & 0x00000001) > 0) odd |= 0xfc00;
	        even = (short int) ((buf[k] >> 4) & 0x000003ff);
		if (((buf[k] >> 14) & 0x00000001) > 0) even |= 0xfc00;
		data[2*k] = even;
	        data[2*k+1] = odd;

		if ((k<(NAUX+1)*16) && ((buf[k] & 0x00000008) != 0))
		{   if (!count) header = header | aux_mask[k%16];
		    else	aux[count-1] =aux[count-1] | aux_mask[k%16];
		}
		if ((k<(NAUX+1)*16) && (k+1)%16==0)
		{   if (count==0)	/* Check Header */
		    {	if ((header | 0x00f0) == 0x03f0)
	    		{   if (pol==HH) flag = 1;
	    		}
	    		else if ((header | 0x00f0) == 0x43f5)
	    		{   if (pol==HV) flag = 1;
			    else	 break;
	    		}
	    		else if ((header | 0x00f0) == 0x83fa)
	    		{   if (pol==VH) flag = 1;
			    else	 break;
	    		}
	    		else if ((header | 0x00f0) == 0xc3ff)
	    		{   if (pol==VV) flag = 1;
			    else	 break;
	    		}
	    		else
	    		{   fprintf(stderr,"Invalid polarization ID.\n");
			    exit(1);
	    		}
		    }
		    count++;
		}
		if ((k>(NAUX+1)*16) && pol!=HH && ((header | 0x00f0)==0x03f0))
		    break;
	    }
/**
Compute range to aimpoint.
**/
	    if (pol==HH) *r2a = 2.0*aux[27] + (aux[28] & 0x7fff)/16384.;
	}

	return(0);
}

void
conv2adts(buf,n)
int	*buf, n;
{
	int	i, word, tmp;

	for (i=0; i<n; i++)
	{   word = ((buf[i] & 0x00000fff) << 4) & 0x0000fff0;
	    tmp = (buf[i] & 0x01ffe000) << 7;
	    word |= (tmp & 0xfff00000);
	    if ((buf[i] & 0x00001000) != 0) word |= 0x00090008;
	    buf[i] = word;
	}
	return;
}

int
find_fillcount(FILE* fp)
{
	int	word, ncount=0;

/**
Start main reading loop.
**/
	while (ncount<LIMIT)
	{   if (fread_bo(&word, sizeof(int), 1, fp, sizeof(int)) != 1) return(-2);
	    if ((word & 0x80000000) != 0) return(0);
	    else			  ncount++;
	}
	
	return(-1);
	
}
