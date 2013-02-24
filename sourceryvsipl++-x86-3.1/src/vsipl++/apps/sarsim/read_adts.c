/*=====================================================================

		Copyright 1994 MIT Lincoln Laboratory

		C.T. Sung, MIT Lincoln Laboratory

	Function to read SAR disk data in the 32-bit version of the
	40-bit ADTS format specified in figure 3 of BM-0 and return one
	polarization of data for one PRI in a supplied data array.
	Data pairs are assumed to be in two's complement format.

	Note: This version is more robust in its polarization ID
	detection by taking the best match of the first AUX word.
	Thus, all 16 bits do not have to match the template for
	polarization detection.

=====================================================================*/

/**
 * Revision 1.2  1998/06/01  20:22:48  anderson
 * *** empty log message ***
 *
 * Revision 1.1  1998/01/12  22:18:36  anderson
 * Initial revision
 *
 * Revision 1.2  1995/08/30  18:49:12  sung
 * Made pol ID detection more robust.
 *
 * Revision 1.1  1995/04/11  20:28:16  sung
 * Initial revision
 *
**/

#include <stdio.h>
#include <math.h>
#include "sarx.h"
#include "read_adts.h"
#include "util_io.h"

#define max(a,b) ((a)>(b)?(a):(b))

#define LIMIT	15000	/* Max number of words before quitting search
				for Barker Code */

	
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
   int		ncsamples)
{
	unsigned short	aux_mask[16], header;
	short int	odd, even;
	int		word, rval, k, id, aflag=0,
			count,		/* Count of decoded Aux words(NAUX+1) */
			aux_save=0,	/* Disables any aux read */
			aux_enable=1,	/* Disables aux read for non-HH pols */
			flag1=0,	/* Set when desired pol ID is found */
			flag2=0;	/* Set after desired pol is read */
/**
Set up initial arrays and data sizes.
**/
	aux_mask[0] = 32768;
	for(k=1; k<16; k++) aux_mask[k] = aux_mask[k-1] >> 1;
/**
Enter main reading loop.  Flag2 is set when the end of the desired
polarization has been read.
**/
	while (flag2==0)
	{
/**
Perform barker code detection.
**/
	    if ((rval=cmp_barker(fp)) < 0)
	    { if (rval==-1) fprintf(stderr,"Unable to find barker sequence.\n");
	      if (rval==-2) fprintf(stderr, "Unable to read input file.\n");
	      return(-1);
	    }
/**
Barker code was found.  Set up the variables for the header word
(polarization ID) and the auxiliary information.  The variable 'count'
keeps track of the current auxiliary word.
**/
	    count = 0;
	    header = 0x0000;
	    aux_enable = 1;
	    if (!aux_save) for(k=0; k<NAUX; k++) aux[k] = 0x0000;
	    for(k=0; k<ncsamples; k++)
	    {   if (4 != fread_bo(&word, 1, 4, fp, 4))
	        {   fprintf(stderr, "Error reading sample %d\n",k);
	            return(-1);
	        }
/**
The 32-bit word is split into 2 16-bit words.  More specifically, bits
20-30 map into the odd word, while bits 4-14 map into the even word.
**/
	        odd = (short int) ((word >> 20) & 0x000003ff);
		if (((word >> 30) & 0x00000001) > 0) odd |= 0xfc00;
	        even = (short int) ((word >> 4) & 0x000003ff);
		if (((word >> 14) & 0x00000001) > 0) even |= 0xfc00;
		data[2*k] = even;
	        data[2*k+1] = odd;
/**
Get header word and auxiliary data if applicable.  An aux word is built
serially from the LSBs of the data words as 16-bit integers.  When a
word is completed, the first aux word tells from which polarization the
current data originates.  The remaining words gives miscellaneous
information about the data.  Currently, we are only interested in range,
and motion sensing parameters inorder to determine which side of the
plane we are looking (squint angle).
**/
	        if (((count < NAUX+1) && aux_enable) || !count)
	        {   if ((word & 0x00000008) > 0) 
		    {   if (!count) header = header | aux_mask[k%16];
			else	    aux[count-1] =aux[count-1] | aux_mask[k%16];
		    }
		    if ((k+1)%16 == 0)
		    {
/**
Check polarization ID.
**/
		        if (count==0)		/* Header word */
		        {   if ((header | 0x00f0) == 0x03f0)	  /* HH */
			    {	if (pol==HH) flag1 = 1;
			    }
			    else if ((header | 0x00f0) == 0x43f5) /* HV */
			    {	if (pol==HV) flag1 = 1;
			    	aux_enable = 0;
			    }
			    else if ((header | 0x00f0) == 0x83fa) /* VH */
			    {	if (pol==VH) flag1 = 1;
			    	aux_enable = 0;
			    }
			    else if ((header | 0x00f0) == 0xc3ff) /* VV */
			    {	if (pol==VV) flag1 = 1;
			        aux_enable = 0;
			    }
			    else
			    {   fprintf(stderr,
					"Cannot determine pole, id=%04x\n", (unsigned)header);
				fprintf(stderr,"Rechecking...\n");
				id = idcheck(header);
				if (id<0)
				{   fprintf(stderr,
					"Fatal Error in polarization id.\n");
				    return(-1);
				}
				else if (id==HH)
				{   if (pol==HH) flag1 = 1;
				    printf("Choosing HH.\n");
				    aflag = 1;
				}
				else if (id==HV)
				{   if (pol==HV) flag1 = 1;
			    	    aux_enable = 0;
				    printf("Choosing HV.\n");
				}
				else if (id==VH)
				{   if (pol==VH) flag1 = 1;
			    	    aux_enable = 0;
				    printf("Choosing VH.\n");
				}
				else if (id==VV)
				{   if (pol==VV) flag1 = 1;
			    	    aux_enable = 0;
				    printf("Choosing VV.\n");
				}
			    }
		        }
		        else if (count==29)	/* Range to aimpoint (m) */
		        {   *r2a = 2.0*aux[count-2] +
				(aux[count-1] & 0x7fff)/16384.;
			    if (aflag) printf("range=%g\n", *r2a);
			    aflag=0;
		        }
		        else if (count==NAUX)
		            aux_save = 1;
		        count++;
		    }
	        }
	    }
/**
If this was the desired polarization, return.
**/
	    if (flag1) flag2 = 1;
	}

	return(0);
}

/**
Check if the current buffer matches the barker code sequence.  Note:
Group 47 does not check the last 2 trailing 0s, thus this code does not
either.  Also, only bits 0 and 4-15 are checked for the sequence.  This
allows the code to be compatible with older forms of input data.
**/
int
cmp_barker(FILE* fp)
{
	int	word, ncount=0, wsize=sizeof(int), buf[2], state=0;
/**
Start main reading loop.
**/
	while (ncount<LIMIT)
	{
/**
Check if word0 = 0.
**/
	    if (state==0)
	    {	if (1 != fread_bo(&word, wsize, 1, fp, wsize)) return(-2);
		ncount++;
		if ((word & 0x0000fff8) != 0x00000000)
		{   state = 0;
		    continue;
		}
		else
		{   state = 1;
		    continue;
		}
	    }
/**
Check if word1 = 0.
**/
	    if (state==1)
	    {	if (1 != fread_bo(&word, wsize, 1, fp, wsize)) return(-2);
		ncount++;
		if ((word & 0x0000fff8) != 0x00000000)
		{   state = 0;
		    continue;
		}
		else
		{   state = 2;
		    continue;
		}
	    }
/**
Check if word2 = 0.
**/
	    if (state==2)
	    {	if (1 != fread_bo(&word, wsize, 1, fp, wsize)) return(-2);
		ncount++;
		if ((word & 0x0000fff8) != 0x00000000)
		{   state = 0;
		    continue;
		}
		else
		{   state = 3;
		    continue;
		}
	    }
/**
Check if word3 = 0.
**/
	    if (state==3)
	    {	if (1 != fread_bo(&word, wsize, 1, fp, wsize)) return(-2);
		ncount++;
		if ((word & 0x0000fff8) != 0x00000000)
		{   state = 0;
		    continue;
		}
		else
		{   state = 4;
		    continue;
		}
	    }
/**
Check if word4 = 0.
**/
	    if (state==4)
	    {	if (1 != fread_bo(&word, wsize, 1, fp, wsize)) return(-2);
		ncount++;
		if ((word & 0x0000fff8) != 0x00000000)
		{   state = 0;
		    continue;
		}
		else
		{   state = 5;
		    continue;
		}
	    }
/**
Check if word5 = 1.
**/
	    if (state==5)
	    {	if (1 != fread_bo(&word, wsize, 1, fp, wsize)) return(-2);
		ncount++;
		if ((word & 0x0000fff8) == 0x00000000)
		{   state = 5;
		    continue;
		}
		else if ((word | 0xffff0007) != 0xffffffff)
		{   state = 0;
		    continue;
		}
	    }
/**
Check if word6 = 1.
**/
	    if (1 != fread_bo(&word, wsize, 1, fp, wsize)) return(-2);
	    ncount++;
	    if ((word & 0x0000fff8) == 0x00000000)
	    {   state = 1;
		continue;
	    }
	    else if ((word | 0xffff0007) != 0xffffffff)
	    {	state = 0;
		continue;
	    }
/**
Check if word7 = 1.
**/
	    if (1 != fread_bo(&word, wsize, 1, fp, wsize)) return(-2);
	    ncount++;
	    if ((word & 0x0000fff8) == 0x00000000)
	    {   state = 1;
		continue;
	    }
	    else if ((word | 0xffff0007) != 0xffffffff)
	    {	state = 0;
		continue;
	    }
/**
Check if word8 = 0.
**/
	    if (1 != fread_bo(&word, wsize, 1, fp, wsize)) return(-2);
	    ncount++;
	    if ((word & 0x0000fff8) != 0x00000000)
	    {	state = 0;
		continue;
	    }
/**
Check if word9 = 1.
**/
	    if (1 != fread_bo(&word, wsize, 1, fp, wsize)) return(-2);
	    ncount++;
	    if ((word & 0x0000fff8) == 0x00000000)
	    {   state = 2;
		continue;
	    }
	    else if ((word | 0xffff0007) != 0xffffffff)
	    {	state = 0;
		continue;
	    }
/**
Check if word10 = 0.
**/
	    if (1 != fread_bo(&word, wsize, 1, fp, wsize)) return(-2);
	    ncount++;
	    if ((word & 0x0000fff8) != 0x00000000)
	    {	state = 0;
		continue;
	    }
/**
Check if word11 = 0.
**/
	    if (1 != fread_bo(&word, wsize, 1, fp, wsize)) return(-2);
	    ncount++;
	    if ((word & 0x0000fff8) != 0x00000000)
	    {	state = 0;
		continue;
	    }
/**
Check if word12 = 1.
**/
	    if (1 != fread_bo(&word, wsize, 1, fp, wsize)) return(-2);
	    ncount++;
	    if ((word & 0x0000fff8) == 0x00000000)
	    {   state = 3;
		continue;
	    }
	    else if ((word | 0xffff0007) != 0xffffffff)
	    {	state = 0;
		continue;
	    }
/**
Check if word13 = 0.
**/
	    if (1 != fread_bo(&word, wsize, 1, fp, wsize)) return(-2);
	    ncount++;
	    if ((word & 0x0000fff8) != 0x00000000)
	    {	state = 0;
		continue;
	    }
/**
Check if word14 = 0.
**/
	    if (1 != fread_bo(&word, wsize, 1, fp, wsize)) return(-2);
	    ncount++;
	    if ((word & 0x0000fff8) != 0x00000000)
	    {	state = 0;
		continue;
	    }
/**
Check if word15 = 0.
**/
	    if (1 != fread_bo(&word, wsize, 1, fp, wsize)) return(-2);
	    ncount++;
	    if ((word & 0x0000fff8) != 0x00000000)
	    {	state = 0;
		continue;
	    }
/**
Check if word16 = 1.
**/
	    if (1 != fread_bo(&word, wsize, 1, fp, wsize)) return(-2);
	    ncount++;
	    if ((word & 0x0000fff8) == 0x00000000)
	    {   state = 4;
		continue;
	    }
	    else if ((word | 0xffff0007) != 0xffffffff)
	    {	state = 0;
		continue;
	    }
/**
Check if word17 = 1.
**/
	    if (1 != fread_bo(&word, wsize, 1, fp, wsize)) return(-2);
	    ncount++;
	    if ((word & 0x0000fff8) == 0x00000000)
	    {   state = 1;
		continue;
	    }
	    else if ((word | 0xffff0007) != 0xffffffff)
	    {	state = 0;
		continue;
	    }
/**
Found Barker Code.  Read last two garbage words and return.
**/
	    if (1 != fread_bo(buf, 2*wsize, 1, fp, wsize)) return(-2);
	    return(0);
	}

	return(-1);
}

int
idcheck(unsigned short hdr)
{
	int	i, pol0, pol1, pol2, pol3, maxpol=-1;

	hdr = hdr|0x00f0;
	pol0 = pol1 = pol2 = pol3 = 0;
	for (i=0; i<16; i++)
	{   if ((((hdr>>i)^(0xfc0f>>i))&0x0001) != 0) pol0++;
	    if ((((hdr>>i)^(0xbc0a>>i))&0x0001) != 0) pol1++;
	    if ((((hdr>>i)^(0x7c05>>i))&0x0001) != 0) pol2++;
	    if ((((hdr>>i)^(0x3c00>>i))&0x0001) != 0) pol3++;
	}
	maxpol = max(pol0,pol1);
	maxpol = max(maxpol,pol2);
	maxpol = max(maxpol,pol3);

	if (maxpol==pol0) return(HH);
	else if (maxpol==pol1) return(HV);
	else if (maxpol==pol2) return(VH);
	else if (maxpol==pol3) return(VV);
	else return(-1);
}
