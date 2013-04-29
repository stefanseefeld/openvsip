/*=====================================================================

		Copyright 1994 MIT Lincoln Laboratory

		C. T. Sung, M.I.T. Lincoln Laboratory

	Program to compare two complex images stored in the SAR output
	format and give histogram values of log error threshold crossings
	vs. threshold in dB.

=====================================================================*/


/*
 * Revision 1.2  1998/06/01  20:22:16  anderson
 * *** empty log message ***
 *
 * Revision 1.2  1996/07/01  14:44:43  anderson
 * Replaced include file sar.h with sarx.h
 *
 * Revision 1.1  1994/09/01  15:50:37  anderson
 * Initial revision
 *	*/

#include <stdio.h>
#include <string.h>

#include "sarx.h"
#include "util_io.h"

#define min(a,b) ((a)<(b)?(a):(b))

int
main(int argc, char** argv)
{
	FILE		*fpin=NULL, *fpref=NULL;
	int		i, j, k, frame_size, index, pol;
	int		frm_cmp1=1, frm_cmp2=1;
	float		val, sum, sq_ang;
	int		hist[201], x1[201];
	short int	aux[NAUX];
	char		fnin[128], fnref[128];
	float		check_threshold;
	int		do_check = 0;
	float		maxsum = -250;

	int		nrange=MAXRANGE;
	int		npulse=MAXPULSE;

	Fcomplex	*inbuf, *refbuf;	/* Main data array.	*/
	double 		refmax = 1.4736e18	/* Reference Peak	*/;
/**
Get input parameters from the command line.
**/
	for (i=1; i<argc; i++)
	{   if (!strcmp(argv[i], "-i"))	/* Input File */
	    {	if (++i < argc)
		{  if (!(fpin=fopen(argv[i],"r")))
		   {	fprintf(stderr,"Can't find input file %s\n", argv[i]);
			exit(1);
		   }
		   else strcpy(fnin, argv[i]);
		}
		else fprintf(stderr, "Invalid value for option %s\n",argv[i-1]);
	    }
	    else if (!strcmp(argv[i], "-ref"))	/* Reference File */
	    {	if (++i < argc)
		{  if (!(fpref=fopen(argv[i],"r")))
		   {   fprintf(stderr,"Can't find reference file %s\n",argv[i]);
		       exit(1);
		   }
		   else strcpy(fnref, argv[i]);
		}
		else fprintf(stderr, "Invalid value for option %s\n",argv[i-1]);
	    }
	    else if (!strcmp(argv[i], "-nrange"))     /* Num. of range cells */
	    {	if (1 == sscanf(argv[++i], "%d", &j))
		    nrange = j;
		else
		    fprintf(stderr, "Invalid value for option %s\n",argv[i-1]);
	    }
	    else if (!strcmp(argv[i], "-npulse"))	/* Num. of pulses */
	    {	if (1 == sscanf(argv[++i], "%d", &j))
		    npulse = j;
		else
		    fprintf(stderr, "Invalid value for option %s\n",argv[i-1]);
	    }
	    else if (!strcmp(argv[i], "-iframe"))	/* Frame of Input */
	    {	if (1 == sscanf(argv[++i], "%d", &j))
		    frm_cmp1 = j;
		else
		    fprintf(stderr, "Invalid value for option %s\n",argv[i-1]);
	    }
	    else if (!strcmp(argv[i], "-rframe"))	/* Frame of Reference */
	    {	if (1 == sscanf(argv[++i], "%d", &j))
		    frm_cmp2 = j;
		else
		    fprintf(stderr, "Invalid value for option %s\n",argv[i-1]);
	    }
	    else if (!strcmp(argv[i], "-chk"))	/* check value */
	    {	if (1 == sscanf(argv[++i], "%f", &check_threshold))
		    do_check = 1;
		else
		    fprintf(stderr, "Invalid value for option %s\n",argv[i-1]);
	    }
	}
	if (!fpin)
	{   fprintf(stderr, "Error:  no input file specified.\n");
	    fprintf(stderr, "Usage: %s -i [input file] -ref [reference file] -iframe [input frames to cmp] -rframe [ref frame to cmp]\n", argv[0]);
	    exit(1);
	}
	if (!fpref)
	{   fprintf(stderr, "Error:  no reference file specified.\n");
	    fprintf(stderr, "Usage: %s -i [input file] -ref [reference file] -iframe [input frames to cmp] -rframe [ref frame to cmp]\n", argv[0]);
	    exit(1);
	}
/**
Compute size of one input frame and allocate memory for the two buffers.
**/
	frame_size = npulse*nrange*sizeof(Fcomplex)
			+ (BCODE_SIZE+NAUX+1)*sizeof(int);
	inbuf = (Fcomplex *) array1fc(1); 	/* File 1 Buffer */
	refbuf = (Fcomplex *) array1fc(1); 	/* File 2 Buffer */
	if (inbuf==NULL || refbuf==NULL)
	{   fprintf(stderr, "Memory allocation error.\n");
	    exit(1);
	}
/**
Check the reference maximum peak.
**/
	if (refmax <= 0.)
	{   fprintf(stderr,"Invalid reference peak.\n");
	    exit(1);
	}
/**
Skip frames if needed.
**/
	if (frm_cmp1 > 1)
	{   printf("Skipping %d frames in input file.\n", frm_cmp1-1);
	    if (fseek(fpin, (frm_cmp1-1)*frame_size, 0) != 0)
	    {   fprintf(stderr, "Error reading input file 1.\n");
		exit(1);
	    }
	}
	if (frm_cmp2 > 1)
	{   printf("Skipping %d frames in reference file.\n", frm_cmp2-1);
	    if (fseek(fpref, (frm_cmp2-1)*frame_size, 0) != 0)
	    {   fprintf(stderr, "Error reading input file 2.\n");
	        exit(1);
	    }
	}

	if (!do_check)
	    printf("Comparing frame %d of %s to frame %d of %s.\n",
		   frm_cmp1, fnin, frm_cmp2, fnref);
/**
Initialize the output histogram and read the input frame headers.
**/
	for(i=0; i<201; i++) hist[i] = 0.;

	if (read_frm_hdr(fpin, aux, &pol, &sq_ang) < 0)
	{   fprintf(stderr,"Error reading input frame header.\n");
	    exit(1);
	}
	if (read_frm_hdr(fpref, aux, &pol, &sq_ang) < 0)
	{   fprintf(stderr,"Error reading ref frame header.\n");
	    exit(1);
	}
/**
Read the data and do a pixel by pixel error comparison.
**/
	for (i=0; i<nrange; i++)
	{   for (j=0; j<npulse; j++)
	    {   if (8 != fread_bo(inbuf, 1, 8, fpin, sizeof(float)))
	        {   fprintf(stderr, "Error reading input file 1.\n");
	    	    exit(1);
	        }
	        if (8 != fread_bo(refbuf, 1, 8, fpref, sizeof(float)))
	        {   fprintf(stderr, "Error reading reference file.\n");
	    	    exit(1);
	        }
/**
Compute the magnitude squared of the pixel difference.
**/
		inbuf[0].r -= refbuf[0].r;
		inbuf[0].i -= refbuf[0].i;
		val = cmagsq(inbuf[0]);
/**
Put this power into dB.
**/
		if (val < 1.e-20) sum = -201.;
		else		sum = 10.*log10(val/(2.*refmax));

		if (sum > maxsum)
		   maxsum = sum;
/**
Fill the histogram for all thresholds between -200 dB and 0 dB that are
less than this pixel error.  These are the thresholds that have been crossed.
**/
		if (sum >= -200.)
		{   index = min(0,floor(sum));
		    for(k=-200; k<=index; k++) hist[k+200] += 1;
		}
	    }
	}
	fclose(fpin);
	fclose(fpref);

	for (i=0; i<201; i++) x1[i] = -200 + i;
/**
The results are simply printed to stdout for now. You can plot this
curve in whatever manner you are used to.
**/
	if (do_check)
	{   
	   if (maxsum < check_threshold) 
	   {
	      printf("%s/fr %d to %s/fr %d: dB: %7.2f (thresh %4.0f)\n",
		     fnin, frm_cmp1, fnref, frm_cmp2,
		     maxsum, check_threshold);
	      return 0;
	      }
	   else
	   {
	      printf("ERROR %s/fr %d to %s/fr %d: dB: %7.2f (thresh %4.0f) - ERROR\n",
		     fnin, frm_cmp1, fnref, frm_cmp2,
		     maxsum, check_threshold);
	      return -1;
	   }
	}
	else
	{
	    printf("\nNumber of Threshold Crossings vs. Threshold (dB)\n");
	    printf("------------------------------------------------\n");

	    for (i=0; i<201; i++) printf("%d	%d\n", x1[i], hist[i]);

	    // array1free(x1);
	}
	exit(0);
}

