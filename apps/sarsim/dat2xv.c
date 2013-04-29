/*=====================================================================

		Copyright 1998 MIT Lincoln Laboratory


		A.H. Anderson, MIT Lincoln Laboratory, 1998

	Program to read 32-bit processed SAR data and convert to
	a format which can be displayed by the program xv.
	This program is derived from dat2view and could be easily
	modified for other display formats.
=====================================================================*/

/**
 * Revision 1.2  1998/07/24  18:05:37  anderson
 * Replaced use of nint in two instructions.
 *
 * Revision 1.1  1998/06/04  17:26:59  anderson
 * Initial revision
 *
**/

#include <stdio.h>
#include <string.h>
#include <malloc.h>
#include "sarx.h"
#include "util_io.h"

int		cmap_size = 256;

char		fnin[128], fnout[128];

unsigned char	**ibuf, *iptr;

/**
Input value defaults.
**/
int		nframe =	4,	/** Number of frames to display **/
		nskip =		0,	/** Number of init frames to skip **/
		sfact =		4,	/** Spoiling factor for display **/
		img_width =	512,	/** Number of image cols to create **/
		img_height =	512;	/** Number of image rows to create **/
float		mindb = 	60.,	/** Minimum pixel values in dB **/
		range =		100.;	/** Range of all pixel values in dB **/


int
main(argc,argv)
int	argc;
char	**argv;
{
	FILE		*fpin=NULL, *fpout=stdout;
	size_t		psize;
	int		swidth, sheight;
	int		i, j, k, l, m, i4, pol, frame=0, nstrip, tstrip, fsize, itmp1, itmp2;
	float		*sum, mag, sq_ang, xf, yf;
	short int	aux[NAUX];		/* Aux info array	*/
	Fcomplex	cbuf[MAXPULSE];		/* SAR frame holder	*/

	int		nrange = MAXRANGE;
	int		npulse = MAXPULSE;
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
	   else if (!strcmp(argv[i], "-o"))	/* Output File */
	    {	if (++i < argc)
		{  if (!(fpout=fopen(argv[i],"w")))
		   {	fprintf(stderr,"Can't open output file %s\n", argv[i]);
			exit(1);
		   }
		   else strcpy(fnout, argv[i]);
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
	    else if (!strcmp(argv[i], "-nframe"))	/* Number of frames to convert */
	    {	if (1 == sscanf(argv[++i], "%d", &j))
		    nframe = j;
		else
		    fprintf(stderr, "Invalid value for option %s\n",argv[i-1]);
	    }
	    else if (!strcmp(argv[i], "-nskip"))	/* Number of frames to skip */
	    {	if (1 == sscanf(argv[++i], "%d", &j))
		    nskip = j;
		else
		    fprintf(stderr, "Invalid value for option %s\n",argv[i-1]);
	    }
	    else if (!strcmp(argv[i], "-mindb"))	/* Minimum dB value */
	    {	if (1 == sscanf(argv[++i], "%g", &xf))
		    mindb = xf;
		else
		    fprintf(stderr, "Invalid value for option %s\n",argv[i-1]);
	    }
	    else if (!strcmp(argv[i], "-range"))	/* dB range */
	    {	if (1 == sscanf(argv[++i], "%g", &xf))
		    range = xf;
		else
		    fprintf(stderr, "Invalid value for option %s\n",argv[i-1]);
	    }
	    else if (!strcmp(argv[i], "-sfact"))	/* Spoiling Factor */
	    {	if (1 == sscanf(argv[++i], "%d", &j))
		    sfact = j;
		else
		    fprintf(stderr, "Invalid value for option %s\n",argv[i-1]);
	    }
	}
	if (!fpin)
	{   fprintf(stderr, "Error:  no input file specified.\n");
	    fprintf(stderr, "Usage: %s -i [input file] (no default)\n\
		-nframe [# of shown frames] (4)\n\
		-nskip [# of skipped frames] (0)\n\
		-sfact [Spoiling factor for display (1=unspoiled)] (4)\n\
		-mindb [min. pixel value in dB] (60)\n\
		-range [range of pixels in dB] (100)\n",argv[0]);
	    exit(1);
	}
	  img_width = npulse*nframe/sfact;
	  img_height = nrange/sfact;
	
/**
Compute the width and height of an image strip created from one frame of SAR output data, the total number of strips in the requested data, and the size of one output pulse in bytes.
**/
	sheight = nrange/sfact;
	swidth = npulse/sfact;
	psize = npulse*(sizeof(Fcomplex));
	tstrip = (int)(floor(img_width*1./(npulse*1./sfact)));

/**
Check the validity of input parameters.
**/
	if (sfact%2 != 0 && sfact!=1)
	{   fprintf(stderr, "Error: spoiling factor must be even or 1.\n");
	    exit(1);
	}
/**
Print input parameters:
**/
	fprintf(stderr, "===========================================================\n");
	fprintf(stderr, "Converting frames (%d-%d) of SAR data file: %s\n",
		nskip+1, nskip+nframe, fnin);
	fprintf(stderr, "Image Dim = (%d, %d)\n", img_width, img_height);
	fprintf(stderr, "Minimum Pixel (db) = %g, Maximum Pixel (db) = %g\n",
		mindb, mindb+range);
	fprintf(stderr, "Spoiling factor = %d\n", sfact);
	fprintf(stderr, "===========================================================\n");
/**

Allocate memory.  Note: this type of allocation routine initializes all
arrays to 0.  
**/
	sum = (float *) array1f(swidth);	     /* Output spoiled sums */
	ibuf = (unsigned char **) array2(img_height,img_width,sizeof(char));
	if (sum==NULL || ibuf==NULL)
	{   fprintf(stderr, "Error: memory allocation error.\n");
	    exit(1);
	}
/**
Skip frames if needed.
**/
	if (nskip > 0)
	{   fprintf(stderr, "Skipping %d frames.\n", nskip);
	    fsize = npulse*nrange*sizeof(Fcomplex)
				+ (BCODE_SIZE+NAUX+1)*sizeof(int);
	    if (fseek(fpin, nskip*fsize, 0) != 0)
	    {   fprintf(stderr, "Error: unable to skip frames.\n");
	        exit(1);
	    }
	    frame += nskip;
	}
/**
Start main loop.
**/
nstrip = 0;
while (++frame <= nframe+nskip)
{	fprintf(stderr, "Reading frame %d.\n", frame);
/**
Read frame header.
**/
	if (read_frm_hdr(fpin, aux, &pol, &sq_ang) < 0)
	{   fprintf(stderr,"Error: unable to read frame header.\n");
	    exit(1);
	}
/**
Read a frame of data, introduce the spoiling factor, and map to an image.
**/
	for (i=0; i<nrange; i+=sfact)
	{   for(j=0; j<swidth; j++) sum[j] = 0.;
	    for (j=0; j<sfact; j++)
	    {	if (psize != fread_bo(cbuf, 1, psize, fpin, sizeof(float)))
		{   fprintf(stderr, "Error reading input file.\n");
		    exit(1);
		}
		for (k=0, l=0; k<npulse; k+=sfact, l++)
		    for (m=0; m<sfact; m++)
			sum[l] += cmag(cbuf[k+m]);
	    }
/**
Take spoiled sums and map to pixel values between 0-CMAP_SIZE-1.
Currently, the dB range and minimum dB are determined by trial and
error and entered at run-time.  (This should change.)  Map the data
according to the side of the plane from which we are looking.
**/
	    xf = 1./ (float)(sfact*sfact);
	    yf = (cmap_size-1.)/range;
/**
If images are moving from left to right...
**/
	    if (sq_ang > 0.) 
	    {	iptr = &ibuf[i/sfact][nstrip*swidth];
	        for (j=0; j<swidth; j++)
	        {   if (sum[j] > 0)
 		    {   mag = 20.0*log10(sum[j]*xf);
			i4 = (mag - mindb)*yf;
			if (i4 < 0) i4 = 0;
			else if (i4 > cmap_size-1) i4 = cmap_size-1;
		    }
		    else i4=0;
		    *(iptr++) = i4;
		}
	    }
/**
If images are moving from right to left...
**/
	    else
	    {	iptr = &ibuf[i/sfact][(tstrip-nstrip)*swidth-1];
	        for (j=0; j<swidth; j++)
	        {   if (sum[j] > 0)
 		    {   mag = 20.0*log10(sum[j]*xf);
			i4 = (mag - mindb)*yf;
			if (i4 < 0) i4 = 0;
			else if (i4 > cmap_size-1) i4 = cmap_size-1;
		    }
		    else i4=0;
		    *(iptr--) = i4;
		}
	    }
	}
nstrip++;
   }
fclose(fpin);
/**
Write header information and data for a file which can be displayed by xv.
Changes for any other format would be make here.
**/

fprintf(fpout,"%c%c%c%c%c%c%c%c%c%c", 86, 73, 69, 87, 0, 0, 0, 1, 0, 0 );

itmp1 = img_height / 256;	
itmp2 = img_height - itmp1 * 256;
fprintf(fpout,"%c%c%c%c", itmp1, itmp2, 0, 0 );

itmp1 = img_width / 256;
itmp2 = img_width - itmp1 * 256;
fprintf(fpout,"%c%c", itmp1, itmp2 );

fprintf(fpout,"%c%c%c%c%c%c%c%c%c%c%c%c", 0, 0, 0, 1,   0, 0, 128, 1,   0, 0, 0, 23 );

for (i=0; i<img_height; i++)
	{for (j=0; j<img_width; j++)
		{
		 fprintf(fpout,"%c",ibuf[i][j]);
		}
	}
fclose(fpout);
fprintf(stderr, "All requested data has been converted.\n");

return 0;
}
