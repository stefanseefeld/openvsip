/*=====================================================================

		Copyright 1994 MIT Lincoln Laboratory

		C.T. Sung, MIT Lincoln Laboratory

	Include constants and functions for use with SAR processing.

=====================================================================*/

/**
 * Revision 1.4  1998/06/05  19:38:27  anderson
 * Another fix to two lines.
 *
 * Revision 1.3  1998/06/05  19:27:09  anderson
 * Put () around NRANGE and NPULSE variables
 *
 * Revision 1.2  1998/06/01  20:25:08  anderson
 * *** empty log message ***
 *
 * Revision 1.1  1998/01/12  22:20:43  anderson
 * Initial revision
 *
**/

#ifndef __sarx_h
#define __sarx_h

#if _MC_EXEC
#  include <exec/sys/stdlib.h>
#else
#  include <malloc.h>
#endif
#include <stdlib.h>
#include <math.h>

/**
Constants
**/
#define	BCODE_SIZE	20
#define NRANGE		(2048/8)		/* Must be 2048 for ADTS data 		*/
#define NPULSE		(512/8)		/* 512 was used in the Benchmark	*/
#define	NCSAMPLES	(NRANGE-16)
#define MAXRANGE	(2048/1)	/* Must be 2048 for ADTS data 	 */
#define MAXPULSE	(512/1)		/* 512 was used in the Benchmark */
#define	MAXCSAMPLES	(MAXRANGE-16)
#define	NAUX		57
#define	NREF		31
#define HH		0
#define HV		1
#define VH		2
#define VV		3


/**
Complex Structures
**/
struct fcomplex { float r; float i; };
struct dcomplex { double r; double i; };

typedef struct fcomplex Fcomplex;
typedef struct dcomplex Dcomplex;

#define cmult(a,b,c)	{ float x9_x,y9_y; x9_x = (b).r; \
			y9_y = (c).r; (a).r = x9_x*y9_y - (b).i*(c).i; \
			(a).i = x9_x*(c).i + (b).i*y9_y; }
#define cdmult(a,b,c)	{ double x9_x,y9_y; x9_x = (b).r; \
			y9_y = (c).r; (a).r = x9_x*y9_y - (b).i*(c).i; \
			(a).i = x9_x*(c).i + (b).i*y9_y; }
#define crmult(a,b,c)	{ (a).r = (b).r*(c); (a).i = (b).i*(c); }
#define cdrmult(a,b,c)	{ (a).r = (b).r*(c); (a).i = (b).i*(c); }

#define cmag(a)		((float) sqrt((double) ((a).r*(a).r + (a).i*(a).i)))
#define cdmag(a)	sqrt((a).r*(a).r + (a).i*(a).i)

#define cmagsq(a)	((a).r*(a).r + (a).i*(a).i)
#define cdmagsq(a)	((a).r*(a).r + (a).i*(a).i)


/**
Array allocation
**/
#define array1(m,s)		calloc((unsigned)m,s)
#define array1free(a)		free((char *) a)

char	**array2(int, int, unsigned);

#define array1f(m)		(float *) calloc((unsigned)m,sizeof(float))
#define array2f(m,n)		(float **) array2(m,n,sizeof(float))

#define array1d(m)		(double *) calloc((unsigned)m,sizeof(double))
#define array2d(m,n)		(double **) array2(m,n,sizeof(double))

#define array1fc(m)	       (Fcomplex *) calloc((unsigned)m,sizeof(Fcomplex))
#define array2fc(m,n)		(Fcomplex **) array2(m,n,sizeof(Fcomplex))

#define array1dc(m)	       (Dcomplex *) calloc((unsigned)m,sizeof(Dcomplex))
#define array2dc(m,n)		(Dcomplex **) array2(m,n,sizeof(Dcomplex))

#define array1si(m)		(short *) calloc((unsigned)m,sizeof(short))
#define array2si(m,n)		(short **) array2(m,n,sizeof(short))

#define array1i(m)		(int *) calloc((unsigned)m,sizeof(int))
#define array2i(m,n)		(int **) array2(m,n,sizeof(int))


/** Signal Processing **/

void
cdft(
   Fcomplex	*data,
   int		nn,
   int		isign);

void
cdfti(
   Fcomplex	*data,
   int		nn,
   int		isign);

void
cdftd(
   Dcomplex	*data,
   int		nn,
   int		isign);

void
cdftdi(
   Dcomplex	*data,
   int		nn,
   int		isign);

// frm_hdr.c
int
read_frm_hdr(
   FILE		*fp,
   short int	*aux,
   int		*pol,
   float	*sq_ang);

int
write_frm_hdr(
   FILE		*fp,
   short int	*aux,
   int		pol);

#endif		/* !__sarx_h */
