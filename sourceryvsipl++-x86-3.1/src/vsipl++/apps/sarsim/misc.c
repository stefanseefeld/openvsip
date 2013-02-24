/*=====================================================================

		Copyright 1994 MIT Lincoln Laboratory

		C.T. Sung, MIT Lincoln Laboratory

	Miscellaneous functions for SAR processing.

=====================================================================*/

/**
 * Revision 1.2  1998/06/01  20:22:28  anderson
 * *** empty log message ***
 *
 * Revision 1.1  1998/01/12  22:18:36  anderson
 * Initial revision
 *
 * Revision 1.1  1995/04/11  20:28:10  sung
 * Initial revision
 *
**/

#include <malloc.h>

char **
array2(
   int		m,
   int		n,
   unsigned	s)
{
	int	i;
	char	**a, **f, *b;

	a = (char **) calloc((unsigned) (m+2), sizeof(char *));
	if (a == (char **)0)
		return(a);

	b = (char *) calloc((unsigned) m*n, s);
	if (b == (char *)0)
	{	free((char *) a);
		return((char **)0);
	}

	a[0] = b;
	f = a + 1;

	for (i=0; i<m; i++)
		f[i] = b + i*n*s;

	return(f);
}
