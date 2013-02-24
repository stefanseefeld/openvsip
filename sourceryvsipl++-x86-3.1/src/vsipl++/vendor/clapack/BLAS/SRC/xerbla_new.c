#include <stdio.h>

int xerbla_(char *srname, int *info)
{
    static char fmt_9999[] =
	    "\n(** On entry to %s parameter number %d had illegal value **)\n";

    printf(fmt_9999,srname,*info);
    exit(0);
}
