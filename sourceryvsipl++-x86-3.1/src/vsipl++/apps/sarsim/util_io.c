/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    util_io.c
    @author  Jules Bergmann
    @date    02/22/2005
    @brief   IO Utilities for SarSim, fread and fwrite with byte-ordering.
*/


/***********************************************************************
  Included Files
***********************************************************************/

#include <stdio.h>
#include <assert.h>
#ifdef _MC_EXEC
#  include <sys/socket.h>
#else
#  include <netinet/in.h>
#endif

#include "util_io.h"

#define BUF_SIZE 1024*1024

char bs_buf[BUF_SIZE];

size_t fwrite_bo1(
   const void	*ptr,
   size_t	size,
   size_t	nmemb,
   FILE		*stream)
{
   size_t i;

   assert(size*nmemb < BUF_SIZE);

   if (size == 1) {
      return fwrite(ptr, size, nmemb, stream);
      }
   else if (size == 2) {
      uint16_t *sptr = (uint16_t*)ptr;

      // reorder data (in-place).
      for (i=0; i<nmemb; ++i)
	 sptr[i] = htons(sptr[i]);
	 
      size_t rv = fwrite(sptr, size, nmemb, stream);

      // put data back into orignal order.
      for (i=0; i<nmemb; ++i)
	 sptr[i] = ntohs(sptr[i]);

      return rv;
      }
   else if (size == 4) {
      uint32_t *sptr = (uint32_t*)ptr;

      // reorder data (in-place)
      for (i=0; i<nmemb; ++i)
	 sptr[i] = htonl(sptr[i]);

      size_t rv = fwrite(sptr, size, nmemb, stream);

      // reorder data (in-place)
      for (i=0; i<nmemb; ++i)
	 sptr[i] = htonl(sptr[i]);

      return rv;
      }
   else {
      assert(0);
      }
   return 0;
}



size_t fwrite_bo(
   const void	*ptr,
   size_t	size,
   size_t	nmemb,
   FILE		*stream,
   int		align)
{
   size_t i;

   size_t n_align = size*nmemb/align;

   assert(n_align*align == size*nmemb);

   if (align == 1) {
      return fwrite(ptr, size, nmemb, stream);
      }
   else if (align == 2) {
      uint16_t *sptr = (uint16_t*)ptr;

      // reorder data (in-place)
      for (i=0; i<n_align; ++i)
	 sptr[i] = htons(sptr[i]);

      size_t rv = fwrite(sptr, size, nmemb, stream);

      // put data back into orignal order.
      for (i=0; i<n_align; ++i)
	 sptr[i] = ntohs(sptr[i]);

      return rv;
      }
   else if (align == 4) {
      uint32_t *sptr = (uint32_t*)ptr;

      // reorder data (in-place)
      for (i=0; i<n_align; ++i)
	 sptr[i] = htonl(sptr[i]);

      size_t rv = fwrite(sptr, size, nmemb, stream);

      // put data back into orignal order.
      for (i=0; i<n_align; ++i)
	 sptr[i] = ntohl(sptr[i]);

      return rv;
      }
   else {
      assert(0);
      }
   return 0;
}



size_t fread_bo1(
   void		*ptr,
   size_t	size,
   size_t	nmemb,
   FILE		*stream)
{
   size_t i;

   if (size == 1) {
      return fread(ptr, size, nmemb, stream);
      }
   else if (size == 2) {
      uint16_t *sptr = (uint16_t*)ptr;

      size_t rv = fread(sptr, size, nmemb, stream);

      for (i=0; i<nmemb; ++i)
	 sptr[i] = ntohs(sptr[i]);
	 
      return rv;
      }
   else if (size == 4) {
      uint32_t *sptr = (uint32_t*)ptr;

      size_t rv = fread(sptr, size, nmemb, stream);

      for (i=0; i<nmemb; ++i)
	 sptr[i] = ntohl(sptr[i]);
	 
      return rv;
      }
   else assert(0);
   return 0;
}



size_t fread_bo(
   void		*ptr,
   size_t	size,
   size_t	nmemb,
   FILE		*stream,
   int		align)
{
   size_t i;

   size_t n_align = size*nmemb/align;
   assert(n_align*align == size*nmemb);

   if (align == 1) {
      return fread(ptr, size, nmemb, stream);
      }
   else if (align == 2) {
      uint16_t *sptr = (uint16_t*)ptr;

      size_t rv = fread(sptr, size, nmemb, stream);

      for (i=0; i<n_align; ++i)
	 sptr[i] = ntohs(sptr[i]);
	 
      return rv;
      }
   else if (align == 4) {
      uint32_t *sptr = (uint32_t*)ptr;

      size_t rv = fread(sptr, size, nmemb, stream);

      for (i=0; i<n_align; ++i)
	 sptr[i] = ntohl(sptr[i]);
	 
      return rv;
      }
   else assert(0);
   return 0;
}
