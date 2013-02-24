/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    util_io.h
    @author  Jules Bergmann
    @date    02/22/2005
    @brief   IO Utilities for SarSim, fread and fwrite with byte-ordering.
*/

#ifndef UTIL_IO_H
#define UTIL_IO_H

size_t fwrite_bo1(
   const void	*ptr,
   size_t	size,
   size_t	nmemb,
   FILE	 	*stream);

size_t fwrite_bo(
   const void	*ptr,
   size_t	size,
   size_t	nmemb,
   FILE		*stream,
   int		align);

size_t fread_bo1(
   void		*ptr,
   size_t	size,
   size_t	nmemb,
   FILE		*stream);

size_t fread_bo(
   void		*ptr,
   size_t	size,
   size_t	nmemb,
   FILE		*stream,
   int		align);

#endif // UTIL_IO_H
