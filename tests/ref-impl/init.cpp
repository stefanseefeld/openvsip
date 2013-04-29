/***********************************************************************

  File:   init.cpp
  Author: Jeffrey D. Oldham, CodeSourcery, LLC.
  Date:   09/02/2003

  Contents: Test initialization of a VSIPL++ program.

Copyright 2005 Georgia Tech Research Corporation, all rights reserved.

A non-exclusive, non-royalty bearing license is hereby granted to all
Persons to copy, distribute and produce derivative works for any
purpose, provided that this copyright notice and following disclaimer
appear on All copies: THIS LICENSE INCLUDES NO WARRANTIES, EXPRESSED
OR IMPLIED, WHETHER ORAL OR WRITTEN, WITH RESPECT TO THE SOFTWARE OR
OTHER MATERIAL INCLUDING, BUT NOT LIMITED TO, ANY IMPLIED WARRANTIES
OF MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE, OR ARISING
FROM A COURSE OF PERFORMANCE OR DEALING, OR FROM USAGE OR TRADE, OR OF
NON-INFRINGEMENT OF ANY PATENTS OF THIRD PARTIES. THE INFORMATION IN
THIS DOCUMENT SHOULD NOT BE CONSTRUED AS A COMMITMENT OF DEVELOPMENT
BY ANY OF THE ABOVE PARTIES.

The US Government has a license under these copyrights, and this
Material may be reproduced by or for the US Government.
***********************************************************************/

/***********************************************************************
  Notes
***********************************************************************/

/*
  This simple VSIPL++ program initializes the VSIPL++ library for use
  ignoring its command-line arguments and terminates.
 */

/***********************************************************************
  Included Files
***********************************************************************/

#include <stdlib.h>
#include <vsip/initfin.hpp>

/***********************************************************************
  Macros
***********************************************************************/

/***********************************************************************
  Forward Declarations
***********************************************************************/

/***********************************************************************
  Type Declarations
***********************************************************************/

/***********************************************************************
  Class Declarations
***********************************************************************/

/***********************************************************************
  Variable Declarations
***********************************************************************/

/***********************************************************************
  Function Declarations
***********************************************************************/

/***********************************************************************
  Variable Definitions
***********************************************************************/

/***********************************************************************
  Inline Function Definitions
***********************************************************************/

/***********************************************************************
  Function Definitions
***********************************************************************/

int
main (int   argc,
      char* argv[])
{
  // Initialize the VSIPL++ library for use.
  // The destructor automatically cleans up the VSIPL++ library.
  vsip::vsipl v (argc, argv);

  return EXIT_SUCCESS;
}
