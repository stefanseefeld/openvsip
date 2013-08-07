/* Copyright (c) 2006, 2011 CodeSourcery, Inc.  All rights reserved. */

/* Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

       * Redistributions of source code must retain the above copyright
         notice, this list of conditions and the following disclaimer.
       * Redistributions in binary form must reproduce the above
         copyright notice, this list of conditions and the following
	 disclaimer in the documentation and/or other materials
	 provided with the distribution.
       * Neither the name of CodeSourcery nor the names of its
         contributors may be used to endorse or promote products
	 derived from this software without specific prior written
	 permission.

   THIS SOFTWARE IS PROVIDED BY CODESOURCERY, INC. "AS IS" AND ANY
   EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
   PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL CODESOURCERY BE LIABLE FOR
   ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
   BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
   OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
   EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  */

/** @file    diffview.cpp
    @author  Don McCoy
    @date    2006-10-29
    @brief   Utility to compare VSIPL++ views to determine equality
*/

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/math.hpp>

#include <vsip_csl/load_view.hpp>
#include <vsip_csl/save_view.hpp>
#include <vsip_csl/error_db.hpp>


using namespace vsip;
using namespace vsip_csl;
using namespace std;


enum data_format_type
{
  COMPLEX_VIEW = 0,
  REAL_VIEW,
  INTEGER_VIEW
};

#if VSIP_BIG_ENDIAN
bool swap_bytes = true;   // Whether or not to swap bytes during file I/O
#else
bool swap_bytes = false;
#endif

template <typename T>
void 
compare(char const* infile, char const* ref, length_type rows, 
  length_type cols);

int
main(int argc, char** argv)
{
  vsip::vsipl init(argc, argv);

  if (argc < 5 || argc > 6)
  {
    cerr << "Usage: " << argv[0] 
         << " [-crn] <input> <reference> <rows> <cols>" << endl;
    return -1;
  }
  else
  {
    data_format_type format = COMPLEX_VIEW;
    if (argc == 6)
    {
      if (0 == strncmp("-c", argv[1], 2))
        format = COMPLEX_VIEW;
      else if (0 == strncmp("-r", argv[1], 2))
        format = REAL_VIEW;
      else if (0 == strncmp("-n", argv[1], 2))
        format = INTEGER_VIEW;
      else
      {
        cerr << "Usage: " << argv[0] 
             << " [-crn] <input> <reference> <rows> <cols>" << endl;
        return -1;
      }
      ++argv;
      --argc;
    }

    if (format == REAL_VIEW)
      compare<float>(argv[1], argv[2], atoi(argv[3]), atoi(argv[4]));
    else if (format == INTEGER_VIEW)
      compare<int>(argv[1], argv[2], atoi(argv[3]), atoi(argv[4]));
    else
      compare<complex<float> >(argv[1], argv[2], atoi(argv[3]), atoi(argv[4])); 

  }

  return 0;
}


template <typename T>
void
compare(char const* infile, char const* ref, length_type rows, 
  length_type cols)
{
  typedef Matrix<T> matrix_type;
  Domain<2> dom(rows, cols);

  matrix_type in(rows, cols);
  in = Load_view<2, T>(infile, dom, vsip::Local_map(), swap_bytes).view();

  matrix_type refv(rows, cols);
  refv = Load_view<2, T>(ref, dom, vsip::Local_map(), swap_bytes).view();

  cout << error_db(in, refv) << endl;
}

