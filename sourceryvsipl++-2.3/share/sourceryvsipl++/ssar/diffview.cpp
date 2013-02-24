/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
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

#if _BIG_ENDIAN
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

