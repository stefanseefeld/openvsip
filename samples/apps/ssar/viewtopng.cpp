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

/** @file    viewtopng.cpp
    @author  Stefan Seefeld
    @date    2008-11-24
    @brief   Utility to convert VSIPL++ views to greyscale png images.
*/

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip_csl/view_cast.hpp>
#include <vsip_csl/load_view.hpp>
#include <vsip_csl/png.hpp>
#include <iostream>
#include <fstream>
#include <string>

using namespace vsip;


enum data_format_type
{
  COMPLEX_MAG = 0,
  COMPLEX_REAL,
  COMPLEX_IMAG,
  SCALAR_FLOAT,
  SCALAR_INTEGER
};

#if VSIP_BIG_ENDIAN
bool const swap_bytes = true;
#else
bool const swap_bytes = false;
#endif

void
convert_to_greyscale(data_format_type format, 
                     std::string const &infile, std::string const &outfile,
                     length_type rows, length_type cols)
{
  using vsip_csl::Load_view;
  typedef Matrix<scalar_f> matrix_type;
  Domain<2> dom(rows, cols);

  matrix_type in(rows, cols);

  if (format == COMPLEX_MAG)
    in = mag(Load_view<2, cscalar_f>(infile.c_str(), dom, Local_map(), swap_bytes).view());
  else if (format == COMPLEX_REAL)
    in = real(Load_view<2, cscalar_f>(infile.c_str(), dom, Local_map(), swap_bytes).view());
  else if (format == COMPLEX_IMAG)
    in = imag(Load_view<2, cscalar_f>(infile.c_str(), dom, Local_map(), swap_bytes).view());
  else if (format == SCALAR_FLOAT)
    in = Load_view<2, scalar_f>(infile.c_str(), dom, Local_map(), swap_bytes).view();
  else if (format == SCALAR_INTEGER)
    in = Load_view<2, scalar_i>(infile.c_str(), dom, Local_map(), swap_bytes).view();
  else
    std::cerr << "Error: format type " << format << " not supported." << std::endl;


  Index<2> idx;
  scalar_f minv = minval(in, idx);
  scalar_f maxv = maxval(in, idx);
  scalar_f scale = (maxv - minv ? maxv - minv : 1.f);

  Matrix<unsigned char> out(rows, cols);
  out = vsip_csl::view_cast<unsigned char>((in - minv) * 255.f / scale);

  std::ofstream ofs(outfile.c_str());
  vsip_csl::png::info info;
  info.width = cols;
  info.height = rows;
  info.depth = 8;
  info.colortype = vsip_csl::png::gray;
  info.compression = 0;
  info.filter = 0;
  info.interlace = vsip_csl::png::none;
  info.rowbytes = cols;
  vsip_csl::png::encoder encoder(ofs.rdbuf(), info);
  encoder.encode(out.block().ptr(), info.height * info.rowbytes);

  // The min and max values are displayed to reveal the scale
  std::cout << infile << " [" << rows << " x " << cols << "] : "
            << "min " << minv << ", max " << maxv << std::endl;
}

void usage(std::ostream &os, std::string const &prog_name)
{
  os << "Usage: " << prog_name
     << " [-c|-r|-i|-s|-n] <input> <output> <rows> <cols>" << std::endl;
}

int
main(int argc, char** argv)
{
  std::string prog_name = argv[0];
  vsip::vsipl init(argc, argv);

  if (argc < 5 || argc > 6)
  {
    usage(std::cerr, prog_name);
    return -1;
  }

  // The default is to create the image using both the real and imaginary 
  // parts by computing the magnitude (default, -c).  Alternatively, the 
  // real or imaginary parts (-r or -i respectively) may be used 
  // individually, or, if the data is already scalar, it MUST be either 
  // single-precision floating point or integer format (-s or -n must be 
  // used to indicate which).
  data_format_type format = COMPLEX_MAG;
  if (argc == 6)
  {
    if (std::string("-c") == argv[1]) format = COMPLEX_MAG;
    else if (std::string("-r") == argv[1]) format = COMPLEX_REAL;
    else if (std::string("-i") == argv[1]) format = COMPLEX_IMAG;
    else if (std::string("-s") == argv[1]) format = SCALAR_FLOAT;
    else if (std::string("-n") == argv[1]) format = SCALAR_INTEGER;
    else
    {
      usage(std::cerr, argv[0]);
      return -1;
    }
    ++argv;
    --argc;
  }
  length_type rows, cols;
  {
    std::istringstream iss(argv[3]);
    iss >> rows;
    if (!iss)
    {
      usage(std::cerr, prog_name);
      return -1;
    }
  }
  {
    std::istringstream iss(argv[4]);
    iss >> cols;
    if (!iss)
    {
      usage(std::cerr, prog_name);
      return -1;
    }
  }
  convert_to_greyscale(format, argv[1], argv[2], rows, cols);
  return 0;
}



