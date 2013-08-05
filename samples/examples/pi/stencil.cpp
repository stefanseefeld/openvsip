/* Copyright (c) 2007, 2011 CodeSourcery, Inc.  All rights reserved. */

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

/** @file    examples/stencil.cpp
    @author  Stefan Seefeld
    @date    2007-04-19
    @brief   VSIPL++ Library: 
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/initfin.hpp>
#include <vsip/matrix.hpp>
#include <vsip/dense.hpp>
#include <vsip/math.hpp>
#include <vsip_csl/png.hpp>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vsip_csl/pi.hpp>

using namespace vsip_csl;
using namespace vsip;

/***********************************************************************
  Type Definitions
***********************************************************************/

typedef char Pixel;

typedef Matrix<Pixel, Dense<2, Pixel> > Image;

/***********************************************************************
  Main Program
***********************************************************************/

int main (int argc, char **argv)
{
  vsipl init(argc, argv);

  if (argc != 3)
  {
    std::cerr << "Usage : " << argv[0] << " <png> <output>" << std::endl;
    return -1;
  }
  png::info info;
  std::ifstream ifs(argv[1]);
  png::decoder decoder(ifs.rdbuf(), info);
  // Make sure the image uses 8-bit grayscale pixels.
  std::cout << "Reading png file :" << '\n'
	    << "  Width : " << info.width << '\n'
	    << "  Height : " << info.height << '\n'
	    << "  Depth : " << info.depth << '\n'
	    << "  Bytes per row : " << info.rowbytes << '\n'
	    << "  Color type : " << info.colortype << std::endl;
  if (info.colortype != png::gray || info.depth != 8)
  {
    std::cerr << "Sorry, pixel format not supported." << std::endl;
    return -1;
  }

  Image image(info.height, info.width);
  decoder.decode(reinterpret_cast<unsigned char *>(image.block().ptr()),
		 sizeof(Pixel) * info.height * info.width);

  Image diff(info.height, info.width);
  pi::Iterator<> i;
  pi::Iterator<> j;
  diff(i,j) = (-1 * image(i - 1, j - 1) + image(i + 1, j - 1)
               -2 * image(i - 1, j) + 2 * image(i + 1, j)
               -1 * image(i - 1, j + 1) + image(i + 1, j + 1));

  std::ofstream ofs(argv[2]);
  png::encoder encoder(ofs.rdbuf(), info);
  encoder.encode(reinterpret_cast<unsigned char *>(diff.block().ptr()),
		 sizeof(Pixel) * info.height * info.rowbytes);
}

