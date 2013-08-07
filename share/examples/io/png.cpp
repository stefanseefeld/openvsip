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


#include <vsip/initfin.hpp>
#include <vsip/matrix.hpp>
#include <vsip/dense.hpp>
#include <vsip/math.hpp>
#include <ovxx/png.hpp>
#include <cmath>
#include <iostream>
#include <fstream>

using namespace ovxx;

/***********************************************************************
  Type Definitions
***********************************************************************/

// Let's assume pixels store 4 channels of 1 byte values.
struct Pixel 
{
  Pixel(char rr, char gg, char bb, char aa) : r(rr), g(gg), b(bb), a(aa) {}
  char r, g, b, a;
};

typedef Matrix<Pixel, Dense<2, Pixel> > Image;

Pixel 
swap_red_green(Pixel const &p) 
{
  return Pixel(p.g, p.r, p.b, p.a);
}



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
  // Make sure the image uses 32-bit RGBA pixels.
  if (info.colortype != png::rgba || info.depth != 8)
  {
    std::cerr << "Sorry, pixel format not supported." << std::endl;
    return -1;
  }
  std::cout << "Reading png file :" << '\n'
	    << "  Width : " << info.width << '\n'
	    << "  Height : " << info.height << '\n'
	    << "  Depth : " << info.depth << '\n'
	    << "  Bytes per row : " << info.rowbytes << '\n'
	    << "  Color type : " << info.colortype << std::endl;

  Image image(info.width, info.height);
  decoder.decode(reinterpret_cast<unsigned char *>(image.block().ptr()),
		 sizeof(Pixel) * info.height * info.width);
  // swap red and green channels
  image = unary<Pixel>(swap_red_green, image);
  std::ofstream ofs(argv[2]);
  png::encoder encoder(ofs.rdbuf(), info);
  encoder.encode(reinterpret_cast<unsigned char *>(image.block().ptr()),
		 sizeof(Pixel) * info.height * info.rowbytes);
}

