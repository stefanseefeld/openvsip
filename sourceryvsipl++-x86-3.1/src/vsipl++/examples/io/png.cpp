/* Copyright (c) 2006 by CodeSourcery, LLC.  All rights reserved. */

/** @file    png.cpp
    @author  Stefan Seefeld
    @date    2006-03-21
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

using namespace vsip_csl;
using namespace vsip;

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

