/* Copyright (c) 2007 by CodeSourcery, Inc.  All rights reserved. */

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


struct Sobel
{
  length_type size(dimension_type) const { return 3;}
  index_type origin(dimension_type) const { return 1;}

  Pixel operator()(Image const &input, index_type y, index_type x) const
  {
    return (-1 * input.get(y - 1, x - 1) + input.get(y + 1, x - 1)
            -2 * input.get(y - 1, x) + 2 * input.get(y + 1, x)
            -1 * input.get(y - 1, x + 1) + input.get(y + 1, x + 1));
  }
};

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
  pi::apply_stencil(image, diff, Sobel());
  std::ofstream ofs(argv[2]);
  png::encoder encoder(ofs.rdbuf(), info);
  encoder.encode(reinterpret_cast<unsigned char *>(diff.block().ptr()),
		 sizeof(Pixel) * info.height * info.rowbytes);
}

