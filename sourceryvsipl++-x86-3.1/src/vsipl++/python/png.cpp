/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    python/png.cpp
    @author  Stefan Seefeld
    @date    2006-09-20
    @brief   VSIPL++ Library: Python bindings.

*/
#include <png.h> // To work around an Ubuntu bug, this needs to come first.
#include <boost/python.hpp>
#include <vsip_csl/png.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <fstream>
#include <stdexcept>

namespace bpl = boost::python;

namespace
{
struct Pixel 
{
  Pixel(char rr, char gg, char bb, char aa) : r(rr), g(gg), b(bb), a(aa) {}
  char r, g, b, a;
};


void encode_png(std::string const &filename, vsip::Matrix<float> image)
{
  vsip_csl::png::info info;
  info.colortype = vsip_csl::png::rgba;
  info.depth = 8;
  info.width = image.size(0);
  info.height = image.size(1);

  std::ofstream ofs(filename.c_str());
  vsip_csl::png::encoder encoder(ofs.rdbuf(), info);
  encoder.encode(reinterpret_cast<unsigned char *>(image.block().ptr()),
		 sizeof(Pixel) * info.height * info.rowbytes);
}

vsip::Matrix<float> decode_png(std::string const &filename)
{
  vsip_csl::png::info info;
  std::ifstream ifs(filename.c_str());
  vsip_csl::png::decoder decoder(ifs.rdbuf(), info);
  // Make sure the image uses 32-bit RGBA pixels.
  if (info.colortype != vsip_csl::png::rgba || info.depth != 8)
    throw std::invalid_argument("Pixel format not supported.");
  vsip::Matrix<float> image(info.width, info.height);
  decoder.decode(reinterpret_cast<unsigned char *>(image.block().ptr()),
		 sizeof(Pixel) * info.height * info.width);
  return image;
}
}

BOOST_PYTHON_MODULE(png)
{
  bpl::def("encode", encode_png);
  bpl::def("decode", decode_png);
}
