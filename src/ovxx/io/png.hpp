//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_io_png_hpp_
#define ovxx_io_png_hpp_

#include <ovxx/dda.hpp>
#include <ovxx/view.hpp>
#include <png.h>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <streambuf>
#include <stdexcept>
#include <cassert>

namespace ovxx
{
namespace png
{
namespace detail
{
void read(png_structp png_ptr, png_bytep image, png_size_t length) 
{
  std::streambuf *input = static_cast<std::streambuf *>(png_get_io_ptr(png_ptr));
  input->sgetn((char *)image, (std::size_t)length);
}

void warning(png_structp, png_const_charp msg)
{
  std::cout << "PNG Warning : " << msg << std::endl;
}

void error(png_structp, png_const_charp msg)
{
  std::cerr << "PNG Error : " << msg << std::endl;
}

void write(png_structp png_ptr, png_bytep image, png_size_t length) 
{
  std::streambuf *sbuf = static_cast<std::streambuf *>(png_get_io_ptr(png_ptr));
  sbuf->sputn((char*)image, (std::size_t)length);
}

void flush(png_structp png_ptr) 
{
  std::streambuf *sbuf = static_cast<std::streambuf *>(png_get_io_ptr(png_ptr));
  sbuf->pubsync();
}

} // namespace ovxx::png::detail

enum color_type
{
  gray = PNG_COLOR_TYPE_GRAY,
  grayalpha = PNG_COLOR_TYPE_GRAY_ALPHA,
  palette = PNG_COLOR_TYPE_PALETTE,
  rgb = PNG_COLOR_TYPE_RGB,
  rgba = PNG_COLOR_TYPE_RGB_ALPHA,
  maskpalette = PNG_COLOR_MASK_PALETTE,
  maskcolor = PNG_COLOR_MASK_COLOR,
  maskalpha = PNG_COLOR_MASK_ALPHA
};

enum interlace_type
{
  none = PNG_INTERLACE_NONE,
  adam7 = PNG_INTERLACE_ADAM7,
  last = PNG_INTERLACE_LAST
};
  
struct info
{
  unsigned long width;
  unsigned long height;
  unsigned long rowbytes;
  unsigned short depth;
  color_type colortype;
  unsigned short compression;
  unsigned short filter;
  interlace_type interlace;
};

class decoder 
{
  static std::size_t const magic_ = 8;
public:
  decoder(std::streambuf *sbuf, info &i)
    : input_(sbuf),
      png_(png_create_read_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0)),
      info_(png_create_info_struct(png_)),
      end_(png_create_info_struct(png_))
  {
    png_byte header[magic_];
    input_->sgetn((char*)header, magic_);
    if (png_sig_cmp(header, 0, magic_))
      throw std::runtime_error("Not a PNG file");
    png_set_sig_bytes(png_, magic_);
    png_set_read_fn(png_, input_, detail::read);
    png_set_error_fn(png_, input_, detail::error, detail::warning);
    png_set_read_status_fn(png_, 0);
    png_read_info(png_, info_);
    png_uint_32 w, h;
    int d, c, in, co, f;
    png_get_IHDR(png_, info_, &w, &h, &d, &c, &in, &co, &f);  
    i.width = w;
    i.height = h;
    i.rowbytes = png_get_rowbytes(png_, info_);
    i.depth = d;
    i.colortype = static_cast<color_type>(c);
    i.compression = co;
    i.filter = f;
    i.interlace = static_cast<interlace_type>(in);
  }
  ~decoder() { png_destroy_read_struct(&png_, &info_, &end_);}

  template <typename T, typename B>
  void decode(Matrix<T, B> image)
  {
    png_uint_32 height = png_get_image_height(png_, info_);
    png_uint_32 width = png_get_image_height(png_, info_);
    OVXX_PRECONDITION(image.size(0) == height);
    OVXX_PRECONDITION(image.size(1) == width);
    png_byte depth = png_get_bit_depth(png_, info_);
    png_byte channels = png_get_channels(png_, info_);
    OVXX_PRECONDITION(sizeof(T) == (static_cast<size_t>(depth) * channels)/8);

    typedef Layout<2, tuple<0,1,2>, dense> layout_type;
    ovxx::dda::Data<B, dda::out, layout_type> data(image.block());
    png_uint_32 rowbytes = png_get_rowbytes(png_, info_);
    std::vector<unsigned char *> rows(height);
    for (png_uint_32 i = 0; i < height; i++)
      rows[i] = reinterpret_cast<unsigned char *>(data.ptr()) + i * rowbytes;

    png_read_image(png_, &*rows.begin());
    png_read_end(png_, end_);
  }

private:
  std::streambuf *input_;
  png_structp png_;
  png_info *info_;
  png_info *end_;
};

class encoder 
{
public:
  encoder(std::streambuf *sb, info const &i)
    : output_(sb),
      png_(png_create_write_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0)),
      info_(png_create_info_struct(png_))
  {
    png_set_write_fn(png_, output_, detail::write, detail::flush);
    png_set_error_fn(png_, output_, detail::error, detail::warning);
    png_set_write_status_fn(png_, 0);

    png_set_IHDR(png_, info_,
		 i.width, i.height, i.depth, i.colortype,
		 i.interlace, i.compression, i.filter);
  }

  ~encoder() { png_destroy_write_struct(&png_, &info_);}

  template <typename T, typename B>
  void encode(Matrix<T, B> image)
  {
    png_uint_32 height = png_get_image_height(png_, info_);
    png_uint_32 width = png_get_image_width(png_, info_);
    OVXX_PRECONDITION(image.size(0) == height);
    OVXX_PRECONDITION(image.size(1) == width);
    png_byte depth = png_get_bit_depth(png_, info_);
    png_byte channels = png_get_channels(png_, info_);
    OVXX_PRECONDITION(sizeof(T) == (static_cast<size_t>(depth) * channels)/8);

    typedef Layout<2, tuple<0,1,2>, dense> layout_type;
    ovxx::dda::Data<B, dda::in, layout_type> data(image.block());
    png_uint_32 rowbytes = png_get_rowbytes(png_, info_);
    std::vector<unsigned char const *> rows(height);
    for (png_uint_32 i = 0; i < height; i++)
      rows[i] = reinterpret_cast<unsigned char const*>(data.ptr()) + i * rowbytes;

    png_write_info(png_, info_);
    png_write_image(png_, const_cast<unsigned char **>(&*rows.begin()));
    png_write_end(png_, 0);
  }

private:
  std::streambuf *output_;
  png_structp png_;
  png_info *info_; 
};

} // namespace ovxx::png
} // namespace ovxx

#endif
