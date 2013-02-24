/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
#ifndef ssar_hpp_
#define ssar_hpp_

#include <vsip/support.hpp>

// Provide a default base type (float or double)
#ifndef SSAR_BASE_TYPE
#define SSAR_BASE_TYPE float
#endif

struct ssar_options
{
  ssar_options(int, char **);

  static void usage(std::string const &prog_name, std::string const &error);

  // Number of process_image iterations to perform (default 1)
  unsigned int loop;
  // Whether or not to swap bytes during file I/O
  bool swap_bytes;

  vsip::scalar_f scale;
  vsip::length_type n;
  vsip::length_type mc;
  vsip::length_type m;

  std::string output;

  std::string data_dir;
  // Files required to be in the data directory
  std::string sar_dimensions;
  std::string raw_sar_data;
  std::string fast_time_filter;
  std::string slow_time_wavenumber;
  std::string slow_time_compressed_aperture_position;
  std::string slow_time_aperture_position;
  std::string slow_time_spatial_frequency;
};

#endif
