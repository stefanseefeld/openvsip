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
