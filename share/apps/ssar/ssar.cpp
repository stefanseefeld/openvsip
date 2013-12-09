/* Copyright (c) 200, 2008, 2011 CodeSourcery, Inc.  All rights reserved. */

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

/// Description: VSIPL++ implementation of HPCS Challenge Benchmarks 
///              Scalable Synthetic Compact Applications - 
///              SSCA #3: Sensor Processing and Knowledge Formation

#include <vsip/initfin.hpp>
#include <vsip/math.hpp>
#include <vsip/signal.hpp>
#include <ovxx/allocator.hpp>
#include "ssar.hpp"
#include "kernel1.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

// Print usage information.
// If an error message is given, print that and exit with -1, otherwise with 0.
void ssar_options::usage(std::string const &prog_name,
			 std::string const &error = std::string())
{
  std::ostream &os = error.empty() ? std::cout : std::cerr;
  if (!error.empty()) os << error << std::endl;
  os << "Usage: " << prog_name 
     << " [-l|--loop <LOOP>] [-o|--output <FILE>] [--swap|--noswap] <data dir> " 
     << std::endl;
  if (error.empty()) std::exit(0);
  else std::exit(-1);
}


ssar_options::ssar_options(int argc, char **argv)
  : loop(1),
#if VSIP_BIG_ENDIAN
    swap_bytes(true)
#else
    swap_bytes(false)
#endif
{
  for (int i = 1; i < argc; ++i)
  {
    std::string arg = argv[i];
    if (arg == "-o" || arg == "--output")
    {
      if (++i == argc) usage(argv[0], "no output argument given");
      else output = argv[i];
    }
    else if (arg == "-l" || arg == "-loop" || arg == "--loop")
    {
      if (++i == argc) usage(argv[0], "no loop argument given");
      else
      {
	std::istringstream iss(argv[i]);
	iss >> loop;
	if (!iss) usage(argv[0], "loop argument not an integer");
      }
    }
    else if (arg == "-swap" || arg == "--swap") swap_bytes = true;
    else if (arg == "-noswap" || arg == "--noswap") swap_bytes = false;
    else if (arg[0] != '-')
    {
      if (data_dir.empty()) data_dir = arg;
      else usage(argv[0], "Invalid non-option argument");
    }
    else usage(argv[0], "Unknown option");
  }

  if (data_dir.empty()) usage(argv[0], "No data dir given");

  data_dir += '/';
  parameters = data_dir + "parameters.hdf5";
  sar_dimensions = data_dir + "dims.txt";
  raw_sar_data = data_dir + "sar.hdf5";

  if (output.empty()) output = data_dir + "image.hdf5";

  std::ifstream ifs(sar_dimensions.c_str());
  if (!ifs) usage(argv[0], "unable to open dimension data file");
  else
  {
    ifs >> scale >> n >> mc >> m;
    if (!ifs) usage(argv[0], "Error reading dimension data");
  }
}

int
main(int argc, char** argv)
{
  vsip::vsipl init(argc, argv);

 #if OVXX_ENABLE_HUGE_PAGE_ALLOCATOR
  std::auto_ptr<ovxx::allocator> allocator
    (new ovxx::huge_page_allocator("/huge/benchmark.bin", 20));
  Local_map huge_map(allocator.get());
#else
  Local_map huge_map;
#endif

  ssar_options opt(argc, argv);

  typedef SSAR_BASE_TYPE T;

  // Setup for Stage 1, Kernel 1 
  ovxx::timer t0;
  Kernel1<T> k1(opt, huge_map); 
  std::cout << "setup:   " << t0.elapsed() << " (s)" << std::endl;

  // Retrieve the raw radar image data from disk.  This Data I/O 
  // component is currently untimed.
  Kernel1<T>::complex_matrix_type s_raw(opt.n, opt.mc, huge_map);
  ovxx::hdf5::read(opt.raw_sar_data, "data", s_raw);

  // Resolve the image.  This Computation component is timed.
  Kernel1<T>::real_matrix_type 
    image(k1.output_size(0), k1.output_size(1), huge_map);

  ovxx::timer t1;
  vsip::Vector<double> process_time(opt.loop);
  for (unsigned int l = 0; l < opt.loop; ++l)
  {
    t1.restart();
    k1.process_image(s_raw, image);
    process_time.put(l, t1.elapsed());
  }

  // Display statistics
  if (opt.loop > 0)
  {
    Index<1> idx;
    double mean = vsip::meanval(process_time);
    std::cout << "loops:   " << opt.loop << std::endl;
    std::cout << "mean:    " << mean << std::endl;
    std::cout << "min:     " << vsip::minval(process_time, idx) << std::endl;
    std::cout << "max:     " << vsip::maxval(process_time, idx) << std::endl;
    std::cout << "std-dev: " << sqrt(vsip::meansqval(process_time - mean)) << std::endl;
  }

  // Store the image on disk for later processing (not timed).
  ovxx::hdf5::write(opt.output, "data", image); 
}
