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

#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/math.hpp>
#include <ovxx/io/hdf5.hpp>

using namespace ovxx;

enum data_format_type
{
  COMPLEX_VIEW = 0,
  REAL_VIEW,
  INTEGER_VIEW
};

template <typename T>
void
compare(std::string const &infile, std::string const &ref)
{
  typedef Matrix<T> matrix_type;

  hdf5::file file(infile, 'r');
  hdf5::dataset ds = file.open_dataset("data");
  Domain<2> dom = ds.query_extent<2>();

  matrix_type in(dom[0].size(), dom[1].size());
  ds.read(in);
  matrix_type refv(dom[0].size(), dom[1].size());
  hdf5::file ref_file(ref, 'r');
  hdf5::dataset ref_ds = ref_file.open_dataset("data");
  ref_ds.read(refv);

  double error = 0.;
  if (anytrue(is_nan(in)) || anytrue(is_nan(refv)))
    error = 201.0;

  Index<2> idx;

  double refmax1 = maxval(magsq(in), idx);
  double refmax2 = maxval(magsq(refv), idx);
  double refmax  = std::max(refmax1, refmax2);
  double maxsum  = maxval(ite(magsq(in - refv) < 1.e-20,
			      -201.0,
			      10.0 * log10(magsq(in - refv)/(2.0*refmax))),
			  idx);
  error = maxsum;

  std::cout << error << " dB" << std::endl;
}

int
main(int argc, char** argv)
{
  vsip::vsipl init(argc, argv);

  if (argc < 3 || argc > 4)
  {
    std::cerr << "Usage: " << argv[0] 
	      << " [-crn] <input> <reference>" << std::endl;
    return -1;
  }
  else
  {
    data_format_type format = COMPLEX_VIEW;
    if (argc == 4)
    {
      if (0 == strncmp("-c", argv[1], 2))
        format = COMPLEX_VIEW;
      else if (0 == strncmp("-r", argv[1], 2))
        format = REAL_VIEW;
      else if (0 == strncmp("-n", argv[1], 2))
        format = INTEGER_VIEW;
      else
      {
	std::cerr << "Usage: " << argv[0] 
		  << " [-crn] <input> <reference>" << std::endl;
        return -1;
      }
      ++argv;
      --argc;
    }

    if (format == REAL_VIEW)
      compare<float>(argv[1], argv[2]);
    else if (format == INTEGER_VIEW)
      compare<int>(argv[1], argv[2]);
    else
      compare<complex<float> >(argv[1], argv[2]);
  }
}
