/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GNU GPL version 2.0 or greater.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   List views contained in .mat file.

#include <vsip/initfin.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/tensor.hpp>
#include <vsip/selgen.hpp>
#include <vsip/map.hpp>

#include <vsip_csl/matlab_file.hpp>
#include <vsip_csl/output.hpp>

#include <iostream>
#include <fstream>
#include <string>

using namespace vsip;
using namespace vsip_csl;

template <dimension_type D, typename T>
void
load_view(Matlab_file &mf, Matlab_file::iterator &iter, Domain<D> const &dom)
{
  typedef Dense<D, T> block_type;
  typedef typename vsip::impl::view_of<block_type>::type view_type;

  block_type block(dom);
  view_type  view(block);

  mf.read_view(view, iter);

  std::cout << view;
}

void version(char const *name)
{
  std::cout << name << ' ' << VSIP_IMPL_MAJOR_VERSION_STRING 
	    << " (Sourcery VSIPL++ " << VSIP_IMPL_VERSION_STRING << ")\n"
	    << " Copyright (C) 2010 CodeSourcery, Inc." << std::endl;
}

int
main(int argc, char** argv)
{
  if (argc != 2)
  {
    std::cerr << "Usage: " << argv[0] << "<filename>" << std::endl;
    return -1;
  }
  std::string arg = argv[1];
  if (arg == "-v" || arg == "--version")
  {
    version(argv[0]);
    return 0;
  }
  char const *filename = argv[1];

  // Create Matlab_file object for 'sample.mat' file.
  Matlab_file mf(filename);
  Matlab_file::iterator cur = mf.begin();
  Matlab_file::iterator end = mf.end();

  Matlab_bin_hdr hdr = mf.header();

  std::cout << "Matlab file: " << filename << '\n'
	    << "  descr  : " << hdr.description << '\n'
	    << "  version: " << hdr.version << '\n'
	    << "  endian : " << hdr.endian 
	    << " (swap: " 
	    << (hdr.endian == ('I'<<8 | 'M') ? "yes" :
		hdr.endian == ('M'<<8 | 'I') ? "no"  : "*unknown*")
	    << ")" << std::endl;

  // Iterate through views in file.
  while (cur != end)
  {
    Matlab_view_header* hdr = *cur;

    std::cout << "view: " << hdr->array_name << std::endl;

    // Dump array_name out by byte
    // for (int i=0; hdr->array_name[i] != 0 && i<128; ++i)
    //   cout << "  [" << i << "]: " << (int)hdr->array_name[i] << endl;

    std::cout << "  dim       : " << hdr->num_dims;
    if (hdr->num_dims > 0)
    {
      char const *sep = " (";
      for (index_type i=0; i<hdr->num_dims; ++i)
      {
	std::cout << sep << hdr->dims[i];
	sep = ", ";
      }
      std::cout << ")";
    }
    std::cout << std::endl;

    std::cout << "  is_complex: " << (hdr->is_complex ? "true" : "false") << '\n'
	      << "  class_type: " << vsip_csl::matlab::class_type(hdr->class_type);
    std::cout << " (" << (int)hdr->class_type << ")" << std::endl;

    if (hdr->class_type == vsip_csl::matlab::mxDOUBLE_CLASS)
    {
      if (hdr->num_dims == 2 && hdr->dims[0] == 1)
	load_view<1, double>(mf, cur, Domain<1>(hdr->dims[1]));
      else if (hdr->num_dims == 2 && hdr->dims[1] == 1)
	load_view<1, double>(mf, cur, Domain<1>(hdr->dims[0]));
      else if (hdr->num_dims == 2)
	load_view<2, double>(mf, cur, Domain<2>(hdr->dims[0], hdr->dims[1]));
//      else if (hdr->num_dims == 3)
//	load_view<3, double>(mf, cur, Domain<3>(hdr->dims[0], hdr->dims[1],
//						hdr->dims[2]));
    }
    else if (hdr->class_type == vsip_csl::matlab::mxSINGLE_CLASS)
    {
      if (hdr->num_dims == 2 && hdr->dims[0] == 1)
	load_view<1, float>(mf, cur, Domain<1>(hdr->dims[1]));
      else if (hdr->num_dims == 2 && hdr->dims[1] == 1)
	load_view<1, float>(mf, cur, Domain<1>(hdr->dims[0]));
      else if (hdr->num_dims == 2)
	load_view<2, float>(mf, cur, Domain<2>(hdr->dims[0], hdr->dims[1]));
//      else if (hdr->num_dims == 3)
//	load_view<3, float>(mf, cur, Domain<3>(hdr->dims[0], hdr->dims[1],
//						hdr->dims[2]));
    }

    ++cur; // Move to next view stored in the file.
  }
} 

