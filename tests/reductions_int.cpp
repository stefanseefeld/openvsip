//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Tests for math reductions, integer and boolean types.

#include "reductions.hpp"

using namespace vsip;
using vsip_csl::equal;
using vsip_csl::sumval;
using vsip_csl::sumsqval;
using vsip_csl::meansqval;
using vsip_csl::meanval;


void
cover_sumval_bool()
{
  typedef bool T;

  test_sumval_bool<Storage<1, T> >(Domain<1>(15), 8);
  
  test_sumval_bool<Storage<2, T, row2_type> >(Domain<2>(15, 17), 8);
  test_sumval_bool<Storage<2, T, col2_type> >(Domain<2>(15, 17), 8);
  
  test_sumval_bool<Storage<3, T, tuple<0, 1, 2> > >(Domain<3>(15, 17, 7), 8);
  test_sumval_bool<Storage<3, T, tuple<0, 2, 1> > >(Domain<3>(15, 17, 7), 8);
  test_sumval_bool<Storage<3, T, tuple<1, 0, 2> > >(Domain<3>(15, 17, 7), 8);
  test_sumval_bool<Storage<3, T, tuple<1, 2, 0> > >(Domain<3>(15, 17, 7), 8);
  test_sumval_bool<Storage<3, T, tuple<2, 0, 1> > >(Domain<3>(15, 17, 7), 8);
  test_sumval_bool<Storage<3, T, tuple<2, 1, 0> > >(Domain<3>(15, 17, 7), 8);

  test_sumval_bool<Storage<1, T, row1_type, Map<Block_dist> > >(Domain<1>(15), 8);
  test_sumval_bool<Storage<1, T, row1_type, Replicated_map<1> > >(Domain<1>(15), 8);
}


int
main(int argc, char** argv)
{
  vsipl init(argc, argv);
   
  cover_sumval<int>();
  cover_sumval_bool();

  cover_sumsqval<int>();
  cover_meanval<int>();
  cover_meansqval<int>();

  // Test some types that the alternate form
  // handles better.
  cover_sumval<unsigned char>();
  cover_sumval<signed char>();
  cover_sumval<unsigned short>();
  cover_sumval<short>();

  cover_sumsqval<unsigned char>();
  cover_sumsqval<signed char>();
  cover_sumsqval<unsigned short>();
  cover_sumsqval<short>();

  cover_meanval<unsigned char>();
  cover_meanval<signed char>();
  cover_meanval<unsigned short>();
  cover_meanval<short>();

  cover_meansqval<unsigned char>();
  cover_meansqval<signed char>();
  cover_meansqval<unsigned short>();
  cover_meansqval<short>();
}
