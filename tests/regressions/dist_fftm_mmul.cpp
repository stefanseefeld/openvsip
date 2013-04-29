//
// Copyright (c) 2007 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#include <algorithm>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/signal.hpp>
#include <vsip/map.hpp>

#include <vsip_csl/test.hpp>

using namespace std;
using namespace vsip;


/***********************************************************************
  Definitions
***********************************************************************/

template <typename T,
	  typename MapT>
void
test_fftm_mmul(
  bool        scale,
  MapT const& map = MapT())
{
  typedef Fftm<T, T, row, fft_fwd, by_value, 1> fftm_type;

  length_type rows = 16;
  length_type cols = 64;

  fftm_type fftm(Domain<2>(rows, cols), scale ? 1.f / cols : 1.f);

  Matrix<T, Dense<2, T, row2_type, MapT> > in (rows, cols,          map);
  Matrix<T, Dense<2, T, row2_type, MapT> > k  (rows, cols, T(   2), map);
  Matrix<T, Dense<2, T, row2_type, MapT> > out(rows, cols, T(-100), map);

  in = T(1);

  out = k * fftm(in); 

  for (index_type r=0; r<rows; ++r)
    test_assert(out.get(r, 0) == T(scale ? 2 : 2*cols));
}



/***********************************************************************
  Main
***********************************************************************/

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

#if 0
  // Enable this section for easier debugging.
  impl::Communicator& comm = impl::default_communicator();
  pid_t pid = getpid();

  cout << "rank: "   << comm.rank()
       << "  size: " << comm.size()
       << "  pid: "  << pid
       << endl;

  // Stop each process, allow debugger to be attached.
  if (comm.rank() == 0) fgetc(stdin);
  comm.barrier();
  cout << "start\n";
#endif


  // Test with local map.

  Local_map lmap;

  test_fftm_mmul<complex<float> >(true,  lmap);


  // Test with map that distributes rows across all processors.
  // (Each processor should have 1 or more rows, unless, NP > 64).
  //
  // 070507: This causes Fft_return_functor to create a local subblock
  //         with the wrong size when np >= 2.


  length_type np = vsip::num_processors();
  Map<> map1(np, 1);

  test_fftm_mmul<complex<float> >(true, map1);


  // Test with map that collects all rows on root processor.
  // (Other processor will have 0 rows, i.e. an empty subblock).
  //
  // 070507: This causes Fft_return_functor to create a local subblock
  //         with the wrong size when np >= 2.

  Map<> map2(1, 1);

  test_fftm_mmul<complex<float> >(true, map2);

  return 0;
}
