/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Unit tests for parallel vector-matrix multiply.

#include <iostream>

#include <vsip/support.hpp>
#include <vsip/initfin.hpp>
#include <vsip/vector.hpp>
#include <vsip/map.hpp>
#include <vsip/parallel.hpp>
#include <vsip/math.hpp>
#include <vsip/domain.hpp>
#include <vsip/signal.hpp>
#include <vsip/core/domain_utils.hpp>

#include <vsip_csl/test.hpp>
#include <vsip_csl/error_db.hpp>
#include "test_ramp.hpp"
#include "util-par.hpp"

using namespace vsip;
using namespace vsip_csl;

template <dimension_type VecDim,
	  dimension_type Dim>
class Check_vmmul
{
public:
  Check_vmmul(vsip::Domain<Dim> const& dom) : dom_(dom) {}

  template <typename T>
  T operator()(T value,
	       vsip::Index<2> const& /*local*/,
	       vsip::Index<2> const& global)
  {
    vsip::index_type i = global[0]*dom_[1].length()+global[1];
    T expected = (VecDim == 0) ? T(global[1] * i) : T(global[0] * i);

    if (value != expected)
    {
      std::cout << "Check_vmmul: MISCOMPARE" << std::endl
		<< "  global  = " << global[0] << ", " << global[1] 
		<< std::endl
		<< "  expected = " << expected << std::endl
		<< "  actual   = " << value << std::endl;
    }
    return value;
  }

private:
  vsip::Domain<Dim> dom_;
};

template <dimension_type Dim,
	  typename       OrderT,
	  typename       T,
	  typename       VecMapT,
	  typename       MatMapT>
void
test_par_vmmul(
  VecMapT const& vec_map,
  MatMapT const& mat_map,
  length_type    rows,
  length_type    cols)
{
  Matrix<T, Dense<2, T, OrderT, MatMapT> >    m  (rows, cols, mat_map);
  Matrix<T, Dense<2, T, OrderT, MatMapT> >    res(rows, cols, mat_map);
  Vector<T, Dense<1, T, row1_type, VecMapT> > v(Dim == 0 ? cols : rows,
						vec_map);

  foreach_point(m, Set_identity<2>(Domain<2>(rows, cols)));
  foreach_point(v, Set_identity<1>(Domain<1>(v.size())));

  res = vmmul<Dim>(v, m);

  foreach_point(res, Check_vmmul<Dim, 2>(Domain<2>(rows, cols)));
}



template <typename OrderT,
	  typename T>
void
par_vmmul_cases()
{
  length_type np, nr, nc;

  get_np_square(np, nr, nc);


  // -------------------------------------------------------------------
  // If vector is global (replicated on all processors),
  // The matrix must not be distributed along the vector

  Replicated_map<1> gmap;
  Map<Block_dist, Block_dist> row_map(np, 1);
  Map<Block_dist, Block_dist> col_map(1,  np);
  Map<Block_dist, Block_dist> chk_map(nr, nc);

  test_par_vmmul<0, OrderT, T>(gmap, row_map, 5, 7);
  // test_par_vmmul<1, OrderT, T>(gmap, row_map, 4, 3); // dist along vector

  // test_par_vmmul<0, OrderT, T>(gmap, col_map, 5, 7); // dist along vector
  test_par_vmmul<1, OrderT, T>(gmap, col_map, 5, 7);

  // test_par_vmmul<0, OrderT, T>(gmap, chk_map, 5, 7); // dist along vector
  // test_par_vmmul<1, OrderT, T>(gmap, chk_map, 5, 7); // dist along vector

  // Likewise for replicated_map
  Replicated_map<1> rmap;
  test_par_vmmul<0, OrderT, T>(rmap, row_map, 5, 7);
  test_par_vmmul<1, OrderT, T>(rmap, col_map, 5, 7);


  // -------------------------------------------------------------------
  // If vector is distributed (not replicated),
  // The matrix must
  //    have the same distribution along the vector
  //    not be distributed in the perpendicular to the vector

  Map<Block_dist> vmap(np);

  test_par_vmmul<0, OrderT, T>(vmap, col_map, 5, 7);
  test_par_vmmul<1, OrderT, T>(vmap, row_map, 5, 7);

  // -------------------------------------------------------------------
  // If vector and matrix are both on single processor
  for (processor_type p=0; p<np; ++p)
  {
    Vector<processor_type> pvec(1);
    pvec(0) = p;
    Map<Block_dist, Block_dist> p1_map(pvec, 1, 1); 

    test_par_vmmul<0, OrderT, T>(p1_map, p1_map, 5, 7);
    test_par_vmmul<1, OrderT, T>(p1_map, p1_map, 5, 7);
  }
}



int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  par_vmmul_cases<row2_type, float>();
  par_vmmul_cases<col2_type, float>();
  par_vmmul_cases<row2_type, complex<float> >();
  par_vmmul_cases<col2_type, complex<float> >();
}
