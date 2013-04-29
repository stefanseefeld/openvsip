/***********************************************************************

  File:   math-reductions.cpp
  Author: Jeffrey D. Oldham, CodeSourcery, LLC.
  Date:   12/29/2003

  Contents: Very, very simple tests of matrix and vector math
    reduction functions.

Copyright 2005 Georgia Tech Research Corporation, all rights reserved.

A non-exclusive, non-royalty bearing license is hereby granted to all
Persons to copy, distribute and produce derivative works for any
purpose, provided that this copyright notice and following disclaimer
appear on All copies: THIS LICENSE INCLUDES NO WARRANTIES, EXPRESSED
OR IMPLIED, WHETHER ORAL OR WRITTEN, WITH RESPECT TO THE SOFTWARE OR
OTHER MATERIAL INCLUDING, BUT NOT LIMITED TO, ANY IMPLIED WARRANTIES
OF MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE, OR ARISING
FROM A COURSE OF PERFORMANCE OR DEALING, OR FROM USAGE OR TRADE, OR OF
NON-INFRINGEMENT OF ANY PATENTS OF THIRD PARTIES. THE INFORMATION IN
THIS DOCUMENT SHOULD NOT BE CONSTRUED AS A COMMITMENT OF DEVELOPMENT
BY ANY OF THE ABOVE PARTIES.

The US Government has a license under these copyrights, and this
Material may be reproduced by or for the US Government.
***********************************************************************/

/***********************************************************************
  Included Files
***********************************************************************/

#include "test.hpp"
#include <vsip/domain.hpp>
#include <vsip/initfin.hpp>
#include <vsip/math.hpp>
#include <vsip/matrix.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <cstdlib>
#include "test-util.hpp"

/***********************************************************************
  Function Definitions
***********************************************************************/

template <template <typename> class Storage>
void
test_vmaxval()
{
   Storage<vsip::scalar_f>
		stor_scalarf(2, 1.0);
   
//  vsip::Vector<vsip::scalar_f>
//		vector_scalarf (2, 1.0);

  vsip::Index<1>
		position_one;

  insist (equal (vsip::maxval (stor_scalarf.view, position_one),
		 static_cast<vsip::scalar_f> (1.0)));
  insist (equal (position_one, vsip::Index<1>(0)));

  stor_scalarf.block().put (1, static_cast<vsip::scalar_f> (2.0));
  insist (equal (vsip::maxval (stor_scalarf.view, position_one),
		 static_cast<vsip::scalar_f> (2.0)));
  insist (equal (position_one, vsip::Index<1>(1)));

  stor_scalarf.block().put (0, static_cast<vsip::scalar_f> (20.0));
  insist (equal (vsip::maxval (stor_scalarf.view, position_one),
		 static_cast<vsip::scalar_f> (20.0)));
  insist (equal (position_one, vsip::Index<1>(0)));
}



void
test_meansqval()
{
  using vsip::Vector;
  using vsip::cscalar_f;
  using vsip::scalar_f;
  using vsip::meansqval;

  // Specification appears to be incorrect.
  Vector<cscalar_f>
		vec (2, cscalar_f (1.0, 0.0));

  vec.put(0, cscalar_f(1.f, 0.f));
  vec.put(1, cscalar_f(1.f, 0.f));
  insist(equal(meansqval(vec), scalar_f(1.0)));

  vec.put(0, cscalar_f(0.f, 1.f));
  vec.put(1, cscalar_f(0.f, 1.f));
  insist(equal(meansqval(vec), scalar_f(1.0)));

  vec.put(0, cscalar_f(1.f, 0.f));
  vec.put(1, cscalar_f(0.f, 2.f));
  insist(equal(meansqval(vec), scalar_f(2.5)));
}



int
main (int argc, char** argv)
{
  vsip::vsipl	init(argc, argv);


  vsip::Vector<bool>
		vector_bool_false (2, false);
  vsip::Vector<bool>
		vector_bool_true (2, true);
  vsip::Vector<bool>
		vector_bool_mixed (2, false);
  vector_bool_mixed.put (0, true);


  vsip::const_Vector<bool>
		const_vector_bool_false (2, false);
  vsip::const_Vector<bool>
		const_vector_bool_true (2, true);
  vsip::const_Vector<bool>
		const_vector_bool_mixed (vector_bool_mixed);


  vsip::Matrix<bool>
		matrix_bool_false (2, 2, false);
  vsip::Matrix<bool>
		matrix_bool_true (2, 2, true);
  vsip::Matrix<bool>
		matrix_bool_mixed (2, 2, false);
  matrix_bool_mixed.put (0, 1, true);
  matrix_bool_mixed.put (1, 0, true);


  vsip::const_Matrix<bool>
		const_matrix_bool_false (2, 2, false);
  vsip::const_Matrix<bool>
		const_matrix_bool_true (2, 2, true);
  vsip::const_Matrix<bool>
		const_matrix_bool_mixed (matrix_bool_mixed);


  /* Test alltrue.  */
  insist (equal (vsip::alltrue (vector_bool_false), false));
  insist (equal (vsip::alltrue (vector_bool_true),  true));
  insist (equal (vsip::alltrue (vector_bool_mixed), false));

  insist (equal (vsip::alltrue (const_vector_bool_false), false));
  insist (equal (vsip::alltrue (const_vector_bool_true),  true));
  insist (equal (vsip::alltrue (const_vector_bool_mixed), false));

  insist (equal (vsip::alltrue (matrix_bool_false), false));
  insist (equal (vsip::alltrue (matrix_bool_true), true));
  insist (equal (vsip::alltrue (matrix_bool_mixed), false));

  insist (equal (vsip::alltrue (const_matrix_bool_false), false));
  insist (equal (vsip::alltrue (const_matrix_bool_true), true));
  insist (equal (vsip::alltrue (const_matrix_bool_mixed), false));

  /* Test meanval.  */

  vsip::Vector<vsip::cscalar_f>
		vector_cscalar (2, vsip::cscalar_f (1.0, 1.0));
  insist (equal (vsip::meanval (vector_cscalar),
		 vsip::cscalar_f (1.0, 1.0)));

  /* Test meansqval.  */
  test_meansqval();

  /* Test sumval.  */

  insist (equal (vsip::sumval (matrix_bool_false),
		 static_cast<vsip::length_type>(0)));
  insist (equal (vsip::sumval (matrix_bool_true),
		 static_cast<vsip::length_type>(4)));
  insist (equal (vsip::sumval (matrix_bool_mixed),
		 static_cast<vsip::length_type>(2)));

  insist (equal (vsip::sumval (const_matrix_bool_false),
		 static_cast<vsip::length_type>(0)));
  insist (equal (vsip::sumval (const_matrix_bool_true),
		 static_cast<vsip::length_type>(4)));
  insist (equal (vsip::sumval (const_matrix_bool_mixed),
		 static_cast<vsip::length_type>(2)));

  /* Test maxval.  */
  test_vmaxval<     VectorStorage>();
  test_vmaxval<ConstVectorStorage>();


  vsip::Matrix<vsip::scalar_f>
		matrix_scalarf (2, 2, 1.0);
  vsip::Index<2>
		position_two;
  insist (equal (vsip::maxval (matrix_scalarf, position_two),
		 static_cast<vsip::scalar_f> (1.0)));
  insist (equal (position_two, vsip::Index<2>(0, 0)));
  matrix_scalarf.put (0, 1, static_cast<vsip::scalar_f> (2.0));
  insist (equal (vsip::maxval (matrix_scalarf, position_two),
		 static_cast<vsip::scalar_f> (2.0)));
  insist (equal (position_two, vsip::Index<2>(0, 1)));
  matrix_scalarf.put (1, 1, static_cast<vsip::scalar_f> (20.0));
  insist (equal (vsip::maxval (matrix_scalarf, position_two),
		 static_cast<vsip::scalar_f> (20.0)));
  insist (equal (position_two, vsip::Index<2>(1, 1)));
  
  return EXIT_SUCCESS;
}
