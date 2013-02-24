/***********************************************************************

  File:   selgen.cpp
  Author: Jeffrey D. Oldham, CodeSourcery, LLC.
  Date:   09/16/2003

  Contents: Very simple tests of the selection and generation
    functions.

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

#include <cstdlib>
#include "test.hpp"
#include <vsip/initfin.hpp>
#include <vsip/selgen.hpp>
#include <vsip/vector.hpp>
#include "test-util.hpp"

/***********************************************************************
  Function Definitions
***********************************************************************/

/* Binary comparison functions.  These functions are used with first
   ().  */

inline bool
equalFloat (vsip::scalar_f first,
	    vsip::scalar_f second) VSIP_NOTHROW
{
  return first == second;
}
		/* Returns:
		     True iff the two operands are equal.  */

inline bool
notEqualInt (vsip::scalar_i first,
	     vsip::scalar_i second) VSIP_NOTHROW
{
  return first != second;
}
		/* Returns:
		     True iff the two operands are not equal.  */

inline bool
equalBool (bool first,
	   bool second) VSIP_NOTHROW
{
  return first > second;
}
		/* Returns:
		     True iff the first operand is greater than the
		     second operand.  */



// -------------------------------------------------------------------- //
template <typename Storage>
void test_indexbool()
{
  vsip::length_type const
		input_length = 17;

  Storage	stor_bool(input_length, false);
			/* The vector to search for non-false values.  */

  vsip::Vector<vsip::Index<1> >
		vector_indices (input_length);
			/* The vector to store the non-false indices.  */
  insist (vsip::indexbool (stor_bool.view, vector_indices) == 0);
  stor_bool.block().put (0, true);
  stor_bool.block().put (3, true);
  insist (vsip::indexbool (stor_bool.view, vector_indices) == 2);
  insist (vector_indices.get (0) == vsip::Index<1>(0));
  insist (vector_indices.get (1) == vsip::Index<1>(3));
  stor_bool.block().put (16, true);
  insist (vsip::indexbool (stor_bool.view, vector_indices) == 3);
  insist (vector_indices.get (0) == vsip::Index<1>(0));
  insist (vector_indices.get (1) == vsip::Index<1>(3));
  insist (vector_indices.get (2) == vsip::Index<1>(16));
}



// -------------------------------------------------------------------- //
template <template <typename> class Storage>
void
test_vgather()
{
  vsip::length_type const
		input_length = 17;

  vsip::Vector<vsip::Index<1> >
		vector_indices (input_length);
			/* The vector to store the non-false indices.  */

  Storage<vsip::scalar_i>
		stor_scalari  (input_length);
//  vsip::Vector<vsip::scalar_i>
//		vector_scalari (input_length);

  for (vsip::index_type idx = input_length; idx-- > 0; ) {
    vector_indices.put (idx, vsip::Index<1>(0));
    stor_scalari.block().put (idx, static_cast<vsip::scalar_i>(idx));
  }
  vsip::Vector<vsip::scalar_i>
		gather_scalari (vsip::gather (stor_scalari.view,
					      vector_indices));
  for (vsip::index_type idx = 0; idx < input_length; ++idx)
    insist (gather_scalari.get (idx) == static_cast<vsip::scalar_i>(0));
  for (vsip::index_type idx = input_length; idx-- > 0; )
    vector_indices.put (idx, vsip::Index<1>(idx));
  gather_scalari = vsip::gather (stor_scalari.view, vector_indices);
  for (vsip::index_type idx = 0; idx < input_length; ++idx)
    insist (gather_scalari.get (idx) == static_cast<vsip::scalar_i>(idx));
}



// -------------------------------------------------------------------- //
template <template <typename> class Storage>
void
test_mgather()
{
   // Require that storage is 2-dimensional
  assert(Storage<vsip::cscalar_f>::dim == 2);

  vsip::length_type const
		input_length = 17;

  Storage<vsip::cscalar_f>
		stor_cscalarf  (input_length);

  vsip::Vector<vsip::Index<2> >
		matrix_indices (input_length);

//  vsip::Matrix<vsip::cscalar_f>
//		matrix_cscalarf (input_length, input_length);

  for (vsip::index_type row_idx = input_length; row_idx-- > 0; ) {
    for (vsip::index_type column_idx = input_length; column_idx-- > 0; )
       stor_cscalarf.block().put (row_idx, column_idx, vsip::cscalar_f (2.0));
  }
  stor_cscalarf.block().put (0, 0, vsip::cscalar_f (1.0));
  for (vsip::index_type idx = input_length; idx-- > 0; )
    matrix_indices.put (idx, vsip::Index<2>(idx, idx));
  vsip::Vector<vsip::cscalar_f>
		gather_cscalarf (vsip::gather (stor_cscalarf.view,
					       matrix_indices));
  insist (gather_cscalarf.get (0) == vsip::cscalar_f (1.0));
  for (vsip::index_type idx = 1; idx < input_length; ++idx)
    insist (gather_cscalarf.get (idx) == vsip::cscalar_f (2.0));
}



int
main (int argc, char** argv)
{
  vsip::vsipl	v(argc, argv);

  /* Begin testing of first ().  */

  /* The vectors will contain default values at all positions except
     position 9.  */
  vsip::length_type const
		input_length = 17;
			/* The length of vectors to test.  */
  vsip::index_type	first_index;
			/* The result of first ().  */
  vsip::index_type const
		answer_index = 9;
			/* The index where the answer should occur.  */

  /* Test first () on a vector of scalar_f.  */
  vsip::Vector<>
		vector_scalar_f_one (input_length, vsip::scalar_f ());
  vector_scalar_f_one.put (answer_index, static_cast<vsip::scalar_f>(1.0));
  vsip::Vector<>
		vector_scalar_f_two (input_length,
				     static_cast<vsip::scalar_f>(1.0));
  first_index = vsip::first (3, &equalFloat,
			     vector_scalar_f_one, vector_scalar_f_two);
  insist (first_index == answer_index);
  first_index = vsip::first (answer_index, &equalFloat,
			     vector_scalar_f_one, vector_scalar_f_two);
  insist (first_index == answer_index);
  first_index = vsip::first (answer_index + 1, &equalFloat,
			     vector_scalar_f_one, vector_scalar_f_two);
  insist (first_index == input_length);
  first_index = vsip::first (input_length, &equalFloat,
			     vector_scalar_f_one, vector_scalar_f_two);
  insist (first_index == input_length);

  /* Test indexbool ().  */
  test_indexbool<     VectorStorage<bool> >();
  test_indexbool<ConstVectorStorage<bool> >();


  /* Test gather.  */
  test_vgather<     VectorStorage>();
  test_vgather<ConstVectorStorage>();
  test_mgather<     MatrixStorage>();
  test_mgather<ConstMatrixStorage>();

  /* Test scatter.  */
  vsip::Vector<vsip::Index<1> >
		vector_indices (input_length);
			/* The vector to store the non-false indices.  */

  vsip::Vector<vsip::scalar_f>
		scatter_scalarf (input_length, 0.0);
  vsip::Vector<vsip::scalar_f>
		vector_scalarf (input_length);
  for (vsip::index_type idx = input_length; idx-- > 0; ) {
    vector_scalarf.put (idx, static_cast<vsip::scalar_f>(idx));
    vector_indices.put (idx, vsip::Index<1>(idx));
  }
  vsip::scatter (vector_scalarf, vector_indices, scatter_scalarf);
  for (vsip::index_type idx = input_length; idx-- > 0; )
    insist (equal (scatter_scalarf.get (idx),
		   static_cast<vsip::scalar_f>(idx)));

  /* Test ramp () to generate a vector of values.  */

  vsip::Vector<>
		vector_ramp_f (vsip::ramp (static_cast<vsip::scalar_f>(0.0),
					   static_cast<vsip::scalar_f>(1.0),
					   input_length));
  insist (vector_ramp_f.size () == input_length);
  insist (vector_ramp_f.get (0) == static_cast<vsip::scalar_f>(0.0));
  insist (vector_ramp_f.get (3) == static_cast<vsip::scalar_f>(3.0));
  insist (vector_ramp_f.get (input_length - 1)
	  == static_cast<vsip::scalar_f>(input_length - 1));

  /* Test clipping.  */

  vsip::Vector<vsip::scalar_i>
		vector_clip_i (vsip::ramp (static_cast<vsip::scalar_i>(0),
					   static_cast<vsip::scalar_i>(1),
					   input_length));

  /* Test clipping of a vector.  */

  vsip::Vector<vsip::scalar_i> vector_clip_answer_i(input_length);
  
  vector_clip_answer_i = clip(vector_clip_i, 3, 15, -73, 73);

  insist (vector_clip_answer_i.size () == input_length);
  insist (vector_clip_answer_i.get (0) == -73);
  insist (vector_clip_answer_i.get (3) == -73);
  insist (vector_clip_answer_i.get (4) == 4);
  insist (vector_clip_answer_i.get (14) == 14);
  insist (vector_clip_answer_i.get (15) == 73);
  insist (vector_clip_answer_i.get (vector_clip_answer_i.size () - 1) == 73);

  /* Test inverse clipping.  */

  vsip::Vector<vsip::scalar_f>
		vector_invclip_f (vsip::ramp (static_cast<vsip::scalar_f>(0.0),
					      static_cast<vsip::scalar_f>(1.0),
					      input_length));

  /* Test inverse clipping of a vector.  */

  vsip::Vector<vsip::scalar_f> vector_invclip_answer_f (input_length);

  vector_invclip_answer_f  = invclip(vector_invclip_f, 3., 9., 15., -73., 73.);

  insist (vector_invclip_answer_f.size () == input_length);
  insist (vector_invclip_answer_f.get (0) == 0.0);
  insist (vector_invclip_answer_f.get (2) == 2.0);
  insist (vector_invclip_answer_f.get (3) == -73.0);
  insist (vector_invclip_answer_f.get (8) == -73.0);
  insist (vector_invclip_answer_f.get (9) == 73.0);
  insist (vector_invclip_answer_f.get (15) == 73.0);
  insist (vector_invclip_answer_f.get (16) == 16.0);

  /* Test swapping.  */

  vsip::Vector<vsip::scalar_i>
		vector_swap_i_one (vsip::ramp
				   (static_cast<vsip::scalar_i>(0),
				    static_cast<vsip::scalar_i>(1),
				    input_length));
  vsip::Vector<vsip::scalar_i>
		vector_swap_i_two (vsip::ramp
				   (static_cast<vsip::scalar_i>(input_length),
				    static_cast<vsip::scalar_i>(1),
				    input_length));

  vsip::swap (vector_swap_i_one, vector_swap_i_two);
  insist (vector_swap_i_one.get (0) ==
	  static_cast<vsip::scalar_i>(input_length));
  insist (vector_swap_i_two.get (0) == 0);
  insist (vector_swap_i_one.get (1)
	  == vector_swap_i_two.get (1)
	  + static_cast<vsip::scalar_i>(input_length));
  insist (vector_swap_i_one.get (input_length - 1)
	  == vector_swap_i_two.get (input_length - 1)
	  + static_cast<vsip::scalar_i>(input_length));

  /* End testing of first ().  */

  return EXIT_SUCCESS;
}
