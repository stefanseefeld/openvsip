/***********************************************************************

  File:   view-math.cpp
  Author: Jules Bergmann, CodeSourcery, LLC.
  Date:   11/23/2002

  Contents: Test view (vector, matrix) math

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

#include <iostream>
#include <cstdlib>
#include "test.hpp"
#include <vsip/complex.hpp>
#include <vsip/initfin.hpp>
#include <vsip/math.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include "test-util.hpp"

using namespace vsip;

#define TEST_XMA 0
#define HAVE_GENERAL_MA 0



/***********************************************************************
  Function Definitions
***********************************************************************/

template <typename T>
T
test_hypot(T x, T y)
{
  return std::sqrt(x * x + y * y);
}

// -------------------------------------------------------------------- //
// test vector combinations
template <typename					T,
	  template <typename, typename> class TClass>
void
test_v()
{
   typedef VectorStorage<T>
		VecT;
   typedef ConstVectorStorage<T>
		ConstVecT;

   TClass<     VecT, VecT>::test();
   TClass<ConstVecT, VecT>::test();
}



// -------------------------------------------------------------------- //
// test vector combinations
template <typename			      T,
	  typename			      TR,
	  template <typename, typename> class TClass>
void
test_tr_v()
{
   typedef VectorStorage<T>
		VecT;
   typedef ConstVectorStorage<T>
		ConstVecT;
   typedef VectorStorage<TR>
		VecTR;

   TClass<     VecT, VecTR>::test();
   TClass<ConstVecT, VecTR>::test();
}



// -------------------------------------------------------------------- //
// test matrix combinations
template <typename			      T,
	  typename			      TR,
	  template <typename, typename> class TClass>
void
test_tr_m()
{
   typedef MatrixStorage<T>
		MatT;
   typedef ConstMatrixStorage<T>
		ConstMatT;
   typedef MatrixStorage<TR>
		MatTR;

   TClass<     MatT, MatTR>::test();
   TClass<ConstMatT, MatTR>::test();
}



// -------------------------------------------------------------------- //
// test matrix combinations
template <typename			      T,
	  template <typename, typename> class TClass>
void
test_m()
{
   test_tr_m<T, T, TClass>();
}



// -------------------------------------------------------------------- //
// test vector/vector combinations
template <typename					T,
	  template <typename, typename, typename> class TClass>
void
test_vv()
{
   typedef VectorStorage<T>
		VecT;
   typedef ConstVectorStorage<T>
		ConstVecT;

   TClass<     VecT,      VecT, VecT>::test();
   TClass<     VecT, ConstVecT, VecT>::test();
   TClass<ConstVecT,      VecT, VecT>::test();
   TClass<ConstVecT, ConstVecT, VecT>::test();
}



// -------------------------------------------------------------------- //
// test vector/vector combinations (with separate types)
//
template <typename					T1,
	  typename					T2,
	  template <typename, typename, typename> class TClass>
void
test_tt_vv()
{
   typedef VectorStorage<T1>
		VecT1;
   typedef ConstVectorStorage<T1>
		ConstVecT1;
   typedef VectorStorage<T2>
		VecT2;
   typedef ConstVectorStorage<T2>
		ConstVecT2;

   typedef typename Promotion<T1, T2>::type
		TR;

   typedef VectorStorage<TR>
		VecTR;


   TClass<     VecT1,      VecT2, VecTR>::test();
   TClass<     VecT1, ConstVecT2, VecTR>::test();
   TClass<ConstVecT1,      VecT2, VecTR>::test();
   TClass<ConstVecT1, ConstVecT2, VecTR>::test();
}



// -------------------------------------------------------------------- //
// test vector/vector combinations (with separate types)
//
template <typename					T1,
	  typename					T2,
	  typename					TR,
	  template <typename, typename, typename> class TClass>
void
test_ttr_vv()
{
   typedef VectorStorage<T1>
		VecT1;
   typedef ConstVectorStorage<T1>
		ConstVecT1;
   typedef VectorStorage<T2>
		VecT2;
   typedef ConstVectorStorage<T2>
		ConstVecT2;

   typedef VectorStorage<TR>
		VecTR;


   TClass<     VecT1,      VecT2, VecTR>::test();
   TClass<     VecT1, ConstVecT2, VecTR>::test();
   TClass<ConstVecT1,      VecT2, VecTR>::test();
   TClass<ConstVecT1, ConstVecT2, VecTR>::test();
}



// -------------------------------------------------------------------- //
// test vector/vector combinations (with separate types)
//
template <typename					T1,
	  typename					T2,
	  template <typename, typename, typename> class TClass>
void
test_tt_vv_noconst()
{
   typedef VectorStorage<T1>
		VecT1;
   typedef VectorStorage<T2>
		VecT2;

   typedef typename Promotion<T1, T2>::type
		TR;

   typedef VectorStorage<TR>
		VecTR;

   TClass<     VecT1,      VecT2, VecTR>::test();
}



// -------------------------------------------------------------------- //
template <typename					T1,
	  typename					T2,
	  template <typename, typename, typename> class TClass>
void
test_tt_sv()
{
   typedef typename Promotion<T1, T2>::type
		TR;

   typedef ScalarStorage<T1>
		ScaT1;

   typedef VectorStorage<T2>
		VecT2;
   typedef ConstVectorStorage<T2>
		ConstVecT2;

   typedef VectorStorage<TR>
		VecTR;

   TClass<ScaT1,      VecT2, VecTR>::test();
   TClass<ScaT1, ConstVecT2, VecTR>::test();
}



// -------------------------------------------------------------------- //
template <typename					T,
	  template <typename, typename, typename> class TClass>
void
test_sv()
{
   test_tt_sv<T, T, TClass>();
}



// -------------------------------------------------------------------- //
template <typename					T1,
	  typename					T2,
	  template <typename, typename, typename> class TClass>
void
test_tt_vs()
{
   typedef typename Promotion<T1, T2>::type
		TR;

   typedef VectorStorage<T1>
		VecT1;
   typedef ConstVectorStorage<T1>
		ConstVecT1;

   typedef ScalarStorage<T2>
		ScaT2;

   typedef VectorStorage<TR>
		VecTR;

   TClass<     VecT1, ScaT2, VecTR>::test();
   TClass<ConstVecT1, ScaT2, VecTR>::test();
}



// -------------------------------------------------------------------- //
template <typename					T,
	  template <typename, typename, typename> class TClass>
void
test_vs()
{
   test_tt_vs<T, T, TClass>();
}



// -------------------------------------------------------------------- //
// test matrix/matrix combinations
template <typename					T,
	  template <typename, typename, typename> class TClass>
void
test_mm()
{
   typedef MatrixStorage<T>
		MatT;
   typedef ConstMatrixStorage<T>
		ConstMatT;

   TClass<     MatT,      MatT, MatT>::test();
   TClass<     MatT, ConstMatT, MatT>::test();
   TClass<ConstMatT,      MatT, MatT>::test();
   TClass<ConstMatT, ConstMatT, MatT>::test();
}



// -------------------------------------------------------------------- //
// test matrix/matrix combinations
template <typename					T1,
	  typename					T2,
	  template <typename, typename, typename> class TClass>
void
test_tt_mm()
{
   typedef MatrixStorage<T1>
		MatT1;
   typedef ConstMatrixStorage<T1>
		ConstMatT1;
   typedef MatrixStorage<T2>
		MatT2;
   typedef ConstMatrixStorage<T2>
		ConstMatT2;

   typedef typename Promotion<T1, T2>::type
		TR;

   typedef MatrixStorage<TR>
		MatTR;

   TClass<     MatT1,      MatT2, MatTR>::test();
   TClass<     MatT1, ConstMatT2, MatTR>::test();
   TClass<ConstMatT1,      MatT2, MatTR>::test();
   TClass<ConstMatT1, ConstMatT2, MatTR>::test();
}



// -------------------------------------------------------------------- //
// test matrix/matrix combinations
template <typename					T1,
	  typename					T2,
	  typename					TR,
	  template <typename, typename, typename> class TClass>
void
test_ttr_mm()
{
   typedef MatrixStorage<T1>
		MatT1;
   typedef ConstMatrixStorage<T1>
		ConstMatT1;
   typedef MatrixStorage<T2>
		MatT2;
   typedef ConstMatrixStorage<T2>
		ConstMatT2;

   typedef MatrixStorage<TR>
		MatTR;

   TClass<     MatT1,      MatT2, MatTR>::test();
   TClass<     MatT1, ConstMatT2, MatTR>::test();
   TClass<ConstMatT1,      MatT2, MatTR>::test();
   TClass<ConstMatT1, ConstMatT2, MatTR>::test();
}



// -------------------------------------------------------------------- //
template <typename					T1,
	  typename					T2,
	  template <typename, typename, typename> class TClass>
void
test_tt_sm()
{
   typedef typename Promotion<T1, T2>::type
		TR;

   typedef ScalarStorage<T1>
		ScaT1;

   typedef MatrixStorage<T2>
		MatT2;
   typedef ConstMatrixStorage<T2>
		ConstMatT2;

   typedef MatrixStorage<TR>
		MatTR;

   TClass<ScaT1,      MatT2, MatTR>::test();
   TClass<ScaT1, ConstMatT2, MatTR>::test();
}



// -------------------------------------------------------------------- //
template <typename					T,
	  template <typename, typename, typename> class TClass>
void
test_sm()
{
   test_tt_sm<T, T, TClass>();
}



// -------------------------------------------------------------------- //
template <typename					T1,
	  typename					T2,
	  template <typename, typename, typename> class TClass>
void
test_tt_ms()
{
   typedef typename Promotion<T1, T2>::type
		TR;

   typedef MatrixStorage<T1>
		MatT1;
   typedef ConstMatrixStorage<T1>
		ConstMatT1;

   typedef ScalarStorage<T2>
		ScaT2;

   typedef MatrixStorage<TR>
		MatTR;

   TClass<     MatT1, ScaT2, MatTR>::test();
   TClass<ConstMatT1, ScaT2, MatTR>::test();
}



// -------------------------------------------------------------------- //
template <typename					T,
	  template <typename, typename, typename> class TClass>
void
test_ms()
{
   test_tt_ms<T, T, TClass>();
}



// -------------------------------------------------------------------- //
template <typename						  T,
	  template <typename, typename, typename, typename> class TClass>
void test_vvv()
{
   typedef VectorStorage<T>
		VecT;
   typedef ConstVectorStorage<T>
		ConstVecT;

   TClass<     VecT,      VecT,      VecT, VecT>::test();
   TClass<     VecT,      VecT, ConstVecT, VecT>::test();
   TClass<     VecT, ConstVecT,      VecT, VecT>::test();
   TClass<     VecT, ConstVecT, ConstVecT, VecT>::test();
   TClass<ConstVecT,      VecT,      VecT, VecT>::test();
   TClass<ConstVecT,      VecT, ConstVecT, VecT>::test();
   TClass<ConstVecT, ConstVecT,      VecT, VecT>::test();
   TClass<ConstVecT, ConstVecT, ConstVecT, VecT>::test();
}



// -------------------------------------------------------------------- //
template <typename						  T,
	  template <typename, typename, typename, typename> class TClass>
void test_vsv()
{
   typedef VectorStorage<T>
		VecT;
   typedef ConstVectorStorage<T>
		ConstVecT;
   typedef ScalarStorage<T>
		ScaT;

   TClass<     VecT,      ScaT,      VecT, VecT>::test();
   TClass<     VecT,      ScaT, ConstVecT, VecT>::test();
   TClass<ConstVecT,      ScaT,      VecT, VecT>::test();
   TClass<ConstVecT,      ScaT, ConstVecT, VecT>::test();
}



// -------------------------------------------------------------------- //
template <typename						  T,
	  template <typename, typename, typename, typename> class TClass>
void test_vvs()
{
   typedef VectorStorage<T>
		VecT;
   typedef ConstVectorStorage<T>
		ConstVecT;
   typedef ScalarStorage<T>
		ScaT;

   TClass<     VecT,      VecT, ScaT, VecT>::test();
   TClass<     VecT, ConstVecT, ScaT, VecT>::test();
   TClass<ConstVecT,      VecT, ScaT, VecT>::test();
   TClass<ConstVecT, ConstVecT, ScaT, VecT>::test();
}



// -------------------------------------------------------------------- //
template <typename						  T,
	  template <typename, typename, typename, typename> class TClass>
void test_vss()
{
   typedef VectorStorage<T>
		VecT;
   typedef ConstVectorStorage<T>
		ConstVecT;
   typedef ScalarStorage<T>
		ScaT;

   TClass<     VecT, ScaT, ScaT, VecT>::test();
   TClass<ConstVecT, ScaT, ScaT, VecT>::test();
}



/* -------------------------------------------------------------------- *
 * Tests for Add -- vector addition					*
 * -------------------------------------------------------------------- */

// -------------------------------------------------------------------- //
// tc_add -- Test addition .
template <typename TStor1,
	  typename TStor2,
	  typename TStorR>
struct tc_add
{
   static void test()
   {
      int const	N = 7;

      typedef typename TStor1::value_type T1;
      typedef typename TStor2::value_type T2;

      TStor1	stor1(N, TestRig<T1>::test_value1());
      TStor2	stor2(N, TestRig<T2>::test_value2());
      TStorR	storR(N);

      // storR.view = stor1.view + stor2.view;
      storR.view = add(stor1.view, stor2.view);

      insist(equal(get_origin(storR.view),
		   TestRig<T1>::test_value1() + TestRig<T2>::test_value2()));
   }
};



void
test_add()
{
   test_tt_vv< scalar_i,  scalar_i, tc_add>(); // QRBS?
   test_tt_vv< scalar_f,  scalar_f, tc_add>();
   test_tt_vv< scalar_f, cscalar_f, tc_add>();
   test_tt_vv<cscalar_f, cscalar_f, tc_add>();

   test_tt_mm< scalar_f,  scalar_f, tc_add>();
   test_tt_mm< scalar_f, cscalar_f, tc_add>();
   test_tt_mm<cscalar_f, cscalar_f, tc_add>();

   test_tt_sv< scalar_i,  scalar_i, tc_add>();
   test_tt_sv< scalar_f,  scalar_f, tc_add>();
   test_tt_sv< scalar_f, cscalar_f, tc_add>();
   test_tt_sv<cscalar_f, cscalar_f, tc_add>();

   test_tt_sm< scalar_f,  scalar_f, tc_add>();
   test_tt_sm< scalar_f, cscalar_f, tc_add>();
   test_tt_sm<cscalar_f, cscalar_f, tc_add>();

   // NRBS:
   //  - VS
   // test_tt_vv<cscalar_f,  scalar_f, tc_add>();
   //
   // test_tt_sm< scalar_i,  scalar_i, tc_add>();
}



// -------------------------------------------------------------------- //
// tc_sub -- Test subtraction .
template <typename TStor1,
	  typename TStor2,
	  typename TStorR>
struct tc_sub
{
   static void test()
   {
      int const	N = 7;

      typedef typename TStor1::value_type T1;
      typedef typename TStor2::value_type T2;

      TStor1	stor1(N, TestRig<T1>::test_value1());
      TStor2	stor2(N, TestRig<T2>::test_value2());
      TStorR	storR(N);

      storR.view = stor1.view - stor2.view;

      insist(equal(get_origin(storR.view),
		   TestRig<T1>::test_value1() - TestRig<T2>::test_value2()));
   }
};



// -------------------------------------------------------------------- //
// tc_mul -- Test multiplication.
template <typename TStor1,
	  typename TStor2,
	  typename TStorR>
struct tc_mul
{
   static void test()
   {
      int const	N = 7;

      typedef typename TStor1::value_type T1;
      typedef typename TStor2::value_type T2;

      TStor1	stor1(N, TestRig<T1>::test_value1());
      TStor2	stor2(N, TestRig<T2>::test_value2());
      TStorR	storR(N);

      storR.view = stor1.view * stor2.view;
      // storR.view = mul(stor1.view, stor2.view);

      insist(equal(get_origin(storR.view),
		   TestRig<T1>::test_value1() * TestRig<T2>::test_value2()));
   }
};



// -------------------------------------------------------------------- //
// tc_div -- Test division.
template <typename TStor1,
	  typename TStor2,
	  typename TStorR>
struct tc_div
{
   static void test()
   {
      int const	N = 7;

      typedef typename TStor1::value_type T1;
      typedef typename TStor2::value_type T2;

      TStor1	stor1(N, TestRig<T1>::test_value1());
      TStor2	stor2(N, TestRig<T2>::test_value2());
      TStorR	storR(N);

      storR.view = stor1.view / stor2.view;
      // storR.view = div(stor1.view, stor2.view);

      insist(equal(get_origin(storR.view),
		   TestRig<T1>::test_value1() / TestRig<T2>::test_value2()));
   }
};



// -------------------------------------------------------------------- //
// tc_jmul -- Test jmultiplication.
template <typename TStor1,
	  typename TStor2,
	  typename TStorR>
struct tc_jmul
{
   static void test()
   {
      int const	N = 7;

      typedef typename TStor1::value_type T1;
      typedef typename TStor2::value_type T2;

      TStor1	stor1(N, TestRig<T1>::test_value1());
      TStor2	stor2(N, TestRig<T2>::test_value2());
      TStorR	storR(N);

      storR.view = jmul(stor1.view, stor2.view);

      insist(equal(get_origin(storR.view),
		   vsip::jmul(TestRig<T1>::test_value1(),
			      TestRig<T2>::test_value2()) ));
   }
};



// -------------------------------------------------------------------- //
// tc_acos -- Test acos.
template <typename TStor1,
	  typename TStorR>
struct tc_acos
{
   static void test()
   {
      int const	N = 7;

      typedef typename TStor1::value_type T1;

      TStor1	stor1(N, TestRig<T1>::test_value1());
      TStorR	storR(N);

      storR.view = acos(stor1.view);

      insist(equal(get_origin(storR.view),
		   acos(TestRig<T1>::test_value1() )));
   }
};


void test_acos()
{
   test_v< scalar_f, tc_acos>();
   test_m< scalar_f, tc_acos>();
}



// -------------------------------------------------------------------- //
// tc_arg -- Test arg.
template <typename TStor1,
	  typename TStorR>
struct tc_arg
{
   static void test()
   {
      int const	N = 7;

      typedef typename TStor1::value_type T1;

      TStor1	stor1(N, TestRig<T1>::test_value1());
      TStorR	storR(N);

      storR.view = arg(stor1.view);

      insist(equal(get_origin(storR.view),
		   arg(TestRig<T1>::test_value1() )));
   }
};


void test_arg()
{
   test_tr_v<cscalar_f, scalar_f, tc_arg>();
   test_tr_m<cscalar_f, scalar_f, tc_arg>();
}

// -------------------------------------------------------------------- //
// tc_atan2 -- Test atan2.
template <typename TStor1,
	  typename TStor2,
	  typename TStorR>
struct tc_atan2
{
   static void test()
   {
      int const	N = 7;

      typedef typename TStor1::value_type T1;
      typedef typename TStor2::value_type T2;

      TStor1	stor1(N, TestRig<T1>::test_value1());
      TStor2	stor2(N, TestRig<T2>::test_value2());
      TStorR	storR(N);

      storR.view = atan2(stor1.view, stor2.view);

      insist(equal(get_origin(storR.view),
		   atan2(TestRig<T1>::test_value1(),
			 TestRig<T2>::test_value2())));
   }
};


void test_atan2()
{
   test_tt_vv< scalar_f,  scalar_f, tc_atan2>();
   test_tt_mm< scalar_f,  scalar_f, tc_atan2>();
}




// -------------------------------------------------------------------- //
// tc_hypot -- Test hypot.
template <typename TStor1,
	  typename TStor2,
	  typename TStorR>
struct tc_hypot
{
   static void test()
   {
      int const	N = 7;

      typedef typename TStor1::value_type T1;
      typedef typename TStor2::value_type T2;

      TStor1	stor1(N, TestRig<T1>::test_value1());
      TStor2	stor2(N, TestRig<T2>::test_value2());
      TStorR	storR(N);

      storR.view = hypot(stor1.view, stor2.view);

      insist(equal(get_origin(storR.view),
		   test_hypot(TestRig<T1>::test_value1(),
			      TestRig<T2>::test_value2())));
   }
};


void test_hypot()
{
   test_tt_vv< scalar_f,  scalar_f, tc_hypot>();
   test_tt_mm< scalar_f,  scalar_f, tc_hypot>();
}



// -------------------------------------------------------------------- //
// tc_max, tc_min -- Test min()/max() .

template <typename TStor1,
	  typename TStor2,
	  typename TStorR>
struct tc_max
{
   static void test()
   {
      int const	N = 7;

      typedef typename TStor1::value_type T1;
      typedef typename TStor2::value_type T2;

      TStor1	stor1(N, TestRig<T1>::test_value1());
      TStor2	stor2(N, TestRig<T2>::test_value2());
      TStorR	storR(N);

      storR.view = max(stor1.view, stor2.view);

      insist(equal(get_origin(storR.view),
		   vsip::max(TestRig<T1>::test_value1(),
			     TestRig<T2>::test_value2())));
   }
};



template <typename TStor1,
	  typename TStor2,
	  typename TStorR>
struct tc_min
{
   static void test()
   {
      int const	N = 7;

      typedef typename TStor1::value_type T1;
      typedef typename TStor2::value_type T2;

      TStor1	stor1(N, TestRig<T1>::test_value1());
      TStor2	stor2(N, TestRig<T2>::test_value2());
      TStorR	storR(N);

      storR.view = min(stor1.view, stor2.view);

      insist(equal(get_origin(storR.view),
		   vsip::min(TestRig<T1>::test_value1(),
			     TestRig<T2>::test_value2())));
   }
};



void test_minmax()
{
   test_tt_vv< scalar_f,  scalar_f, tc_max>();
   test_tt_mm< scalar_f,  scalar_f, tc_max>();

   test_tt_vv< scalar_f,  scalar_f, tc_min>();
   test_tt_mm< scalar_f,  scalar_f, tc_min>();

   // NO cscalar_f or scalar_i?
}



// -------------------------------------------------------------------- //
// tc_maxmg -- Test maxmg .
template <typename TStor1,
	  typename TStor2,
	  typename TStorR>
struct tc_maxmg
{
   static void test()
   {
      int const	N = 7;

      typedef typename TStor1::value_type T1;
      typedef typename TStor2::value_type T2;

      TStor1	stor1(N, TestRig<T1>::test_value1());
      TStor2	stor2(N, TestRig<T2>::test_value2());
      TStorR	storR(N);

      storR.view = maxmg(stor1.view, stor2.view);

      insist(equal(get_origin(storR.view),
		   vsip::maxmg(TestRig<T1>::test_value1(),
			       TestRig<T2>::test_value2())));
   }
};



void test_maxmg()
{
   test_tt_vv< scalar_f,  scalar_f, tc_maxmg>();
   test_tt_mm< scalar_f,  scalar_f, tc_maxmg>();
}



// -------------------------------------------------------------------- //
// tc_minmg -- Test minmg .
template <typename TStor1,
	  typename TStor2,
	  typename TStorR>
struct tc_minmg
{
   static void test()
   {
      int const	N = 7;

      typedef typename TStor1::value_type T1;
      typedef typename TStor2::value_type T2;

      TStor1	stor1(N, TestRig<T1>::test_value1());
      TStor2	stor2(N, TestRig<T2>::test_value2());
      TStorR	storR(N);

      storR.view = minmg(stor1.view, stor2.view);

      insist(equal(get_origin(storR.view),
		   vsip::minmg(TestRig<T1>::test_value1(),
			       TestRig<T2>::test_value2())));
   }
};



void test_minmg()
{
   test_tt_vv< scalar_f,  scalar_f, tc_minmg>();
   test_tt_mm< scalar_f,  scalar_f, tc_minmg>();
}



// -------------------------------------------------------------------- //
// tc_maxmgsq -- Test maxmgsq.
template <typename TStor1,
	  typename TStor2,
	  typename TStorR>
struct tc_maxmgsq
{
   static void test()
   {
      int const	N = 7;

      typedef typename TStor1::value_type T1;
      typedef typename TStor2::value_type T2;

      TStor1	stor1(N, TestRig<T1>::test_value1());
      TStor2	stor2(N, TestRig<T2>::test_value2());
      TStorR	storR(N);

      storR.view = maxmgsq(stor1.view, stor2.view);

      insist(equal(get_origin(storR.view),
		   vsip::maxmgsq(TestRig<T1>::test_value1(),
				 TestRig<T2>::test_value2())));
   }
};



void test_maxmgsq()
{
   test_ttr_vv<cscalar_f, cscalar_f, scalar_f, tc_maxmgsq>();
   test_ttr_mm<cscalar_f, cscalar_f, scalar_f, tc_maxmgsq>();
}



// -------------------------------------------------------------------- //
// tc_minmgsq -- Test minmgsq.
template <typename TStor1,
	  typename TStor2,
	  typename TStorR>
struct tc_minmgsq
{
   static void test()
   {
      int const	N = 7;

      typedef typename TStor1::value_type T1;
      typedef typename TStor2::value_type T2;

      TStor1	stor1(N, TestRig<T1>::test_value1());
      TStor2	stor2(N, TestRig<T2>::test_value2());
      TStorR	storR(N);

      storR.view = minmgsq(stor1.view, stor2.view);

      insist(equal(get_origin(storR.view),
		   vsip::minmgsq(TestRig<T1>::test_value1(),
				 TestRig<T2>::test_value2())));
   }
};



void test_minmgsq()
{
   test_ttr_vv<cscalar_f, cscalar_f, scalar_f, tc_minmgsq>();
   test_ttr_mm<cscalar_f, cscalar_f, scalar_f, tc_minmgsq>();
}



// -------------------------------------------------------------------- //
// tc_mag -- Test mag (logical and).
template <typename TStor1,
	  typename TStorR>
struct tc_mag
{
   static void test()
   {
      int const	N = 7;

      typedef typename TStor1::value_type T1;

      typedef T1 RT;

      TStor1	stor1(N, TestRig<T1>::test_value1());
      TStorR	storR(N);

      storR.view = mag(stor1.view);

      insist(equal(get_origin(storR.view),
		   vsip::mag(TestRig<T1>::test_value1())));
   }
};



// -------------------------------------------------------------------- //
// tc_am -- Test vector add multiply.
template <typename TStor1,
	  typename TStor2,
	  typename TStor3,
	  typename TStorR>
struct tc_am
{
   static void test()
   {
      int const	N = 7;

      typedef typename TStor1::value_type T1;
      typedef typename TStor2::value_type T2;
      typedef typename TStor3::value_type T3;

      TStor1	stor1(N, TestRig<typename TStor1::value_type>::test_value1());
      TStor2	stor2(N, TestRig<typename TStor2::value_type>::test_value2());
      TStor3	stor3(N, TestRig<typename TStor3::value_type>::test_value3());
      TStorR	storR(N);

      storR.view = am(stor1.view, stor2.view, stor3.view);

      insist(equal(get_origin(storR.view),
		   (  TestRig<T1>::test_value1() + TestRig<T2>::test_value2())
		   * TestRig<T3>::test_value3()));
   }
};









// -------------------------------------------------------------------- //
// test add-multiply cases
void test_am()
{
   test_vvv< scalar_f, tc_am>();
   test_vvv<cscalar_f, tc_am>();

   test_vsv< scalar_f, tc_am>();
   test_vsv<cscalar_f, tc_am>();

   // NRBS:
   //  - types other than scalar_f and cscalar_f
   //  - views other than vector
   //  - VVS or VSS
}



// -------------------------------------------------------------------- //
// tc_ma -- Test multiply-add.
template <typename TStor1,
	  typename TStor2,
	  typename TStor3,
	  typename TStorR>
struct tc_ma
{
   static void test()
   {
      int const	N = 7;

      typedef typename TStor1::value_type T1;
      typedef typename TStor2::value_type T2;
      typedef typename TStor3::value_type T3;

      TStor1	stor1(N, TestRig<typename TStor1::value_type>::test_value1());
      TStor2	stor2(N, TestRig<typename TStor2::value_type>::test_value2());
      TStor3	stor3(N, TestRig<typename TStor3::value_type>::test_value3());
      TStorR	storR(N);

      storR.view = ma(stor1.view, stor2.view, stor3.view);

      insist(equal(get_origin(storR.view),
		   (  TestRig<T1>::test_value1() * TestRig<T2>::test_value2())
		   + TestRig<T3>::test_value3()));
   }
};



// test multiply-add cases
void test_ma()
{
   test_vvv< scalar_f, tc_ma>();
   test_vvv<cscalar_f, tc_ma>();

   test_vsv< scalar_f, tc_ma>();
   test_vsv<cscalar_f, tc_ma>();

   test_vvs< scalar_f, tc_ma>();
   test_vvs<cscalar_f, tc_ma>();

   test_vss< scalar_f, tc_ma>();
   test_vss<cscalar_f, tc_ma>();

   // NRBS:
   //  - types other than scalar_f and cscalar_f
   //  - views other than vector
}



// -------------------------------------------------------------------- //
// tc_msb -- Test multiply-subtract.
template <typename TStor1,
	  typename TStor2,
	  typename TStor3,
	  typename TStorR>
struct tc_msb
{
   static void test()
   {
      int const	N = 7;

      typedef typename TStor1::value_type T1;
      typedef typename TStor2::value_type T2;
      typedef typename TStor3::value_type T3;

      TStor1	stor1(N, TestRig<typename TStor1::value_type>::test_value1());
      TStor2	stor2(N, TestRig<typename TStor2::value_type>::test_value2());
      TStor3	stor3(N, TestRig<typename TStor3::value_type>::test_value3());
      TStorR	storR(N);

      storR.view = msb(stor1.view, stor2.view, stor3.view);

      insist(equal(get_origin(storR.view),
		   (  TestRig<T1>::test_value1() * TestRig<T2>::test_value2())
		   - TestRig<T3>::test_value3()));
   }
};



// test multiply-add cases
void test_msb()
{
   test_vvv< scalar_f, tc_msb>();
   test_vvv<cscalar_f, tc_msb>();

   // Only msb VVV is RBS
}



// -------------------------------------------------------------------- //
// tc_xma -- Test (ExperiMENTAL) multiply-add.

#if TEST_XMA
template <typename TStor1,
	  typename TStor2,
	  typename TStor3,
	  typename TStorR>
struct tc_xma
{
   static void test()
   {
      int const	N = 7;

      typedef typename TStor1::value_type T1;
      typedef typename TStor2::value_type T2;
      typedef typename TStor3::value_type T3;

      TStor1	stor1(N, TestRig<typename TStor1::value_type>::test_value1());
      TStor2	stor2(N, TestRig<typename TStor2::value_type>::test_value2());
      TStor3	stor3(N, TestRig<typename TStor3::value_type>::test_value3());
      TStorR	storR(N);

      storR.view = xma(stor1.view, stor2.view, stor3.view);

      insist(equal(get_origin(storR.view),
		   (  TestRig<T1>::test_value1() * TestRig<T2>::test_value2())
		   + TestRig<T3>::test_value3()));
   }
};
#endif


// test multiply-add cases
void test_xma()
{
#if TEST_XMA
   test_vvv< scalar_f, tc_xma>();
   test_vvv<cscalar_f, tc_xma>();

   test_vsv< scalar_f, tc_xma>();
   test_vsv<cscalar_f, tc_xma>();

   test_vvs< scalar_f, tc_xma>();
   test_vvs<cscalar_f, tc_xma>();

   test_vss< scalar_f, tc_xma>();
   test_vss<cscalar_f, tc_xma>();

   // NRBS:
   //  - types other than scalar_f and cscalar_f
   //  - views other than vector
#endif
}






// -------------------------------------------------------------------- //
// test add
void test_sub()
{
   test_vv< scalar_f, tc_sub>();
   test_vv<cscalar_f, tc_sub>();

   test_tt_sv< scalar_i,  scalar_i, tc_sub>();
   test_tt_sv< scalar_f,  scalar_f, tc_sub>();
   test_tt_sv< scalar_f, cscalar_f, tc_sub>();
   test_tt_sv<cscalar_f, cscalar_f, tc_sub>();

   test_tt_sm< scalar_f,  scalar_f, tc_sub>();
   test_tt_sm< scalar_f, cscalar_f, tc_sub>();
   test_tt_sm<cscalar_f, cscalar_f, tc_sub>();

   // NRBS:
   //  - VS
}



// -------------------------------------------------------------------- //
// test multiply.
void test_mul()
{
   test_tt_vv< scalar_i,  scalar_i, tc_mul>();
   test_tt_vv< scalar_f,  scalar_f, tc_mul>();
   test_tt_vv< scalar_f, cscalar_f, tc_mul>();
   test_tt_vv<cscalar_f, cscalar_f, tc_mul>();

   test_tt_mm< scalar_f,  scalar_f, tc_mul>();
   test_tt_mm< scalar_f, cscalar_f, tc_mul>();
   test_tt_mm<cscalar_f, cscalar_f, tc_mul>();

   test_tt_sv< scalar_f,  scalar_f, tc_mul>();
   test_tt_sv< scalar_f, cscalar_f, tc_mul>();
   test_tt_sv<cscalar_f, cscalar_f, tc_mul>();

   test_tt_sm< scalar_f,  scalar_f, tc_mul>();
   test_tt_sm< scalar_f, cscalar_f, tc_mul>();
   test_tt_sm<cscalar_f, cscalar_f, tc_mul>();

   // NRBS:
   // test_tt_vv<cscalar_f,  scalar_f, tc_mul>();
   // test_tt_mm< scalar_i,  scalar_i, tc_mul>();
}



// -------------------------------------------------------------------- //
// test division
void test_div()
{
   test_tt_vv< scalar_f,  scalar_f, tc_div>();
   test_tt_vv< scalar_f, cscalar_f, tc_div>();
   test_tt_vv<cscalar_f,  scalar_f, tc_div>();
   test_tt_vv<cscalar_f, cscalar_f, tc_div>();

   test_tt_mm< scalar_f,  scalar_f, tc_div>();
   test_tt_mm< scalar_f, cscalar_f, tc_div>();
   test_tt_mm<cscalar_f,  scalar_f, tc_div>();
   test_tt_mm<cscalar_f, cscalar_f, tc_div>();

   test_tt_sv< scalar_f,  scalar_f, tc_div>();
   test_tt_sv< scalar_f, cscalar_f, tc_div>();
   test_tt_sv<cscalar_f, cscalar_f, tc_div>();

   test_tt_sm< scalar_f,  scalar_f, tc_div>();
   test_tt_sm< scalar_f, cscalar_f, tc_div>();
   test_tt_sm<cscalar_f, cscalar_f, tc_div>();

   test_tt_vs< scalar_f,  scalar_f, tc_div>();
   test_tt_vs<cscalar_f,  scalar_f, tc_div>();

   test_tt_ms< scalar_f,  scalar_f, tc_div>();
   test_tt_ms<cscalar_f,  scalar_f, tc_div>();

   // NRBS:
   //  - VS
}



// -------------------------------------------------------------------- //
// test j-multiply.
void test_jmul()
{
   test_tt_vv<cscalar_f, cscalar_f, tc_jmul>();
   test_tt_mm<cscalar_f, cscalar_f, tc_jmul>();
}



// -------------------------------------------------------------------- //
// tc_land -- Test land (logical and).
template <typename TStor1,
	  typename TStor2,
	  typename TStorR>
struct tc_land
{
   static void test()
   {
      int const	N = 7;

      typedef typename TStor1::value_type T1;
      typedef typename TStor2::value_type T2;

      typedef typename Promotion<T1, T2>::type RT;

      TStor1	stor1(N, TestRig<T1>::test_value1());
      TStor2	stor2(N, TestRig<T2>::test_value2());
      TStorR	storR(N);

      storR.view = land(stor1.view, stor2.view);

      insist(equal(get_origin(storR.view),
		   (RT)(TestRig<T1>::test_value1() && TestRig<T2>::test_value2())));
   }
};



void test_land()
{
   test_vv<    bool, tc_land>();
   test_vv<scalar_i, tc_land>();

   // RBS, but not implemented by tvcpp
   // test_mm<    bool, tc_land>();
   // test_mm<scalar_i, tc_land>();
}



// -------------------------------------------------------------------- //
// tc_fmod -- Test land (logical and).
template <typename TStor1,
	  typename TStor2,
	  typename TStorR>
struct tc_fmod
{
   static void test()
   {
      int const	N = 7;

      typedef typename TStor1::value_type T1;
      typedef typename TStor2::value_type T2;

      typedef typename Promotion<T1, T2>::type RT;

      TStor1	stor1(N, TestRig<T1>::test_value1());
      TStor2	stor2(N, TestRig<T2>::test_value2());
      TStorR	storR(N);

      storR.view = fmod(stor1.view, stor2.view);

      insist(equal(get_origin(storR.view),
		   fmod(TestRig<T1>::test_value1(), TestRig<T2>::test_value2())));
   }
};



void test_fmod()
{
   test_vv<scalar_f, tc_fmod>();
   test_mm<scalar_f, tc_fmod>();
}



// -------------------------------------------------------------------- //
// test mag.
void test_mag()
{
   test_tr_v< scalar_f, scalar_f, tc_mag>();
   test_tr_v<cscalar_f, scalar_f, tc_mag>();

   test_tr_m< scalar_f, scalar_f, tc_mag>();
   test_tr_m<cscalar_f, scalar_f, tc_mag>();

   // RBS, but not implemented by tvcpp
   // test_mm<    bool, tc_land>();
   // test_mm<scalar_i, tc_land>();
}



// -------------------------------------------------------------------- //
// test_ma_vvv -- Test vector multiply add
template <typename		    T,
	  template <typename> class ViewStorage>
void
test_ma_vvv()
{
   using namespace vsip;

   int const	N = 7;

   ViewStorage<T>
		data(N);

   data.viewR = ma(data.view1, data.view2, data.view3);

   insist(equal(get_origin(data.viewR),
		(TestRig<T>::test_value1() * TestRig<T>::test_value2())
		+ TestRig<T>::test_value3()));
}



// -------------------------------------------------------------------- //
// test_ma_vsv -- Test scalar multiply, vector add
template <typename		    T,
	  template <typename> class ViewStorage>
void
test_ma_vsv()
{
   using namespace vsip;

   int const	N = 7;

   ViewStorage<T>
		data(N);

   data.viewR = ma(data.view1, TestRig<T>::test_value2(), data.view3);

   insist(equal(get_origin(data.viewR),
		TestRig<T>::test_value1() * TestRig<T>::test_value2()
		+ TestRig<T>::test_value3()));
}



// -------------------------------------------------------------------- //
// test_ma_vsv -- Test vector multiply, scalar add
template <typename		    T,
	  template <typename> class ViewStorage>
void
test_ma_vvs()
{
   using namespace vsip;

   int const	N = 7;

   ViewStorage<T>
		data(N);

   data.viewR = ma(data.view1, data.view2, TestRig<T>::test_value3());

   insist(equal(get_origin(data.viewR),
		TestRig<T>::test_value1() * TestRig<T>::test_value2()
		+ TestRig<T>::test_value3()));
}



// -------------------------------------------------------------------- //
// test_ma_vsv -- Test scalar multiply, scalar add
template <typename		    T,
	  template <typename> class ViewStorage>
void
test_ma_vss()
{
   using namespace vsip;

   int const	N = 7;

   ViewStorage<T>
		data(N);

   data.viewR = ma(data.view1, TestRig<T>::test_value2(),
		   TestRig<T>::test_value3());

   insist(equal(get_origin(data.viewR),
		TestRig<T>::test_value1() * TestRig<T>::test_value2()
		+ TestRig<T>::test_value3()));
}



/* -------------------------------------------------------------------- *
 * Tests for Expoavg -- exponential average				*
 * -------------------------------------------------------------------- */

template <typename TStor1,
	  typename TStor2,
	  typename TStorR>
struct tc_expoavg
{
   static void test()
   {
      int const	N = 7;

      typedef typename TStor1::value_type T1;
      typedef typename TStor2::value_type T2;

      TStor1	stor1(N, TestRig<T1>::test_value1());
      TStor2	stor2(N, TestRig<T2>::test_value2());
      TStorR	storR(N);

      scalar_f	alpha = 0.25;

      storR.view = expoavg(alpha, stor1.view, stor2.view);

      insist(equal(get_origin(storR.view),
		   (alpha         * TestRig<T1>::test_value1()) +
		   ((1.f - alpha) * TestRig<T2>::test_value2())));
   }
};



void
test_expoavg()
{
   test_tt_vv< scalar_f,  scalar_f, tc_expoavg>();
   test_tt_vv<cscalar_f, cscalar_f, tc_expoavg>();

   test_tt_mm< scalar_f,  scalar_f, tc_expoavg>();
   test_tt_mm<cscalar_f, cscalar_f, tc_expoavg>();
}



// tc_expoavg -- Testcase for exponential average.
template <typename		    T,
	  template <typename> class ViewStorage>
void
tc_expoavg_old()
{
   using namespace vsip;

   int const	N = 7;

   ViewStorage<T>
		data(N);

   scalar_f	alpha = 0.25;

   data.viewR = expoavg(alpha, data.view2, data.view3);

   insist(equal(get_origin(data.viewR),
		(alpha         * TestRig<T>::test_value2()) +
		((1.f - alpha) * TestRig<T>::test_value3())));
}



void
test_expoavg_old()
{
   tc_expoavg_old<vsip::scalar_f, MVectorStorage>();
   tc_expoavg_old<vsip::scalar_f, MVector12Storage>();
   tc_expoavg_old<vsip::scalar_f, MMatrixStorage>();
   tc_expoavg_old<vsip::scalar_f, MSubVectorStorage>();
   tc_expoavg_old<vsip::scalar_f, MColVectorStorage>();

   tc_expoavg_old<vsip::cscalar_f, MVectorStorage>();
   tc_expoavg_old<vsip::cscalar_f, MVector12Storage>();
   tc_expoavg_old<vsip::cscalar_f, MMatrixStorage>();
   tc_expoavg_old<vsip::cscalar_f, MSubVectorStorage>();
   tc_expoavg_old<vsip::cscalar_f, MColVectorStorage>();

   vsip::Vector<vsip::scalar_f>
		vector_scalarf (7, 3.4);

   vector_scalarf = 1.0;
   check_entry (vsip::expoavg (static_cast<vsip::scalar_f>(0.5),
			       vector_scalarf, vector_scalarf),
		2,
		static_cast<vsip::scalar_f>(1.0));
}




// -------------------------------------------------------------------- //
int
main (int argc, char** argv)
{
  vsip::vsipl	init(argc, argv);
  vsip::Vector<vsip::scalar_f>
		vector_scalarf (7, 3.4);
  vsip::Vector<vsip::cscalar_f>
		vector_cscalarf (7, vsip::cscalar_f (3.3, 3.3));
  vsip::Vector<vsip::scalar_i>
		vector_scalari (7, 3);
#ifdef XBOOL
  vsip::Vector<bool>
		vector_bool (7, false);
#endif

  /* Test assignment.  */
  vector_scalarf = 2.2;
  check_entry (vector_scalarf, 2, static_cast<vsip::scalar_f>(2.2));

  /* Test assignment of one vector's values to another.  */
  vsip::Vector<> vector_lhs (7, 0.0);
  check_entry (vector_lhs, 2, static_cast<vsip::scalar_f>(0.0));
  vector_lhs = vector_scalarf;
  check_entry (vector_lhs, 2, static_cast<vsip::scalar_f>(2.2));

  /* Test assignment of a scalar to a vector.  */
  vector_scalarf = 0.0;
  check_entry (vector_scalarf, 2, static_cast<vsip::scalar_f>(0.0));

  /* Test arccosine.  This should yield a vector of 1.0's.  */
  vector_scalarf = 1.0;
  vector_scalarf = vsip::acos (vector_scalarf);
  check_entry (vector_scalarf, 2, static_cast<vsip::scalar_f>(0.0));

  /* Test add.  */
  test_add();
  test_sub();
  test_mul();
  test_div();
  test_jmul();
  test_land();

  test_acos();
  test_arg();
  test_atan2();
  test_hypot();
  test_minmax();

  /* Test am.  */
  test_am();
  vector_scalarf = 3.0;
  vector_scalarf = am (vector_scalarf, vector_scalarf, vector_scalarf);
  check_entry (vector_scalarf, 2, static_cast<vsip::scalar_f>(18.0));

  /* Test arg.  */
  vector_cscalarf = vsip::cscalar_f (3.3, 0.0);
  vector_scalarf = vsip::arg (vector_cscalarf);
  check_entry (vector_scalarf, 2, static_cast<vsip::scalar_f>(0.0));

  /* Test ceil.  */
  vector_scalarf = 2.1;
  vector_scalarf = vsip::ceil (vector_scalarf);
  check_entry (vector_scalarf, 3, static_cast<vsip::scalar_f>(3.0));

#ifdef XBOOL
  /* Test eq.  */
  vector_bool = vsip::eq (vector_scalari, vector_scalari);
  check_entry (vector_bool, 2, true);
  check_entry (vsip::eq (vector_scalari, vector_scalari), 2, true);
  vector_bool = vsip::eq (vector_scalarf, vector_scalarf);
  check_entry (vector_bool, 2, true);
#if 0 /* tvcpp0p8 does not define the underlying function.  */
  insist (vsip::eq (vector_cscalarf, vector_cscalarf));
#endif /* tvcpp0p8 does not define the underlying function.  */
#endif

  /* Test euler.  */
  vector_scalarf = 0.0;
  vector_cscalarf = vsip::euler (vector_scalarf);
  check_entry (vector_cscalarf, 2, vsip::cscalar_f (1.0));

  /* Test fmod.  */
  test_fmod();
  vector_scalarf = 3.4;
  check_entry (vsip::fmod (vector_scalarf, vector_scalarf), 3,
	       static_cast<vsip::scalar_f>(0.0));

  /* Test imag.  */
  vector_cscalarf = vsip::cscalar_f (3.3, 3.2);
  vector_scalarf = vsip::imag (vector_cscalarf);
  check_entry (vector_scalarf, 2, static_cast<vsip::scalar_f>(3.2));

  /* Test magsq.  */
  vector_cscalarf = vsip::cscalar_f (3.3, 0.0);
  vector_scalarf = vsip::magsq (vector_cscalarf);
  check_entry (vector_scalarf, 2,
	       static_cast<vsip::scalar_f>(3.3) *
	       static_cast<vsip::scalar_f>(3.3));
  vector_cscalarf = vsip::cscalar_f (0.0, 3.3);
  vector_scalarf = vsip::magsq (vector_cscalarf);
  check_entry (vector_scalarf, 2,
	       static_cast<vsip::scalar_f>(3.3) *
	       static_cast<vsip::scalar_f>(3.3));

  /* Test mag. */
  test_mag();
  test_maxmg();
  test_minmg();
  test_maxmgsq();
  test_minmgsq();

  /* Test ne and !=.  */
  check_entry (ne (vector_scalarf, vector_scalarf),
	       2,
	       false);
  check_entry (vector_scalarf != vector_scalarf,
	       2,
	       false);
#if 0 /* tvcpp0p8 does not define the underlying function.  */
  check_entry (ne (vector_cscalarf, vector_cscalarf),
	       2,
	       false);
#endif /* tvcpp0p8 does not define the underlying function.  */

  /* Test ne and -.  */
  vector_scalarf = 3.4;
  check_entry (neg (vector_scalarf),
	       2,
	       static_cast<vsip::scalar_f>(-3.4));
  check_entry (-vector_scalarf,
	       2,
	       static_cast<vsip::scalar_f>(-3.4));

#ifdef XBOOL
  /* Test lnot.  */
  vector_bool.put (3, true);
  vector_bool = vsip::lnot (vector_bool);
  check_entry (vector_bool, 3, false);
#endif

  /* Test sub.  */
  vector_scalarf = 3.4;
  check_entry (vector_scalarf - vector_scalarf, 2,
	       static_cast<vsip::scalar_f>(0.0));

  /* Test adding a scalar to a vector.  */
  vector_scalarf = 3.4;
  check_entry (vsip::add (static_cast<vsip::scalar_f>(-3.4), vector_scalarf),
	       2,
	       static_cast<vsip::scalar_f>(0.0));


  test_ma();
  test_msb();
  test_xma();


  // Test ma (multiply-add) combinations.
  // vvv: vector multiply, vector add
#if HAVE_GENERAL_MA
  // The following cases are not required by the spec:
  test_ma_vvv<vsip::scalar_i,  MVectorStorage>(); // scalar_i not reg by spec
  test_ma_vvv<vsip::scalar_f,  MMatrixStorage>(); // matrix not req by spec
  test_ma_vvv<vsip::cscalar_f, MMatrixStorage>(); // matrix not req by spec
#endif

  test_ma_vvv<vsip::scalar_f,  MVectorStorage>();
  test_ma_vvv<vsip::scalar_f,  MVector12Storage>();
  test_ma_vvv<vsip::scalar_f,  MSubVectorStorage>();
  test_ma_vvv<vsip::scalar_f,  MColVectorStorage>();

  test_ma_vvv<vsip::cscalar_f, MVectorStorage>();
  test_ma_vvv<vsip::cscalar_f, MVector12Storage>();
  test_ma_vvv<vsip::cscalar_f, MSubVectorStorage>();
  test_ma_vvv<vsip::cscalar_f, MColVectorStorage>();

  // vsv: scalar multiply, vector add
  test_ma_vsv<vsip::scalar_f,  MVectorStorage>();
  test_ma_vsv<vsip::cscalar_f, MVectorStorage>();

  // vvs: vector multiply, scalar add
  test_ma_vvs<vsip::scalar_f,  MVectorStorage>();
  test_ma_vvs<vsip::cscalar_f, MVectorStorage>();

  // vss: scalar multiply, scalar add
  test_ma_vss<vsip::scalar_f,  MVectorStorage>();
  test_ma_vss<vsip::cscalar_f, MVectorStorage>();

  /* Test ma with vector, scalar, scalar.  */
  vector_scalarf = 1.0;
  vector_scalarf = vsip::ma (vector_scalarf,
			     static_cast<vsip::scalar_f>(-1.0),
			     static_cast<vsip::scalar_f>(2.0));
  check_entry (vsip::ma (vector_scalarf,
			 static_cast<vsip::scalar_f>(-1.0),
			 static_cast<vsip::scalar_f>(2.0)),
	       2,
	       static_cast<vsip::scalar_f>(1.0));

  /* Test expoavg.  */
  test_expoavg();
  test_expoavg_old();
  

  /* Test arithmetic on vectors.  */
  check_entry (static_cast<vsip::scalar_f>(0.5) + vector_scalarf,
	       2,
	       static_cast<vsip::scalar_f>(1.5));

  vector_scalarf = 1.1;
  vector_scalarf = static_cast<vsip::scalar_f>(2.0) * vector_scalarf
    + vector_scalarf;
  check_entry (vector_scalarf, 2, static_cast<vsip::scalar_f>(3.3));
  vector_scalarf = 1.1;
  vector_scalarf = static_cast<vsip::scalar_f>(2.0) * vector_scalarf
    + (static_cast<vsip::scalar_f>(4.0) + vector_scalarf);
  check_entry (vector_scalarf, 2, static_cast<vsip::scalar_f>(7.3));

  /* Test incrementing a vector.  */
  vector_scalarf = 0.0;
  vector_scalarf += static_cast<vsip::scalar_f>(1.1);
  check_entry (vector_scalarf, 2, static_cast<vsip::scalar_f>(1.1));

  /* Test adding two vectors of complex numbers.  */
  vsip::Vector<vsip::cscalar_f>
		vector_complex_a (4, vsip::cscalar_f(1.0, 1.0));
  vector_complex_a += vector_complex_a;
  check_entry (vector_complex_a, 2, vsip::cscalar_f(2.0, 2.0));
  vsip::Vector<vsip::cscalar_f>
		vector_complex_b (4, vsip::cscalar_f(1.0, 1.0));
  vector_complex_b += vector_complex_a;
  check_entry (vector_complex_b, 2, vsip::cscalar_f(3.0, 3.0));

#if 0 /* The VSIPL++ specification does not require supporting
	 addition of vectors with different value types except for a
	 few special cases.  */
  /* Test addition of vector with complex numbers and non-complex
     numbers.  */
  vsip::Vector<vsip::scalar_i>
		vector_int (4, -17);
  vector_complex_b = vector_complex_b + vector_int;
  check_entry (vector_complex_b, 2, vsip::cscalar_f(-14.0, 3.0));
  vector_complex_b += static_cast<vsip::scalar_f>(-13.3);
  check_entry (vector_complex_b, 2, vsip::cscalar_f(-0.7, 3.0));
#endif

  return EXIT_SUCCESS;
}
