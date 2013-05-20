//
// Copyright (c) 2005, 2006, 2007 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

/// Description
///   Coverage tests for expressions.

#ifndef common_hpp_
#define common_hpp_

#include <vsip/support.hpp>
#include <vsip/complex.hpp>
#include <vsip/vector.hpp>
#include <vsip/math.hpp>
#include <vsip/random.hpp>
#include <test.hpp>
#include <storage.hpp>

using namespace ovxx;

template <typename T> struct split_or_array
{ static storage_format_type const value = array;};

template <typename T> struct split_or_array<complex<T> >
{ static storage_format_type const value = split_complex;};

template <typename T> struct interleaved_or_array
{ static storage_format_type const value = array;};

template <typename T> struct interleaved_or_array<complex<T> >
{ static storage_format_type const value = interleaved_complex;};

enum range_type
{
  nonzero,
  posval,
  anyval
};

template <typename T>
struct Get_value
{
  static T at(vsip::index_type arg, vsip::index_type i, range_type rt=anyval)
  {
    if (i == 0) return T(arg+3);

    // return T((2+arg)*i + (1+arg));
    vsip::Rand<T> rand(5*i + arg);
    T value =  rand.randu();
    if ((rt == nonzero || rt== posval) && value == T(0))
      value += T(1);
    if (rt == posval && value < T(0))
      value = -value;
    return value;
  }
};

template <typename T>
struct Get_value<vsip::complex<T> >
{
  static vsip::complex<T> at(
    vsip::index_type arg,
    vsip::index_type i,
    range_type rt=anyval)
  {
    vsip::Rand<T> rand(5*i + arg);
    vsip::complex<T> value = vsip::complex<T>(rand.randu(), rand.randu());
    if (rt == nonzero && value == vsip::complex<T>(0))
      value += vsip::complex<T>(1);
    return value;
  }
};

template <>
struct Get_value<bool>
{
  static bool at(
    vsip::index_type arg,
    vsip::index_type i,
    range_type =anyval)
  {
    vsip::Rand<float> rand(5*i + arg);
    return rand.randu() > 0.5;
  }
};



/***********************************************************************
  Unary Operator Tests
***********************************************************************/

#if VERBOSE
#  define DEBUG_UNARY(NAME, OP)						\
  {									\
    std::cout << "Test"#NAME << std::endl				\
	      << "  at pos  : " << i << std::endl			\
	      << "  expected: " << expected << std::endl		\
	      << "  got     : " << get_nth(view2, i) << std::endl	\
      ;									\
    vsip_csl::assign_diagnostics(view2, OP (view1));		        \
  }
#else
#  define DEBUG_UNARY(NAME, OP)						\
  {}
#endif

// Test structure for Unary operator
//
// Where
//   NAME is the suffix of the test class (Test_NAME)
//   OP is the unary operator for a view
//   CHKOP is the unary operator for an element
//   RT is the permissible range of values (nonzero, posval, anyval)

#define TEST_UNARY(NAME, OP, CHKOP, RT)					\
struct Test_##NAME							\
{									\
  template <typename View1,						\
	    typename View2>						\
  static void								\
  exec(									\
    View1 view1,							\
    View2 view2)							\
  {									\
    length_type size = get_size(view2);					\
    test_assert(is_scalar<View1>::value || get_size(view1) == size);	\
									\
    typedef typename Value_type_of<View1>::type T1;			\
    typedef typename Value_type_of<View2>::type T2;			\
  									\
    for (index_type i=0; i<get_size(view1); ++i)			\
      put_nth(view1, i, Get_value<T1>::at(0, i, RT));			\
    									\
    view2 = OP (view1);							\
									\
    for (index_type i=0; i<get_size(view2); ++i)			\
    {									\
      T2 expected = 							\
        CHKOP (Get_value<T1>::at(0, is_scalar<View1>::value ? 0 : i, RT));\
      if (!(equal(get_nth(view2, i), expected)))			\
	DEBUG_UNARY(NAME, OP)						\
      test_assert(equal(get_nth(view2, i), expected));			\
    }									\
  }									\
};



/***********************************************************************
  Binary Operator Tests
***********************************************************************/

#define TEST_BINARY_OP(NAME, OP, CHKOP, RT)				\
struct Test_##NAME							\
{									\
  template <typename View1,						\
	    typename View2,						\
	    typename View3>						\
  static void								\
  exec(									\
    View1 view1,							\
    View2 view2,							\
    View3 view3)							\
  {									\
    length_type size = get_size(view3);					\
    test_assert(is_scalar<View1>::value || get_size(view1) == size);	\
    test_assert(is_scalar<View2>::value || get_size(view2) == size);	\
									\
    typedef typename Value_type_of<View1>::type T1;			\
    typedef typename Value_type_of<View2>::type T2;			\
    typedef typename Value_type_of<View3>::type T3;			\
  									\
    for (index_type i=0; i<get_size(view1); ++i)			\
      put_nth(view1, i, Get_value<T1>::at(0, i));			\
    for (index_type i=0; i<get_size(view2); ++i)			\
      put_nth(view2, i, Get_value<T2>::at(1, i, RT));			\
    									\
    view3 = view1 OP view2;						\
									\
    for (index_type i=0; i<get_size(view3); ++i)			\
    {									\
      T3 expected =							\
        (Get_value<T1>::at(0, is_scalar<View1>::value ? 0 : i)		\
	 CHKOP								\
	 Get_value<T2>::at(1, is_scalar<View2>::value ? 0 : i, RT));	\
      test_assert(equal(get_nth(view3, i), expected));			\
    }									\
  }									\
};



#define TEST_BINARY_FUNC(NAME, FUN, CHKFUN, RT)				\
struct Test_##NAME							\
{									\
  template <typename View1,						\
	    typename View2,						\
	    typename View3>						\
  static void								\
  exec(									\
    View1 view1,							\
    View2 view2,							\
    View3 view3)							\
  {									\
    length_type size = get_size(view3);					\
    test_assert(is_scalar<View1>::value || get_size(view1) == size);	\
    test_assert(is_scalar<View2>::value || get_size(view2) == size);	\
									\
    typedef typename Value_type_of<View1>::type T1;			\
    typedef typename Value_type_of<View2>::type T2;			\
    typedef typename Value_type_of<View3>::type T3;			\
  									\
    for (index_type i=0; i<get_size(view1); ++i)			\
      put_nth(view1, i, Get_value<T1>::at(0, i));			\
    for (index_type i=0; i<get_size(view2); ++i)			\
      put_nth(view2, i, Get_value<T2>::at(1, i, RT));			\
    									\
    view3 = FUN(view1, view2);						\
									\
    for (index_type i=0; i<get_size(view3); ++i)			\
    {									\
      T3 expected =							\
        CHKFUN(Get_value<T1>::at(0, is_scalar<View1>::value ? 0 : i),	\
	       Get_value<T2>::at(1, is_scalar<View2>::value ? 0 : i, RT));\
      test_assert(equal(get_nth(view3, i), expected));			\
    }									\
  }									\
};



/***********************************************************************
  Ternary Operator Tests
***********************************************************************/

#if VERBOSE
#  define TEST_TERNARY(NAME, FCN, OP1, OP2, CHKOP1, CHKOP2)		\
struct Test_##NAME							\
{									\
  template <typename View1,						\
	    typename View2,						\
	    typename View3,						\
	    typename View4>						\
  static void								\
  exec(									\
    View1 view1,							\
    View2 view2,							\
    View3 view3,							\
    View4 view4)	/* Result */					\
  {									\
    length_type size = get_size(view4);					\
    test_assert(is_scalar<View1>::value || get_size(view1) == size);	\
    test_assert(is_scalar<View2>::value || get_size(view2) == size);	\
    test_assert(is_scalar<View3>::value || get_size(view3) == size);	\
    									\
    typedef typename Value_type_of<View1>::type T1;			\
    typedef typename Value_type_of<View2>::type T2;			\
    typedef typename Value_type_of<View3>::type T3;			\
    typedef typename Value_type_of<View4>::type T4;			\
									\
    for (index_type i=0; i<get_size(view1); ++i)			\
      put_nth(view1, i, Get_value<T1>::at(0, i));			\
    for (index_type i=0; i<get_size(view2); ++i)			\
      put_nth(view2, i, Get_value<T2>::at(1, i));			\
    for (index_type i=0; i<get_size(view3); ++i)			\
      put_nth(view3, i, Get_value<T2>::at(2, i));			\
    									\
    view4 = FCN(view1, view2, view3);					\
    									\
    for (index_type i=0; i<get_size(view4); ++i)			\
    {									\
      T4 expected =							\
	(       Get_value<T1>::at(0, is_scalar<View1>::value ? 0 : i)	\
	 CHKOP1 Get_value<T2>::at(1, is_scalar<View2>::value ? 0 : i))	\
	 CHKOP2 Get_value<T3>::at(2, is_scalar<View3>::value ? 0 : i);	\
      if (!equal(get_nth(view4, i), expected))				\
      {									\
	std::cout							\
	  << "TEST_TERNARY FCN FAILURE\n"				\
	  << "  i       : " << i << "\n"				\
	  << "  at(0, i): "						\
	  << Get_value<T1>::at(0, is_scalar<View1>::value ? 0 : i)	\
	  << "\n"							\
	  << "  at(1, i): "						\
	  << Get_value<T1>::at(1, is_scalar<View1>::value ? 0 : i)	\
	  << "\n"							\
	  << "  at(2, i): "						\
	  << Get_value<T1>::at(2, is_scalar<View1>::value ? 0 : i)	\
	  << "\n"							\
	  << "  result: " << get_nth(view4, i) << "\n"			\
	  << "  expected: " << expected << "\n"				\
	  ;								\
	/* vsip_csl::assign_diagnostics(view4,				\
	   FCN(view1, view2, view3)); */				\
      }									\
      test_assert(equal(get_nth(view4, i), expected));			\
    }									\
    									\
    view4 = T4();							\
    view4 = (view1 OP1 view2) OP2 view3;				\
    									\
    for (index_type i=0; i<get_size(view4); ++i)			\
    {									\
      T4 expected =							\
	(       Get_value<T1>::at(0, is_scalar<View1>::value ? 0 : i)	\
	 CHKOP1 Get_value<T2>::at(1, is_scalar<View2>::value ? 0 : i))	\
	 CHKOP2 Get_value<T3>::at(2, is_scalar<View3>::value ? 0 : i);	\
      if (!equal(get_nth(view4, i), expected))				\
      {									\
	std::cout							\
	  << "TEST_TERNARY OP1/OP2 FAILURE\n"				\
	  << "  i       : " << i << "\n"				\
	  << "  at(0, i): "						\
	  << Get_value<T1>::at(0, is_scalar<View1>::value ? 0 : i)	\
	  << "\n"							\
	  << "  at(1, i): "						\
	  << Get_value<T1>::at(1, is_scalar<View1>::value ? 0 : i)	\
	  << "\n"							\
	  << "  at(2, i): "						\
	  << Get_value<T1>::at(2, is_scalar<View1>::value ? 0 : i)	\
	  << "\n"							\
	  << "  result: " << get_nth(view4, i) << "\n"			\
	  << "  expected: " << expected << "\n"				\
	  ;								\
	/* vsip_csl::assign_diagnostics(view4,				\
	   (view1 OP1 view2) OP2 view3); */ 				\
      }									\
      test_assert(equal(get_nth(view4, i), expected));			\
    }									\
  }									\
};
#else
#  define TEST_TERNARY(NAME, FCN, OP1, OP2, CHKOP1, CHKOP2)		\
struct Test_##NAME							\
{									\
  template <typename View1,						\
	    typename View2,						\
	    typename View3,						\
	    typename View4>						\
  static void								\
  exec(									\
    View1 view1,							\
    View2 view2,							\
    View3 view3,							\
    View4 view4)	/* Result */					\
  {									\
    length_type size = get_size(view4);					\
    test_assert(is_scalar<View1>::value || get_size(view1) == size);	\
    test_assert(is_scalar<View2>::value || get_size(view2) == size);	\
    test_assert(is_scalar<View3>::value || get_size(view3) == size);	\
    									\
    typedef typename Value_type_of<View1>::type T1;			\
    typedef typename Value_type_of<View2>::type T2;			\
    typedef typename Value_type_of<View3>::type T3;			\
    typedef typename Value_type_of<View4>::type T4;			\
									\
    for (index_type i=0; i<get_size(view1); ++i)			\
      put_nth(view1, i, Get_value<T1>::at(0, i));			\
    for (index_type i=0; i<get_size(view2); ++i)			\
      put_nth(view2, i, Get_value<T2>::at(1, i));			\
    for (index_type i=0; i<get_size(view3); ++i)			\
      put_nth(view3, i, Get_value<T2>::at(2, i));			\
    									\
    view4 = FCN(view1, view2, view3);					\
    									\
    for (index_type i=0; i<get_size(view4); ++i)			\
    {									\
      T4 expected =							\
	(       Get_value<T1>::at(0, is_scalar<View1>::value ? 0 : i)	\
	 CHKOP1 Get_value<T2>::at(1, is_scalar<View2>::value ? 0 : i))	\
	 CHKOP2 Get_value<T3>::at(2, is_scalar<View3>::value ? 0 : i);	\
      test_assert(equal(get_nth(view4, i), expected));			\
    }									\
    									\
    view4 = T4();							\
    view4 = (view1 OP1 view2) OP2 view3;				\
    									\
    for (index_type i=0; i<get_size(view4); ++i)			\
    {									\
      T4 expected =							\
	(       Get_value<T1>::at(0, is_scalar<View1>::value ? 0 : i)	\
	 CHKOP1 Get_value<T2>::at(1, is_scalar<View2>::value ? 0 : i))	\
	 CHKOP2 Get_value<T3>::at(2, is_scalar<View3>::value ? 0 : i);	\
      test_assert(equal(get_nth(view4, i), expected));			\
    }									\
  }									\
};
#endif



/***********************************************************************
  Test Drivers
***********************************************************************/

template <typename       Test_class,
	  typename       Stor1,
	  typename       Stor2,
	  vsip::dimension_type Dim>
void
do_case2(vsip::Domain<Dim> dom)
{
  Stor1 stor1(dom);
  Stor2 stor2(dom);

  Test_class::exec(stor1.view, stor2.view);
}



template <typename       Test_class,
	  typename       Stor1,
	  typename       Stor2,
	  typename       Stor3,
	  vsip::dimension_type Dim>
void
do_case3(vsip::Domain<Dim> dom)
{
  Stor1 stor1(dom);
  Stor2 stor2(dom);
  Stor3 stor3(dom);

  Test_class::exec(stor1.view, stor2.view, stor3.view);
}



template <typename       Test_class,
	  typename       Stor1,
	  typename       Stor2,
	  typename       Stor3,
	  vsip::dimension_type Dim>
void
do_case3_left_ip_helper(vsip::Domain<Dim> dom, vsip::impl::true_type)
{
  Stor1 stor1(dom);
  Stor2 stor2(dom);

  Test_class::exec(stor1.view, stor2.view, stor1.view);
}


template <typename       Test_class,
	  typename       Stor1,
	  typename       Stor2,
	  typename       Stor3,
	  vsip::dimension_type Dim>
void
do_case3_left_ip_helper(vsip::Domain<Dim>, vsip::impl::false_type)
{
}



// Test left operand in-place (A op B -> A).

template <typename       Test_class,
	  typename       Stor1,
	  typename       Stor2,
	  typename       Stor3,
	  vsip::dimension_type Dim>
void
do_case3_left_ip(vsip::Domain<Dim> dom)
{
  do_case3_left_ip_helper<Test_class, Stor1, Stor2, Stor3, Dim>(
	dom,
	vsip::impl::integral_constant<bool, vsip::impl::is_same<Stor1, Stor3>::value>());
}



template <typename       Test_class,
	  typename       Stor1,
	  typename       Stor2,
	  typename       Stor3,
	  vsip::dimension_type Dim>
void
do_case3_right_ip_helper(vsip::Domain<Dim> dom, vsip::impl::true_type)
{
  Stor1 stor1(dom);
  Stor2 stor2(dom);

  Test_class::exec(stor1.view, stor2.view, stor2.view);
}


template <typename       Test_class,
	  typename       Stor1,
	  typename       Stor2,
	  typename       Stor3,
	  vsip::dimension_type Dim>
void
do_case3_right_ip_helper(vsip::Domain<Dim>, vsip::impl::false_type)
{
}



// Test right operand in-place (A op B -> A).

template <typename       Test_class,
	  typename       Stor1,
	  typename       Stor2,
	  typename       Stor3,
	  vsip::dimension_type Dim>
void
do_case3_right_ip(vsip::Domain<Dim> dom)
{
  do_case3_right_ip_helper<Test_class, Stor1, Stor2, Stor3, Dim>(
	dom,
	vsip::impl::integral_constant<bool, vsip::impl::is_same<Stor2, Stor3>::value>());
}



template <typename       Test_class,
	  typename       Stor1,
	  typename       Stor2,
	  typename       Stor3,
	  typename       Stor4,
	  vsip::dimension_type Dim>
void
do_case4(vsip::Domain<Dim> dom)
{
  Stor1 stor1(dom);
  Stor2 stor2(dom);
  Stor3 stor3(dom);
  Stor4 stor4(dom);

  Test_class::exec(stor1.view, stor2.view, stor3.view, stor4.view);
}



template <typename Test_class,
	  typename T1,
	  typename T2>
void
vector_cases2_rt()
{
  typedef Storage<0, T1>            sca1_t;

  typedef Storage<1, T1>            vec1_t;
  typedef Storage<1, T2>            vec2_t;

  typedef Storage<1, T1, row1_type, Replicated_map<1> > gvec1_t;
  typedef Storage<1, T2, row1_type, Replicated_map<1> > gvec2_t;

  typedef Storage<1, T1, row1_type, Map<> >         dvec1_t;
  typedef Storage<1, T2, row1_type, Map<> >         dvec2_t;

  typedef Storage<1, T1, row1_type, Local_map, split_or_array<T1>::value> spl1_t;
  typedef Storage<1, T2, row1_type, Local_map, split_or_array<T2>::value> spl2_t;

  typedef Row_vector<T1, row2_type> row1_t;
  typedef Row_vector<T2, row2_type> row2_t;

  typedef Diag_vector<T1, row2_type> dia1_t;
  typedef Diag_vector<T2, row2_type> dia2_t;

  Domain<1> dom(11);
  
  do_case2<Test_class, vec1_t, vec2_t>(dom);

  do_case2<Test_class, sca1_t, vec2_t>(dom);

  do_case2<Test_class, row1_t, vec2_t>(dom);
  do_case2<Test_class, vec1_t, row2_t>(dom);
  
  do_case2<Test_class, dia1_t, vec2_t>(dom);
  
  do_case2<Test_class, spl1_t, spl2_t>(dom);

  // distributed cases
#ifdef OVXX_PARALLEL
  do_case2<Test_class, gvec1_t, gvec2_t>(dom);
  do_case2<Test_class,  sca1_t, gvec2_t>(dom);

  do_case2<Test_class, dvec1_t, dvec2_t>(dom);
  do_case2<Test_class,  sca1_t, dvec2_t>(dom);

  do_case2<Test_class, gvec1_t, dvec2_t>(dom);
  do_case2<Test_class, dvec1_t, gvec2_t>(dom);
#endif
}



template <typename Test_class,
	  typename T1>
void
vector_cases2()
{
  vector_cases2_rt<Test_class, T1, T1>();
}



template <typename Test_class,
	  typename T1>
void
vector_cases2_mix()
{
  typedef T1 T2;

  typedef Storage<1, T1, row1_type, Local_map, interleaved_or_array<T1>::value> int1_t;
  typedef Storage<1, T2, row1_type, Local_map, interleaved_or_array<T2>::value> int2_t;

  typedef Storage<1, T1, row1_type, Local_map, split_or_array<T1>::value> spl1_t;
  typedef Storage<1, T2, row1_type, Local_map, split_or_array<T2>::value> spl2_t;

  vsip::Domain<1> dom(11);
  
  do_case2<Test_class, int1_t, spl2_t>(dom);
  do_case2<Test_class, spl1_t, int2_t>(dom);
}



// Vector 2-operand -> 1 result cases, with specified return type.

template <typename Test_class,
	  typename T1,
	  typename T2,
	  typename T3>
void
vector_cases3_rt()
{
  typedef Storage<0, T1>            sca1_t;
  typedef Storage<0, T2>            sca2_t;

  typedef Storage<1, T1>            vec1_t;
  typedef Storage<1, T2>            vec2_t;
  typedef Storage<1, T3>            vec3_t;

  typedef Storage<1, T1, row1_type, Replicated_map<1> > gvec1_t;
  typedef Storage<1, T2, row1_type, Replicated_map<1> > gvec2_t;
  typedef Storage<1, T3, row1_type, Replicated_map<1> > gvec3_t;

  typedef Storage<1, T1, row1_type, Map<> > dvec1_t;
  typedef Storage<1, T2, row1_type, Map<> > dvec2_t;
  typedef Storage<1, T3, row1_type, Map<> > dvec3_t;

  typedef Row_vector<T1, col2_type> row1_t;
  typedef Row_vector<T2, col2_type> row2_t;
  typedef Row_vector<T3, col2_type> row3_t;

  vsip::Domain<1> dom(11);
  
  do_case3<Test_class, vec1_t, vec2_t, vec3_t>(dom);

  do_case3_left_ip <Test_class, vec1_t, vec2_t, vec3_t>(dom);
  do_case3_right_ip<Test_class, vec1_t, vec2_t, vec3_t>(dom);

  do_case3<Test_class, sca1_t, vec2_t, vec3_t>(dom);
  do_case3<Test_class, vec1_t, sca2_t, vec3_t>(dom);

  do_case3_right_ip<Test_class, sca1_t, vec2_t, vec3_t>(dom);
  do_case3_left_ip <Test_class, vec1_t, sca2_t, vec3_t>(dom);

  do_case3<Test_class, row1_t, vec2_t, vec3_t>(dom);
  do_case3<Test_class, vec1_t, row2_t, vec3_t>(dom);
  do_case3<Test_class, vec1_t, vec2_t, row3_t>(dom);

  do_case3<Test_class, row1_t, sca2_t, vec3_t>(dom);
  do_case3<Test_class, sca1_t, row2_t, vec3_t>(dom);

  do_case3_right_ip<Test_class, row1_t, sca2_t, vec3_t>(dom);
  do_case3_left_ip <Test_class, sca1_t, row2_t, vec3_t>(dom);

  // distributed cases
#ifdef OVXX_PARALLEL
  do_case3<Test_class, gvec1_t, gvec2_t, gvec3_t>(dom);
  do_case3<Test_class,  sca1_t, gvec2_t, gvec3_t>(dom);
  do_case3<Test_class, gvec1_t,  sca2_t, gvec3_t>(dom);

  do_case3<Test_class, dvec1_t, dvec2_t, dvec3_t>(dom);
  do_case3<Test_class,  sca1_t, dvec2_t, dvec3_t>(dom);
  do_case3<Test_class, dvec1_t,  sca2_t, dvec3_t>(dom);
#endif
}



// Vector 2-operand -> 1 result cases, with promotion return type.

template <typename Test_class,
	  typename T1,
	  typename T2>
void
vector_cases3()
{
  typedef typename vsip::Promotion<T1, T2>::type  T3;

  vector_cases3_rt<Test_class, T1, T2, T3>();
}



template <typename Test_class,
	  typename T1,
	  typename T2>
void
vector_cases3_bool()
{
  vector_cases3_rt<Test_class, T1, T2, bool>();
}



template <typename Test_class,
	  typename T1,
	  typename T2,
	  typename T3>
void
vector_cases4()
{
  typedef typename Promotion<T1, typename Promotion<T2, T3>::type>::type  T4;

  // Scalars
  typedef Storage<0, T1>            sca1_t;
  typedef Storage<0, T2>            sca2_t;
  typedef Storage<0, T3>            sca3_t;

  // Regular views
  typedef Storage<1, T1>            vec1_t;
  typedef Storage<1, T2>            vec2_t;
  typedef Storage<1, T3>            vec3_t;
  typedef Storage<1, T4>            vec4_t;

  typedef Row_vector<T1, row2_type> row1_t;
  typedef Row_vector<T2, row2_type> row2_t;
  typedef Row_vector<T3, row2_type> row3_t;
  typedef Row_vector<T4, row2_type> row4_t;

  Domain<1> dom(11);
  
  do_case4<Test_class, vec1_t, vec2_t, vec3_t, vec4_t>(dom);
  do_case4<Test_class, sca1_t, vec2_t, vec3_t, vec4_t>(dom);
  do_case4<Test_class, vec1_t, sca2_t, vec3_t, vec4_t>(dom);
  do_case4<Test_class, vec1_t, vec2_t, sca3_t, vec4_t>(dom);

  do_case4<Test_class, row1_t, vec2_t, vec3_t, vec4_t>(dom);
  do_case4<Test_class, vec1_t, row2_t, vec3_t, vec4_t>(dom);
  do_case4<Test_class, vec1_t, vec2_t, row3_t, vec4_t>(dom);
  do_case4<Test_class, vec1_t, vec2_t, vec3_t, row4_t>(dom);
}



template <typename Test_class,
	  typename T1,
	  typename T2>
void
matrix_cases3()
{
  using namespace vsip;
  typedef typename Promotion<T1, T2>::type  T3;

  typedef Storage<0, T1>                  ss_1_t;
  typedef Storage<0, T2>                  ss_2_t;

  typedef Storage<2, T1, row2_type>       mr_1_t;
  typedef Storage<2, T2, row2_type>       mr_2_t;
  typedef Storage<2, T3, row2_type>       mr_3_t;

  typedef Storage<2, T1, col2_type>       mc_1_t;
  typedef Storage<2, T2, col2_type>       mc_2_t;
  typedef Storage<2, T3, col2_type>       mc_3_t;

  typedef Transpose_matrix<T1, row2_type> tr_1_t;
  typedef Transpose_matrix<T2, row2_type> tr_2_t;
  typedef Transpose_matrix<T3, row2_type> tr_3_t;

  Domain<2> dom(7, 11);

  do_case3<Test_class, ss_1_t, mr_2_t, mr_3_t>(dom);
  do_case3<Test_class, mr_1_t, ss_2_t, mr_3_t>(dom);

  do_case3<Test_class, ss_1_t, mc_2_t, mc_3_t>(dom);
  do_case3<Test_class, mc_1_t, ss_2_t, mc_3_t>(dom);
  
  do_case3<Test_class, mr_1_t, mr_2_t, mr_3_t>(dom);

  do_case3<Test_class, tr_1_t, mr_2_t, mr_3_t>(dom);
  do_case3<Test_class, mr_1_t, tr_2_t, mr_3_t>(dom);
  do_case3<Test_class, mr_1_t, mr_2_t, tr_3_t>(dom);
}



template <typename Test_class,
	  typename T1,
	  typename T2,
	  typename T3>
void
matrix_cases4()
{
  using namespace vsip;
  typedef typename Promotion<T1, typename Promotion<T2, T3>::type>::type  T4;

  // Scalars
  typedef Storage<0, T1>            sc_1_t;
  typedef Storage<0, T2>            sc_2_t;
  typedef Storage<0, T3>            sc_3_t;

  // Regular views
  typedef Storage<2, T1, row2_type>       mr_1_t;
  typedef Storage<2, T2, row2_type>       mr_2_t;
  typedef Storage<2, T3, row2_type>       mr_3_t;
  typedef Storage<2, T4, row2_type>       mr_4_t;

  typedef Transpose_matrix<T1, row2_type> tr_1_t;
  typedef Transpose_matrix<T2, row2_type> tr_2_t;
  typedef Transpose_matrix<T3, row2_type> tr_3_t;
  typedef Transpose_matrix<T4, row2_type> tr_4_t;

  Domain<2> dom(7, 11);
  
  do_case4<Test_class, mr_1_t, mr_2_t, mr_3_t, mr_4_t>(dom);

  // SVV, VSV, VVS
  do_case4<Test_class, sc_1_t, mr_2_t, mr_3_t, mr_4_t>(dom);
  do_case4<Test_class, mr_1_t, sc_2_t, mr_3_t, mr_4_t>(dom);
  do_case4<Test_class, mr_1_t, mr_2_t, sc_3_t, mr_4_t>(dom);

  // VSS, SVS, SSV
  do_case4<Test_class, mr_1_t, sc_2_t, sc_3_t, mr_4_t>(dom);
  do_case4<Test_class, sc_1_t, mr_2_t, sc_3_t, mr_4_t>(dom);
  do_case4<Test_class, sc_1_t, sc_2_t, mr_3_t, mr_4_t>(dom);
}


template <typename TestC>
void
test_ternary()
{
  using vsip::complex;

  vector_cases4<TestC, float,           float,           float>();
  vector_cases4<TestC, complex<float>,  complex<float>,  complex<float> >();
#if VSIP_IMPL_TEST_LEVEL > 0
  vector_cases4<TestC, double,          double,          double>();
  vector_cases4<TestC, complex<double>, complex<double>, complex<double> >();
#endif

#if VSIP_IMPL_TEST_LEVEL > 0
  matrix_cases4<TestC, float, float, float>();
#endif
}




#endif // TESTS_COMMON_COVERAGE_HPP
