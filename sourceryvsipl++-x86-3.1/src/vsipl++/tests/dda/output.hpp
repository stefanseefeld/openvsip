/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    tests/extdata-output.cpp
    @author  Jules Bergmann
    @date    04/13/2005
    @brief   VSIPL++ Library: Utilities to print extdata related types.
*/

#ifndef VSIP_TESTS_EXTDATA_OUTPUT_HPP
#define VSIP_TESTS_EXTDATA_OUTPUT_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <string>
#include <sstream>
#include <complex>

#define USE_TYPEID 0

#if USE_TYPEID
#  include <typeinfo>
#endif

#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/core/expr/unary_block.hpp>



/***********************************************************************
  Defintions
***********************************************************************/

template <typename T>
struct Type_name
{
  static std::string name()
  {
#if USE_TYPEID
    return typeid(T).name();
#else
    return "#unknown#";
#endif
  }
}
;

template <typename T>
struct Type_name<T const>
{
  static std::string name()
  {
    std::ostringstream s;
    s << Type_name<T>::name() << " const";
    return s.str();
  }
};

#define TYPE_NAME(TYPE, NAME)				\
  template <>						\
  struct Type_name<TYPE> {				\
    static std::string name() { return NAME; }		\
  };

template <typename T>
struct Type_name<std::complex<T> >
{
  static std::string name()
  {
    std::ostringstream s;
    s << "complex<" << Type_name<T>::name() << ">";
    return s.str();
  }
};

TYPE_NAME(int,    "int")
TYPE_NAME(float,  "float")
TYPE_NAME(double, "double")

TYPE_NAME(vsip::dda::impl::Direct_access_tag,   "Direct_access_tag")
TYPE_NAME(vsip::dda::impl::Reorder_access_tag,  "Reorder_access_tag")
TYPE_NAME(vsip::dda::impl::Copy_access_tag,     "Copy_access_tag")
TYPE_NAME(vsip::dda::impl::Flexible_access_tag, "Flexible_access_tag")

TYPE_NAME(vsip::row1_type,   "tuple<0, 1, 2>")

/***********************************************************************
  Storage Type
***********************************************************************/

template <vsip::storage_format_type C,
	  typename T>
struct Type_name<vsip::impl::Storage<C, T> >
{
  static std::string name()
  {
    std::ostringstream s;
    s << "Storage<"
      << C << ", "
      << Type_name<T>::name() << ">";
    return s.str();
  }
};



/***********************************************************************
  Blocks
***********************************************************************/

template <typename Block, vsip::dimension_type D>
struct Type_name<vsip::impl::Sliced_block<Block, D> >
{
  static std::string name()
  {
    std::ostringstream s;
    s << "Sliced_block<" << D << ">";
    return s.str();
  }
};

template <typename                  Block,
          template <typename> class Extractor>
struct Type_name<vsip::impl::Component_block<Block, Extractor> >
{
  static std::string name()
  {
    std::ostringstream s;
    s << "Component_block<"
      << Type_name<Block>::name() << ", "
      << Type_name<Extractor<typename Block::value_type> >::name() << ">";
    return s.str();
  }
};

template <typename Cplx>
struct Type_name<vsip::impl::Real_extractor<Cplx> >
{ static std::string name() { return std::string("Real_extractor"); } };

template <typename Cplx>
struct Type_name<vsip::impl::Imag_extractor<Cplx> >
{ static std::string name() { return std::string("Imag_extractor"); } };

template <vsip::dimension_type Dim,
	  typename    T,
	  typename    Order,
	  typename    Map>
struct Type_name<vsip::Dense<Dim, T, Order, Map> >
{
  static std::string name()
  {
    std::ostringstream s;
    s << "Dense<" 
      << Dim << ", "
      << Type_name<T>::name() << ", "
      << Type_name<Order>::name() << ", "
      << Type_name<Map>::name() << ">";
    return s.str();
 }
};

template <vsip::dimension_type Dim,
	  typename             T,
	  typename             LP,
	  typename             Map>
struct Type_name<vsip::impl::Strided<Dim, T, LP, Map> >
{
  static std::string name() { return std::string("Strided<>"); }
};

template <template <typename> class O, typename B, bool E>
struct Type_name<vsip::impl::expr::Unary<O, B, E> >
{
  static std::string name() { return std::string("expr::Unary<>"); }
};

template <vsip::dimension_type D, typename T>
struct Type_name<vsip::impl::expr::Scalar<D, T> >
{
  static std::string name()
  {
    std::ostringstream s;
    s << "expr::Scalar<" << D << ", " << Type_name<T>::name() << ">";
    return s.str();
 }
};


TYPE_NAME(vsip::Block_dist,  "Block_dist")
TYPE_NAME(vsip::Cyclic_dist, "Cyclic_dist")

template <typename Dist0,
	  typename Dist1,
	  typename Dist2>
struct Type_name<vsip::Map<Dist0, Dist1, Dist2> >
{
  static std::string name()
  {
    std::ostringstream s;
    s << "Map<" 
      << Type_name<Dist0>::name() << ", "
      << Type_name<Dist1>::name() << ", "
      << Type_name<Dist2>::name()
      << ">";
    return s.str();
  }
};



template <vsip::dimension_type Dim>
struct Type_name<vsip::Replicated_map<Dim> >
{
  static std::string name()
  {
    std::ostringstream s;
    s << "Replicated_map<" << Dim << ">";
    return s.str();
  }
};

template <vsip::dimension_type Dim>
struct Type_name<vsip::Local_or_global_map<Dim> >
{
  static std::string name()
  {
    std::ostringstream s;
    s << "Local_or_global_map<" << Dim << ">";
    return s.str();
  }
};

TYPE_NAME(vsip::Local_map, "Local_map")

#undef TYPE_NAME



template <typename LP,
	  typename Block>
std::string
access_type(
  Block const&,
  LP    const& = LP())
{
  typedef typename vsip::dda::impl::Choose_access<Block, LP>::type
		da_type;

  return Type_name<da_type>::name();
}



template <typename LP>
void
print_layout(std::ostream& out)
{
  out << "  dim  = " << LP::dim << std::endl
      << "  pack  = " << LP::packing << std::endl
      << "  order = " << Type_name<typename LP::order_type>::name() << std::endl
      << "  cmplx = " << LP::storage_format
      << std::endl
    ;
}

#endif // VSIP_TESTS_EXTDATA_OUTPUT_HPP
