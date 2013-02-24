/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    python/signal/iir.cpp
    @author  Stefan Seefeld
    @date    2009-09-20
    @brief   VSIPL++ Library: Python bindings for signal module.

*/
#include <boost/python.hpp>
#include <boost/noncopyable.hpp>
#include <vsip/signal.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include "../block.hpp"
#include <memory>

namespace pyvsip
{
template <typename T>
class iir_base
{
  typedef vsip::Vector<T, Block<1, T> > view_type;
  typedef vsip::Matrix<T, Block<2, T> > matrix_type;

public:
  virtual ~iir_base() {};

  virtual vsip::length_type kernel_size() const = 0;
  virtual vsip::length_type input_size() const = 0;
  virtual vsip::obj_state state() const = 0;
  virtual void reset() = 0;
  virtual void op(view_type, view_type) = 0;
};

template <typename T, vsip::obj_state S>
class iir : public iir_base<T>
{
  typedef vsip::Vector<T, Block<1, T> > view_type;
  typedef vsip::Matrix<T, Block<2, T> > matrix_type;

public:
  iir(matrix_type a, matrix_type b, vsip::length_type l)
    : iir_(a, b, l) {}

  virtual vsip::length_type kernel_size() const { return iir_.kernel_size();}
  virtual vsip::length_type input_size() const { return iir_.input_size();}
  virtual vsip::obj_state state() const { return S;}
  virtual void reset() { iir_.reset();}
  virtual void op(view_type i, view_type o) { iir_(i, o);}

private:
  vsip::Iir<T, S> iir_;
};

template <typename T>
std::auto_ptr<iir_base<T> >
create_iir(vsip::Matrix<T, Block<2, T> > a,
	   vsip::Matrix<T, Block<2, T> > b,
	   vsip::length_type l,
	   vsip::obj_state s,
	   unsigned int /*n*/, vsip::alg_hint_type /*h*/)
{
  typedef std::auto_ptr<iir_base<T> > ap;
  if (s == vsip::state_no_save)
    return ap(new iir<T, vsip::state_no_save>(a, b, l));
  else
    return ap(new iir<T, vsip::state_save>(a, b, l));
}

template <typename T>
void define_iir(char const *type_name)
{
  typedef iir_base<T> iir_type;

  bpl::class_<iir_type, std::auto_ptr<iir_type>, boost::noncopyable>
    iir(type_name, bpl::no_init);
  iir.def("__init__", bpl::make_constructor(create_iir<T>));
  iir.add_property("kernel_size", &iir_type::kernel_size);
  iir.add_property("input_size", &iir_type::input_size);
  iir.add_property("state", &iir_type::state);
  iir.def("reset", &iir_type::reset);
  iir.def("__call__", &iir_type::op);
}
}

BOOST_PYTHON_MODULE(iir)
{
  using namespace pyvsip;

  define_iir<float>("iirf");
  define_iir<double>("iird");
  define_iir<std::complex<float> >("ciirf");
  define_iir<std::complex<double> >("ciird");
}
