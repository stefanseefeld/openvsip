/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    python/signal/corr.cpp
    @author  Stefan Seefeld
    @date    2009-09-20
    @brief   VSIPL++ Library: Python bindings for signal module.

*/
#include <boost/python.hpp>
#include <boost/noncopyable.hpp>
#include <vsip/signal.hpp>
#include <vsip/vector.hpp>
#include "../block.hpp"
#include <memory>
#include <string>

namespace pyvsip
{
template <typename T>
class corr_base
{
  typedef vsip::Vector<T, Block<1, T> > view_type;

public:
  virtual ~corr_base() {};

  virtual vsip::length_type reference_size() const = 0;
  virtual vsip::length_type input_size() const = 0;
  virtual vsip::support_region_type support() const = 0;
  virtual vsip::length_type output_size() const = 0;
  virtual void op(vsip::bias_type, view_type, view_type, view_type) = 0;
};

template <typename T, vsip::support_region_type R>
class corr : public corr_base<T>
{
  typedef vsip::Vector<T, Block<1, T> > view_type;

public:
  corr(vsip::length_type r, vsip::length_type i)
    : corr_(r, i) {}

  virtual vsip::length_type reference_size() const { return corr_.reference_size().size();}
  virtual vsip::length_type input_size() const { return corr_.input_size().size();}
  virtual vsip::support_region_type support() const { return corr_.support();}
  virtual vsip::length_type output_size() const { return corr_.output_size().size();}
  virtual void op(vsip::bias_type b, view_type r, view_type i, view_type o) { corr_(b, r, i, o);}

private:
  vsip::Correlation<vsip::const_Vector, R, T> corr_;
};

template <typename T>
std::auto_ptr<corr_base<T> >
create_corr(vsip::length_type r,
	    vsip::length_type i,
	    vsip::support_region_type s,
	    unsigned int /*n*/, vsip::alg_hint_type /*h*/)
{
  using vsip::support_full;
  using vsip::support_same;
  using vsip::support_min;

  typedef std::auto_ptr<corr_base<T> > ap;
  if (s == vsip::support_full)
    return ap(new corr<T, support_full>(r, i));
  else if (s == vsip::support_same)
    return ap(new corr<T, support_same>(r, i));
  else
    return ap(new corr<T, support_min>(r, i));
}

template <typename T>
void define_corr(char const *type_name)
{
  typedef corr_base<T> corr_type;

  bpl::class_<corr_type, std::auto_ptr<corr_type>, boost::noncopyable>
    corr(type_name, bpl::no_init);
  corr.def("__init__", bpl::make_constructor(create_corr<T>));
  corr.add_property("reference_size", &corr_type::reference_size);
  corr.add_property("support", &corr_type::support);
  corr.add_property("input_size", &corr_type::input_size);
  corr.add_property("output_size", &corr_type::output_size);
  corr.def("__call__", &corr_type::op);

}
}

BOOST_PYTHON_MODULE(corr)
{
  using namespace pyvsip;

  define_corr<float>("corrf");
  define_corr<double>("corrd");
  define_corr<std::complex<float> >("ccorrf");
  define_corr<std::complex<double> >("ccorrd");
}
