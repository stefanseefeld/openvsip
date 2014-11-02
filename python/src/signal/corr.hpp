//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef signal_corr_hpp_
#define signal_corr_hpp_

#include <ovxx/python/block.hpp>
#include <vsip/signal.hpp>

namespace pyvsip
{
namespace bpl = boost::python;

template <typename T>
class corr_base
{
  typedef ovxx::python::Block<1, T> block_type;

public:
  virtual ~corr_base() {};

  virtual vsip::length_type reference_size() const = 0;
  virtual vsip::length_type input_size() const = 0;
  virtual vsip::support_region_type support() const = 0;
  virtual vsip::length_type output_size() const = 0;
  virtual void op(vsip::bias_type, block_type const &, block_type const &, block_type &) = 0;
};

template <typename T, vsip::support_region_type R>
class corr : public corr_base<T>
{
  typedef ovxx::python::Block<1, T> B;
  typedef vsip::Vector<T, B> V;

public:
  corr(vsip::length_type r, vsip::length_type i)
    : corr_(r, i) {}

  virtual vsip::length_type reference_size() const { return corr_.reference_size().size();}
  virtual vsip::length_type input_size() const { return corr_.input_size().size();}
  virtual vsip::support_region_type support() const { return corr_.support();}
  virtual vsip::length_type output_size() const { return corr_.output_size().size();}
  virtual void op(vsip::bias_type b, B const &r, B const &i, B &o)
  { corr_(b, V(const_cast<B&>(r)), V(const_cast<B&>(i)), V(o));}

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
void define_corr()
{
  typedef corr_base<T> corr_type;

  bpl::class_<corr_type, std::auto_ptr<corr_type>, boost::noncopyable>
    corr("corr", bpl::no_init);
  corr.def("__init__", bpl::make_constructor(create_corr<T>));
  corr.add_property("reference_size", &corr_type::reference_size);
  corr.add_property("support", &corr_type::support);
  corr.add_property("input_size", &corr_type::input_size);
  corr.add_property("output_size", &corr_type::output_size);
  corr.def("__call__", &corr_type::op);

}
}

#endif
