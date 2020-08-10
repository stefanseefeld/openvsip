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
class iir_base
{
  typedef ovxx::python::Block<1, T> block_type;

public:
  virtual ~iir_base() {};

  virtual vsip::length_type kernel_size() const = 0;
  virtual vsip::length_type input_size() const = 0;
  virtual vsip::obj_state state() const = 0;
  virtual void reset() = 0;
  virtual void op(block_type const &, block_type &) = 0;
};

template <typename T, vsip::obj_state S>
class iir : public iir_base<T>
{
  typedef ovxx::python::Block<2, T> B2;
  typedef vsip::Matrix<T, B2> M;
  typedef ovxx::python::Block<1, T> B;
  typedef vsip::Vector<T, B> V;

public:
  iir(B2 const &a, B2 const &b, vsip::length_type l)
    : iir_(M(const_cast<B2&>(a)), M(const_cast<B2&>(b)), l) {}

  virtual vsip::length_type kernel_size() const { return iir_.kernel_size();}
  virtual vsip::length_type input_size() const { return iir_.input_size();}
  virtual vsip::obj_state state() const { return S;}
  virtual void reset() { iir_.reset();}
  virtual void op(B const &i, B &o) { iir_(V(const_cast<B&>(i)), V(o));}

private:
  vsip::Iir<T, S> iir_;
};

template <typename T>
std::unique_ptr<iir_base<T> >
create_iir(ovxx::python::Block<2, T> const &a,
	   ovxx::python::Block<2, T> const &b,
	   vsip::length_type l,
	   vsip::obj_state s,
	   unsigned int /*n*/, vsip::alg_hint_type /*h*/)
{
  typedef std::unique_ptr<iir_base<T> > ap;
  if (s == vsip::state_no_save)
    return ap(new iir<T, vsip::state_no_save>(a, b, l));
  else
    return ap(new iir<T, vsip::state_save>(a, b, l));
}

template <typename T>
void define_iir()
{
  typedef iir_base<T> iir_type;

  bpl::class_<iir_type, std::unique_ptr<iir_type>, boost::noncopyable>
    iir("iir", bpl::no_init);
  iir.def("__init__", bpl::make_constructor(create_iir<T>));
  iir.add_property("kernel_size", &iir_type::kernel_size);
  iir.add_property("input_size", &iir_type::input_size);
  iir.add_property("state", &iir_type::state);
  iir.def("reset", &iir_type::reset);
  iir.def("__call__", &iir_type::op);
}
}

#endif
