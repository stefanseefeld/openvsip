//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef signal_conv_hpp_
#define signal_conv_hpp_

#include <ovxx/python/block.hpp>
#include <vsip/signal.hpp>
#include <ovxx/domain_utils.hpp>

namespace pyvsip
{
namespace bpl = boost::python;

template <typename T>
class conv_base
{
  typedef ovxx::python::Block<1, T> block_type;

public:
  virtual ~conv_base() {};

  virtual vsip::length_type kernel_size() const = 0;
  virtual vsip::symmetry_type symmetry() const = 0;
  virtual vsip::length_type input_size() const = 0;
  virtual vsip::support_region_type support() const = 0;
  virtual vsip::length_type output_size() const = 0;
  virtual vsip::length_type decimation() const = 0;
  virtual void op(block_type const &, block_type &) = 0;
};

template <typename T,
	  vsip::symmetry_type S, vsip::support_region_type R>
class conv : public conv_base<T>
{
  typedef ovxx::python::Block<1, T> B;
  typedef vsip::Vector<T, B> V;

public:
  conv(B const &v, vsip::length_type i, vsip::length_type d)
    : conv_(V(const_cast<B&>(v)), i, d) {}

  virtual vsip::length_type kernel_size() const { return conv_.kernel_size().size();}
  virtual vsip::symmetry_type symmetry() const { return conv_.symmetry();}
  virtual vsip::length_type input_size() const { return conv_.input_size().size();}
  virtual vsip::support_region_type support() const { return conv_.support();}
  virtual vsip::length_type output_size() const { return conv_.output_size().size();}
  virtual vsip::length_type decimation() const { return conv_.decimation();}
  virtual void op(B const &i, B &o) { conv_(V(const_cast<B&>(i)), V(o));}

private:
  vsip::Convolution<vsip::const_Vector, S, R, T> conv_;
};

template <typename T>
std::unique_ptr<conv_base<T> >
create_conv(ovxx::python::Block<1, T> const &c,
	    vsip::symmetry_type s,
	    vsip::length_type i,
	    vsip::length_type d,
	    vsip::support_region_type r,
	    unsigned int /*n*/, vsip::alg_hint_type /*h*/)
{
  using vsip::support_full;
  using vsip::support_same;
  using vsip::support_min;

  using vsip::nonsym;
  using vsip::sym_even_len_odd;
  using vsip::sym_even_len_even;


  typedef std::unique_ptr<conv_base<T> > ap;
  if (r == vsip::support_full)
  {
    if (s == vsip::sym_even_len_odd)
      return ap(new conv<T, sym_even_len_odd, support_full>(c, i, d));
    else if (s == vsip::sym_even_len_even)
      return ap(new conv<T, sym_even_len_even, support_full>(c, i, d));
    else
      return ap(new conv<T, nonsym, support_full>(c, i, d));
  }
  else if (r == vsip::support_same)
  {
    if (s == vsip::sym_even_len_odd)
      return ap(new conv<T, sym_even_len_odd, support_same>(c, i, d));
    else if (s == vsip::sym_even_len_even)
      return ap(new conv<T, sym_even_len_even, support_same>(c, i, d));
    else
      return ap(new conv<T, nonsym, support_same>(c, i, d));
  }
  else
  {
    if (s == vsip::sym_even_len_odd)
      return ap(new conv<T, sym_even_len_odd, support_min>(c, i, d));
    else if (s == vsip::sym_even_len_even)
      return ap(new conv<T, sym_even_len_even, support_min>(c, i, d));
    else
      return ap(new conv<T, nonsym, support_min>(c, i, d));
  }
}

template <typename T>
void define_conv()
{
  typedef conv_base<T> conv_type;

  bpl::class_<conv_type, std::unique_ptr<conv_type>, boost::noncopyable>
    conv("conv", bpl::no_init);
  conv.def("__init__", bpl::make_constructor(create_conv<T>));
  conv.add_property("kernel_size", &conv_type::kernel_size);
  conv.add_property("symmetry", &conv_type::symmetry);
  conv.add_property("support", &conv_type::support);
  conv.add_property("decimation", &conv_type::decimation);
  conv.add_property("input_size", &conv_type::input_size);
  conv.add_property("output_size", &conv_type::output_size);
  conv.def("__call__", &conv_type::op);

}

}

#endif
