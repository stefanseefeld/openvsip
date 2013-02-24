/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   VSIPL++ Library: Python bindings for convolution.

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
class conv_base
{
  typedef vsip::Vector<T, Block<1, T> > view_type;

public:
  virtual ~conv_base() {};

  virtual vsip::length_type kernel_size() const = 0;
  virtual vsip::symmetry_type symmetry() const = 0;
  virtual vsip::length_type input_size() const = 0;
  virtual vsip::support_region_type support() const = 0;
  virtual vsip::length_type output_size() const = 0;
  virtual vsip::length_type decimation() const = 0;
  virtual void op(view_type, view_type) = 0;
};

template <typename T,
	  vsip::symmetry_type S, vsip::support_region_type R>
class conv : public conv_base<T>
{
  typedef vsip::Vector<T, Block<1, T> > view_type;

public:
  conv(view_type v, vsip::length_type i, vsip::length_type d)
    : conv_(v, i, d) {}

  virtual vsip::length_type kernel_size() const { return conv_.kernel_size().size();}
  virtual vsip::symmetry_type symmetry() const { return conv_.symmetry();}
  virtual vsip::length_type input_size() const { return conv_.input_size().size();}
  virtual vsip::support_region_type support() const { return conv_.support();}
  virtual vsip::length_type output_size() const { return conv_.output_size().size();}
  virtual vsip::length_type decimation() const { return conv_.decimation();}
  virtual void op(view_type i, view_type o) { conv_(i, o);}

private:
  vsip::Convolution<vsip::const_Vector, S, R, T> conv_;
};

template <typename T>
std::auto_ptr<conv_base<T> >
create_conv(vsip::Vector<T, Block<1, T> > c,
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


  typedef std::auto_ptr<conv_base<T> > ap;
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

bpl::object
create_conv_from_type(bpl::object c,
		      vsip::symmetry_type s,
		      vsip::length_type i,
		      vsip::length_type d,
		      vsip::support_region_type r,
		      unsigned int n, vsip::alg_hint_type h)
{
  bpl::extract<vsip::Vector<float, Block<1, float> > &> ef(c);
  bpl::extract<vsip::Vector<double, Block<1, double> > &> ed(c);
  bpl::extract<vsip::Vector<std::complex<float>, Block<1, std::complex<float> > > &> ecf(c);
  bpl::extract<vsip::Vector<std::complex<double>, Block<1, std::complex<double> > > &> ecd(c);
  if (ef.check())
  {
    vsip::Vector<float, Block<1, float> > &coeff = ef();
    return bpl::object(create_conv<float>(coeff, s, i, d, r, n, h));
  }
  else if (ed.check())
  {
    vsip::Vector<double, Block<1, double> > &coeff = ed();
    return bpl::object(create_conv<double>(coeff, s, i, d, r, n, h));
  }
  else if (ecf.check())
  {
    vsip::Vector<std::complex<float>, Block<1, std::complex<float> > > &coeff = ecf();
    return bpl::object(create_conv<std::complex<float> >(coeff, s, i, d, r, n, h));
  }
  else if (ecd.check())
  {
    vsip::Vector<std::complex<double>, Block<1, std::complex<double> > > &coeff = ecd();
    return bpl::object(create_conv<std::complex<double> >(coeff, s, i, d, r, n, h));
  }
  else throw std::runtime_error("unsupported type");
}

template <typename T>
void define_conv(char const *type_name)
{
  typedef conv_base<T> conv_type;

  bpl::class_<conv_type, std::auto_ptr<conv_type>, boost::noncopyable>
    conv(type_name, bpl::no_init);
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

BOOST_PYTHON_MODULE(conv)
{
  using namespace pyvsip;

  define_conv<float>("convf");
  define_conv<double>("convd");
  define_conv<std::complex<float> >("cconvf");
  define_conv<std::complex<double> >("cconvd");
  bpl::def("convolution", create_conv_from_type);
}
