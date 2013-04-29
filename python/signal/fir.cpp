/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    python/signal/fir.cpp
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
class fir_base
{
  typedef vsip::Vector<T, Block<1, T> > view_type;

public:
  virtual ~fir_base() {};

  virtual vsip::length_type kernel_size() const = 0;
  virtual vsip::symmetry_type symmetry() const = 0;
  virtual vsip::length_type input_size() const = 0;
  virtual vsip::length_type output_size() const = 0;
  virtual vsip::length_type decimation() const = 0;
  virtual vsip::obj_state state() const = 0;
  virtual void reset() = 0;
  virtual void op(view_type, view_type) = 0;
};

template <typename T, vsip::symmetry_type S, vsip::obj_state C>
class fir : public fir_base<T>
{
  typedef vsip::Vector<T, Block<1, T> > view_type;

public:
  fir(view_type k, vsip::length_type l, vsip::length_type d)
    : fir_(k, l, d) {}

  virtual vsip::length_type kernel_size() const { return fir_.kernel_size();}
  virtual vsip::symmetry_type symmetry() const { return S;}
  virtual vsip::length_type input_size() const { return fir_.input_size();}
  virtual vsip::length_type output_size() const { return fir_.output_size();}
  virtual vsip::length_type decimation() const { return fir_.decimation();}
  virtual vsip::obj_state state() const { return C;}
  virtual void reset() { fir_.reset();}
  virtual void op(view_type i, view_type o) { fir_(i, o);}

private:
  vsip::Fir<T, S, C> fir_;
};

template <typename T>
std::auto_ptr<fir_base<T> >
create_fir(vsip::Vector<T, Block<1, T> > k,
	   vsip::symmetry_type s,
	   vsip::length_type l,
	   vsip::length_type d,
	   vsip::obj_state c,
	   unsigned int /*n*/, vsip::alg_hint_type /*h*/)
{
  using vsip::nonsym;
  using vsip::sym_even_len_odd;
  using vsip::sym_even_len_even;

  using vsip::state_no_save;
  using vsip::state_save;

  typedef std::auto_ptr<fir_base<T> > ap;
  if (s == sym_even_len_even)
  {
    if (c == state_no_save)
      return ap(new fir<T, sym_even_len_even, state_no_save>(k, l, d));
    else
      return ap(new fir<T, sym_even_len_even, state_save>(k, l, d));
  }
  else if (s == sym_even_len_odd)
  {
    if (c == state_no_save)
      return ap(new fir<T, sym_even_len_odd, state_no_save>(k, l, d));
    else
      return ap(new fir<T, sym_even_len_odd, state_save>(k, l, d));
  }
  else
  {
    if (c == state_no_save)
      return ap(new fir<T, nonsym, state_no_save>(k, l, d));
    else
      return ap(new fir<T, nonsym, state_save>(k, l, d));
  }
}

template <typename T>
void define_fir(char const *type_name)
{
  typedef fir_base<T> fir_type;

  bpl::class_<fir_type, std::auto_ptr<fir_type>, boost::noncopyable>
    fir(type_name, bpl::no_init);
  fir.def("__init__", bpl::make_constructor(create_fir<T>));
  fir.add_property("kernel_size", &fir_type::kernel_size);
  fir.add_property("input_size", &fir_type::input_size);
  fir.add_property("output_size", &fir_type::output_size);
  fir.add_property("decimation", &fir_type::decimation);
  fir.add_property("symmetry", &fir_type::symmetry);
  fir.add_property("state", &fir_type::state);
  fir.def("reset", &fir_type::reset);
  fir.def("__call__", &fir_type::op);
}
}

BOOST_PYTHON_MODULE(fir)
{
  using namespace pyvsip;

  define_fir<float>("firf");
  define_fir<double>("fird");
  define_fir<std::complex<float> >("cfirf");
  define_fir<std::complex<double> >("cfird");
}
