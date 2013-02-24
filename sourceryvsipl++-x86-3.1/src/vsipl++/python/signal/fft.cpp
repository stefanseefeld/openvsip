/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/// Description
///   VSIPL++ Library: Python bindings for Fft<> classes.

#include <boost/python.hpp>
#include <boost/noncopyable.hpp>
#include <vsip/signal.hpp>
#include <vsip/vector.hpp>
#include "../block.hpp"
#include <memory>
#include <string>

namespace pyvsip
{
template <typename I, typename O, int D> class fft_base;

template <typename T>
class fft_base<T, std::complex<T>, vsip::fft_fwd>
{
  typedef std::complex<T> C;
  typedef vsip::Vector<T, Block<1, T> > input_view_type;
  typedef vsip::Vector<C, Block<1, C> > output_view_type;

public:
  virtual ~fft_base() {}
  virtual vsip::length_type input_size() const = 0;
  virtual vsip::length_type output_size() const = 0;
  virtual T scale() const = 0;
  virtual bool forward() const = 0;
  virtual void op(input_view_type, output_view_type) = 0;
};

template <typename T>
class fft_base<std::complex<T>, T, vsip::fft_inv>
{
  typedef std::complex<T> C;
  typedef vsip::Vector<C, Block<1, C> > input_view_type;
  typedef vsip::Vector<T, Block<1, T> > output_view_type;

public:
  virtual ~fft_base() {}
  virtual vsip::length_type input_size() const = 0;
  virtual vsip::length_type output_size() const = 0;
  virtual T scale() const = 0;
  virtual bool forward() const = 0;
  virtual void op(input_view_type, output_view_type) = 0;
};

template <typename T, int D>
class fft_base<std::complex<T>, std::complex<T>, D>
{
  typedef std::complex<T> C;
  typedef vsip::Vector<C, Block<1, C> > view_type;

public:
  virtual ~fft_base() {}
  virtual vsip::length_type input_size() const = 0;
  virtual vsip::length_type output_size() const = 0;
  virtual T scale() const = 0;
  virtual bool forward() const = 0;
  virtual void op(view_type, view_type) = 0;
  virtual void ip(view_type) = 0;
};

template <typename I, typename O, int D, unsigned N, vsip::alg_hint_type H>
class fft;

template <typename T, int D, unsigned N, vsip::alg_hint_type H>
class fft<std::complex<T>, std::complex<T>, D, N, H>
  : public fft_base<std::complex<T>, std::complex<T>, D>
{
  typedef std::complex<T> C;
  typedef vsip::Vector<C, Block<1, C> > view_type;

public:
  fft(vsip::length_type l, T s) : fft_(l, s) {}
  virtual vsip::length_type input_size() const { return fft_.input_size().size();}
  virtual vsip::length_type output_size() const { return fft_.output_size().size();}
  virtual T scale() const { return fft_.scale();}
  virtual bool forward() const { return fft_.forward();}
  virtual void op(view_type i, view_type o) { fft_(i, o);}
  virtual void ip(view_type io) { fft_(io);}
private:
  vsip::Fft<vsip::Vector, std::complex<T>, std::complex<T>,
            D, vsip::by_reference, N, H> fft_;
};

template <typename T, unsigned N, vsip::alg_hint_type H>
class fft<T, std::complex<T>, vsip::fft_fwd, N, H>
  : public fft_base<T, std::complex<T>, vsip::fft_fwd>
{
  typedef std::complex<T> C;
  typedef vsip::Vector<T, Block<1, T> > input_view_type;
  typedef vsip::Vector<C, Block<1, C> > output_view_type;

public:
  fft(vsip::length_type l, T s) : fft_(l, s) {}
  virtual vsip::length_type input_size() const { return fft_.input_size().size();}
  virtual vsip::length_type output_size() const { return fft_.output_size().size();}
  virtual T scale() const { return fft_.scale();}
  virtual bool forward() const { return fft_.forward();}
  virtual void op(input_view_type i, output_view_type o) { fft_(i, o);}
private:
  vsip::Fft<vsip::Vector, T, std::complex<T>, 0, vsip::by_reference, N, H> fft_;
};

template <typename T, unsigned N, vsip::alg_hint_type H>
class fft<std::complex<T>, T, vsip::fft_inv, N, H>
  : public fft_base<std::complex<T>, T, vsip::fft_inv>
{
  typedef std::complex<T> C;
  typedef vsip::Vector<C, Block<1, C> > input_view_type;
  typedef vsip::Vector<T, Block<1, T> > output_view_type;

public:
  fft(vsip::length_type l, T s) : fft_(l, s) {}
  virtual vsip::length_type input_size() const { return fft_.input_size().size();}
  virtual vsip::length_type output_size() const { return fft_.output_size().size();}
  virtual T scale() const { return fft_.scale();}
  virtual bool forward() const { return fft_.forward();}
  virtual void op(input_view_type i, output_view_type o) {fft_(i, o);}
private:
  vsip::Fft<vsip::Vector, std::complex<T>, T, 0, vsip::by_reference, N, H> fft_;
};

template <typename I, typename O, int D>
std::auto_ptr<fft_base<I, O, D> > 
create_fft(vsip::length_type length,
	   typename vsip::impl::scalar_of<I>::type scale,
	   unsigned n, vsip::alg_hint_type h)
{
  using vsip::impl::fft::patient;
  using vsip::impl::fft::measure;
  using vsip::impl::fft::estimate;

  using vsip::alg_noise;
  using vsip::alg_space;
  using vsip::alg_time;

  typedef std::auto_ptr<fft_base<I, O, D> > ap;

  switch (h)
  {
    case alg_noise:
      if (n >= patient) 
	return ap(new fft<I, O, D, patient, alg_noise>(length, scale));
      else if (n >= measure) 
	return ap(new fft<I, O, D, measure, alg_noise>(length, scale));
      else if (n >= estimate)
	return ap(new fft<I, O, D, estimate, alg_noise>(length, scale));
      else  
	return ap(new fft<I, O, D, 0, alg_noise>(length, scale));
      break;
    case alg_space:
      if (n >= patient) 
	return ap(new fft<I, O, D, patient, alg_space>(length, scale));
      else if (n >= measure) 
	return ap(new fft<I, O, D, measure, alg_space>(length, scale));
      else if (n >= estimate)
	return ap(new fft<I, O, D, estimate, alg_space>(length, scale));
      else
        return ap(new fft<I, O, D, 0, alg_space>(length, scale));
      break;
    default:
      if (n >= patient) 
	return ap(new fft<I, O, D, patient, alg_time>(length, scale));
      else if (n >= measure) 
	return ap(new fft<I, O, D, measure, alg_time>(length, scale));
      else if (n >= estimate)
	return ap(new fft<I, O, D, estimate, alg_time>(length, scale));
      else
	return ap(new fft<I, O, D, 0, alg_time>(length, scale));
      break;
  }
}

bpl::object create_fft_from_type(bpl::object type, int D,
				 vsip::length_type length,
				 double scale,
				 unsigned n, vsip::alg_hint_type h)
{
  if (PyType_Check(type.ptr()))
  {
    if (type.ptr() == (PyObject*)&PyFloat_Type ||
	type.ptr() == (PyObject*)&PyFloat64ArrType_Type)
      if (D == vsip::fft_fwd)
	return bpl::object(create_fft<double, std::complex<double>, vsip::fft_fwd>
			   (length, scale, n, h));
      else
	return bpl::object(create_fft<std::complex<double>, double, vsip::fft_inv>
			   (length, scale, n, h));
    else if (type.ptr() == (PyObject*)&PyComplex_Type)
      if (D == vsip::fft_fwd)
	return bpl::object(create_fft<std::complex<double>, std::complex<double>, vsip::fft_fwd>
			   (length, scale, n, h));
      else
	return bpl::object(create_fft<std::complex<double>, std::complex<double>, vsip::fft_inv>
			   (length, scale, n, h));
    else throw std::runtime_error("unsupported type");
  }
  else throw std::runtime_error("argument 1 not a type");
}

template <typename T>
void define_fft(char typecode)
{
  typedef std::complex<T> C;
  typedef fft_base<C, C, vsip::fft_fwd> fft_type;
  typedef fft_base<C, C, vsip::fft_inv> ifft_type;
  typedef fft_base<T, C, vsip::fft_fwd> rfft_type;
  typedef fft_base<C, T, vsip::fft_inv> irfft_type;

  std::string type_name = "fft";
  type_name += typecode;
  bpl::class_<fft_type, std::auto_ptr<fft_type>, boost::noncopyable>
    fft(type_name.c_str(), bpl::no_init);
  fft.def("__init__", bpl::make_constructor(create_fft<C, C, vsip::fft_fwd>));
  fft.add_property("input_size", &fft_type::input_size);
  fft.add_property("output_size", &fft_type::output_size);
  fft.add_property("scale", &fft_type::scale);
  fft.add_property("forward", &fft_type::forward);
  fft.def("__call__", &fft_type::op);
  fft.def("__call__", &fft_type::ip);

  type_name = "ifft";
  type_name += typecode;
  bpl::class_<ifft_type, std::auto_ptr<ifft_type>, boost::noncopyable>
    ifft(type_name.c_str(), bpl::no_init);
  ifft.def("__init__", bpl::make_constructor(create_fft<C, C, vsip::fft_inv>));
  ifft.add_property("input_size", &ifft_type::input_size);
  ifft.add_property("output_size", &ifft_type::output_size);
  ifft.add_property("scale", &ifft_type::scale);
  ifft.add_property("forward", &ifft_type::forward);
  ifft.def("__call__", &ifft_type::op);
  ifft.def("__call__", &ifft_type::ip);

  type_name = "rfft";
  type_name += typecode;
  bpl::class_<rfft_type, std::auto_ptr<rfft_type>, boost::noncopyable>
    rfft(type_name.c_str(), bpl::no_init);
  rfft.def("__init__", bpl::make_constructor(create_fft<T, C, vsip::fft_fwd>));
  rfft.add_property("input_size", &rfft_type::input_size);
  rfft.add_property("output_size", &rfft_type::output_size);
  rfft.add_property("scale", &rfft_type::scale);
  rfft.add_property("forward", &rfft_type::forward);
  rfft.def("__call__", &rfft_type::op);

  type_name = "irfft";
  type_name += typecode;
  bpl::class_<irfft_type, std::auto_ptr<irfft_type>, boost::noncopyable>
    irfft(type_name.c_str(), bpl::no_init);
  irfft.def("__init__", bpl::make_constructor(create_fft<C, T, vsip::fft_inv>));
  irfft.add_property("input_size", &irfft_type::input_size);
  irfft.add_property("output_size", &irfft_type::output_size);
  irfft.add_property("scale", &irfft_type::scale);
  irfft.add_property("forward", &irfft_type::forward);
  irfft.def("__call__", &irfft_type::op);
}
}

BOOST_PYTHON_MODULE(fft)
{
  using namespace pyvsip;

  define_fft<float>('f');
  define_fft<double>('d');
  bpl::def("fft", pyvsip::create_fft_from_type);
}
