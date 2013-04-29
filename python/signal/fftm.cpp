/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/// Description
///   VSIPL++ Library: Python bindings for Fftm<> classes.

#include <boost/python.hpp>
#include <boost/noncopyable.hpp>
#include <vsip/signal.hpp>
#include <vsip/matrix.hpp>
#include "../block.hpp"
#include <memory>
#include <string>

namespace pyvsip
{
bpl::tuple as_tuple(vsip::Domain<2> const &d)
{ return bpl::make_tuple(d[0].size(), d[1].size());}

template <typename I, typename O, int D> class fftm_base;

template <typename T>
class fftm_base<T, std::complex<T>, vsip::fft_fwd>
{
  typedef std::complex<T> C;
  typedef vsip::Matrix<T, Block<2, T> > input_view_type;
  typedef vsip::Matrix<C, Block<2, C> > output_view_type;

public:
  virtual ~fftm_base() {}
  virtual bpl::tuple input_size() const = 0;
  virtual bpl::tuple output_size() const = 0;
  virtual T scale() const = 0;
  virtual bool forward() const = 0;
  virtual int axis() const = 0;
  virtual void op(input_view_type, output_view_type) = 0;
};

template <typename T>
class fftm_base<std::complex<T>, T, vsip::fft_inv>
{
  typedef std::complex<T> C;
  typedef vsip::Matrix<C, Block<2, C> > input_view_type;
  typedef vsip::Matrix<T, Block<2, T> > output_view_type;

public:
  virtual ~fftm_base() {}
  virtual bpl::tuple input_size() const = 0;
  virtual bpl::tuple output_size() const = 0;
  virtual T scale() const = 0;
  virtual bool forward() const = 0;
  virtual int axis() const = 0;
  virtual void op(input_view_type, output_view_type) = 0;
};

template <typename T, int D>
class fftm_base<std::complex<T>, std::complex<T>, D>
{
  typedef std::complex<T> C;
  typedef vsip::Matrix<C, Block<2, C> > view_type;

public:
  virtual ~fftm_base() {}
  virtual bpl::tuple input_size() const = 0;
  virtual bpl::tuple output_size() const = 0;
  virtual T scale() const = 0;
  virtual bool forward() const = 0;
  virtual int axis() const = 0;
  virtual void op(view_type, view_type) = 0;
  virtual void ip(view_type) = 0;
};

template <typename I, typename O, int A, int D, unsigned N, vsip::alg_hint_type H>
class fftm;

template <typename T, int A, int D, unsigned N, vsip::alg_hint_type H>
class fftm<std::complex<T>, std::complex<T>, A, D, N, H>
  : public fftm_base<std::complex<T>, std::complex<T>, D>
{
  typedef std::complex<T> C;
  typedef vsip::Matrix<C, Block<2, C> > view_type;

public:
  fftm(vsip::length_type r, vsip::length_type c, T s)
    : fftm_(vsip::Domain<2>(r, c), s) {}
  virtual bpl::tuple input_size() const { return as_tuple(fftm_.input_size());}
  virtual bpl::tuple output_size() const { return as_tuple(fftm_.output_size());}
  virtual T scale() const { return fftm_.scale();}
  virtual bool forward() const { return fftm_.forward();}
  virtual int axis() const { return A;}
  virtual void op(view_type i, view_type o) { fftm_(i, o);}
  virtual void ip(view_type io) { fftm_(io);}
private:
  vsip::Fftm<std::complex<T>, std::complex<T>,
	     A, D, vsip::by_reference, N, H> fftm_;
};

template <typename T, int A, unsigned N, vsip::alg_hint_type H>
class fftm<T, std::complex<T>, A, vsip::fft_fwd, N, H>
  : public fftm_base<T, std::complex<T>, vsip::fft_fwd>
{
  typedef std::complex<T> C;
  typedef vsip::Matrix<T, Block<2, T> > input_view_type;
  typedef vsip::Matrix<C, Block<2, C> > output_view_type;

public:
  fftm(vsip::length_type r, vsip::length_type c, T s)
    : fftm_(vsip::Domain<2>(r, c), s) {}
  virtual bpl::tuple input_size() const { return as_tuple(fftm_.input_size());}
  virtual bpl::tuple output_size() const { return as_tuple(fftm_.output_size());}
  virtual T scale() const { return fftm_.scale();}
  virtual bool forward() const { return fftm_.forward();}
  virtual int axis() const { return A;}
  virtual void op(input_view_type i, output_view_type o) { fftm_(i, o);}
private:
  vsip::Fftm<T, std::complex<T>, A, vsip::fft_fwd, vsip::by_reference, N, H> fftm_;
};

template <typename T, int A, unsigned N, vsip::alg_hint_type H>
class fftm<std::complex<T>, T, A, vsip::fft_inv, N, H>
  : public fftm_base<std::complex<T>, T, vsip::fft_inv>
{
  typedef std::complex<T> C;
  typedef vsip::Matrix<C, Block<2, C> > input_view_type;
  typedef vsip::Matrix<T, Block<2, T> > output_view_type;

public:
  fftm(vsip::length_type r, vsip::length_type c, T s)
    : fftm_(vsip::Domain<2>(r, c), s) {}
  virtual bpl::tuple input_size() const { return as_tuple(fftm_.input_size());}
  virtual bpl::tuple output_size() const { return as_tuple(fftm_.output_size());}
  virtual T scale() const { return fftm_.scale();}
  virtual int axis() const { return A;}
  virtual bool forward() const { return fftm_.forward();}
  virtual void op(input_view_type i, output_view_type o) {fftm_(i, o);}
private:
  vsip::Fftm<std::complex<T>, T, A, vsip::fft_inv, vsip::by_reference, N, H> fftm_;
};

template <typename I, typename O, int D>
std::auto_ptr<fftm_base<I, O, D> > 
create_fftm(vsip::length_type rows,
	    vsip::length_type cols,
	    typename vsip::impl::scalar_of<I>::type scale,
	    int a, unsigned n, vsip::alg_hint_type h)
{
  using vsip::impl::fft::patient;
  using vsip::impl::fft::measure;
  using vsip::impl::fft::estimate;

  using vsip::alg_noise;
  using vsip::alg_space;
  using vsip::alg_time;

  typedef std::auto_ptr<fftm_base<I, O, D> > ap;

  switch (h)
  {
    case alg_noise:
      if (n >= patient) 
	return a == 0 ?
	  ap(new fftm<I, O, 0, D, patient, alg_noise>(row, cols, scale)) :
	  ap(new fftm<I, O, 1, D, patient, alg_noise>(row, cols, scale)); 
      else if (n >= measure)
	return a == 0 ?
	  ap(new fftm<I, O, 0, D, measure, alg_noise>(row, cols, scale)) :
	  ap(new fftm<I, O, 1, D, measure, alg_noise>(row, cols, scale));
      else if (n >= estimate)
	return a == 0 ?
	  ap(new fftm<I, O, 0, D, estimate, alg_noise>(row, cols, scale)) :
	  ap(new fftm<I, O, 1, D, estimate, alg_noise>(row, cols, scale));
      else  
	return a == 0 ?
	  ap(new fftm<I, O, 0, D, 0, alg_noise>(row, cols, scale)) :
	  ap(new fftm<I, O, 1, D, 0, alg_noise>(row, cols, scale));
      break;
    case alg_space:
      if (n >= patient) 
	return a == 0 ?
	  ap(new fftm<I, O, 0, D, patient, alg_space>(row, cols, scale)) :
	  ap(new fftm<I, O, 1, D, patient, alg_space>(row, cols, scale));
      else if (n >= measure) 
	return a == 0 ?
	  ap(new fftm<I, O, 0, D, measure, alg_space>(row, cols, scale)) :
	  ap(new fftm<I, O, 1, D, measure, alg_space>(row, cols, scale));
      else if (n >= estimate)
	return a == 0 ?
	  ap(new fftm<I, O, 0, D, estimate, alg_space>(row, cols, scale)) :
	  ap(new fftm<I, O, 1, D, estimate, alg_space>(row, cols, scale));
      else
	return a == 0 ?
	  ap(new fftm<I, O, 0, D, 0, alg_space>(row, cols, scale)) :
	  ap(new fftm<I, O, 1, D, 0, alg_space>(row, cols, scale));
      break;
    default:
      if (n >= patient) 
	return a == 0 ?
	  ap(new fftm<I, O, 0, D, patient, alg_time>(row, cols, scale)) :
	  ap(new fftm<I, O, 1, D, patient, alg_time>(row, cols, scale));
      else if (n >= measure) 
	return a == 0 ?
	  ap(new fftm<I, O, 0, D, measure, alg_time>(row, cols, scale)) :
	  ap(new fftm<I, O, 1, D, measure, alg_time>(row, cols, scale));
      else if (n >= estimate)
	return a == 0 ?
	  ap(new fftm<I, O, 0, D, estimate, alg_time>(row, cols, scale)) :
	  ap(new fftm<I, O, 1, D, estimate, alg_time>(row, cols, scale));
      else
	return a == 0 ?
	  ap(new fftm<I, O, 0, D, 0, alg_time>(row, cols, scale)) :
	  ap(new fftm<I, O, 1, D, 0, alg_time>(row, cols, scale));
      break;
  }
}

bpl::object create_fftm_from_type(bpl::object type, int D,
				  vsip::length_type rows,
				  vsip::length_type cols,
				  double scale,
				  int a, unsigned n, vsip::alg_hint_type h)
{
  if (PyType_Check(type.ptr()))
  {
    if (type.ptr() == (PyObject*)&PyFloat_Type)
      if (D == vsip::fft_fwd)
	return bpl::object(create_fftm<double, std::complex<double>, vsip::fft_fwd>
			   (rows, cols, scale, a, n, h));
      else
	return bpl::object(create_fftm<std::complex<double>, double, vsip::fft_inv>
			   (rows, cols, scale, a, n, h));
    else if (type.ptr() == (PyObject*)&PyComplex_Type)
      if (D == vsip::fft_fwd)
	return bpl::object(create_fftm<std::complex<double>, std::complex<double>, vsip::fft_fwd>
			   (rows, cols, scale, a, n, h));
      else
	return bpl::object(create_fftm<std::complex<double>, std::complex<double>, vsip::fft_inv>
			   (rows, cols, scale, a, n, h));
    else throw std::runtime_error("unsupported type");
  }
  else throw std::runtime_error("argument 1 not a type");
}

template <typename T>
void define_fftm(char typecode)
{
  typedef std::complex<T> C;
  typedef fftm_base<C, C, vsip::fft_fwd> fftm_type;
  typedef fftm_base<C, C, vsip::fft_inv> ifftm_type;
  typedef fftm_base<T, C, vsip::fft_fwd> rfftm_type;
  typedef fftm_base<C, T, vsip::fft_inv> irfftm_type;

  std::string type_name = "fftm";
  type_name += typecode;
  bpl::class_<fftm_type, std::auto_ptr<fftm_type>, boost::noncopyable>
    fftm(type_name.c_str(), bpl::no_init);
  fftm.def("__init__", bpl::make_constructor(create_fftm<C, C, vsip::fft_fwd>));
  fftm.add_property("input_size", &fftm_type::input_size);
  fftm.add_property("output_size", &fftm_type::output_size);
  fftm.add_property("scale", &fftm_type::scale);
  fftm.add_property("forward", &fftm_type::forward);
  fftm.add_property("axis", &fftm_type::axis);
  fftm.def("__call__", &fftm_type::op);
  fftm.def("__call__", &fftm_type::ip);

  type_name = "ifftm";
  type_name += typecode;
  bpl::class_<ifftm_type, std::auto_ptr<ifftm_type>, boost::noncopyable>
    ifftm(type_name.c_str(), bpl::no_init);
  ifftm.def("__init__", bpl::make_constructor(create_fftm<C, C, vsip::fft_inv>));
  ifftm.add_property("input_size", &ifftm_type::input_size);
  ifftm.add_property("output_size", &ifftm_type::output_size);
  ifftm.add_property("scale", &ifftm_type::scale);
  ifftm.add_property("forward", &ifftm_type::forward);
  ifftm.add_property("axis", &ifftm_type::axis);
  ifftm.def("__call__", &ifftm_type::op);
  ifftm.def("__call__", &ifftm_type::ip);

  type_name = "rfftm";
  type_name += typecode;
  bpl::class_<rfftm_type, std::auto_ptr<rfftm_type>, boost::noncopyable>
    rfftm(type_name.c_str(), bpl::no_init);
  rfftm.def("__init__", bpl::make_constructor(create_fftm<T, C, vsip::fft_fwd>));
  rfftm.add_property("input_size", &rfftm_type::input_size);
  rfftm.add_property("output_size", &rfftm_type::output_size);
  rfftm.add_property("scale", &rfftm_type::scale);
  rfftm.add_property("forward", &rfftm_type::forward);
  rfftm.add_property("axis", &rfftm_type::axis);
  rfftm.def("__call__", &rfftm_type::op);

  type_name = "irfftm";
  type_name += typecode;
  bpl::class_<irfftm_type, std::auto_ptr<irfftm_type>, boost::noncopyable>
    irfftm(type_name.c_str(), bpl::no_init);
  irfftm.def("__init__", bpl::make_constructor(create_fftm<C, T, vsip::fft_inv>));
  irfftm.add_property("input_size", &irfftm_type::input_size);
  irfftm.add_property("output_size", &irfftm_type::output_size);
  irfftm.add_property("scale", &irfftm_type::scale);
  irfftm.add_property("forward", &irfftm_type::forward);
  irfftm.add_property("axis", &irfftm_type::axis);
  irfftm.def("__call__", &irfftm_type::op);
}
}

BOOST_PYTHON_MODULE(fftm)
{
  using namespace pyvsip;

  define_fftm<float>('f');
  define_fftm<double>('d');
  bpl::def("fftm", pyvsip::create_fftm_from_type);
}
