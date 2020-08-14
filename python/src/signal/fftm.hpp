//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef signal_fftm_hpp_
#define signal_fftm_hpp_

#include <ovxx/python/block.hpp>
#include <vsip/signal.hpp>
#include <ovxx/domain_utils.hpp>
#include "../unique_ptr.hpp"

namespace pyvsip
{
namespace bpl = boost::python;

bpl::tuple as_tuple(vsip::Domain<2> const &d)
{ return bpl::make_tuple(d[0].size(), d[1].size());}

template <typename I, typename O, int D> class fftm_base;

template <typename T, int D>
class fftm_base<vsip::complex<T>, vsip::complex<T>, D>
{
  typedef vsip::complex<T> C;
  typedef ovxx::python::Block<2, C> B;

public:
  virtual ~fftm_base() {}
  virtual bpl::tuple input_size() const = 0;
  virtual bpl::tuple output_size() const = 0;
  virtual T scale() const = 0;
  virtual bool forward() const = 0;
  virtual int axis() const = 0;
  virtual void op(B const &, B &) = 0;
  virtual void ip(B &) = 0;
};

template <typename T>
class fftm_base<T, vsip::complex<T>, vsip::fft_fwd>
{
  typedef vsip::complex<T> C;
  typedef ovxx::python::Block<2, T> IB;
  typedef ovxx::python::Block<2, C> OB;

public:
  virtual ~fftm_base() {}
  virtual bpl::tuple input_size() const = 0;
  virtual bpl::tuple output_size() const = 0;
  virtual T scale() const = 0;
  virtual bool forward() const = 0;
  virtual int axis() const = 0;
  virtual void op(IB const &, OB &) = 0;
};

template <typename T>
class fftm_base<vsip::complex<T>, T, vsip::fft_inv>
{
  typedef vsip::complex<T> C;
  typedef ovxx::python::Block<2, C> IB;
  typedef ovxx::python::Block<2, T> OB;

public:
  virtual ~fftm_base() {}
  virtual bpl::tuple input_size() const = 0;
  virtual bpl::tuple output_size() const = 0;
  virtual T scale() const = 0;
  virtual bool forward() const = 0;
  virtual int axis() const = 0;
  virtual void op(IB const &, OB &) = 0;
};

template <typename I, typename O, int A, int D, unsigned N, vsip::alg_hint_type H>
class fftm;

template <typename T, int A, int D, unsigned N, vsip::alg_hint_type H>
class fftm<vsip::complex<T>, vsip::complex<T>, A, D, N, H>
  : public fftm_base<vsip::complex<T>, vsip::complex<T>, D>
{
  typedef vsip::complex<T> C;
  typedef ovxx::python::Block<2, C> B;
  typedef vsip::Matrix<C, B> V;

public:
  fftm(vsip::length_type r, vsip::length_type c, T s)
    : fftm_(vsip::Domain<2>(r, c), s) 
  {}
  virtual bpl::tuple input_size() const { return as_tuple(fftm_.input_size());}
  virtual bpl::tuple output_size() const { return as_tuple(fftm_.output_size());}
  virtual T scale() const { return fftm_.scale();}
  virtual bool forward() const { return fftm_.forward();}
  virtual int axis() const { return A;}
  virtual void op(B const &i, B &o) { fftm_(V(const_cast<B &>(i)), V(o));}
      virtual void ip(B &io) { fftm_(V(io));}
private:
  vsip::Fftm<vsip::complex<T>, vsip::complex<T>,
	     A, D, vsip::by_reference, N, H> fftm_;
};

template <typename T, int A, unsigned N, vsip::alg_hint_type H>
class fftm<T, vsip::complex<T>, A, vsip::fft_fwd, N, H>
  : public fftm_base<T, vsip::complex<T>, vsip::fft_fwd>
{
  typedef vsip::complex<T> C;
  typedef ovxx::python::Block<2, T> IB;
  typedef vsip::Matrix<T, IB> IV;
  typedef ovxx::python::Block<2, C> OB;
  typedef vsip::Matrix<C, OB> OV;

public:
  fftm(vsip::length_type r, vsip::length_type c, T s)
    : fftm_(vsip::Domain<2>(r, c), s) {}
  virtual bpl::tuple input_size() const { return as_tuple(fftm_.input_size());}
  virtual bpl::tuple output_size() const { return as_tuple(fftm_.output_size());}
  virtual T scale() const { return fftm_.scale();}
  virtual bool forward() const { return fftm_.forward();}
  virtual int axis() const { return A;}
  virtual void op(IB const &i, OB &o) { fftm_(IV(const_cast<IB&>(i)), OV(o));}
private:
  vsip::Fftm<T, vsip::complex<T>, A, vsip::fft_fwd, vsip::by_reference, N, H> fftm_;
};

template <typename T, int A, unsigned N, vsip::alg_hint_type H>
class fftm<vsip::complex<T>, T, A, vsip::fft_inv, N, H>
  : public fftm_base<vsip::complex<T>, T, vsip::fft_inv>
{
  typedef vsip::complex<T> C;
  typedef ovxx::python::Block<2, C> IB;
  typedef vsip::Matrix<C, IB> IV;
  typedef ovxx::python::Block<2, T> OB;
  typedef vsip::Matrix<T, OB> OV;

public:
  fftm(vsip::length_type r, vsip::length_type c, T s)
    : fftm_(vsip::Domain<2>(r, c), s) {}
  virtual bpl::tuple input_size() const { return as_tuple(fftm_.input_size());}
  virtual bpl::tuple output_size() const { return as_tuple(fftm_.output_size());}
  virtual T scale() const { return fftm_.scale();}
  virtual int axis() const { return A;}
  virtual bool forward() const { return fftm_.forward();}
  virtual void op(IB const &i, OB &o) {fftm_(IV(const_cast<IB&>(i)), OV(o));}
private:
  vsip::Fftm<vsip::complex<T>, T, A, vsip::fft_inv, vsip::by_reference, N, H> fftm_;
};

template <typename I, typename O, int D>
std::unique_ptr<fftm_base<I, O, D> > 
create_fftm(vsip::length_type rows,
	    vsip::length_type cols,
	    typename ovxx::scalar_of<I>::type scale,
	    int a, unsigned n, vsip::alg_hint_type h)
{
  using ovxx::signal::fft::patient;
  using ovxx::signal::fft::measure;
  using ovxx::signal::fft::estimate;

  using vsip::alg_noise;
  using vsip::alg_space;
  using vsip::alg_time;

  typedef std::unique_ptr<fftm_base<I, O, D> > ap;

  switch (h)
  {
    case alg_noise:
      if (n >= patient) 
	return a == 0 ?
	  ap(new fftm<I, O, 0, D, patient, alg_noise>(rows, cols, scale)) :
	  ap(new fftm<I, O, 1, D, patient, alg_noise>(rows, cols, scale)); 
      else if (n >= measure)
	return a == 0 ?
	  ap(new fftm<I, O, 0, D, measure, alg_noise>(rows, cols, scale)) :
	  ap(new fftm<I, O, 1, D, measure, alg_noise>(rows, cols, scale));
      else if (n >= estimate)
	return a == 0 ?
	  ap(new fftm<I, O, 0, D, estimate, alg_noise>(rows, cols, scale)) :
	  ap(new fftm<I, O, 1, D, estimate, alg_noise>(rows, cols, scale));
      else  
	return a == 0 ?
	  ap(new fftm<I, O, 0, D, 0, alg_noise>(rows, cols, scale)) :
	  ap(new fftm<I, O, 1, D, 0, alg_noise>(rows, cols, scale));
      break;
    case alg_space:
      if (n >= patient) 
	return a == 0 ?
	  ap(new fftm<I, O, 0, D, patient, alg_space>(rows, cols, scale)) :
	  ap(new fftm<I, O, 1, D, patient, alg_space>(rows, cols, scale));
      else if (n >= measure) 
	return a == 0 ?
	  ap(new fftm<I, O, 0, D, measure, alg_space>(rows, cols, scale)) :
	  ap(new fftm<I, O, 1, D, measure, alg_space>(rows, cols, scale));
      else if (n >= estimate)
	return a == 0 ?
	  ap(new fftm<I, O, 0, D, estimate, alg_space>(rows, cols, scale)) :
	  ap(new fftm<I, O, 1, D, estimate, alg_space>(rows, cols, scale));
      else
	return a == 0 ?
	  ap(new fftm<I, O, 0, D, 0, alg_space>(rows, cols, scale)) :
	  ap(new fftm<I, O, 1, D, 0, alg_space>(rows, cols, scale));
      break;
    default:
      if (n >= patient) 
	return a == 0 ?
	  ap(new fftm<I, O, 0, D, patient, alg_time>(rows, cols, scale)) :
	  ap(new fftm<I, O, 1, D, patient, alg_time>(rows, cols, scale));
      else if (n >= measure) 
	return a == 0 ?
	  ap(new fftm<I, O, 0, D, measure, alg_time>(rows, cols, scale)) :
	  ap(new fftm<I, O, 1, D, measure, alg_time>(rows, cols, scale));
      else if (n >= estimate)
	return a == 0 ?
	  ap(new fftm<I, O, 0, D, estimate, alg_time>(rows, cols, scale)) :
	  ap(new fftm<I, O, 1, D, estimate, alg_time>(rows, cols, scale));
      else
	return a == 0 ?
	  ap(new fftm<I, O, 0, D, 0, alg_time>(rows, cols, scale)) :
	  ap(new fftm<I, O, 1, D, 0, alg_time>(rows, cols, scale));
      break;
  }
}

template <typename T>
void define_real_fftm()
{
  typedef vsip::complex<T> C;
  typedef fftm_base<T, C, vsip::fft_fwd> fftm_type;
  typedef fftm_base<C, T, vsip::fft_inv> ifftm_type;

  bpl::class_<fftm_type, std::unique_ptr<fftm_type>, boost::noncopyable>
    fftm("fftm", bpl::no_init);
  fftm.def("__init__", bpl::make_constructor(create_fftm<T, C, vsip::fft_fwd>));
  fftm.add_property("input_size", &fftm_type::input_size);
  fftm.add_property("output_size", &fftm_type::output_size);
  fftm.add_property("scale", &fftm_type::scale);
  fftm.add_property("forward", &fftm_type::forward);
  fftm.add_property("axis", &fftm_type::axis);
  fftm.def("__call__", &fftm_type::op);

  bpl::class_<ifftm_type, std::unique_ptr<ifftm_type>, boost::noncopyable>
    ifftm("ifftm", bpl::no_init);
  ifftm.def("__init__", bpl::make_constructor(create_fftm<C, T, vsip::fft_inv>));
  ifftm.add_property("input_size", &ifftm_type::input_size);
  ifftm.add_property("output_size", &ifftm_type::output_size);
  ifftm.add_property("scale", &ifftm_type::scale);
  ifftm.add_property("forward", &ifftm_type::forward);
  ifftm.add_property("axis", &ifftm_type::axis);
  ifftm.def("__call__", &ifftm_type::op);
}

template <typename T>
void define_complex_fftm()
{
  typedef vsip::complex<T> C;
  typedef fftm_base<C, C, vsip::fft_fwd> fftm_type;
  typedef fftm_base<C, C, vsip::fft_inv> ifftm_type;

  bpl::class_<fftm_type, std::unique_ptr<fftm_type>, boost::noncopyable>
    fftm("fftm", bpl::no_init);
  fftm.def("__init__", bpl::make_constructor(create_fftm<C, C, vsip::fft_fwd>));
  fftm.add_property("input_size", &fftm_type::input_size);
  fftm.add_property("output_size", &fftm_type::output_size);
  fftm.add_property("scale", &fftm_type::scale);
  fftm.add_property("forward", &fftm_type::forward);
  fftm.add_property("axis", &fftm_type::axis);
  fftm.def("__call__", &fftm_type::op);
  fftm.def("__call__", &fftm_type::ip);

  bpl::class_<ifftm_type, std::unique_ptr<ifftm_type>, boost::noncopyable>
    ifftm("ifftm", bpl::no_init);
  ifftm.def("__init__", bpl::make_constructor(create_fftm<C, C, vsip::fft_inv>));
  ifftm.add_property("input_size", &ifftm_type::input_size);
  ifftm.add_property("output_size", &ifftm_type::output_size);
  ifftm.add_property("scale", &ifftm_type::scale);
  ifftm.add_property("forward", &ifftm_type::forward);
  ifftm.add_property("axis", &ifftm_type::axis);
  ifftm.def("__call__", &ifftm_type::op);
  ifftm.def("__call__", &ifftm_type::ip);
}
}

#endif
