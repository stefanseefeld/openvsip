//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef signal_fft_hpp_
#define signal_fft_hpp_

#include <ovxx/python/block.hpp>
#include <vsip/signal.hpp>
#include <ovxx/domain_utils.hpp>

namespace pyvsip
{
namespace bpl = boost::python;

template <typename I, typename O, int D> class fft_base;

template <typename T, int D>
class fft_base<vsip::complex<T>, vsip::complex<T>, D>
{
  typedef vsip::complex<T> C;
  typedef ovxx::python::Block<1, C> block_type;

public:
  virtual ~fft_base() {}
  virtual vsip::length_type input_size() const = 0;
  virtual vsip::length_type output_size() const = 0;
  virtual T scale() const = 0;
  virtual bool forward() const = 0;
  virtual void op(block_type const &, block_type &) = 0;
  virtual void ip(block_type &) = 0;
};

template <typename T>
class fft_base<T, vsip::complex<T>, vsip::fft_fwd>
{
  typedef vsip::complex<T> C;
  typedef ovxx::python::Block<1, T> input_block_type;
  typedef ovxx::python::Block<1, C> output_block_type;

public:
  virtual ~fft_base() {}
  virtual vsip::length_type input_size() const = 0;
  virtual vsip::length_type output_size() const = 0;
  virtual T scale() const = 0;
  virtual bool forward() const = 0;
  virtual void op(input_block_type const &, output_block_type &) = 0;
};

template <typename T>
class fft_base<vsip::complex<T>, T, vsip::fft_inv>
{
  typedef vsip::complex<T> C;
  typedef ovxx::python::Block<1, C> input_block_type;
  typedef ovxx::python::Block<1, T> output_block_type;

public:
  virtual ~fft_base() {}
  virtual vsip::length_type input_size() const = 0;
  virtual vsip::length_type output_size() const = 0;
  virtual T scale() const = 0;
  virtual bool forward() const = 0;
  virtual void op(input_block_type const &, output_block_type &) = 0;
};




template <typename I, typename O, int D, unsigned N, vsip::alg_hint_type H>
class fft;

template <typename T, int D, unsigned N, vsip::alg_hint_type H>
class fft<vsip::complex<T>, vsip::complex<T>, D, N, H>
  : public fft_base<vsip::complex<T>, vsip::complex<T>, D>
{
  typedef vsip::complex<T> C;
  typedef ovxx::python::Block<1, C> block_type;
  typedef vsip::Vector<C, block_type> V;
public:
  fft(vsip::length_type l, T s) : fft_(l, s) {}
  virtual vsip::length_type input_size() const { return fft_.input_size().size();}
  virtual vsip::length_type output_size() const { return fft_.output_size().size();}
  virtual T scale() const { return fft_.scale();}
  virtual bool forward() const { return fft_.forward();}
  virtual void op(block_type const &i, block_type &o) { fft_(V(const_cast<block_type&>(i)), V(o));}
  virtual void ip(block_type &io) { fft_(V(io));}
private:
  vsip::Fft<vsip::Vector, vsip::complex<T>, vsip::complex<T>,
            D, vsip::by_reference, N, H> fft_;
};

template <typename T, unsigned N, vsip::alg_hint_type H>
class fft<T, vsip::complex<T>, vsip::fft_fwd, N, H>
  : public fft_base<T, vsip::complex<T>, vsip::fft_fwd>
{
  typedef vsip::complex<T> C;
  typedef ovxx::python::Block<1, T> input_block_type;
  typedef ovxx::python::Block<1, C> output_block_type;
  typedef vsip::Vector<T, input_block_type> IV;
  typedef vsip::Vector<C, output_block_type> OV;

public:
  fft(vsip::length_type l, T s) : fft_(l, s) {}
  virtual vsip::length_type input_size() const { return fft_.input_size().size();}
  virtual vsip::length_type output_size() const { return fft_.output_size().size();}
  virtual T scale() const { return fft_.scale();}
  virtual bool forward() const { return fft_.forward();}
  virtual void op(input_block_type const &i, output_block_type &o) { fft_(IV(const_cast<input_block_type&>(i)), OV(o));}
private:
  vsip::Fft<vsip::Vector, T, vsip::complex<T>, 0, vsip::by_reference, N, H> fft_;
};

template <typename T, unsigned N, vsip::alg_hint_type H>
class fft<vsip::complex<T>, T, vsip::fft_inv, N, H>
  : public fft_base<vsip::complex<T>, T, vsip::fft_inv>
{
  typedef vsip::complex<T> C;
  typedef ovxx::python::Block<1, C> input_block_type;
  typedef ovxx::python::Block<1, T> output_block_type;
  typedef vsip::Vector<C, input_block_type> IV;
  typedef vsip::Vector<T, output_block_type> OV;

public:
  fft(vsip::length_type l, T s) : fft_(l, s) {}
  virtual vsip::length_type input_size() const { return fft_.input_size().size();}
  virtual vsip::length_type output_size() const { return fft_.output_size().size();}
  virtual T scale() const { return fft_.scale();}
  virtual bool forward() const { return fft_.forward();}
  virtual void op(input_block_type const &i, output_block_type &o) {fft_(IV(const_cast<input_block_type&>(i)), OV(o));}
private:
  vsip::Fft<vsip::Vector, vsip::complex<T>, T, 0, vsip::by_reference, N, H> fft_;
};

template <typename I, typename O, int D>
std::auto_ptr<fft_base<I, O, D> > 
create_fft(vsip::length_type length,
	   typename ovxx::scalar_of<I>::type scale,
	   unsigned n, vsip::alg_hint_type h)
{
  using ovxx::signal::fft::patient;
  using ovxx::signal::fft::measure;
  using ovxx::signal::fft::estimate;

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

template <typename T>
void define_real_fft()
{
  typedef vsip::complex<T> C;
  typedef fft_base<T, C, vsip::fft_fwd> fft_type;
  typedef fft_base<C, T, vsip::fft_inv> ifft_type;

  bpl::class_<fft_type, std::auto_ptr<fft_type>, boost::noncopyable>
    fft("fft", bpl::no_init);
  fft.def("__init__", bpl::make_constructor(create_fft<T, C, vsip::fft_fwd>));
  fft.add_property("input_size", &fft_type::input_size);
  fft.add_property("output_size", &fft_type::output_size);
  fft.add_property("scale", &fft_type::scale);
  fft.add_property("forward", &fft_type::forward);
  fft.def("__call__", &fft_type::op);

  bpl::class_<ifft_type, std::auto_ptr<ifft_type>, boost::noncopyable>
    ifft("ifft", bpl::no_init);
  ifft.def("__init__", bpl::make_constructor(create_fft<C, T, vsip::fft_inv>));
  ifft.add_property("input_size", &ifft_type::input_size);
  ifft.add_property("output_size", &ifft_type::output_size);
  ifft.add_property("scale", &ifft_type::scale);
  ifft.add_property("forward", &ifft_type::forward);
  ifft.def("__call__", &ifft_type::op);
}

template <typename T>
void define_complex_fft()
{
  typedef vsip::complex<T> C;
  typedef fft_base<C, C, vsip::fft_fwd> fft_type;
  typedef fft_base<C, C, vsip::fft_inv> ifft_type;

  bpl::class_<fft_type, std::auto_ptr<fft_type>, boost::noncopyable>
    fft("fft", bpl::no_init);
  fft.def("__init__", bpl::make_constructor(create_fft<C, C, vsip::fft_fwd>));
  fft.add_property("input_size", &fft_type::input_size);
  fft.add_property("output_size", &fft_type::output_size);
  fft.add_property("scale", &fft_type::scale);
  fft.add_property("forward", &fft_type::forward);
  fft.def("__call__", &fft_type::op);
  fft.def("__call__", &fft_type::ip);

  bpl::class_<ifft_type, std::auto_ptr<ifft_type>, boost::noncopyable>
    ifft("ifft", bpl::no_init);
  ifft.def("__init__", bpl::make_constructor(create_fft<C, C, vsip::fft_inv>));
  ifft.add_property("input_size", &ifft_type::input_size);
  ifft.add_property("output_size", &ifft_type::output_size);
  ifft.add_property("scale", &ifft_type::scale);
  ifft.add_property("forward", &ifft_type::forward);
  ifft.def("__call__", &ifft_type::op);
  ifft.def("__call__", &ifft_type::ip);
}

}

#endif
