//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_CORE_SIGNAL_IIR_HPP
#define VSIP_CORE_SIGNAL_IIR_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/core/signal/types.hpp>
#include <vsip/core/profile.hpp>

/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

template <typename      T = VSIP_DEFAULT_VALUE_TYPE,
	  obj_state     C = state_save,
	  unsigned      N = 0,
	  alg_hint_type H = alg_time>
class Iir : impl::profile::Accumulator<impl::profile::signal>
{
  typedef impl::profile::Accumulator<impl::profile::signal> accumulator_type;

public:
  static obj_state const continuous_filtering = C;

  template <typename Block0, typename Block1>
  Iir(const_Matrix<T, Block0> b, const_Matrix<T, Block1> a, length_type i)
    VSIP_THROW((std::bad_alloc))
  : accumulator_type(impl::signal_detail::Description<1, T>::tag("Iir", i), 
                     impl::signal_detail::Op_count_iir<T>::value(i,
								 a.size(0))),
    b_(b.size(0), b.size(1)),
    a_(a.size(0), a.size(1)),
    w_(b_.size(0), 2, T()),
    input_size_(i)
  {
    assert(b_.size(0) == a_.size(0));
    assert(b_.size(1) == 3);
    assert(a_.size(1) == 2);
    
    b_ = b;
    a_ = a;
  }

  Iir(Iir const &iir) VSIP_THROW((std::bad_alloc))
    : accumulator_type(iir),
      b_(iir.b_.size(0), 3),
      a_(iir.a_.size(0), 2),
      w_(b_.size(0), 2),
      input_size_(iir.input_size_)
  {
    b_ = iir.b_;
    a_ = iir.a_;
    w_ = iir.w_;
  }

  Iir& operator=(Iir const &iir) VSIP_THROW((std::bad_alloc))
  {
    accumulator_type::operator= (iir);
    assert(this->kernel_size() == iir.kernel_size());

    b_ = iir.b_;
    a_ = iir.a_;
    w_ = iir.w_;

    input_size_ = iir.input_size_;

    return *this;
  }

  length_type kernel_size()  const VSIP_NOTHROW { return 2 * a_.size(0); }
  length_type filter_order() const VSIP_NOTHROW { return 2 * a_.size(0); }
  length_type input_size()   const VSIP_NOTHROW { return input_size_; }
  length_type output_size()  const VSIP_NOTHROW { return input_size_; }

  // Specification has both member function and member static const
  // variable with same name, which is not allowed in C++.  Since the
  // member variable can be used in constant expressions (such as
  // template parameters), as well as in situations where the function
  // can be used, we implement the variable.
  //
  // obj_state continuous_filtering() const VSIP_NOTHROW;

  template <typename Block0, typename Block1>
  Vector<T, Block1> operator()(
    const_Vector<T, Block0>,
    Vector<T, Block1>)
    VSIP_NOTHROW;

  void reset() VSIP_NOTHROW { w_ = T();}

public:

  float impl_performance(char const * what) const  VSIP_NOTHROW
  {
    if      (!strcmp(what, "mops"))  return this->event_.mflops();
    else if (!strcmp(what, "time"))  return this->event_.total();
    else if (!strcmp(what, "count")) return this->event_.count();
    else return 0.f;
  }

private:
  Matrix<T>   b_;
  Matrix<T>   a_;
  Matrix<T>   w_;
  length_type input_size_;
};



/***********************************************************************
  Definitions
***********************************************************************/

template <typename      T,
	  obj_state     C,
	  unsigned      N,
	  alg_hint_type H>
template <typename      Block0,
	  typename      Block1>
Vector<T, Block1>
Iir<T, C, N, H>::operator()(const_Vector<T, Block0> data,
                            Vector<T, Block1> out)
  VSIP_NOTHROW
{
  typename accumulator_type::Scope scope(*this);

  index_type const a1 = 0;
  index_type const a2 = 1;

  index_type const b0 = 0;
  index_type const b1 = 1;
  index_type const b2 = 2;

  index_type const w1 = 0;
  index_type const w2 = 1;

  assert(data.size() == this->input_size());
  assert(out.size()  == this->output_size());

  length_type const m_max = a_.size(0);

  for (index_type i=0; i<out.size(); ++i)
  {
    T val = data(i);

    for (index_type m=0; m < m_max; ++m)
    {
      T w0 = val
	   - a_.get(m, a1) * w_.get(m, w1)
	   - a_.get(m, a2) * w_.get(m, w2);

      val  = b_.get(m, b0) * w0
	   + b_.get(m, b1) * w_.get(m, w1)
	   + b_.get(m, b2) * w_.get(m, w2);

      w_.put(m, w2, w_.get(m, w1));
      w_.put(m, w1, w0);
    }

    out.put(i, val);
  }

  if (C == state_no_save)
    this->reset();

  return out;
}



} // namespace vsip

#endif // VSIP_CORE_SIGNAL_IIR_HPP
