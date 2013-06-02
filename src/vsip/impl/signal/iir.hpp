//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef vsip_impl_signal_iir_hpp_
#define vsip_impl_signal_iir_hpp_

#include <vsip/support.hpp>
#include <vsip/impl/signal/types.hpp>

namespace vsip
{

template <typename      T = VSIP_DEFAULT_VALUE_TYPE,
	  obj_state     C = state_save,
	  unsigned      N = 0,
	  alg_hint_type H = alg_time>
class Iir
{
public:
  static obj_state const continuous_filtering = C;

  template <typename B1, typename B2>
  Iir(const_Matrix<T, B1> b, const_Matrix<T, B2> a, length_type i)
    VSIP_THROW((std::bad_alloc))
  : b_(b.size(0), b.size(1)),
    a_(a.size(0), a.size(1)),
    w_(b_.size(0), 2, T()),
    input_size_(i)
  {
    OVXX_PRECONDITION(b_.size(0) == a_.size(0));
    OVXX_PRECONDITION(b_.size(1) == 3);
    OVXX_PRECONDITION(a_.size(1) == 2);
    
    b_ = b;
    a_ = a;
  }

  Iir(Iir const &iir) VSIP_THROW((std::bad_alloc))
    : b_(iir.b_.size(0), 3),
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
    OVXX_PRECONDITION(this->kernel_size() == iir.kernel_size());

    b_ = iir.b_;
    a_ = iir.a_;
    w_ = iir.w_;

    input_size_ = iir.input_size_;

    return *this;
  }

  length_type kernel_size()  const VSIP_NOTHROW { return 2 * a_.size(0);}
  length_type filter_order() const VSIP_NOTHROW { return 2 * a_.size(0);}
  length_type input_size()   const VSIP_NOTHROW { return input_size_;}
  length_type output_size()  const VSIP_NOTHROW { return input_size_;}

  // Specification has both member function and member static const
  // variable with same name, which is not allowed in C++.  Since the
  // member variable can be used in constant expressions (such as
  // template parameters), as well as in situations where the function
  // can be used, we implement the variable.
  //
  // obj_state continuous_filtering() const VSIP_NOTHROW;

  template <typename B1, typename B2>
  Vector<T, B2> operator()(const_Vector<T, B1>, Vector<T, B2>)
    VSIP_NOTHROW;

  void reset() VSIP_NOTHROW { w_ = T();}

private:
  Matrix<T>   b_;
  Matrix<T>   a_;
  Matrix<T>   w_;
  length_type input_size_;
};

template <typename      T,
	  obj_state     C,
	  unsigned      N,
	  alg_hint_type H>
template <typename B1, typename B2>
Vector<T, B2>
Iir<T, C, N, H>::operator()(const_Vector<T, B1> data, Vector<T, B2> out)
  VSIP_NOTHROW
{
  index_type const a1 = 0;
  index_type const a2 = 1;

  index_type const b0 = 0;
  index_type const b1 = 1;
  index_type const b2 = 2;

  index_type const w1 = 0;
  index_type const w2 = 1;

  OVXX_PRECONDITION(data.size() == this->input_size());
  OVXX_PRECONDITION(out.size()  == this->output_size());

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

#endif
