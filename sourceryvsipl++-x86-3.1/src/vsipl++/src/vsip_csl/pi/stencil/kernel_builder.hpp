/* Copyright (c) 2007 by CodeSourcery, Inc.  All rights reserved. */

/// Description
///   Harness to generate kernel from expression.

#ifndef vsip_csl_pi_stencil_kernel_builder_hpp_
#define vsip_csl_pi_stencil_kernel_builder_hpp_

#include <vsip_csl/pi/expr.hpp>

namespace vsip_csl
{
namespace pi
{
namespace stencil
{

//
// Kernel builder: initialize kernel coefficients.
//
template <typename T>
struct Kernel
{
public:
  Kernel(vsip::index_type y, vsip::index_type x,
         vsip::length_type h, vsip::length_type w)
    : y_(y), x_(x), height_(h), width_(w), data_(new T[h*w])
  {
    for (int i = h*w - 1; i >= 0; --i) data_[i] = T(0);
  }
  Kernel(Kernel const &k)
    : y_(k.y_), x_(k.x_), height_(k.height_), width_(k.width_),
      data_(new T[height_*width_])
  {
    for (int i = height_*width_ - 1; i >= 0; --i) data_[i] = T(0);
  }
  ~Kernel() { delete [] data_;}
  Kernel &operator+=(Kernel const &k)
  {
    for (int i = height_*width_ - 1; i >= 0; --i) data_[i] += k.data_[i];
    return *this;
  }
  Kernel &operator-=(Kernel const &k)
  {
    for (int i = height_*width_ - 1; i >= 0; --i) data_[i] -= k.data_[i];
    return *this;
  }
  Kernel &operator*=(T s)
  {
    for (int i = height_*width_ - 1; i >= 0; --i) data_[i] *= s;
    return *this;
  }
  Kernel &operator/=(T s)
  {
    for (int i = height_*width_ - 1; i >= 0; --i) data_[i] /= s;
    return *this;
  }

  vsip::index_type origin(vsip::dimension_type d) const 
  { return d == 0 ? y_ : x_;}
  vsip::length_type size(vsip::dimension_type d) const 
  { return d == 0 ? height_ : width_;}

  T const &operator() (vsip::index_type y, vsip::index_type x) const
  { return data_[x + y * width_];}
  T &operator() (vsip::index_type y, vsip::index_type x)
  { return data_[x + y * width_];}
private:
  vsip::index_type y_, x_;
  vsip::length_type height_, width_;
  T *data_;
};

template <typename E, typename T> 
struct Kernel_builder
{
  static void apply(E e, Kernel<T>& k);
};

template <typename L, typename R, typename T>
void build_kernel(Binary<expr::op::Add, L, R> e, Kernel<T>& k)
{
  Kernel_builder<L, T>::apply(e.arg1(), k);
  Kernel_builder<R, T>::apply(e.arg2(), k);
}

template <typename L, typename R, typename T>
void build_kernel(Binary<expr::op::Sub, L, R> e, Kernel<T>& k)
{
  Kernel_builder<L, T>::apply(e.arg1(), k);
  Kernel<T> k2(k);
  Kernel_builder<R, T>::apply(e.arg2(), k2);
  k -= k2;
}

// Assume S to be a scalar (the only case in which multiplication is defined).
template <template <typename, typename> class O, typename L, typename R,
          typename S, typename T>
void build_kernel(Binary<expr::op::Mult, Binary<O, L, R>, S> e,
                  Kernel<T>& k)
{
  Kernel<T> k1(k);
  Kernel_builder<Binary<O, L, R>, T>::apply(e.arg1(), k1);
  k1 *= T(e.arg2());
  k += k1;
}

template <typename S,
	  template <typename, typename> class O, typename L, typename R,
          typename T>
void build_kernel(Binary<expr::op::Mult, Scalar<S>, Binary<O, L, R> > e,
                  Kernel<T>& k)
{
  Kernel<T> k1(k);
  Kernel_builder<Binary<O, L, R>, T>::apply(e.arg2(), k1);
  k1 *= T(e.arg1().value());
  k += k1;
}

template <template <typename, typename> class O, typename L, typename R,
          typename S, typename T>
void build_kernel(Binary<expr::op::Div, Binary<O, L, R>, Scalar<S> > e,
                  Kernel<T>& k)
{
  Kernel<T> k1(k);
  Kernel_builder<Binary<O, L, R>, T>::apply(e.arg1(), k1);
  k1 /= T(e.arg2().value());
  k += k1;
}

template <typename B, typename I, typename J,
          typename S, typename T>
void build_kernel(Binary<expr::op::Mult, Call<B, I, J>, Scalar<S> > e,
                  Kernel<T>& k)
{
  Kernel<T> k1(k);
  Kernel_builder<Call<B, I, J>, T>::apply(e.arg1(), k1);
  k1 *= T(e.arg2().value());
  k += k1;
}

template <typename S, 
          typename B, typename I, typename J,
          typename T>
void build_kernel(Binary<expr::op::Mult, Scalar<S>, Call<B, I, J> > e,
                  Kernel<T>& k)
{
  Kernel<T> k1(k);
  Kernel_builder<Call<B, I, J>, T>::apply(e.arg2(), k1);
  k1 *= T(e.arg1().value());
  k += k1;
}

template <typename B, typename I, typename J,
          typename S, typename T>
void build_kernel(Binary<expr::op::Div, Call<B, I, J>, Scalar<S> > e,
                  Kernel<T>& k)
{
  Kernel<T> k1(k);
  Kernel_builder<Call<B, I, J>, T>::apply(e.arg1(), k1);
  k1 /= T(e.arg2().value());
  k += k1;
}

template <int I>
vsip::stride_type offset(Iterator<I>) { return I;}

vsip::stride_type offset(Offset o) { return o.i;}

template <typename B, typename I, typename J, typename T>
void build_kernel(Call<B, I, J> e, Kernel<T>& k)
{
  vsip::index_type y = k.origin(0) + offset(e.i());
  vsip::index_type x = k.origin(1) + offset(e.j());
  k(y, x) += T(1);
};

template <typename E, typename T>
void 
Kernel_builder<E, T>::apply(E e, Kernel<T>& k) 
{ build_kernel(e, k);}

} // namespace vsip_csl::pi::stencil
} // namespace vsip_csl::pi
} // namespace vsip_csl

#endif
