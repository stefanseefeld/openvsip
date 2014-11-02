//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef math_matvec_api_hpp_
#define math_matvec_api_hpp_

#include <ovxx/python/block.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/math.hpp>

namespace pyvsip
{
using namespace ovxx;
using namespace ovxx::python;

template <typename T>
T dot(Block<1, T> const &b1, Block<1, T> const &b2)
{
  typedef Block<1, T> B;
  return vsip::dot(Vector<T, B>(const_cast<B&>(b1)), Vector<T, B>(const_cast<B&>(b2)));
}

template <typename T>
boost::shared_ptr<Block<2, T> > trans(Block<2, T> const&b)
{
  typedef Block<2, T> B;
  boost::shared_ptr<B> block_ptr(new B(Domain<2>(b.size(2, 1), b.size(2, 0))));
  Matrix<T, B> mout(*block_ptr);
  Matrix<T, B> min(const_cast<B&>(b));
  mout = trans(min);
  return block_ptr;
}

template <typename T>
boost::shared_ptr<Block<2, T> > herm(Block<2, T> const&b)
{
  typedef Block<2, T> B;
  boost::shared_ptr<B> block_ptr(new B(Domain<2>(b.size(2, 1), b.size(2, 0))));
  Matrix<T, B> mout(*block_ptr);
  Matrix<T, B> min(const_cast<B&>(b));
  mout = herm(min);
  return block_ptr;
}

template <typename T>
boost::shared_ptr<Block<1, T> > vmprod(Block<1, T> const &a, Block<2, T> const &b)
{
  typedef Block<1, T> B1;
  typedef Block<2, T> B2;
  boost::shared_ptr<B1> block_ptr(new B1(b.size(2, 1)));
  Vector<T, B1> mout(*block_ptr);
  mout = vsip::prod(Vector<T, B1>(const_cast<B1&>(a)), Matrix<T, B2>(const_cast<B2&>(b)));
  return block_ptr;
}

template <typename T>
boost::shared_ptr<Block<1, T> > mvprod(Block<2, T> const &a, Block<1, T> const &b)
{
  typedef Block<2, T> B2;
  typedef Block<1, T> B1;
  boost::shared_ptr<B1> block_ptr(new B1(a.size(2, 0)));
  Vector<T, B1> mout(*block_ptr);
  mout = vsip::prod(Matrix<T, B2>(const_cast<B2&>(a)), Vector<T, B1>(const_cast<B1&>(b)));
  return block_ptr;
}

template <typename T>
boost::shared_ptr<Block<2, T> > mmprod(Block<2, T> const &a, Block<2, T> const &b)
{
  typedef Block<2, T> B;
  boost::shared_ptr<B> block_ptr(new B(Domain<2>(a.size(2, 0), b.size(2, 1))));
  Matrix<T, B> mout(*block_ptr);
  mout = vsip::prod(Matrix<T, B>(const_cast<B&>(a)), Matrix<T, B>(const_cast<B&>(b)));
  return block_ptr;
}

template <typename T>
boost::shared_ptr<Block<2, T> > vmmul(int axis, Block<1, T> const &a, Block<2, T> const &b)
{
  typedef Block<1, T> B1;
  typedef Block<2, T> B2;
  boost::shared_ptr<B2> block_ptr(new B2(vsip::Domain<2>(b.size(2, 0), b.size(2, 1))));
  Matrix<T, B2> mout(*block_ptr);
  if (axis==0)
    mout = vsip::vmmul<0>(Vector<T, B1>(const_cast<B1&>(a)), Matrix<T, B2>(const_cast<B2&>(b)));
  else
    mout = vsip::vmmul<1>(Vector<T, B1>(const_cast<B1&>(a)), Matrix<T, B2>(const_cast<B2&>(b)));
  return block_ptr;
}




template <typename T>
void define_complex_api(T) {}

template <typename T>
void define_complex_api(complex<T>)
{
  bpl::def("herm", herm<complex<T> >);
}

template <typename T>
void define_api()
{
  bpl::def("dot", dot<T>);
  bpl::def("trans", trans<T>);
  bpl::def("prod", vmprod<T>);
  bpl::def("prod", mvprod<T>);
  bpl::def("prod", mmprod<T>);
  bpl::def("vmmul", vmmul<T>);
  define_complex_api(T());
}

}


#endif
