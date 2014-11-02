//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef math_solvers_hpp_
#define math_solvers_hpp_

#include <ovxx/python/block.hpp>
#include <vsip/solvers.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>

namespace pyvsip
{
namespace bpl = boost::python;

template <typename T>
void covsol(ovxx::python::Block<2, T> &a,
	    ovxx::python::Block<2, T> const &b,
	    ovxx::python::Block<2, T> &x)
{
  typedef ovxx::python::Block<2, T> B;
  typedef vsip::Matrix<T, B> M;
  vsip::covsol(M(a), M(const_cast<B&>(b)), M(x));
}

template <typename T>
void llsqsol(ovxx::python::Block<2, T> &a,
	     ovxx::python::Block<2, T> const &b,
	     ovxx::python::Block<2, T> &x)
{
  typedef ovxx::python::Block<2, T> B;
  typedef vsip::Matrix<T, B> M;
  vsip::llsqsol(M(a), M(const_cast<B&>(b)), M(x));
}

template <typename T>
void toepsol(ovxx::python::Block<1, T> const &t,
	     ovxx::python::Block<1, T> const &b,
	     ovxx::python::Block<1, T> &y,
	     ovxx::python::Block<1, T> &x)
{
  typedef ovxx::python::Block<1, T> B;
  typedef vsip::Vector<T, B> V;
  vsip::toepsol(V(const_cast<B&>(t)), V(const_cast<B&>(b)), V(y), V(x));
}

template <typename T>
class lud
{
  typedef ovxx::python::Block<2, T> B;
  typedef vsip::Matrix<T, B> M;
public:
  lud(vsip::length_type l) : impl_(l) {}
  vsip::length_type length() const { return impl_.length();}
  bool decompose(B &b) { return impl_.decompose(M(b));}
  bool solve(vsip::mat_op_type op, B const &b, B &x)
  {
    using namespace vsip;
    switch (op)
    {
      case mat_ntrans: return impl_.template solve<mat_ntrans>(M(const_cast<B&>(b)), M(x));
      case mat_trans: return impl_.template solve<mat_trans>(M(const_cast<B&>(b)), M(x));
      case mat_herm: return impl_.template solve<mat_herm>(M(const_cast<B&>(b)), M(x));
      case mat_conj: return impl_.template solve<mat_conj>(M(const_cast<B&>(b)), M(x));
    }
  }
private:
  vsip::lud<T, vsip::by_reference> impl_;
};

template <typename T>
void define_lud()
{
  typedef lud<T> lud_type;
  bpl::class_<lud_type, std::auto_ptr<lud_type>, boost::noncopyable>
    lud("lud", bpl::init<vsip::length_type>());
  lud.def("decompose", &lud_type::decompose);
  lud.def("solve", &lud_type::solve);
}

template <typename T>
class chold
{
  typedef ovxx::python::Block<2, T> B;
  typedef vsip::Matrix<T, B> M;
public:
  chold(vsip::mat_uplo uplo, vsip::length_type l) : impl_(uplo, l) {}
  vsip::length_type length() const { return impl_.length();}
  vsip::mat_uplo uplo() const { return impl_.uplo();}
  bool decompose(B &b) { return impl_.decompose(M(b));}
  bool solve(B const &b, B &x) { return impl_.solve(M(const_cast<B&>(b)), M(x));}
private:
  vsip::chold<T, vsip::by_reference> impl_;
};

template <typename T>
void define_chold()
{
  typedef chold<T> chold_type;
  bpl::class_<chold_type, std::auto_ptr<chold_type>, boost::noncopyable>
    chold("chold", bpl::init<vsip::mat_uplo, vsip::length_type>());
  chold.def("decompose", &chold_type::decompose);
  chold.def("solve", &chold_type::solve);
}

template <typename T>
class qrd
{
  typedef ovxx::python::Block<2, T> B;
  typedef vsip::Matrix<T, B> M;
public:
  qrd(vsip::length_type rows, vsip::length_type cols, vsip::storage_type s) : impl_(rows, cols, s) {}
  vsip::length_type rows() const { return impl_.rows();}
  vsip::length_type columns() const { return impl_.columns();}
  vsip::storage_type qstorage() const { return impl_.qstorage();}
  bool decompose(B &b) { return impl_.decompose(M(b));}
  bool prodq(vsip::mat_op_type op, vsip::product_side_type side, B const &b, B &x)
  {
    using namespace vsip;
    switch (op)
    {
      case mat_ntrans: return this->prodq_<mat_ntrans>(side, b, x);
      case mat_trans: return this->prodq_<mat_trans>(side, b, x);
      case mat_herm: return this->prodq_<mat_herm>(side, b, x);
      case mat_conj: return this->prodq_<mat_conj>(side, b, x);
    }
  }
  bool rsol(vsip::mat_op_type op, B const &b, T alpha, B &x)
  {
    using namespace vsip;
    switch (op)
    {
      case mat_ntrans: return impl_.template rsol<mat_ntrans>(M(const_cast<B&>(b)), alpha, M(x));
      case mat_trans: return impl_.template rsol<mat_trans>(M(const_cast<B&>(b)), alpha, M(x));
      case mat_herm: return impl_.template rsol<mat_herm>(M(const_cast<B&>(b)), alpha, M(x));
      case mat_conj: return impl_.template rsol<mat_conj>(M(const_cast<B&>(b)), alpha, M(x));
    }
  }
  bool covsol(B const &b, B &x) { return impl_.covsol(M(const_cast<B&>(b)), M(x));}
  bool lsqsol(B const &b, B &x) { return impl_.lsqsol(M(const_cast<B&>(b)), M(x));}
private:
  template <vsip::mat_op_type tr>
  bool prodq_(vsip::product_side_type side, B const &b, B &x)
  {
    if (side == vsip::mat_lside)
      return impl_.template prodq<tr, vsip::mat_lside>(M(const_cast<B&>(b)), M(x));
    else
      return impl_.template prodq<tr, vsip::mat_rside>(M(const_cast<B&>(b)), M(x));
  }
  vsip::qrd<T, vsip::by_reference> impl_;
};

template <typename T>
void define_qrd()
{
  typedef qrd<T> qrd_type;
  bpl::class_<qrd_type, std::auto_ptr<qrd_type>, boost::noncopyable>
    qrd("qrd", bpl::init<vsip::length_type, vsip::length_type, vsip::storage_type>());
  qrd.def("decompose", &qrd_type::decompose);
  qrd.def("prodq", &qrd_type::prodq);
  qrd.def("rsol", &qrd_type::rsol);
  qrd.def("covsol", &qrd_type::covsol);
  qrd.def("lsqsol", &qrd_type::lsqsol);
}

// SVD call traits. For the non-complex type we generate a runtime error when the wrong operation is called.
// Passing to vsip::svd might otherwise have caused a compile-time error.
template <typename T> struct svd_call_traits;

template <typename T>
class svd
{
  typedef typename ovxx::scalar_of<T>::type S;
  typedef ovxx::python::Block<1, S> B1;
  typedef ovxx::python::Block<2, T> B2;
  typedef vsip::Vector<S, B1> V;
  typedef vsip::Matrix<T, B2> M;
public:
  svd(vsip::length_type rows, vsip::length_type cols,
      vsip::storage_type ust, vsip::storage_type vst)
    : impl_(rows, cols, ust, vst) {}
  vsip::length_type rows() const { return impl_.rows();}
  vsip::length_type columns() const { return impl_.columns();}
  vsip::storage_type ustorage() const { return impl_.ustorage();}
  vsip::storage_type vstorage() const { return impl_.vstorage();}
  bool decompose(B2 &b, B1 &dest) { return impl_.decompose(M(b), V(dest));}
  bool produ(vsip::mat_op_type op, vsip::product_side_type side, B2 const &b, B2 &x)
  {
    return svd_call_traits<T>::produ(*this, op, side, b, x);
  }
  bool prodv(vsip::mat_op_type op, vsip::product_side_type side, B2 const &b, B2 &x)
  {
    return svd_call_traits<T>::prodv(*this, op, side, b, x);
  }
  bool u(vsip::index_type low, vsip::index_type high, B2 &dest) { return impl_.u(low, high, M(dest));}
  bool v(vsip::index_type low, vsip::index_type high, B2 &dest) { return impl_.v(low, high, M(dest));}

  template <vsip::mat_op_type tr>
  bool produ_(vsip::product_side_type side, B2 const &b, B2 &x)
  {
    if (side == vsip::mat_lside)
      return impl_.template produ<tr, vsip::mat_lside>(M(const_cast<B2&>(b)), M(x));
    else
      return impl_.template produ<tr, vsip::mat_rside>(M(const_cast<B2&>(b)), M(x));
  }
  template <vsip::mat_op_type tr>
  bool prodv_(vsip::product_side_type side, B2 const &b, B2 &x)
  {
    if (side == vsip::mat_lside)
      return impl_.template prodv<tr, vsip::mat_lside>(M(const_cast<B2&>(b)), M(x));
    else
      return impl_.template prodv<tr, vsip::mat_rside>(M(const_cast<B2&>(b)), M(x));
  }
private:
  vsip::svd<T, vsip::by_reference> impl_;
};

template <typename T>
struct svd_call_traits
{
  typedef svd<T> svd_type;
  typedef ovxx::python::Block<2, T> B;

  static bool produ(svd_type &svd, vsip::mat_op_type op, vsip::product_side_type side, B const &b, B &x)
  {
    using namespace vsip;
    switch (op)
    {
      case mat_ntrans: return svd.template produ_<mat_ntrans>(side, b, x);
      case mat_trans: return svd.template produ_<mat_trans>(side, b, x);
      default: throw std::invalid_argument("invalid operation");
    }
  }
  static bool prodv(svd_type &svd, vsip::mat_op_type op, vsip::product_side_type side, B const &b, B &x)
  {
    using namespace vsip;
    switch (op)
    {
      case mat_ntrans: return svd.template prodv_<mat_ntrans>(side, b, x);
      case mat_trans: return svd.template prodv_<mat_trans>(side, b, x);
      default: throw std::invalid_argument("invalid operation");
    }
  }
};

template <typename T>
struct svd_call_traits<vsip::complex<T> >
{
  typedef svd<T> svd_type;
  typedef ovxx::python::Block<2, T> B;

  static bool produ(svd_type &svd, vsip::mat_op_type op, vsip::product_side_type side, B const &b, B &x)
  {
    using namespace vsip;
    switch (op)
    {
      case mat_ntrans: return svd.template produ_<mat_ntrans>(side, b, x);
      case mat_trans: return svd.template produ_<mat_trans>(side, b, x);
      case mat_herm: return svd.template produ_<mat_herm>(side, b, x);
      case mat_conj: return svd.template produ_<mat_conj>(side, b, x);
    }
  }
  static bool prodv(svd_type &svd, vsip::mat_op_type op, vsip::product_side_type side, B const &b, B &x)
  {
    using namespace vsip;
    switch (op)
    {
      case mat_ntrans: return svd.template prodv_<mat_ntrans>(side, b, x);
      case mat_trans: return svd.template prodv_<mat_trans>(side, b, x);
      case mat_herm: return svd.template prodv_<mat_herm>(side, b, x);
      case mat_conj: return svd.template prodv_<mat_conj>(side, b, x);
    }
  }
};


template <typename T>
void define_svd()
{
  typedef svd<T> svd_type;
  bpl::class_<svd_type, std::auto_ptr<svd_type>, boost::noncopyable>
    svd("svd", bpl::init<vsip::length_type, vsip::length_type, vsip::storage_type, vsip::storage_type>());
  svd.def("decompose", &svd_type::decompose);
  svd.def("produ", &svd_type::produ);
  svd.def("prodv", &svd_type::prodv);
  svd.def("u", &svd_type::u);
  svd.def("v", &svd_type::v);
}


}

#endif
