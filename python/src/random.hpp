//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef random_hpp_
#define random_hpp_

#include <ovxx/python/block.hpp>
#include <vsip/random.hpp>

namespace pyvsip
{
using namespace ovxx;
using namespace ovxx::python;

template <typename T>
class rand
{
  typedef Block<1, T> vblock_type;
  typedef std::shared_ptr<vblock_type> vblock_ptr;
  typedef Block<2, T> mblock_type;
  typedef std::shared_ptr<mblock_type> mblock_ptr;

public:
  rand(vsip::index_type seed, bool portable)
    : rand_(seed, portable) {}
  T randu() { return rand_.randu();}
  T randn() { return rand_.randn();}
  bpl::object vrandu(vsip::length_type l)
  { return make_block(rand_.randu(l));}
  bpl::object mrandu(vsip::length_type r, vsip::length_type c)
  { return make_block(rand_.randu(r, c));}
  bpl::object vrandn(vsip::length_type l)
  { return make_block(rand_.randn(l));}
  bpl::object mrandn(vsip::length_type r, vsip::length_type c)
  { return make_block(rand_.randn(r, c));}

private:
  template <typename B>
  bpl::object make_block(const_Vector<T, B> v)
  {
    vblock_ptr b(new vblock_type(v.size()));
    Vector<T, vblock_type> vtmp(*b);
    vtmp = v;
    return bpl::object(b);
  }
  template <typename B>
  bpl::object make_block(const_Matrix<T, B> m)
  {
    mblock_ptr b(new mblock_type(Domain<2>(m.size(0), m.size(1))));
    Matrix<T, mblock_type> mtmp(*b);
    mtmp = m;
    return bpl::object(b);
  }

  vsip::Rand<T> rand_;
};

template <typename T>
void define_rand()
{
  typedef rand<T> rand_type;

  bpl::class_<rand_type, boost::noncopyable>
    rand("rand", bpl::init<vsip::index_type, bool>());
  rand.def("randu", &rand_type::randu);
  rand.def("randu", &rand_type::vrandu);
  rand.def("randu", &rand_type::mrandu);
  rand.def("randn", &rand_type::randn);
  rand.def("randn", &rand_type::vrandn);
  rand.def("randn", &rand_type::mrandn);
}
}

#endif
