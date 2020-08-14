//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef dda_api_hpp_
#define dda_api_hpp_

#include <ovxx/python/block.hpp>
#include <ovxx/opencl/dda.hpp>

namespace pyvsip
{
using namespace ovxx;
using namespace ovxx::python;
namespace ocl = ovxx::opencl;

template <dimension_type D, typename T>
class dda_data_base
{
public:
  virtual ocl::buffer ptr() = 0;
  virtual void sync_out() = 0;
};

template <dimension_type D, typename T, vsip::dda::sync_policy S>
class dda_data : public dda_data_base<D, T>
{
  typedef Block<D, T> block_type;
  typedef Layout<D, tuple<0,1,2>, dense> layout_type;
  typedef ocl::Data<block_type, S, layout_type> data_type;
public:
  dda_data(block_type &b) : data_(b) {}
  virtual ocl::buffer ptr() { return data_.ptr();}
  virtual void sync_out() { data_.sync_out();}
private:
  data_type data_;
};

template <dimension_type D, typename T>
std::unique_ptr<dda_data_base<D, T> >
create_data(Block<D, T> &b, vsip::dda::sync_policy s)
{
  std::unique_ptr<dda_data_base<D, T> > data;
  switch (s)
  {
    case vsip::dda::in:
      data.reset(new dda_data<D, T, vsip::dda::in>(b));
      break;
    case vsip::dda::out:
      data.reset(new dda_data<D, T, vsip::dda::out>(b));
      break;
    case vsip::dda::inout:
      data.reset(new dda_data<D, T, vsip::dda::inout>(b));
      break;
  }
  return data;
}

template <dimension_type D, typename T>
void define_dda(char const *type_name)
{
  typedef dda_data_base<D, T> data_type;

  bpl::class_<data_type, boost::noncopyable> data(type_name, bpl::no_init);
  data.def("__init__", bpl::make_constructor(create_data<D, T>));
  data.def("sync_out", &data_type::sync_out);
  data.def("buf", &data_type::ptr);
}

} // namespace pyvsip

#endif
