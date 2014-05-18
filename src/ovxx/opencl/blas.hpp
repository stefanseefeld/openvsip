//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_opencl_blas_hpp_
#define ovxx_opencl_blas_hpp_

#include <ovxx/opencl/dda.hpp>
#include <clAmdBlas.h>

namespace ovxx
{
namespace opencl
{

template <typename T>
inline T
dot(int n, buffer x, int incx, buffer y, int incy,
    buffer scratch, command_queue queue);

#define OVXX_OPENCL_DOT(T, B)			       \
template <>					       \
inline T					       \
dot(int n, buffer x, int incx, buffer y, int incy,     \
    buffer scratch, command_queue queue)	       \
{						       \
  /*  if (incx < 0) x += incx * (n-1);*/	       \
  /*if (incy < 0) y += incy * (n-1);*/		       \
  buffer res(queue.context(), sizeof(T), buffer::write);\
  B(n, res.get(), 0,				       \
    x.get(), 0, incx, y.get(), 0, incy,		       \
    scratch.get(), 1, &queue.get(), 0, 0, 0);	       \
  T result;					       \
  queue.read(res, &result, 1);			       \
  return result;				       \
}

OVXX_OPENCL_DOT(float, clAmdBlasSdot)
OVXX_OPENCL_DOT(double, clAmdBlasDdot)
OVXX_OPENCL_DOT(complex<float>, clAmdBlasCdotu)
OVXX_OPENCL_DOT(complex<double>, clAmdBlasZdotu)

#undef OVXX_OPENCL_DOT

} // namespace ovxx::opencl

namespace dispatcher
{

template <typename T, typename B0, typename B1>
struct Evaluator<op::dot, be::opencl, T(B0 const&, B1 const&)>
{
  static bool const ct_valid = 
    //    blas::traits<T>::valid &&
    is_same<T, typename B0::value_type>::value &&
    is_same<T, typename B1::value_type>::value &&
    opencl::Data<B0, dda::in>::ct_cost == 0 &&
    opencl::Data<B1, dda::in>::ct_cost == 0 &&
    !is_split_block<B0>::value &&
    !is_split_block<B1>::value;

  static bool rt_valid(B0 const&, B1 const&) { return true;}

  static T exec(B0 const &a, B1 const &b)
  {
    using namespace opencl;
    OVXX_PRECONDITION(a.size(1, 0) == b.size(1, 0));
    
    Data<B0, dda::in> data_a(a);
    Data<B1, dda::in> data_b(b);
    buffer scratch(default_context(), a.size(1,0)*sizeof(T), buffer::read_write);

    T r = dot<T>(a.size(1, 0),
		 data_a.ptr(), data_a.stride(0),
		 data_b.ptr(), data_b.stride(0),
		 scratch, default_queue());
    return r;
  }
};


} // namespace ovxx::dispatcher
} // namespace ovxx

#endif
