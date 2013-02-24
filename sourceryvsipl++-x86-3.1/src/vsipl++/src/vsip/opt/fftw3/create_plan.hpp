/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/fftw3/create_plan.hpp
    @author  Assem Salama
    @date    2007-04-13
    @brief   VSIPL++ Library: File that has create_plan struct
*/
#ifndef VSIP_OPT_FFTW3_CREATE_PLAN_HPP
#define VSIP_OPT_FFTW3_CREATE_PLAN_HPP

#include <vsip/dense.hpp>

#include <vsip/opt/fftw3/fftw_support.hpp>

namespace vsip
{
namespace impl
{
namespace fftw3
{

// This is a helper struct to create temporary buffers used durring plan
// creation.
template <storage_format_type, typename T>
struct Cmplx_buffer;

// intereaved complex
template <typename T>
struct Cmplx_buffer<interleaved_complex, T>
{
  std::complex<T> *ptr() { return buffer_.get(); }

  Cmplx_buffer(length_type size) : buffer_(32, size)
  {}
  aligned_array<std::complex<T> > buffer_;
};

// split complex
template <typename T>
struct Cmplx_buffer<split_complex, T>
{
  Cmplx_buffer(length_type size) :
    buffer_r_(32, size),
    buffer_i_(32, size)
  {}

  std::pair<T*,T*> ptr()
  { return std::pair<T*, T*>(buffer_r_.get(), buffer_i_.get()); }

  aligned_array<T>  buffer_r_;
  aligned_array<T>  buffer_i_;
};

// Convert form axis to tuple
template <dimension_type Dim>
Rt_tuple tuple_from_axis(int A);

template <>
Rt_tuple tuple_from_axis<1>(int /*A*/) { return Rt_tuple(0,1,2); }
template <>
Rt_tuple tuple_from_axis<2>(int A) 
{
  switch (A)
  {
    case 0:  return Rt_tuple(1,0,2);
    default: return Rt_tuple(0,1,2);
  };
}

template <>
Rt_tuple tuple_from_axis<3>(int A) 
{
  switch (A)
  {
    case 0:  return Rt_tuple(2,1,0);
    case 1:  return Rt_tuple(0,2,1);
    default: return Rt_tuple(0,1,2);
  };
}

// This is a helper strcut to create plans
template<storage_format_type>
struct Create_plan;

// interleaved
template<>
struct Create_plan<interleaved_complex>
{

  // create function for complex -> complex
  template <typename PlanT, typename IodimT,
            typename T, dimension_type Dim>
  static PlanT
  create(std::complex<T>* ptr1, std::complex<T>* ptr2,
         int exp, int flags, Domain<Dim> const& size)
  {
    int sz[Dim];
    for(dimension_type i=0;i<Dim;i++) sz[i] = size[i].size();
    return create_fftw_plan(Dim, sz, ptr1,ptr2,exp,flags);
  }

  // create function for real -> complex
  template <typename PlanT, typename IodimT,
            typename T, dimension_type Dim>
  static PlanT
  create(T* ptr1, std::complex<T>* ptr2,
         int A, int flags, Domain<Dim> const& size)
  {
    int sz[Dim];
    for(dimension_type i=0;i<Dim;i++) sz[i] = size[i].size();
    if(A != Dim-1) std::swap(sz[A], sz[Dim-1]);
    return create_fftw_plan(Dim,sz,ptr1,ptr2,flags);
  }

  // create function for complex -> real
  template <typename PlanT, typename IodimT,
            typename T, dimension_type Dim>
  static PlanT
  create(std::complex<T>* ptr1, T* ptr2,
         int A, int flags, Domain<Dim> const& size)
  {
    int sz[Dim];
    for(dimension_type i=0;i<Dim;i++) sz[i] = size[i].size();
    if(A != Dim-1) std::swap(sz[A], sz[Dim-1]);
    return create_fftw_plan(Dim,sz,ptr1,ptr2,flags);
  }

  static storage_format_type const storage_format = interleaved_complex;  

};

// split
template<>
struct Create_plan<split_complex>
{

  // create for complex -> complex
  template <typename PlanT, typename IodimT,
            typename T, dimension_type Dim>
  static PlanT
  create(std::pair<T*,T*> ptr1, std::pair<T*,T*> ptr2,
         int /*exp*/, int flags, Domain<Dim> const& size)
  {
    IodimT iodims[Dim];

    Applied_layout<Layout<Dim, typename Row_major<Dim>::type,
      Aligned_packing<VSIP_IMPL_ALLOC_ALIGNMENT>::value,
      split_complex> >
    app_layout(size);

    for (index_type i=0;i<Dim;i++) 
    { 
      iodims[i].n = app_layout.size(i);
      iodims[i].is = iodims[i].os = app_layout.stride(i);
    }

    return create_fftw_plan(Dim, iodims, ptr1,ptr2, flags);

  }

  // create for real -> complex
  template <typename PlanT, typename IodimT,
            typename T, dimension_type Dim>
  static PlanT
  create(T *ptr1, std::pair<T*, T*> ptr2, 
         int A, int flags, Domain<Dim> const& size)
  {
    IodimT iodims[Dim];

    Applied_layout<Rt_layout<Dim> >
       app_layout(Rt_layout<Dim>(aligned,
                                 tuple_from_axis<Dim>(A),
                                 split_complex,
                                 0),
              size, sizeof(T));

    for (index_type i=0;i<Dim;i++) 
    { 
      iodims[i].n = app_layout.size(i);
      iodims[i].is = iodims[i].os = app_layout.stride(i); 
    }

    return create_fftw_plan(Dim, iodims, ptr1,ptr2, flags);
  }

  // create for complex -> real
  template <typename PlanT, typename IodimT,
            typename T, dimension_type Dim>
  static PlanT
  create(std::pair<T*,T*> ptr1, T* ptr2,
         int A, int flags, Domain<Dim> const& size)
  {
    IodimT iodims[Dim];

    Applied_layout<Rt_layout<Dim> >
       app_layout(Rt_layout<Dim>(aligned,
                                 tuple_from_axis<Dim>(A),
                                 split_complex,
                                 0),
              size, sizeof(T));

    for (index_type i=0;i<Dim;i++) 
    { 
      iodims[i].n = app_layout.size(i);
      iodims[i].is = iodims[i].os = app_layout.stride(i);
    }

    return create_fftw_plan(Dim, iodims, ptr1,ptr2, flags);
  }

  static storage_format_type const storage_format = split_complex;  
};


} // namespace vsip::impl::fftw3
} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_OPT_FFTW3_CREATE_PLAN_HPP
