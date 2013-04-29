/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/// Description
///   VSIPL++ Library: Unit-test for run-time DDA

#define VERBOSE 0

#if VERBOSE
#  include <iostream>
#endif

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/map.hpp>
#include <vsip/dda.hpp>

#include <vsip_csl/test.hpp>
#include "../util.hpp"

using namespace vsip;
using vsip_csl::equal;

using vsip::impl::Rt_layout;
using vsip::impl::Rt_tuple;
using vsip::impl::Storage;
using vsip::impl::Rt_data;
using vsip::impl::Applied_layout;
using vsip::impl::Length;
using vsip::impl::extent;



/***********************************************************************
  Definitions
***********************************************************************/

// Utility functions to return a unique value for each index in
// a view.  Overloaded so that tests can work for any-dimension view.

inline index_type value1(Index<1> const& idx) { return idx[0]; }
inline index_type value1(Index<2> const& idx) { return 100*idx[0] + idx[1]; }
inline index_type value1(Index<3> const& idx)
{ return 10000*idx[0] + 100*idx[1] + idx[2]; }

inline index_type value2(Index<1> const& idx) { return 2*idx[0]; }
inline index_type value2(Index<2> const& idx) { return 100*idx[1] + idx[0]; }
inline index_type value2(Index<3> const& idx)
{ return 10000*idx[2] + 100*idx[0] + idx[1]; }


template <dimension_type Dim,
          typename       ExtDataT>
inline stride_type
offset(Index<Dim> const& idx,
       ExtDataT const&   ext)
{
  stride_type off = stride_type();

  for (dimension_type d=0; d<Dim; ++d)
    off += idx[d] * ext.stride(d);

  return off;
}



// Test that Rt_layout matches Layout.
template <typename       LayoutT,
	  dimension_type Dim>
void
test_layout(Rt_layout<Dim> rtl)
{
  static pack_type const packing = LayoutT::packing;
  static storage_format_type storage_format = LayoutT::storage_format;
  typedef typename LayoutT::order_type order_type;

  test_assert(rtl.dim     == LayoutT::dim);
  test_assert(rtl.packing == packing);
  test_assert(rtl.storage_format == storage_format);
  test_assert(rtl.order.impl_dim0 == LayoutT::order_type::impl_dim0);
  test_assert(rtl.order.impl_dim1 == LayoutT::order_type::impl_dim1);
  test_assert(rtl.order.impl_dim2 == LayoutT::order_type::impl_dim2);
  test_assert(rtl.alignment == impl::is_packing_aligned<packing>::alignment);
}



// Test run-time external data access (assuming that data is either
// not complex or is interleaved-complex).

template <typename       T,
	  typename       LayoutT,
	  dda::sync_policy Sync,
	  dimension_type Dim>
void
t_rtex(
  Domain<Dim> const& dom,
  Rt_tuple           order,
  pack_type pack,
  storage_format_type    cformat,
  bool               alloc,
  int                cost,
  bool force_copy = false)
{
  typedef impl::Strided<Dim, T, LayoutT> block_type;
  typedef typename impl::view_of<block_type>::type view_type;

  view_type view = create_view<view_type>(dom);

  Rt_layout<Dim> blk_rtl = vsip::impl::block_layout<Dim>(view.block());
  test_layout<LayoutT>(blk_rtl);

  Length<Dim> len = impl::extent(dom);
  for (Index<Dim> idx; valid(len, idx); next(len, idx))
    put(view, idx, T(value1(idx)));

  Rt_layout<Dim> rt_layout;

  rt_layout.packing = pack;
  rt_layout.order = order; 
  rt_layout.storage_format = cformat;
  rt_layout.alignment = (pack == aligned) ? 16 : 0;

  // Pre-allocate temporary buffer.
  T* buffer = 0;
  if (alloc)
  {
    Applied_layout<Rt_layout<Dim> > app_layout(rt_layout, len, sizeof(T));
    length_type total_size = app_layout.total_size();
    buffer = new T[total_size];
  }

  {
    Rt_data<block_type, dda::inout> data(view.block(), rt_layout, buffer);

    T* ptr = data.ptr().as_inter();

    test_assert(cost == data.cost());
    if (alloc && cost != 0)
      test_assert(ptr == buffer);

#if VERBOSE
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "cost: " << data.cost() << std::endl;

    for (index_type i=0; i< data.size(); ++i)
      std::cout << i << ": " << ptr[i] << std::endl;
#endif


    for (Index<Dim> idx; valid(len,idx); next(len, idx))
    {
      test_assert(equal(ptr[offset(idx, data)], get(view, idx)));
      ptr[offset(idx, data)] = T(value2(idx));
    }
  }

  if (alloc)
    delete[] buffer;

  if (Sync == dda::inout)
  {
    for (Index<Dim> idx; valid(len,idx); next(len, idx))
      test_assert(equal(get(view, idx), T(value2(idx))));
  }
  else if (Sync == dda::in)
  {
    for (Index<Dim> idx; valid(len,idx); next(len, idx))
      test_assert(equal(get(view, idx), T(value1(idx))));
  }
}



// Test run-time external data access (assuming that data is complex,
// either interleaved or split).

template <typename       T,
	  typename       LayoutT,
	  dda::sync_policy Sync,
	  dimension_type Dim>
void
t_rtex_c(
  Domain<Dim> const& dom,
  Rt_tuple           order,
  pack_type       pack,
  storage_format_type    cformat,
  int                cost,
  bool               alloc,
  bool force_copy = false)
{
  typedef impl::Strided<Dim, T, LayoutT>              block_type;
  typedef typename impl::view_of<block_type>::type view_type;

  view_type view = create_view<view_type>(dom);

  Rt_layout<Dim> blk_rtl = vsip::impl::block_layout<Dim>(view.block());
  test_layout<LayoutT>(blk_rtl);

  Length<Dim> len = impl::extent(dom);
  for (Index<Dim> idx; valid(len, idx); next(len, idx))
    put(view, idx, T(value1(idx)));

  Rt_layout<Dim> rt_layout;

  rt_layout.packing = pack;
  rt_layout.order = order; 
  rt_layout.storage_format = cformat;
  rt_layout.alignment = (pack == aligned) ? 16 : 0;

  // Pre-allocate temporary buffer.
  T* buffer = 0;
  if (alloc)
  {
    Applied_layout<Rt_layout<Dim> > app_layout(rt_layout, len, sizeof(T));
    length_type total_size = app_layout.total_size();
    buffer = new T[total_size];
  }

  if (Sync == dda::in)
  {
    Rt_data<block_type, dda::in> data(view.block(), force_copy, rt_layout, buffer);

#if VERBOSE
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "cost: " << cost << "  vs data: " << data.cost() << std::endl;
#endif

    test_assert(cost == data.cost());

    if (rt_layout.storage_format == interleaved_complex)
    {
      typedef Storage<interleaved_complex, T> storage_type;
      typename storage_type::const_type ptr = data.ptr().as_inter();
#if VERBOSE
      for (index_type i=0; i<data.size(); ++i)
	std::cout << i << ": " << ptr[i] << std::endl;
#endif
      if (alloc && cost != 0)
	test_assert(ptr == buffer);

      for (Index<Dim> idx; valid(len,idx); next(len, idx))
      {
	test_assert(equal(ptr[offset(idx, data)], get(view, idx)));
	if (force_copy)
	{
	  // Make sure we are in fact scribbling over a copy.
	  T *nc_ptr = const_cast<T*>(ptr);
	  nc_ptr[offset(idx, data)] = T(value2(idx));
	}
      }
    }
    else /* rt_layout.storage_format == split_complex */
    {
      typedef Storage<split_complex, T> storage_type;
      typedef typename vsip::impl::scalar_of<T>::type scalar_type;
      typename storage_type::const_type ptr = data.ptr().as_split();
#if VERBOSE
      for (index_type i=0; i< data.size(); ++i)
	std::cout << i << ": " << ptr.first[i] << "," << ptr.second[i]
		  << std::endl;
#endif
      if (alloc && cost != 0) 
	test_assert(reinterpret_cast<T const *>(ptr.first) == buffer);

      for (Index<Dim> idx; valid(len,idx); next(len, idx))
      {
	test_assert(
	  equal(storage_type::get(ptr, offset(idx, data)), get(view, idx)));
	if (force_copy)
	{
	  // Make sure we are in fact scribbling over a copy.
	  typename storage_type::type nc_ptr = 
	    dda::impl::const_cast_<typename storage_type::type>(ptr);
	  storage_type::put(nc_ptr, offset(idx, data), T(value2(idx)));
	}
      }
    }
  }
  else
  {
    Rt_data<block_type, Sync> data(view.block(), rt_layout, buffer);

#if VERBOSE
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "cost: " << data.cost() << std::endl;
#endif

    test_assert(cost == data.cost());

    if (rt_layout.storage_format == interleaved_complex)
    {
      typedef Storage<interleaved_complex, T> storage_type;
      typedef typename storage_type::type ptr_type;
      // Since this 'else' block is compiled also for Sync==in, we have to const-cast,
      // even if in the case where this branch is taken Data::ptr_type is in fact non-const.
      ptr_type ptr = dda::impl::const_cast_<ptr_type>(data.ptr().as_inter());
#if VERBOSE
      for (index_type i=0; i< data.size(); ++i)
	std::cout << i << ": " << ptr[i] << std::endl;
#endif
      if (alloc && cost != 0)
	test_assert(ptr == buffer);

      for (Index<Dim> idx; valid(len,idx); next(len, idx))
      {
	test_assert(equal(ptr[offset(idx, data)], get(view, idx)));
	ptr[offset(idx, data)] = T(value2(idx));
      }
    }
    else /* rt_layout.storage_format == split_complex */
    {
      typedef Storage<split_complex, T> storage_type;
      typedef typename storage_type::type ptr_type;
      ptr_type ptr = dda::impl::const_cast_<ptr_type>(data.ptr().as_split());
#if VERBOSE
      for (index_type i=0; i<data.size(); ++i)
	std::cout << i << ": " << ptr.first[i] << "," << ptr.second[i]
		  << std::endl;
#endif
      if (alloc && cost != 0) 
	test_assert(reinterpret_cast<T *>(ptr.first) == buffer);

      for (Index<Dim> idx; valid(len,idx); next(len, idx))
      {
	test_assert(
	  equal(storage_type::get(ptr, offset(idx, data)), get(view, idx)));
	storage_type::put(ptr, offset(idx, data), T(value2(idx)));
      }
    }
  }

  if (alloc)
    delete[] buffer;

  if (Sync == dda::inout)
  {
    for (Index<Dim> idx; valid(len,idx); next(len, idx))
      test_assert(equal(get(view, idx), T(value2(idx))));
  }
  else if (Sync == dda::in)
  {
    for (Index<Dim> idx; valid(len,idx); next(len, idx))
      test_assert(equal(get(view, idx), T(value1(idx))));
  }
}



template <typename T>
void
test_noncomplex(
  Domain<2> const& d,		// size of matrix
  bool             a)		// pre-allocate buffer or not.
{
  using vsip::Layout;
  using vsip::dense;
  using vsip::interleaved_complex;

  Rt_tuple r1_v = Rt_tuple(row1_type());
  Rt_tuple r2_v = Rt_tuple(row2_type());
  Rt_tuple c2_v = Rt_tuple(col2_type());

  typedef row1_type r1_t;
  typedef row2_type r2_t;
  typedef col2_type c2_t;

  storage_format_type const cif = interleaved_complex;
  storage_format_type const csf = split_complex;

  Domain<1> d1(d[0]);

  // Ask for interleaved_complex
  t_rtex<T, Layout<1,r1_t,dense,cif>, dda::inout>(d1,r1_v,dense,cif,a,0);
  t_rtex<T, Layout<1,r1_t,dense,csf>, dda::inout>(d1,r1_v,dense,cif,a,0);

  // Check that split_complex is ignored since type is non-complex.
  t_rtex<T, Layout<1,r1_t,dense,cif>, dda::inout>(d1,r1_v,dense,csf,a,0);
  t_rtex<T, Layout<1,r1_t,dense,csf>, dda::inout>(d1,r1_v,dense,csf,a,0);

  t_rtex<T, Layout<2,r2_t,dense,cif>, dda::inout>(d,r2_v,dense,cif,a,0);
  t_rtex<T, Layout<2,r2_t,dense,cif>, dda::inout>(d,c2_v,dense,cif,a,2);
  t_rtex<T, Layout<2,c2_t,dense,cif>, dda::inout>(d,r2_v,dense,cif,a,2);
  t_rtex<T, Layout<2,c2_t,dense,cif>, dda::inout>(d,c2_v,dense,cif,a,0);
}
  

template <typename       T,
	  dimension_type D>
void
test(
  Domain<D> const& d,		// size of matrix
  bool             a)		// pre-allocate buffer or not.
{
  typedef complex<T> CT;

  using vsip::Layout;
  static vsip::storage_format_type const split = vsip::split_complex;
  static vsip::storage_format_type const inter = vsip::interleaved_complex;

  typedef typename impl::Row_major<D>::type row;
  typedef typename impl::Col_major<D>::type col;

  // dda::inout cases --------------------------------------------------

  t_rtex_c<CT, Layout<D,row,dense,inter>, dda::inout> (d,row(),dense,inter,0,a);
  t_rtex_c<CT, Layout<D,row,dense,inter>, dda::inout> (d,row(),dense,split,2,a);
  t_rtex_c<CT, Layout<D,row,dense,split>, dda::inout> (d,row(),dense,inter,2,a);
  t_rtex_c<CT, Layout<D,row,dense,split>, dda::inout> (d,row(),dense,split,0,a);

  if (D > 1)
  {
  // These tests only make sense if row and col are different.
  t_rtex_c<CT, Layout<D,row,dense,inter>, dda::inout> (d,col(),dense,inter,2,a);
  t_rtex_c<CT, Layout<D,row,dense,inter>, dda::inout> (d,col(),dense,split,2,a);
  t_rtex_c<CT, Layout<D,row,dense,split>, dda::inout> (d,col(),dense,inter,2,a);
  t_rtex_c<CT, Layout<D,row,dense,split>, dda::inout> (d,col(),dense,split,2,a);

  t_rtex_c<CT, Layout<D,col,dense,inter>, dda::inout> (d,col(),dense,inter,0,a);
  t_rtex_c<CT, Layout<D,col,dense,inter>, dda::inout> (d,col(),dense,split,2,a);
  t_rtex_c<CT, Layout<D,col,dense,split>, dda::inout> (d,col(),dense,inter,2,a);
  t_rtex_c<CT, Layout<D,col,dense,split>, dda::inout> (d,col(),dense,split,0,a);

  t_rtex_c<CT, Layout<D,col,dense,inter>, dda::inout> (d,row(),dense,inter,2,a);
  t_rtex_c<CT, Layout<D,col,dense,inter>, dda::inout> (d,row(),dense,split,2,a);
  t_rtex_c<CT, Layout<D,col,dense,split>, dda::inout> (d,row(),dense,inter,2,a);
  t_rtex_c<CT, Layout<D,col,dense,split>, dda::inout> (d,row(),dense,split,2,a);
  }


  // force-copy cases -------------------------------------------

  t_rtex_c<CT, Layout<D,row,dense,inter>, dda::in> (d,row(),dense,inter,2,a,true);
  t_rtex_c<CT, Layout<D,row,dense,inter>, dda::in> (d,row(),dense,split,2,a,true);
  t_rtex_c<CT, Layout<D,row,dense,split>, dda::in> (d,row(),dense,inter,2,a,true);
  t_rtex_c<CT, Layout<D,row,dense,split>, dda::in> (d,row(),dense,split,2,a,true);

  if (D > 1)
  {
  // These tests only make sense if row and col are different.
  t_rtex_c<CT, Layout<D,row,dense,inter>, dda::in> (d,col(),dense,inter,2,a,true);
  t_rtex_c<CT, Layout<D,row,dense,inter>, dda::in> (d,col(),dense,split,2,a,true);
  t_rtex_c<CT, Layout<D,row,dense,split>, dda::in> (d,col(),dense,inter,2,a,true);
  t_rtex_c<CT, Layout<D,row,dense,split>, dda::in> (d,col(),dense,split,2,a,true);

  t_rtex_c<CT, Layout<D,col,dense,inter>, dda::in> (d,col(),dense,inter,2,a,true);
  t_rtex_c<CT, Layout<D,col,dense,inter>, dda::in> (d,col(),dense,split,2,a,true);
  t_rtex_c<CT, Layout<D,col,dense,split>, dda::in> (d,col(),dense,inter,2,a,true);
  t_rtex_c<CT, Layout<D,col,dense,split>, dda::in> (d,col(),dense,split,2,a,true);

  t_rtex_c<CT, Layout<D,col,dense,inter>, dda::in> (d,row(),dense,inter,2,a,true);
  t_rtex_c<CT, Layout<D,col,dense,inter>, dda::in> (d,row(),dense,split,2,a,true);
  t_rtex_c<CT, Layout<D,col,dense,split>, dda::in> (d,row(),dense,inter,2,a,true);
  t_rtex_c<CT, Layout<D,col,dense,split>, dda::in> (d,row(),dense,split,2,a,true);
  }



  // dda::in (read-only) cases ------------------------------------------

  t_rtex_c<CT, Layout<D,row,dense,inter>, dda::in> (d,row(),dense,inter,0,a);
  t_rtex_c<CT, Layout<D,row,dense,inter>, dda::in> (d,row(),dense,split,2,a);
  t_rtex_c<CT, Layout<D,row,dense,split>, dda::in> (d,row(),dense,inter,2,a);
  t_rtex_c<CT, Layout<D,row,dense,split>, dda::in> (d,row(),dense,split,0,a);

  if (D > 1)
  {
  // These tests only make sense if row and col are different.
  t_rtex_c<CT, Layout<D,row,dense,inter>, dda::in> (d,col(),dense,inter,2,a);
  t_rtex_c<CT, Layout<D,row,dense,inter>, dda::in> (d,col(),dense,split,2,a);
  t_rtex_c<CT, Layout<D,row,dense,split>, dda::in> (d,col(),dense,inter,2,a);
  t_rtex_c<CT, Layout<D,row,dense,split>, dda::in> (d,col(),dense,split,2,a);

  t_rtex_c<CT, Layout<D,col,dense,inter>, dda::in> (d,col(),dense,inter,0,a);
  t_rtex_c<CT, Layout<D,col,dense,inter>, dda::in> (d,col(),dense,split,2,a);
  t_rtex_c<CT, Layout<D,col,dense,split>, dda::in> (d,col(),dense,inter,2,a);
  t_rtex_c<CT, Layout<D,col,dense,split>, dda::in> (d,col(),dense,split,0,a);

  t_rtex_c<CT, Layout<D,col,dense,inter>, dda::in> (d,row(),dense,inter,2,a);
  t_rtex_c<CT, Layout<D,col,dense,inter>, dda::in> (d,row(),dense,split,2,a);
  t_rtex_c<CT, Layout<D,col,dense,split>, dda::in> (d,row(),dense,inter,2,a);
  t_rtex_c<CT, Layout<D,col,dense,split>, dda::in> (d,row(),dense,split,2,a);
  }
}



template <dimension_type D>
void
test_types(Domain<D> const& dom, bool alloc)
{
  test<float> (dom, alloc);
#if VSIP_IMPL_TEST_LEVEL > 0
  test<short> (dom, alloc);
  test<int>   (dom, alloc);
  test<double>(dom, alloc);
#endif
}



int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  test_noncomplex<float>(Domain<2>(4, 8), true);
  test_noncomplex<float>(Domain<2>(4, 8), false);

  test_types(Domain<1>(4), true);
  test_types(Domain<1>(4), false);

  test_types(Domain<2>(4, 8), true);
  test_types(Domain<2>(4, 8), false);

  test_types(Domain<3>(6, 8, 12), true);
  test_types(Domain<3>(6, 8, 12), false);
}

