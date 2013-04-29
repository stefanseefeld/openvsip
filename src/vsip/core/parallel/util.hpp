//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef VSIP_CORE_PARALLEL_UTIL_HPP
#define VSIP_CORE_PARALLEL_UTIL_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/parallel/services.hpp>
#include <vsip/support.hpp>
#include <vsip/core/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/domain.hpp>



/***********************************************************************
  Definitions
***********************************************************************/

namespace vsip
{

namespace impl
{

// Evaluate a function object foreach local patch of a distributed view.

// Requires:
//   VIEW is a distributed view.
//   FCN is a function object,
//     Arguments:
//       PATCH_VIEW - a patch view
//       GDOM - the global domain of the patch
//     Return value is ignored

template <template <typename, typename> class ViewT,
	  typename                            T,
	  typename                            BlockT,
	  typename                            FuncT>
void
foreach_patch(
  ViewT<T, BlockT> view,
  FuncT            fcn)
{
  typedef typename BlockT::map_type         map_t;
  typedef typename BlockT::local_block_type local_block_t;

  dimension_type const dim = ViewT<T, BlockT>::dim;

  BlockT&      block = view.block();
  map_t const& map   = view.block().map();

  index_type sb = map.subblock();

  if (sb != no_subblock)
  {
    ViewT<T, local_block_t> local_view = get_local_view(view, sb);

    for (index_type p=0; p<num_patches(view, sb); ++p)
    {
      Domain<dim> ldom = local_domain(view, sb, p);
      Domain<dim> gdom = global_domain(view, sb, p);

      fcn(local_view.get(ldom), gdom);
    }
  }
}



// Evaluate a function object foreach local element of a distributed view.

// Requires:
//   VIEW is a distributed view.
//   FCN is a function object that is invoked for each local element,
//     Arguments:
//       VALUE - the value at the local element (type T),
//       L_IDX - the local index of the element in the local subblock,
//       G_IDX - the global index of the element in the disrtributed object.
//     Return Value 
//       FCN should return a value of type T to update the local element.

template <typename ViewT,
	  typename FuncT>
void
foreach_point(
  ViewT view,
  FuncT fcn)
{
  dimension_type const dim = ViewT::dim;
  typedef typename ViewT::block_type::map_type map_t;

  map_t const& map = view.block().map();

  index_type sb = map.subblock();
  if (sb != no_subblock)
  {
    typename ViewT::local_type local_view = get_local_view(view);

    for (index_type p=0; p<num_patches(view, sb); ++p)
    {
      Domain<dim> ldom = local_domain(view, sb, p);
      Domain<dim> gdom = global_domain(view, sb, p);

      Length<dim> ext = extent(ldom);
      for (Index<dim> idx; valid(ext,idx); next(ext, idx))
      {
	Index<dim> l_idx = domain_nth(ldom, idx);
	Index<dim> g_idx = domain_nth(gdom, idx);

	put(local_view, l_idx, fcn(get(local_view, l_idx), l_idx, g_idx));
      }
    }
  }
}



template <typename T, typename Block>
void
buf_send(
  Communicator&          comm,
  processor_type         dest_proc,
  const_Vector<T, Block> data)
{
  T* raw = new T[data.size()];

  for (index_type i=0; i<data.size(); ++i)
    raw[i] = data.get(i);

  comm.buf_send(dest_proc, raw, data.size());

  delete[] raw;
}



template <typename T,
	  typename Block>
inline void
buf_send(
  Communicator&          comm,
  processor_type         dest_proc,
  const_Matrix<T, Block> data)
{
  T* raw = new T[data.size()];

  for (index_type r=0; r<data.size(0); ++r)
    for (index_type c=0; c<data.size(1); ++c)
      raw[r*data.size(1)+c] = data.get(r, c);

  comm.buf_send(dest_proc, raw, data.size());

  delete[] raw;
}



template <typename T,
	  typename Block>
void
recv(
  Communicator&    comm,
  processor_type   src_proc,
  Vector<T, Block> data)
{
  T* raw = new T[data.size()];

  comm.recv(src_proc, raw, data.size());

  for (index_type i=0; i<data.size(); ++i)
    data.put(i, raw[i]);

  delete[] raw;
}



template <typename T,
	  typename Block>
void
recv(
  Communicator&    comm,
  processor_type   src_proc,
  Matrix<T, Block> data)
{
  T* raw = new T[data.size()];

  comm.recv(src_proc, raw, data.size());

  for (index_type r=0; r<data.size(0); ++r)
    for (index_type c=0; c<data.size(1); ++c)
      data.put(r, c, raw[r*data.size(1)+c]);

  delete[] raw;
}

} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_CORE_PARALLEL_UTIL_HPP
