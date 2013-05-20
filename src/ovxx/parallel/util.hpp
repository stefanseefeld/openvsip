//
// Copyright (c) 2005, 2006 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_parallel_util_hpp_
#define ovxx_parallel_util_hpp_

#include <ovxx/parallel/service.hpp>
#include <vsip/support.hpp>
#include <vsip/impl/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/domain.hpp>

namespace ovxx
{
namespace parallel
{

// Evaluate a function object foreach local patch of a distributed view.

// Requires:
//   VIEW is a distributed view.
//   FCN is a function object,
//     Arguments:
//       PATCH_VIEW - a patch view
//       GDOM - the global domain of the patch
//     Return value is ignored
template <template <typename, typename> class V,
	  typename T, typename B, typename F>
void
foreach_patch(V<T, B> view, F fcn)
{
  typedef typename B::map_type map_type;
  typedef typename B::local_block_type local_block_type;

  dimension_type const dim = V<T, B>::dim;

  B &block = view.block();
  map_type const &map   = view.block().map();

  index_type sb = map.subblock();

  if (sb != no_subblock)
  {
    V<T, local_block_type> local_view = ovxx::get_local_view(view, sb);

    for (index_type p = 0; p < num_patches(view, sb); ++p)
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

template <typename V, typename F>
void foreach_point(V view, F fcn)
{
  dimension_type const dim = V::dim;
  typedef typename V::block_type::map_type map_type;

  map_type const &map = view.block().map();

  index_type sb = map.subblock();
  if (sb != no_subblock)
  {
    typename V::local_type local_view = ovxx::get_local_view(view);

    for (index_type p = 0; p < num_patches(view, sb); ++p)
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

template <typename T, typename B>
void
buf_send(Communicator &comm, processor_type dest_proc, const_Vector<T, B> data)
{
  T *raw = new T[data.size()];

  for (index_type i=0; i<data.size(); ++i)
    raw[i] = data.get(i);

  comm.buf_send(dest_proc, raw, data.size());

  delete[] raw;
}



template <typename T, typename B>
inline void
buf_send(Communicator &comm, processor_type dest_proc, const_Matrix<T, B> data)
{
  T *raw = new T[data.size()];

  for (index_type r=0; r<data.size(0); ++r)
    for (index_type c=0; c<data.size(1); ++c)
      raw[r*data.size(1)+c] = data.get(r, c);

  comm.buf_send(dest_proc, raw, data.size());

  delete[] raw;
}



template <typename T, typename B>
void
recv(Communicator &comm, processor_type src_proc, Vector<T, B> data)
{
  T *raw = new T[data.size()];

  comm.recv(src_proc, raw, data.size());

  for (index_type i=0; i<data.size(); ++i)
    data.put(i, raw[i]);

  delete[] raw;
}



template <typename T, typename B>
void
recv(Communicator &comm, processor_type src_proc, Matrix<T, B> data)
{
  T *raw = new T[data.size()];

  comm.recv(src_proc, raw, data.size());

  for (index_type r=0; r<data.size(0); ++r)
    for (index_type c=0; c<data.size(1); ++c)
      data.put(r, c, raw[r*data.size(1)+c]);

  delete[] raw;
}

} // namespace ovxx::parallel
} // namespace ovxx

#endif
