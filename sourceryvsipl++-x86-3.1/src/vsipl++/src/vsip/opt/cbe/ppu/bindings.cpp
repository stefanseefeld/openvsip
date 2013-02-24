/* Copyright (c) 2006, 2007, 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/cbe/ppu/bindings.cpp
    @author  Stefan Seefeld
    @date    2006-12-29
    @brief   VSIPL++ Library: Wrappers and traits to bridge with IBMs CBE SDK.
*/

#include <memory>
#include <algorithm>

#include <vsip/core/config.hpp>
#include <vsip/math.hpp>
#include <vsip/opt/cbe/ppu/bindings.hpp>
#include <vsip/opt/cbe/ppu/task_manager.hpp>
#include <vsip/opt/cbe/ppu/util.hpp>
#include <vsip/opt/cbe/overlay_params.h>
#include <vsip/opt/cbe/vmmul_params.h>
extern "C"
{
#include <libspe2.h>
}

namespace vsip
{
namespace impl
{
namespace cbe
{

// Scalar row-wise vmmul
//
// Computes: R(r, c) = V(c) * M(r, c)
//
// Arguments
//   - m_stride is the distance between input rows (or cols)
//   - r_stride is the distance between output rows (or cols)
//   - lines expresses the number of rows (cols)
//   - length is the size of the input vector 
//   - mult expresses the number of lines that can be grouped
//       together for efficiency

template <typename T>
void
vmmul_row(T const* V, T const* M, T* R, 
	  stride_type m_stride, stride_type r_stride, 
	  length_type lines, length_type length, length_type mult)
{
  assert(length >= VSIP_IMPL_MIN_VMMUL_SIZE);

  static char* code_ea = 0;
  static int   code_size;

  if (code_ea == 0) lwp::load_plugin(code_ea, code_size, "plugins/chalfast_f.plg");

  Task_manager *mgr = Task_manager::instance();

  int op = is_complex<T>::value ? overlay_cvmmul_row_f : overlay_vmmul_row_f;

  std::auto_ptr<lwp::Task> task =
    mgr->reserve_lwp_task(VSIP_IMPL_OVERLAY_BUFFER_SIZE, 2,
			  (uintptr_t)code_ea, code_size, op);

  Vmmul_params vp;

  vp.cmd           = op;
  vp.input_stride  = m_stride;
  vp.output_stride = r_stride;

  length_type spes            = mgr->num_spes();
  length_type vectors_per_spe = lines / spes;
  assert(vectors_per_spe * spes <= lines);
  length_type const granularity = VSIP_IMPL_CBE_DMA_GRANULARITY / sizeof(T);

  // break large vmmul into smaller vmmuls of MAX size
  index_type pos = 0;
  while (pos<length && (length-pos > VSIP_IMPL_MIN_VMMUL_SIZE))
  {
    length_type my_length = std::min<length_type>(
      length - pos, VSIP_IMPL_MAX_VMMUL_SIZE);

    if (my_length % granularity != 0)
      my_length -= my_length % granularity;

    vp.length = my_length;
    assert(my_length >= VSIP_IMPL_MIN_VMMUL_SIZE);

    vp.ea_input_vector  = ea_from_ptr(V + pos);
    vp.ea_input_matrix  = ea_from_ptr(M + pos);
    vp.ea_output_matrix = ea_from_ptr(R + pos);
    vp.shift = 0;

    for (index_type i=0; i<spes && i<lines; ++i)
    {
      // If chunks don't divide evenly, give the first SPEs one extra.
      length_type lines_per_spe = (i < lines % spes) ? vectors_per_spe + 1
                                                     : vectors_per_spe;

      lwp::Workblock block = task.get()->create_workblock(lines_per_spe);
      block.set_parameters(vp);
      block.enqueue();

      vp.ea_input_matrix  += sizeof(T) * lines_per_spe * m_stride;
      vp.ea_output_matrix += sizeof(T) * lines_per_spe * r_stride;
    }

    pos += my_length;
  }

  // Cleanup
  if (length-pos > 0)
  {
    for (index_type r=0; r<lines; ++r)
    {
      for (index_type c=pos; c<length; ++c)
      {
	R[c] = V[c] * M[c];
      }
      M += m_stride;
      R += r_stride;
    }
  }
  task.get()->sync();
}



//template void vmmul_row(
//  float const* V, float const* M, float* R, 
//  stride_type m_stride, stride_type r_stride, length_type length, length_type lines);
template void vmmul_row(
  std::complex<float> const* V, std::complex<float> const* M, std::complex<float>* R, 
  stride_type m_stride, stride_type r_stride, length_type length, length_type lines,
  length_type mult);




template <typename T>
void
vmmul_row(std::pair<T const *, T const *> const& V,
	  T const *M,
	  std::pair<T*, T*> const& R,
	  stride_type m_stride,
	  stride_type r_stride, 
	  length_type lines,
	  length_type length,
	  length_type mult)
{
#if USE_FUNCTIONAL_VERSION
  T* Vr = V.first;
  T* Vi = V.second;
  T* Rr = R.first;
  T* Ri = R.second;

  for (index_type r = 0; r < lines; ++r)
  {
    for (index_type c = 0; c < length; ++c)
    {
      Rr[c] = Vr[c] * M[c];
      Ri[c] = Vi[c] * M[c];
    }
    M += m_stride;
    Rr += r_stride;
    Ri += r_stride;
  }
#else
  static char* code_ea = 0;
  static int   code_size;

  if (code_ea == 0) lwp::load_plugin(code_ea, code_size, "plugins/rzvmmul_row_f.plg");

  Task_manager *mgr = Task_manager::instance();
  length_type spes = mgr->num_spes();

  std::auto_ptr<lwp::Task> task =
    mgr->reserve_lwp_task(VSIP_IMPL_OVERLAY_BUFFER_SIZE, 2,
			  (uintptr_t)code_ea, code_size,
			  overlay_rzvmmul_row_f);

  Vmmul_split_params vp;
  vp.cmd           = overlay_rzvmmul_row_f;
  vp.input_stride  = m_stride;
  vp.output_stride = r_stride;
  vp.mult = mult;

  // If multiple rows will fit into the SPEs memory at a time, then
  // group them together and send fewer workblocks.
  if (mult > 1)
  {
    assert(length >= VSIP_IMPL_MIN_VMMUL_SIZE);
    vp.length = length;

    length_type lines_per_spe = (lines / mult / spes) ? 
      (lines / mult / spes) * mult : mult;
    assert(lines >= lines_per_spe);
    length_type groups = lines / lines_per_spe;

    vp.ea_input_vector_re  = ea_from_ptr(V.first);
    vp.ea_input_vector_im  = ea_from_ptr(V.second);
    vp.ea_input_matrix_re  = ea_from_ptr(M);
    vp.ea_input_matrix_im  = 0;
    vp.ea_output_matrix_re = ea_from_ptr(R.first);
    vp.ea_output_matrix_im = ea_from_ptr(R.second);
    vp.shift = 0;

    // Dispatch as many groups as are possible to each SPE
    for (index_type i = 0; i < spes && i < groups; ++i)
    {
      lwp::Workblock block = task.get()->create_workblock(lines_per_spe / mult);
      block.set_parameters(vp);
      block.enqueue();
      
      vp.ea_input_matrix_re  += sizeof(T) * lines_per_spe * m_stride;
      vp.ea_output_matrix_re += sizeof(T) * lines_per_spe * r_stride;
      vp.ea_output_matrix_im += sizeof(T) * lines_per_spe * r_stride;
    }
    
    // Clean up, getting all the lines that wouldn't fit cleanly 
    // onto the SPEs.
    index_type start_row = groups * lines_per_spe;

    T const* Vr = V.first;
    T const* Vi = V.second;
    T const* Mr = M + start_row * m_stride;
    T* Rr = R.first + start_row * r_stride;
    T* Ri = R.second + start_row * r_stride;
    for (index_type r = start_row; r < lines; ++r)
    {
      for (index_type c = 0; c < length; ++c)
      {
        Rr[c] = Vr[c] * Mr[c];
        Ri[c] = Vi[c] * Mr[c];
      }
      Mr += m_stride;
      Rr += r_stride;
      Ri += r_stride;
    }
  }
  else 
  {
    // break large vmmul into smaller vmmuls of MAX size
    length_type vectors_per_spe = lines / spes;
    length_type const granularity = VSIP_IMPL_CBE_DMA_GRANULARITY / sizeof(T);

    index_type pos = 0;
    while (pos<length && (length-pos > VSIP_IMPL_MIN_VMMUL_SIZE))
    {
      length_type my_length = std::min<length_type>(
        length - pos, VSIP_IMPL_MAX_VMMUL_SIZE);

      if (my_length % granularity != 0)
        my_length -= my_length % granularity;

      vp.length = my_length;
      assert(my_length >= VSIP_IMPL_MIN_VMMUL_SIZE);

      vp.ea_input_vector_re  = ea_from_ptr(V.first  + pos);
      vp.ea_input_vector_im  = ea_from_ptr(V.second + pos);
      vp.ea_input_matrix_re  = ea_from_ptr(M + pos);
      vp.ea_input_matrix_im  = 0;
      vp.ea_output_matrix_re = ea_from_ptr(R.first  + pos);
      vp.ea_output_matrix_im = ea_from_ptr(R.second + pos);
      vp.shift = 0;

      for (index_type i=0; i<spes && i<lines; ++i)
      {
        // If chunks don't divide evenly, give the first SPEs one extra.
        length_type lines_per_spe = (i < lines % spes) ? vectors_per_spe + 1
          : vectors_per_spe;

        lwp::Workblock block = task.get()->create_workblock(lines_per_spe);
        block.set_parameters(vp);
        block.enqueue();

        vp.ea_input_matrix_re  += sizeof(T) * lines_per_spe * m_stride;
        vp.ea_input_matrix_im  += sizeof(T) * lines_per_spe * m_stride;
        vp.ea_output_matrix_re += sizeof(T) * lines_per_spe * r_stride;
        vp.ea_output_matrix_im += sizeof(T) * lines_per_spe * r_stride;
      }

      pos += my_length;
    }
    if (length - pos > 0)
    {
      T const* Vr = V.first;
      T const* Vi = V.second;
      T const* Mr = M;
      T* Rr = R.first;
      T* Ri = R.second;

      for (index_type r = 0; r < lines; ++r)
      {
        for (index_type c = pos; c < length; ++c)
        {
          Rr[c] = Vr[c] * Mr[c];
          Ri[c] = Vi[c] * Mr[c];
        }
        Mr += m_stride;
        Rr += r_stride;
        Ri += r_stride;
      }
    }
  } // end if large vmmul
  task.get()->sync();
#endif
}

template
void
vmmul_row(std::pair<float const *, float const *> const& V,
	  float const *M,
	  std::pair<float*, float*> const& R,
	  stride_type m_stride, stride_type r_stride, 
	  length_type lines, length_type length, length_type mult);


// Split-complex row-wise vmmul
//
// Computes: R(r, c) = V(c) * M(r, c)
//
// Arguments
//   - m_stride is the distance between input rows (or cols)
//   - r_stride is the distance between output rows (or cols)
//   - lines expresses the number of rows (cols)
//   - length is the size of the input vector 
//   - mult expresses the number of lines that can be grouped
//       together for efficiency

template <typename T>
void
vmmul_row(std::pair<T const *, T const *> const& V,
	  std::pair<T const *, T const *> const& M,
	  std::pair<T*, T*> const& R,
	  stride_type m_stride,
	  stride_type r_stride, 
	  length_type lines,
	  length_type length,
	  length_type mult)
{
  static char* code_ea = 0;
  static int   code_size;

  if (code_ea == 0) lwp::load_plugin(code_ea, code_size, "plugins/zhalfast_f.plg");

  Task_manager *mgr = Task_manager::instance();
  length_type spes = mgr->num_spes();

  std::auto_ptr<lwp::Task> task =
    mgr->reserve_lwp_task(VSIP_IMPL_OVERLAY_BUFFER_SIZE, 2,
			  (uintptr_t)code_ea, code_size,
			  overlay_zvmmul_row_f);

  Vmmul_split_params vp;
  vp.cmd           = overlay_zvmmul_row_f;
  vp.input_stride  = m_stride;
  vp.output_stride = r_stride;
  vp.mult = mult;

  // If multiple rows will fit into the SPEs memory at a time, then
  // group them together and send fewer workblocks.
  if (mult > 1)
  {
    assert(length >= VSIP_IMPL_MIN_VMMUL_SIZE);
    vp.length = length;

    length_type lines_per_spe = (lines / mult / spes) ? 
      (lines / mult / spes) * mult : mult;
    assert(lines >= lines_per_spe);
    length_type groups = lines / lines_per_spe;

    vp.ea_input_vector_re  = ea_from_ptr(V.first);
    vp.ea_input_vector_im  = ea_from_ptr(V.second);
    vp.ea_input_matrix_re  = ea_from_ptr(M.first);
    vp.ea_input_matrix_im  = ea_from_ptr(M.second);
    vp.ea_output_matrix_re = ea_from_ptr(R.first);
    vp.ea_output_matrix_im = ea_from_ptr(R.second);
    vp.shift = 0;

    // Dispatch as many groups as are possible to each SPE
    for (index_type i = 0; i < spes && i < groups; ++i)
    {
      lwp::Workblock block = task.get()->create_workblock(lines_per_spe / mult);
      block.set_parameters(vp);
      block.enqueue();
      
      vp.ea_input_matrix_re  += sizeof(T) * lines_per_spe * m_stride;
      vp.ea_input_matrix_im  += sizeof(T) * lines_per_spe * m_stride;
      vp.ea_output_matrix_re += sizeof(T) * lines_per_spe * r_stride;
      vp.ea_output_matrix_im += sizeof(T) * lines_per_spe * r_stride;
    }
    
    // Clean up, getting all the lines that wouldn't fit cleanly 
    // onto the SPEs.
    index_type start_row = groups * lines_per_spe;

    T const * Vr = V.first;
    T const * Vi = V.second;
    T const * Mr = M.first + start_row * m_stride;
    T const * Mi = M.second + start_row * m_stride;
    T* Rr = R.first + start_row * r_stride;
    T* Ri = R.second + start_row * r_stride;
    for (index_type r = start_row; r < lines; ++r)
    {
      for (index_type c = 0; c < length; ++c)
      {
        float tmp = Vr[c] * Mr[c] - Vi[c] * Mi[c];
        Ri[c]     = Vr[c] * Mi[c] + Vi[c] * Mr[c];
        Rr[c]     = tmp;
      }
      Mr += m_stride;
      Mi += m_stride;
      Rr += r_stride;
      Ri += r_stride;
    }
  }
  else 
  {
    // break large vmmul into smaller vmmuls of MAX size
    length_type vectors_per_spe = lines / spes;
    length_type const granularity = VSIP_IMPL_CBE_DMA_GRANULARITY / sizeof(T);

    index_type pos = 0;
    while (pos<length && (length-pos > VSIP_IMPL_MIN_VMMUL_SIZE))
    {
      length_type my_length = std::min<length_type>(
        length - pos, VSIP_IMPL_MAX_VMMUL_SIZE);

      if (my_length % granularity != 0)
        my_length -= my_length % granularity;

      vp.length = my_length;
      assert(my_length >= VSIP_IMPL_MIN_VMMUL_SIZE);

      vp.ea_input_vector_re  = ea_from_ptr(V.first  + pos);
      vp.ea_input_vector_im  = ea_from_ptr(V.second + pos);
      vp.ea_input_matrix_re  = ea_from_ptr(M.first  + pos);
      vp.ea_input_matrix_im  = ea_from_ptr(M.second + pos);
      vp.ea_output_matrix_re = ea_from_ptr(R.first  + pos);
      vp.ea_output_matrix_im = ea_from_ptr(R.second + pos);
      vp.shift = 0;

      for (index_type i=0; i<spes && i<lines; ++i)
      {
        // If chunks don't divide evenly, give the first SPEs one extra.
        length_type lines_per_spe = (i < lines % spes) ? vectors_per_spe + 1
          : vectors_per_spe;

        lwp::Workblock block = task.get()->create_workblock(lines_per_spe);
        block.set_parameters(vp);
        block.enqueue();

        vp.ea_input_matrix_re  += sizeof(T) * lines_per_spe * m_stride;
        vp.ea_input_matrix_im  += sizeof(T) * lines_per_spe * m_stride;
        vp.ea_output_matrix_re += sizeof(T) * lines_per_spe * r_stride;
        vp.ea_output_matrix_im += sizeof(T) * lines_per_spe * r_stride;
      }

      pos += my_length;
    }
    if (length-pos > 0)
    {
      T const * Vr = V.first;
      T const * Vi = V.second;
      T const * Mr = M.first;
      T const * Mi = M.second;
      T* Rr = R.first;
      T* Ri = R.second;

      for (index_type r=0; r<lines; ++r)
      {
        for (index_type c=pos; c<length; ++c)
        {
          float tmp = Vr[c] * Mr[c] - Vi[c] * Mi[c];
          Ri[c]     = Vr[c] * Mi[c] + Vi[c] * Mr[c];
          Rr[c]     = tmp;
        }
        Mr += m_stride;
        Mi += m_stride;
        Rr += r_stride;
        Ri += r_stride;
      }
    }
  } // end if large vmmul
  task.get()->sync();
}

template
void
vmmul_row(std::pair<float const *, float const *> const& V,
	  std::pair<float const *, float const *> const& M,
	  std::pair<float*, float*> const& R,
	  stride_type m_stride, stride_type r_stride, 
	  length_type lines, length_type length, length_type mult);



// Split-complex real-scalar column-wise vmmul
//
// Computes: R(r, c) = V(r) * M(r, c)
//
// Arguments
//   - m_stride is the distance between input rows (or cols)
//   - r_stride is the distance between output rows (or cols)
//   - lines expresses the number of rows (cols)
//   - length is the size of the input vector 

template <typename T>
void
vmmul_col(T const*V,
	  std::pair<T const *, T const *> const& M,
	  std::pair<T *, T *> const& R,
	  stride_type m_stride,
	  stride_type r_stride, 
	  length_type lines,
	  length_type length)
{
#if USE_FUNCTIONAL_VERSION
  T const * Mr = M.first;
  T const * Mi = M.second;
  T* Rr = R.first;
  T* Ri = R.second;

  for (index_type r=0; r<lines; ++r)
  {
    for (index_type c=0; c<length; ++c)
    {
      Rr[c] = V[r] * Mr[c];
      Ri[c] = V[r] * Mi[c];
    }
    Mr += m_stride;
    Mi += m_stride;
    Rr += r_stride;
    Ri += r_stride;
  }
#else
  static char* code_ea = 0;
  static int   code_size;

  if (code_ea == 0) lwp::load_plugin(code_ea, code_size, "plugins/zrvmmul_col_f.plg");

  Task_manager *mgr = Task_manager::instance();

  std::auto_ptr<lwp::Task> task =
    mgr->reserve_lwp_task(VSIP_IMPL_OVERLAY_BUFFER_SIZE, 2,
			  (uintptr_t)code_ea, code_size, 
			  overlay_zrvmmul_col_f);

  Vmmul_split_params vp;

  vp.cmd       = overlay_zrvmmul_col_f;

  assert(4*sizeof(T)*VSIP_IMPL_MAX_VMMUL_SIZE/2 <= VSIP_IMPL_OVERLAY_BUFFER_SIZE);
  assert(8 < VSIP_IMPL_OVERLAY_DTL_SIZE);

  length_type spes          = mgr->num_spes();
  length_type lines_per_spe = lines / spes;
  if (lines_per_spe == 0 || lines_per_spe % 4 != 0)
    lines_per_spe += 4 - (lines_per_spe % 4);

  for (length_type pos=0; pos<length; pos += 2048)
  {
    length_type this_length = std::min<length_type>(length-pos, 2048);

    vp.length = this_length;
    vp.ea_input_vector_re  = ea_from_ptr(V);
    vp.ea_input_vector_im  = 0;

    if (vp.ea_input_vector_re % 16 != 0)
    {
      int shift = (vp.ea_input_vector_re % 16);
      vp.ea_input_vector_re -= shift;
      vp.shift = shift / sizeof(float);
    }
    else
      vp.shift = 0;

    vp.input_stride        = m_stride;
    vp.output_stride       = r_stride;
    vp.ea_input_matrix_re  = ea_from_ptr(M.first  + pos);
    vp.ea_input_matrix_im  = ea_from_ptr(M.second + pos);
    vp.ea_output_matrix_re = ea_from_ptr(R.first  + pos);
    vp.ea_output_matrix_im = ea_from_ptr(R.second + pos);

    for (length_type rem_lines=lines; rem_lines > 0;)
    {
      // If chunks don't divide evenly, give the first SPEs one extra.
      length_type my_lines = (lines_per_spe < rem_lines) ? lines_per_spe
                                                         : rem_lines;

      lwp::Workblock block = task.get()->create_workblock(my_lines);
      block.set_parameters(vp);
      block.enqueue();

      vp.ea_input_vector_re  += sizeof(T) * my_lines;
      vp.ea_input_matrix_re  += sizeof(T) * my_lines * m_stride;
      vp.ea_input_matrix_im  += sizeof(T) * my_lines * m_stride;
      vp.ea_output_matrix_re += sizeof(T) * my_lines * r_stride;
      vp.ea_output_matrix_im += sizeof(T) * my_lines * r_stride;
      rem_lines -= my_lines;
    }
  }
  task.get()->sync();
#endif
}

template
void
vmmul_col(float const*                     V,
	  std::pair<float const *, float const *> const& M,
	  std::pair<float*, float*> const& R,
	  stride_type m_stride,
	  stride_type r_stride, 
	  length_type lines, 
	  length_type length);



// Split-complex real-scalar column-wise vmmul
//
// Computes R(r, c) = V(r) * M(r, c)
//
// Arguments
//   - m_stride is the distance between input rows (or cols)
//   - r_stride is the distance between output rows (or cols)
//   - lines expresses the number of rows (cols)
//   - length is the size of the input vector 

template <typename T>
void
vmmul_col(std::pair<T const *, T const *> const& V,
	  std::pair<T const *, T const *> const& M,
	  std::pair<T*, T*> const& R,
	  stride_type m_stride,
	  stride_type r_stride, 
	  length_type lines,
	  length_type length)
{
#if USE_FUNCTIONAL_VERSION
  T const * Vr = V.first;
  T const * Vi = V.second;
  T const * Mr = M.first;
  T const * Mi = M.second;
  T* Rr = R.first;
  T* Ri = R.second;

  for (index_type r=0; r<lines; ++r)
  {
    for (index_type c=0; c<length; ++c)
    {
      T tmp = Vr[r] * Mr[c] - Vi[r] * Mi[c];
      Ri[c] = Vr[r] * Mi[c] + Vi[r] * Mr[c];
      Rr[c] = tmp;
    }
    Mr += m_stride;
    Mi += m_stride;
    Rr += r_stride;
    Ri += r_stride;
  }
#else
  static char* code_ea = 0;
  static int   code_size;

  if (code_ea == 0) lwp::load_plugin(code_ea, code_size, "plugins/zvmmul_col_f.plg");

  Task_manager *mgr = Task_manager::instance();

  std::auto_ptr<lwp::Task> task =
    mgr->reserve_lwp_task(VSIP_IMPL_OVERLAY_BUFFER_SIZE, 2,
			  (uintptr_t)code_ea, code_size, 
			  overlay_zvmmul_col_f);

  Vmmul_split_params vp;

  vp.cmd       = overlay_zvmmul_col_f;

  assert(4*sizeof(T)*VSIP_IMPL_MAX_VMMUL_SIZE/2 <= VSIP_IMPL_OVERLAY_BUFFER_SIZE);
  assert(8 < VSIP_IMPL_OVERLAY_DTL_SIZE);

  length_type spes          = mgr->num_spes();
  length_type lines_per_spe = lines / spes;
  if (lines_per_spe == 0 || lines_per_spe % 4 != 0)
    lines_per_spe += 4 - (lines_per_spe % 4);

  for (length_type pos=0; pos<length; pos += 2048)
  {
    length_type this_length = std::min<length_type>(length-pos, 2048);

    vp.length = this_length;
    vp.ea_input_vector_re  = ea_from_ptr(V.first);
    vp.ea_input_vector_im  = ea_from_ptr(V.second);

    assert(vp.ea_input_vector_re % 16 == vp.ea_input_vector_im % 16);
    if (vp.ea_input_vector_re % 16 != 0)
    {
      int shift = (vp.ea_input_vector_re % 16);
      vp.ea_input_vector_re -= shift;
      vp.ea_input_vector_im -= shift;
      vp.shift = shift / sizeof(float);
    }
    else
      vp.shift = 0;

    vp.input_stride        = m_stride;
    vp.output_stride       = r_stride;
    vp.ea_input_matrix_re  = ea_from_ptr(M.first  + pos);
    vp.ea_input_matrix_im  = ea_from_ptr(M.second + pos);
    vp.ea_output_matrix_re = ea_from_ptr(R.first  + pos);
    vp.ea_output_matrix_im = ea_from_ptr(R.second + pos);

    for (length_type rem_lines=lines; rem_lines > 0;)
    {
      // If chunks don't divide evenly, give the first SPEs one extra.
      length_type my_lines = (lines_per_spe < rem_lines) ? lines_per_spe
                                                         : rem_lines;

      lwp::Workblock block = task.get()->create_workblock(my_lines);
      block.set_parameters(vp);
      block.enqueue();

      vp.ea_input_vector_re  += sizeof(T) * my_lines;
      vp.ea_input_vector_im  += sizeof(T) * my_lines;
      vp.ea_input_matrix_re  += sizeof(T) * my_lines * m_stride;
      vp.ea_input_matrix_im  += sizeof(T) * my_lines * m_stride;
      vp.ea_output_matrix_re += sizeof(T) * my_lines * r_stride;
      vp.ea_output_matrix_im += sizeof(T) * my_lines * r_stride;
      rem_lines -= my_lines;
    }
  }
  task.get()->sync();
#endif
}

template
void vmmul_col(std::pair<float const *, float const *> const& V,
	       std::pair<float const *, float const *> const& M,
	       std::pair<float *, float *> const& R,
	       stride_type m_stride, stride_type r_stride, 
	       length_type lines, length_type length);
  
} // namespace vsip::impl::cbe
} // namespace vsip::impl
} // namespace vsip
