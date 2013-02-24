/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    benchmarks/hpec-kernel/cfar.cpp
    @author  Don McCoy
    @date    2006-04-21
    @brief   VSIPL++ Library: Constant False Alarm Rate Detection - High
             Performance Embedded Computing (HPEC) Kernel-Level Benchmarks

    Briefly, this problem involves finding targets based on data within a 
    three-dimensional cube of 'beam locations', 'range gates' and 'doppler 
    bins'.  It does this by comparing the signal in a given cell to that of
    nearby cells in order to avoid false-detection of targets.  The range 
    gate parameter is varied when considering 'nearby' cells.  A certain 
    number of guard cells are skipped, resulting in a computation that sums
    the values from two thick slices of this data cube (one on either side 
    of the slice for a particular range gate).  The HPEC PCA Kernel-Level 
    benchmark paper has a diagram that shows one cell under consideration.
    Please refer to it if needed.

    The algorithm involves these basic steps:
     - compute the squares of all the values in the data cube
     - for each range gate:
      - sum the squares of desired values around the current range gate
      - compute the normalized power for each cell in the slice
      - search for values that exceed a certain threshold

    Some of the code relates to boundary conditions (near either end of the 
    'range gates' parameter), but otherwise it follows the above description. 
*/

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>
#include <memory>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/tensor.hpp>
#include <vsip/random.hpp>
#include <vsip/selgen.hpp>
#include <vsip_csl/test-precision.hpp>

#include "benchmarks.hpp"

using namespace vsip;
using vsip_csl::Precision_traits;

#ifdef VSIP_IMPL_SOURCERY_VPP
#  define PARALLEL_CFAR 1
#else
#  define PARALLEL_CFAR 0
#endif


/***********************************************************************
  Definitions
***********************************************************************/

template <typename T,
	  typename ImplTag>
struct t_cfar_base;

struct ImplSlice;   // All range cells processed together
struct ImplVector;  // Each range gate vector processed independently
struct ImplHybrid;  // A cache-efficient combination of the above 
                    // approaches using SIMD instructions


/***********************************************************************
  CFAR Implementations
***********************************************************************/

template <typename T>
struct t_cfar_base<T, ImplSlice> : Benchmark_base
{
  char const *what() { return "t_cfar_sweep_range<T, ImplSlice>"; }

  template <typename Block>
  void
  cfar_detect(
    Tensor<T, Block>    cube,
    Tensor<T, Block>    cpow,
    Matrix<Index<2> >   located,
    Vector<length_type> count,
    length_type         loop,
    float&              time)
  {
    length_type const c = cfar_gates_;
    length_type const g = guard_cells_;
    length_type const beams = cube.size(0);
    length_type const dbins = cube.size(1);
    length_type const gates = cube.size(2);

    vsip::impl::profile::Timer t1;
    t1.start();
    for (index_type l=0; l<loop; ++l)
    {
      length_type gates_used = 0;
      T inv_gates;

      // Compute the square of all values in the data cube.  This is 
      // done in advance once, as the values are needed many times
      // (approximately twice as many times as the number of guard cells)
      cpow = sq(cube);

      // Create space to hold sums of squares
      Matrix<T> sum(beams, dbins, T());
      Matrix<T> tmp(beams, dbins);

      index_type k;
      for ( k = 0; k < (g + c + 1); ++k )
      {
        // Case 0: Initialize
        if ( k == 0 )
        {
          gates_used = c;
          for ( length_type lnd = g; lnd < g + c; ++lnd )
            sum += cpow(whole_domain, whole_domain, 1 + lnd);
        }
        // Case 1: No cell included on left side of CFAR; 
        // very close to left boundary 
        else if ( k < (g + 1) )
        {
          gates_used = c;
          sum += cpow(whole_domain, whole_domain, k+g+c);
          sum -= cpow(whole_domain, whole_domain, k+g);
        }
        // Case 2: Some cells included on left side of CFAR;
        // close to left boundary 
        else
        {
          gates_used = c + k - (g + 1);
          sum += cpow(whole_domain, whole_domain, k+g+c);
          sum -= cpow(whole_domain, whole_domain, k+g);
          sum += cpow(whole_domain, whole_domain, k-(g+1));
        }
        inv_gates = (1.0 / gates_used);
        tmp = sum * inv_gates;
        tmp = max(tmp, Precision_traits<T>::eps);
        tmp = cpow(whole_domain, whole_domain, k) / tmp;
        count(k) = impl::indexbool( tmp > this->mu_, located.row(k) );
      }

      for ( k = (g + c + 1); (k + (g + c)) < gates; ++k )
      {
        // Case 3: All cells included on left and right side of CFAR
        // somewhere in the middle of the range vector
        gates_used = 2 * c;
        sum += cpow(whole_domain, whole_domain, k+g+c);
        sum -= cpow(whole_domain, whole_domain, k+g);
        sum += cpow(whole_domain, whole_domain, k-(g+1));
        sum -= cpow(whole_domain, whole_domain, k-(c+g+1));

        inv_gates = (1.0 / gates_used);
        tmp = sum * inv_gates;
        tmp = max(tmp, Precision_traits<T>::eps);
        tmp = cpow(whole_domain, whole_domain, k) / tmp;
        count(k) = impl::indexbool( tmp > this->mu_, located.row(k) );
      }

      for ( k = gates - (g + c); k < gates; ++k )
      {
        // Case 4: Some cells included on right side of CFAR;
        // close to right boundary
        if ( (k + g) < gates )
        {
          gates_used = c + gates - (k + g);
          sum -= cpow(whole_domain, whole_domain, k+g);
          sum += cpow(whole_domain, whole_domain, k-(g+1));
          sum -= cpow(whole_domain, whole_domain, k-(c+g+1));
        }
        // Case 5: No cell included on right side of CFAR; 
        // very close to right boundary 
        else
        {
          gates_used = c;
          sum += cpow(whole_domain, whole_domain, k-(g+1));
          sum -= cpow(whole_domain, whole_domain, k-(c+g+1));
        }
        inv_gates = (1.0 / gates_used);
        tmp = sum * inv_gates;
        tmp = max(tmp, Precision_traits<T>::eps);
        tmp = cpow(whole_domain, whole_domain, k) / tmp;
        count(k) = impl::indexbool( tmp > this->mu_, located.row(k) );
      }    
    }
    t1.stop();
    time = t1.delta();
  }


  t_cfar_base(length_type beams, length_type bins, 
    length_type cfar_gates, length_type guard_cells)
    : beams_(beams), dbins_(bins), cfar_gates_(cfar_gates), 
      guard_cells_(guard_cells), ntargets_(30), mu_(100)
  {}

protected:
  // Member data
  length_type const beams_;         // Number of beam locations
  length_type const dbins_;         //   "   "   doppler bins
  length_type const cfar_gates_;    //   "   "   ranges gates to consider
  length_type const guard_cells_;   //   "   "   cells to skip near target
  length_type const ntargets_;      //   "   "   targets
  length_type const mu_;            // Threshold for determining targets
};


template <typename T>
struct t_cfar_base<T, ImplVector> : public Benchmark_base
{
  char const *what() { return "t_cfar_sweep_range<T, ImplVector>"; }

  template <typename Block>
  void
  cfar_detect(
    Tensor<T, Block>    cube,
    Tensor<T, Block>    cpow,
    Matrix<Index<2> >   located,
    Vector<length_type> count,
    length_type         loop,
    float&              time)
  {
    length_type const c = cfar_gates_;
    length_type const g = guard_cells_;
    length_type const beams = cube.size(0);
    length_type const dbins = cube.size(1);
    length_type const gates = cube.size(2);
    T const eps = Precision_traits<T>::eps;

    typedef typename 
      Tensor<T, Block>::template subvector<0, 1>::impl_type  subvector_type;

    vsip::impl::profile::Timer t1;
    t1.start();
    for (index_type l=0; l<loop; ++l)
    {
      // Clear count for targets found per gate.
      count = 0;

      // Extract a single vector containing all the range cells for a 
      // particular beam location and doppler bin.
      subvector_type cpow_vec = cpow(0, 0, whole_domain);

      for ( index_type i = 0; i < beams; ++i )
        for ( index_type j = 0; j < dbins; ++j )
        {
          length_type gates_used;
          index_type k;
          T inv_gates;
          T sum = T();

          // Compute the square of all values in the data cube.  This is 
          // done in advance for each vector once, as the values are needed
          // many times (approximately twice as many times as the number of
          // guard cells)
          cpow_vec = sq(cube(i, j, whole_domain));

          for ( k = 0; k < (g + c + 1); ++k )
          {
            // Case 0: Initialize
            if ( k == 0 )
            {
              gates_used = c;
              for ( length_type lnd = g; lnd < g + c; ++lnd )
                sum += cpow_vec.get(1 + lnd);
            }
            // Case 1: No cell included on left side of CFAR; 
            // very close to left boundary 
            else if ( k < (g + 1) )
            {
              gates_used = c;
              sum += cpow_vec.get(k+g+c)   - cpow_vec.get(k+g);
            }
            // Case 2: Some cells included on left side of CFAR;
            // close to left boundary 
            else
            {
              gates_used = c + k - (g + 1);
              sum += cpow_vec.get(k+g+c)   - cpow_vec.get(k+g) 
                + cpow_vec.get(k-(g+1));
            }
            inv_gates = (1.0 / gates_used);

            if ( cpow_vec.get(k) / max((sum * inv_gates), eps) >
              this->mu_ )
              located.row(k).put(count(k)++, Index<2>(i, j));
          }

          gates_used = 2 * c;
          inv_gates = (1.0 / gates_used);
          for ( k = (g + c + 1); (k + (g + c)) < gates; ++k )
          {
            // Case 3: All cells included on left and right side of CFAR;
            // somewhere in the middle of the range vector
            sum += cpow_vec.get(k+g+c)     - cpow_vec.get(k+g) 
              + cpow_vec.get(k-(g+1))   - cpow_vec.get(k-(c+g+1));

            if ( cpow_vec.get(k) / max((sum * inv_gates), eps) >
              this->mu_ )
              located.row(k).put(count(k)++, Index<2>(i, j));
          }

          for ( k = gates - (g + c); k < gates; ++k )
          {
            // Case 4: Some cells included on right side of CFAR;
            // close to right boundary
            if ( (k + g) < gates )
            {
              gates_used = c + gates - (k + g);
              sum +=                             - cpow_vec.get(k+g) 
                + cpow_vec.get(k-(g+1)) - cpow_vec.get(k-(c+g+1));
            }
            // Case 5: No cell included on right side of CFAR; 
            // very close to right boundary 
            else
            {
              gates_used = c;
              sum += cpow_vec.get(k-(g+1)) - cpow_vec.get(k-(c+g+1));
            }
            inv_gates = (1.0 / gates_used);
            if ( cpow_vec.get(k) / max((sum * inv_gates), eps) >
              this->mu_ )
              located.row(k).put(count(k)++, Index<2>(i, j));
          }    
        }
    }
    t1.stop();
    time = t1.delta();
  }

  t_cfar_base(length_type beams, length_type bins, 
    length_type cfar_gates, length_type guard_cells)
    : beams_(beams), dbins_(bins), cfar_gates_(cfar_gates), 
      guard_cells_(guard_cells), ntargets_(30), mu_(100)
  {}

protected:
  // Member data
  length_type const beams_;         // Number of beam locations
  length_type const dbins_;         //   "   "   doppler bins
  length_type const cfar_gates_;    //   "   "   ranges gates to consider
  length_type const guard_cells_;   //   "   "   cells to skip near target
  length_type const ntargets_;      //   "   "   targets
  length_type const mu_;            // Threshold for determining targets
};


/***********************************************************************
  t_cfar_base<T, ImplHybrid>  (using SIMD)
***********************************************************************/

// This uses GCC's vector extensions, in particular the builtin operators
// such as '+', '/', etc.  These are only supported in GCC 4.x and above.
#if __GNUC__ >= 4
#  if defined(__SSE__)
static const int zero[4] __attribute__((aligned(16))) =
    { 0, 0, 0, 0 };

typedef float v4sf __attribute__ ((vector_size(16)));

inline v4sf
load_scalar(float& scalar)
{
  return _mm_load1_ps(&scalar);
}

inline v4sf
max(v4sf a, v4sf b)
{
  return _mm_max_ps(a, b);
}

inline bool
anytrue(v4sf a)
{
  return _mm_movemask_ps(a) != 0;
}

inline v4sf
gt(v4sf a, v4sf b)
{
  return _mm_cmpgt_ps(a, b);
}

template <typename T>
struct t_cfar_base<T, ImplHybrid> : public Benchmark_base
{
  char const *what() { return "t_cfar_sweep_range<T, ImplHybrid>"; }

  template <typename Block>
  void
  cfar_detect(
    Tensor<T, Block>    cube,
    Tensor<T, Block> /* cpow */,
    Matrix<Index<2> >   located,
    Vector<length_type> count,
    length_type         loop,
    float&              time)
  {
    length_type const c = cfar_gates_;
    length_type const g = guard_cells_;
    length_type const beams = cube.size(0);
    length_type const dbins = cube.size(1);
    length_type const gates = cube.size(2);

    Vector<v4sf> strip(gates);

    T eps = Precision_traits<T>::eps;
    T fmu = this->mu_;
    v4sf v_eps  = load_scalar(eps);
    v4sf v_mu   = load_scalar(fmu);

    vsip::impl::profile::Timer t1;
    t1.start();
    for (index_type l=0; l<loop; ++l)
    {
      // Clear count for targets found per gate.
      count = 0;

      for ( index_type i = 0; i < beams; ++i )
        for ( index_type j = 0; j < dbins; j+=4 )
        {
          length_type gates_used;
          index_type k;
          T inv_gates;
          v4sf sum = *(__v4sf*)zero;

          for (index_type aa=0; aa<gates; ++aa)
          {
            v4sf x;
            float* f = (float*)&x;
            f[0] = cube.get(i, j+0, aa);
            f[1] = cube.get(i, j+1, aa);
            f[2] = cube.get(i, j+2, aa);
            f[3] = cube.get(i, j+3, aa);
            strip.put(aa, x*x);
          }

          for ( k = 0; k < (g + c + 1); ++k )
          {
            // Case 0: Initialize
            if ( k == 0 )
            {
              gates_used = c;
              for ( length_type lnd = g; lnd < g + c; ++lnd )
                sum = sum + strip.get(1 + lnd);
            }
            // Case 1: No cell included on left side of CFAR; 
            // very close to left boundary 
            else if ( k < (g + 1) )
            {
              gates_used = c;
              sum += strip.get(k+g+c)   - strip.get(k+g);
            }
            // Case 2: Some cells included on left side of CFAR;
            // close to left boundary 
            else
            {
              gates_used = c + k - (g + 1);
              sum += strip.get(k+g+c)   - strip.get(k+g) 
                + strip.get(k-(g+1));
            }

            inv_gates = (1.0 / gates_used);
            v4sf factor = load_scalar(inv_gates);
            v4sf tmp;
            tmp      = strip.get(k) / max(sum * factor, v_eps);
            tmp      = _mm_cmpgt_ps(tmp, v_mu);
            if (anytrue(tmp))
            {
              float* vsum = (float*)& tmp;
              for (int aa=0; aa<4; ++aa)
              {
                if (vsum[aa])
                  located.row(k).put(count(k)++, Index<2>(i, j+aa));
              }
            }
          }

          gates_used = 2 * c;
          inv_gates = (1.0 / gates_used);
          v4sf v_inv_gates = load_scalar(inv_gates);
          for ( k = (g + c + 1); (k + (g + c)) < gates; ++k )
          {
            // Case 3: All cells included on left and right side of CFAR;
            // somewhere in the middle of the range vector
            sum += strip.get(k+g+c)   - strip.get(k+g) 
              + strip.get(k-(g+1)) - strip.get(k-(c+g+1));

            v4sf tmp = gt(strip.get(k) / max(sum * v_inv_gates, v_eps), v_mu);
            if (anytrue(tmp))
            {
              float* vsum = (float*)& tmp;
              for (int aa=0; aa<4; ++aa)
              {
                if (vsum[aa])
                  located.row(k).put(count(k)++, Index<2>(i, j+aa));
              }
            }
          }

          for ( k = gates - (g + c); k < gates; ++k )
          {
            // Case 4: Some cells included on right side of CFAR;
            // close to right boundary
            if ( (k + g) < gates )
            {
              gates_used = c + gates - (k + g);
              sum +=                    - strip.get(k+g) 
                +  strip.get(k-(g+1)) - strip.get(k-(c+g+1));
            }
            // Case 5: No cell included on right side of CFAR; 
            // very close to right boundary 
            else
            {
              gates_used = c;
              sum += strip.get(k-(g+1)) - strip.get(k-(c+g+1));
            }
            inv_gates = (1.0 / gates_used);
            v4sf factor = load_scalar(inv_gates);
            v4sf tmp;
            tmp      = strip.get(k) / _mm_max_ps(sum * factor, v_eps);
            tmp      = _mm_cmpgt_ps(tmp, v_mu);
            int hit  = _mm_movemask_ps(tmp);
            if (hit)
            {
              float* vsum = (float*)& tmp;
              for (int aa=0; aa<4; ++aa)
              {
                if (vsum[aa])
                  located.row(k).put(count(k)++, Index<2>(i, j+aa));
              }
            }
          }    
        }
    }
    t1.stop();
    time = t1.delta();
  }


  t_cfar_base(length_type beams, length_type bins, 
    length_type cfar_gates, length_type guard_cells)
    : beams_(beams), dbins_(bins), cfar_gates_(cfar_gates), 
      guard_cells_(guard_cells), ntargets_(30), mu_(100)
  {}

protected:
  // Member data
  length_type const beams_;         // Number of beam locations
  length_type const dbins_;         //   "   "   doppler bins
  length_type const cfar_gates_;    //   "   "   ranges gates to consider
  length_type const guard_cells_;   //   "   "   cells to skip near target
  length_type const ntargets_;      //   "   "   targets
  length_type const mu_;            // Threshold for determining targets
};


#  endif // defined(__SSE__)
#endif // __GNUC__ >= 4




/***********************************************************************
  Benchmark driver defintions.
***********************************************************************/

template <typename T,
          typename ImplTag,
 	  typename OrderT = tuple<0, 1, 2> >
struct t_cfar_sweep_range : public t_cfar_base<T, ImplTag>
{
  int ops_per_point(length_type /*size*/)
  { 
    int ops = vsip::impl::Ops_info<T>::sqr + vsip::impl::Ops_info<T>::mul
        + 4 * vsip::impl::Ops_info<T>::add + vsip::impl::Ops_info<T>::div; 
    return (this->beams_ * this->dbins_ * ops);
  }
  int riob_per_point(length_type) { return -1; }
  int wiob_per_point(length_type) { return -1; }
  int mem_per_point(length_type) 
    { return this->beams_ * this->dbins_ * sizeof(T); }

#if PARALLEL_CFAR
    typedef Map<Block_dist, Block_dist, Block_dist>  root_map_type;
    typedef Dense<3, T, OrderT, root_map_type>       root_block_type;
    typedef Tensor<T, root_block_type>               root_view_type;
#else
    typedef Dense<3, T, OrderT>         root_block_type;
    typedef Tensor<T, root_block_type>  root_view_type;
#endif

  void 
  initialize_cube(root_view_type& root)
  {
    length_type const gates = root.size(2);

#if PARALLEL_CFAR
    // The processor set contains only one processor, hence the map 
    // has only one subblock.
    Vector<processor_type> pset0(1);
    pset0(0) = processor_set()(0);
    root_map_type root_map(pset0, 1, 1);
    
    // Create some test data
    root_view_type root_cube(this->beams_, this->dbins_, gates, root_map);
#else
    root_view_type root_cube(this->beams_, this->dbins_, gates);
#endif

    // Only the root processor need initialize the target array
#if PARALLEL_CFAR
    if (root_map.subblock() != no_subblock)
#endif
    {
      // First, place a background of uniform noise 
      index_type seed = 1;
      T max_val = T(1);
      T min_val = T(1 / sqrt(T(2)));
      Rand<T> gen(seed, 0);
      for ( length_type i = 0; i < this->beams_; ++i )
        for ( length_type j = 0; j < this->dbins_; ++j )
          for ( length_type k = 0; k < gates; ++k )
            root_cube.local().put(i, j, k, 
              T(max_val - min_val) * gen.randu() + min_val);
      
      // Place several targets within the data cube
      Matrix<index_type> placed(this->ntargets_, 3);
      for ( length_type t = 0; t < this->ntargets_; ++t )
      {
        int b = static_cast<int>(gen.randu() * (this->beams_ - 2)) + 1;
        int d = static_cast<int>(gen.randu() * (this->dbins_ - 2)) + 1;
        int r = static_cast<int>(gen.randu() * (gates - 2)) + 1;
        
        root_cube.local().put(b, d, r, T(50.0));
        placed(t, 0) = b;
        placed(t, 1) = d;
        placed(t, 2) = r;
      }
    }
    
    root = root_cube;
  }


  template <typename Block>
  void
  cfar_verify(
    Tensor<T, Block>    cube,
    Matrix<Index<2> >   located,
    Vector<length_type> count)
  {
    // Sum all the targets found on each processor
    length_type l_total_found = 0;
    for ( index_type i = 0; i < cube.size(2); ++i )
      for ( index_type j = 0; j < count(i); ++j )
      {
	test_assert( cube.get(located.get(i, j)[0], 
                              located.get(i, j)[1], i) == T(50.0) );
	++l_total_found;
      }

#if PARALLEL_CFAR
    // Create a vector with one element on each processor.
    length_type np = num_processors();
    length_type lp = local_processor();
    Vector<length_type, Dense<1, length_type, row1_type, Map<> > >
      sum(np, Map<>(np));
    sum.put(lp, l_total_found);

    // Parallel reduction.
    length_type total_found = sumval(sum);
#else
    length_type lp = 0;
    length_type total_found = l_total_found;
#endif

    // Warn if we don't find all the targets.
    if( total_found != this->ntargets_ && lp == 0 )
      std::cerr << "only found " << total_found
		<< " out of " << this->ntargets_
		<< std::endl;
  }


  void operator()(length_type size, length_type loop, float& time)
  {
    length_type beams = this->beams_;
    length_type dbins = this->dbins_;
    length_type gates = size;

    // The number of range gates must be sufficiently greater than the sum
    // of CFAR gates and guard cells.  If not, the radar signal processing 
    // parameters are flawed!
    test_assert( 2 * (this->cfar_gates_ + this->guard_cells_) < gates );
    
    // Create a "root" view for initialization.  Only the first processor
    // will access the data.
    root_view_type root(beams, dbins, gates);
    initialize_cube(root);

    // Create a (possibly distributed) view for computation.  Also create a 
    // temporary cube with an identical map to hold squared values.
#if PARALLEL_CFAR
    typedef Map<Block_dist, Block_dist, Whole_dist>  map_type;
    typedef Dense<3, T, OrderT, map_type>            block_type;
    typedef Tensor<T, block_type>                    view_type;
    typedef typename view_type::local_type           local_type;

    processor_type np = num_processors();
    map_type map = map_type(Block_dist(np), Block_dist(1), Whole_dist());

    view_type dist_cube(beams, dbins, gates, map);
    view_type dist_cpow(beams, dbins, gates, map);

    dist_cube = root;

    local_type cube = dist_cube.local();
    local_type cpow = dist_cpow.local();
#else
    typedef Dense<3, T, OrderT>     block_type;
    typedef Tensor<T, block_type>   view_type;
    typedef view_type local_type;

    view_type& cube = root;
    view_type cpow(beams, dbins, gates);
#endif


    // Create a place to store the locations of targets that are found
    Matrix<Index<2> > located(gates, this->ntargets_, Index<2>());
    Vector<length_type> count(gates);
    
    // Process the data cube and time it
    cfar_detect(cube, cpow, located, count, loop, time);

    // Verify targets detected
    cfar_verify(cube, located, count);
  }


  t_cfar_sweep_range(length_type beams, length_type bins,
                 length_type cfar_gates, length_type guard_cells)
   : t_cfar_base<T, ImplTag>(beams, bins, cfar_gates, guard_cells)
  {}
};


void
defaults(Loop1P& loop)
{
  loop.loop_start_ = 2;
  loop.progression_ = linear;
  loop.prog_scale_ = 100;
  loop.start_ = 2;
  loop.stop_ = 9;
}


template <> float  Precision_traits<float>::eps = 0.0;
template <> double Precision_traits<double>::eps = 0.0;

int
test(Loop1P& loop, int what)
{
  Precision_traits<float>::compute_eps();
  Precision_traits<double>::compute_eps();

/* From PCA Kernel-Level Benchmarks Project Report:

                  Parameter sets for the CFAR Kernel Benchmark.
Name     Description                     Set 1  Set 2 Set 3 Set 4 Units
Nbm      Number of beams                  16     48    48    16   beams
Nrg      Number of range gates            64    3500  1909  9900  range gates
Ndop     Number of doppler bins           24    128    64    16   doppler bins
Ntgts    Number of targets that will be   30     30    30    30   targets
         pseudo-randomly distributed
         in Radar data cube
Ncfar    Number of CFAR range gates        5     10    10    20   range gates
G        CFAR guard cells                  4      8     8    16   range gates
mu       Detection sensitivity factor    100    100   100   100
W        Workload                        0.17   150    41    18   Mﬂop
*/

  typedef float F;
  typedef double D;
  typedef tuple<2, 0, 1>  S;  // Slice storage order
  typedef tuple<1, 0, 2>  V;  // Vector storage order
  typedef tuple<0, 1, 2>  H;  // Hybrid storage order

  switch (what)
  {
  // parameters are number of: beams, doppler bins, CFAR range gates and 
  // CFAR guard cells respectively
  case  1: loop(t_cfar_sweep_range<F, ImplSlice, S>(16,  24,  5,  4)); break;
  case  2: loop(t_cfar_sweep_range<F, ImplSlice, S>(48, 128, 10,  8)); break;
  case  3: loop(t_cfar_sweep_range<F, ImplSlice, S>(48,  64, 10,  8)); break;
  case  4: loop(t_cfar_sweep_range<F, ImplSlice, S>(16,  16, 20, 16)); break;

  case 11: loop(t_cfar_sweep_range<D, ImplSlice, S>(16,  24,  5,  4)); break;
  case 12: loop(t_cfar_sweep_range<D, ImplSlice, S>(48, 128, 10,  8)); break;
  case 13: loop(t_cfar_sweep_range<D, ImplSlice, S>(48,  64, 10,  8)); break;
  case 14: loop(t_cfar_sweep_range<D, ImplSlice, S>(16,  16, 20, 16)); break;

  case 21: loop(t_cfar_sweep_range<F, ImplVector, V>(16,  24,  5,  4)); break;
  case 22: loop(t_cfar_sweep_range<F, ImplVector, V>(48, 128, 10,  8)); break;
  case 23: loop(t_cfar_sweep_range<F, ImplVector, V>(48,  64, 10,  8)); break;
  case 24: loop(t_cfar_sweep_range<F, ImplVector, V>(16,  16, 20, 16)); break;

  case 31: loop(t_cfar_sweep_range<D, ImplVector, V>(16,  24,  5,  4)); break;
  case 32: loop(t_cfar_sweep_range<D, ImplVector, V>(48, 128, 10,  8)); break;
  case 33: loop(t_cfar_sweep_range<D, ImplVector, V>(48,  64, 10,  8)); break;
  case 34: loop(t_cfar_sweep_range<D, ImplVector, V>(16,  16, 20, 16)); break;

#if __GNUC__ >= 4
#  if defined(__SSE__)
  case 41: loop(t_cfar_sweep_range<F, ImplHybrid, H>(16,  24,  5,  4)); break;
  case 42: loop(t_cfar_sweep_range<F, ImplHybrid, H>(48, 128, 10,  8)); break;
  case 43: loop(t_cfar_sweep_range<F, ImplHybrid, H>(48,  64, 10,  8)); break;
  case 44: loop(t_cfar_sweep_range<F, ImplHybrid, H>(16,  16, 20, 16)); break;
#  endif // defined(__SSE__)
#endif // __GNUC__ >= 4

  case 0:
    std::cout
      << "cfar -- Constant False Alarm Rate Detection\n"
      << "\n"
      << "  type\n"
      << "    F   float\n"
      << "    D   double\n"
      << "\n"
      << "  storage\n"
      << "    S   slice\n"
      << "    V   vector\n"
      << "    H   hybrid\n"
      << "                                   CFAR  CFAR\n"
      << "                           doppler range guard\n"
      << "        type storage beams   bins  gates cells\n"
      << "        ---- ------- ----- ------- ----- -----\n"
      << "   -1:    F     S      16      24     5     4\n"
      << "   -2:    F     S      48     128    10     8\n"
      << "   -3:    F     S      48      64    10     8\n"
      << "   -4:    F     S      16      16    20    16\n"
      << "\n"
      << "  -11:    D     S      16      24     5     4\n"
      << "  -12:    D     S      48     128    10     8\n"
      << "  -13:    D     S      48      64    10     8\n"
      << "  -14:    D     S      16      16    20    16\n"
      << "\n"
      << "  -21:    F     V      16      24     5     4\n"
      << "  -22:    F     V      48     128    10     8\n"
      << "  -23:    F     V      48      64    10     8\n"
      << "  -24:    F     V      16      16    20    16\n"
      << "\n"
      << "  -31:    D     V      16      24     5     4\n"
      << "  -32:    D     V      48     128    10     8\n"
      << "  -33:    D     V      48      64    10     8\n"
      << "  -34:    D     V      16      16    20    16\n"
      << "\n"
      << "  -41:    F     H      16      24     5     4   SSE\n"
      << "  -42:    F     H      48     128    10     8   SSE\n"
      << "  -43:    F     H      48      64    10     8   SSE\n"
      << "  -44:    F     H      16      16    20    16   SSE\n"
      ;

  default: 
    return 0;
  }
  return 1;
}
 
