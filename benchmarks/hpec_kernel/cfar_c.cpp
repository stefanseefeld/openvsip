/* Copyright (c) 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

/// Description
///   Constant False Alarm Rate Detection - High
///   Performance Embedded Computing (HPEC) Kernel-Level Benchmarks
///   C version of benchmark.
///
///   Briefly, this problem involves finding targets based on data within a 
///   three-dimensional cube of 'beam locations', 'range gates' and 'doppler 
///   bins'.  It does this by comparing the signal in a given cell to that of
///   nearby cells in order to avoid false-detection of targets.  The range 
///   gate parameter is varied when considering 'nearby' cells.  A certain 
///   number of guard cells are skipped, resulting in a computation that sums
///   the values from two thick slices of this data cube (one on either side 
///   of the slice for a particular range gate).  The HPEC PCA Kernel-Level 
///   benchmark paper has a diagram that shows one cell under consideration.
///   Please refer to it if needed.
///
///   The algorithm involves these basic steps:
///    - compute the squares of all the values in the data cube
///    - for each range gate:
///    - sum the squares of desired values around the current range gate
///    - compute the normalized power for each cell in the slice
///    - search for values that exceed a certain threshold
///
///   Some of the code relates to boundary conditions (near either end of the 
///   'range gates' parameter), but otherwise it follows the above description. 

#include <iostream>
#if defined(__SSE__)
#  include <xmmintrin.h>
#endif

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
  cfar function tests
***********************************************************************/

template <typename T>
struct t_cfar_base : Benchmark_base
{
  int ops_per_point(length_type /*size*/)
  { 
    int ops = vsip::impl::Ops_info<T>::sqr + vsip::impl::Ops_info<T>::mul
        + 4 * vsip::impl::Ops_info<T>::add + vsip::impl::Ops_info<T>::div; 
    return (beams_ * dbins_ * ops);
  }
  int riob_per_point(length_type) { return -1; }
  int wiob_per_point(length_type) { return -1; }
  int mem_per_point(length_type) 
    { return this->beams_ * this->dbins_ * sizeof(T); }

#if PARALLEL_CFAR
  typedef Map<Block_dist, Block_dist, Block_dist>  root_map_type;
  typedef Dense<3, T, row3_type, root_map_type>    root_block_type;
  typedef Tensor<T, root_block_type>               root_view_type;
  typedef typename root_view_type::local_type      local_type;
#else
  typedef Dense<3, T, row3_type>       root_block_type;
  typedef Tensor<T, root_block_type>   root_view_type;
  typedef root_view_type               local_type;
#endif


  void 
  initialize_cube(root_view_type& root)
  {
    length_type const& gates = root.size(2);

#if PARALLEL_CFAR
    // The processor set contains only one processor, hence the map 
    // has only one subblock.
    Vector<processor_type> pset0(1);
    pset0(0) = processor_set()(0);
    root_map_type root_map(pset0, 1, 1);
    
    // Create some test data
    root_view_type root_cube(beams_, dbins_, gates, root_map);
#else
    root_view_type root_cube(beams_, dbins_, gates);
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
      for ( length_type i = 0; i < beams_; ++i )
        for ( length_type j = 0; j < dbins_; ++j )
          for ( length_type k = 0; k < gates; ++k )
            root_cube.local().put(i, j, k, 
              T(max_val - min_val) * gen.randu() + min_val);
      
      // Place several targets within the data cube
      Matrix<index_type> placed(ntargets_, 3);
      for ( length_type t = 0; t < ntargets_; ++t )
      {
        int b = static_cast<int>(gen.randu() * (beams_ - 2)) + 1;
        int d = static_cast<int>(gen.randu() * (dbins_ - 2)) + 1;
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
    Tensor<T, Block>   l_cube,
    Matrix<Index<2> >  located,
    length_type        count[])
  {
    // Create a vector with one element on each processor.
    length_type np = num_processors();
    Vector<length_type, Dense<1, length_type, row1_type, Map<> > >
      sum(np, Map<>(np));

    length_type l_total_found = 0;
    for ( index_type i = 0; i < l_cube.size(2); ++i )
      for ( index_type j = 0; j < count[i]; ++j )
      {
	test_assert( l_cube.get(located.get(i, j)[0], 
				located.get(i, j)[1], i) == T(50.0) );
	++l_total_found;
      }
    sum.put(local_processor(), l_total_found);

    // Parallel reduction.
    length_type total_found = sumval(sum);

    // Warn if we don't find all the targets.
    if( total_found != this->ntargets_ && local_processor() == 0 )
      std::cerr << "only found " << total_found
		<< " out of " << this->ntargets_
		<< std::endl;
  }

  t_cfar_base(length_type beams, length_type bins)
    : beams_(beams), dbins_(bins), ntargets_(30)
  {}


protected:
  // Member data
  length_type const beams_;    // Number of beam locations
  length_type const dbins_;    // Number of doppler bins
  length_type const ntargets_; // Number of targets
};



/***********************************************************************
  cfar_by_vector_c 
***********************************************************************/

template <typename T>
struct t_cfar_by_vector_c : public t_cfar_base<T>
{
  char const *what() { return "t_cfar_by_vector_c"; }

#if PARALLEL_CFAR
  typedef Map<Block_dist, Block_dist, Block_dist>  root_map_type;
  typedef Dense<3, T, row3_type, root_map_type>    root_block_type;
  typedef Tensor<T, root_block_type>               root_view_type;
#else
  typedef Dense<3, T, row3_type>       root_block_type;
  typedef Tensor<T, root_block_type>   root_view_type;
#endif


  template <typename Block>
  void
  cfar_detect(
    Tensor<T, Block>   cube,
    Tensor<T, Block>   cpow,
    Matrix<Index<2> >  located,
    length_type        count[])
  {
    typedef typename 
      Tensor<T, Block>::template subvector<0, 1>::type  subvector_type;

    length_type c = cfar_gates_;
    length_type g = guard_cells_;
    length_type gates_used = 0;

    // The number of range gates must be sufficiently greater than the sum
    // of CFAR gates and guard cells.  If not, the radar signal processing 
    // parameters are flawed!
    length_type gates = cube.size(2);
    test_assert( 2 * (c + g) < gates );


    // Compute the square of all values in the data cube.  This is 
    // done in advance once, as the values are needed many times
    // (approximately twice as many times as the number of guard cells)
    // cpow = sq(cube);

    // Clear scratch space used to hold sums of squares and counts for 
    // targets found per gate.
    T sum = T();
    index_type k;
    for ( k = 0; k < gates; ++k )
      count[k] = 0;

    dda::Data<Block, dda::inout> data_cube(cube.block());

    subvector_type cpow_vec = cpow(0, 0, whole_domain);
    dda::Data<typename subvector_type::block_type, dda::inout>
      data_cpow_vec(cpow_vec.block());

    float* p_cube = data_cube.ptr();
    float* p_cpow = data_cpow_vec.ptr();

    length_type l_beams = cube.size(0);
    length_type l_dbins = cube.size(1);
    for ( index_type i = 0; i < l_beams; ++i )
    {
      for ( index_type j = 0; j < l_dbins; ++j )
      {
	sum = T();

	for (index_type aa=0; aa<gates; ++aa)
	  p_cpow[aa] = p_cube[aa] * p_cube[aa];
	p_cube += gates;
	  

        for ( k = 0; k < (g + c + 1); ++k )
        {
          // Case 0: Initialize
          if ( k == 0 )
          {
            gates_used = c;
            for ( length_type lnd = g; lnd < g + c; ++lnd )
              sum += p_cpow[1 + lnd];
          }
          // Case 1: No cell included on left side of CFAR; 
          // very close to left boundary 
          else if ( k < (g + 1) )
          {
            gates_used = c;
            sum += p_cpow[k+g+c]   - p_cpow[k+g];
          }
          // Case 2: Some cells included on left side of CFAR;
          // close to left boundary 
          else
          {
            gates_used = c + k - (g + 1);
            sum += p_cpow[k+g+c]  - p_cpow[k+g]
                 + p_cpow[k-(g+1)];
          }
          T inv_gates = (1.0 / gates_used);
          if ( p_cpow[k] / max((sum * inv_gates), Precision_traits<T>::eps) >
               this->mu_ )
            located.row(k).put(count[k]++, Index<2>(i, j));
        }

        gates_used = 2 * c;
        T inv_gates = (1.0 / gates_used);
        for ( k = (g + c + 1); (k + (g + c)) < gates; ++k )
        {
          // Case 3: All cells included on left and right side of CFAR;
          // somewhere in the middle of the range vector
          sum += p_cpow[k+g+c]     - p_cpow[k+g] 
               + p_cpow[k-(g+1)]   - p_cpow[k-(c+g+1)];

          if ( p_cpow[k] / max((sum * inv_gates), Precision_traits<T>::eps) >
               this->mu_ )
            located.row(k).put(count[k]++, Index<2>(i, j));
        }

        for ( k = gates - (g + c); k < gates; ++k )
        {
          // Case 4: Some cells included on right side of CFAR;
          // close to right boundary
          if ( (k + g) < gates )
          {
            gates_used = c + gates - (k + g);
            sum +=                             - p_cpow[k+g] 
                 + p_cpow[k-(g+1)] - p_cpow[k-(c+g+1)];
          }
          // Case 5: No cell included on right side of CFAR; 
          // very close to right boundary 
          else
          {
            gates_used = c;
            sum += p_cpow[k-(g+1)] - p_cpow[k-(c+g+1)];
          }
          T inv_gates = (1.0 / gates_used);
          if ( p_cpow[k] / max((sum * inv_gates), Precision_traits<T>::eps) >
               this->mu_ )
            located.row(k).put(count[k]++, Index<2>(i, j));
        }    
      }
    }
  }


  void operator()(length_type size, length_type loop, float& time)
  {
    length_type beams = this->beams_;
    length_type dbins = this->dbins_;
    length_type gates = size;
    
    // Create a "root" view for each that will give the first
    // processor access to all of the data.  
    root_view_type root(beams, dbins, gates);
    initialize_cube(root);
    
#if PARALLEL_CFAR
    // Create the distributed views that will give each processor a 
    // subset of the data
    typedef Map<Block_dist, Block_dist, Whole_dist>  map_type;
    typedef Dense<3, T, row3_type, map_type>         block_type;
    typedef Tensor<T, block_type>                    view_type;
    typedef typename view_type::local_type           local_type;

    processor_type np = num_processors();
    map_type map = map_type(Block_dist(np), Block_dist(1), Whole_dist());

    view_type cube(beams, dbins, gates, map);
    cube = root;

    // Create temporary to hold squared values
    view_type cpow(beams, dbins, gates, map);
#else
    typedef Dense<3, T, col1_type>  block_type;
    typedef Tensor<T, block_type>   view_type;
    typedef view_type local_type;

    view_type& cube = root;

    view_type cpow(beams, dbins, gates);
#endif

    local_type l_cube = LOCAL(cube);
    local_type l_cpow = LOCAL(cpow);
    // length_type l_beams  = l_cube.size(0);

    // And a place to hold found targets
    Matrix<Index<2> > located(gates, this->ntargets_, Index<2>());
    length_type *count = new length_type[gates];

    
    // Run the test and time it
    vsip_csl::profile::Timer t1;
    t1.start();
    for (index_type l=0; l<loop; ++l)
    {
      cfar_detect(l_cube, l_cpow, located, count);
    }
    t1.stop();
    time = t1.delta();


    // Verify targets detected
    cfar_verify(l_cube, located, count);

    delete[] count;
  }


  t_cfar_by_vector_c(length_type beams, length_type bins,
                     length_type cfar_gates, length_type guard_cells)
   : t_cfar_base<T>(beams, bins),
     cfar_gates_(cfar_gates), guard_cells_(guard_cells), 
     mu_(100)
  {}

public:
  // Member data
  length_type cfar_gates_;    // Number of ranges gates to consider
  length_type guard_cells_;   // Number of cells to skip near target
  length_type mu_;            // Threshold for determining targets
};



/***********************************************************************
  cfar_by_vector_csimd
***********************************************************************/
#if defined(__SSE__)

static const int zero[4] __attribute__((aligned(16))) =
    { 0, 0, 0, 0 };

template <typename T>
struct t_cfar_by_vector_csimd : public t_cfar_base<T>
{
  char const *what() { return "t_cfar_by_vector_csimd"; }

#if PARALLEL_CFAR
  typedef Map<Block_dist, Block_dist, Block_dist>  root_map_type;
  typedef Dense<3, T, row3_type, root_map_type>    root_block_type;
  typedef Tensor<T, root_block_type>               root_view_type;
#else
  typedef Dense<3, T, row3_type>       root_block_type;
  typedef Tensor<T, root_block_type>   root_view_type;
#endif


  template <typename Block>
  void
  cfar_detect(
    Tensor<T, Block>   cube,
    Tensor<T, Block>   cpow,
    Matrix<Index<2> >  located,
    length_type        count[])
  {
    typedef typename 
      Tensor<T, Block>::template subvector<0, 1>::type  subvector_type;

    length_type c = cfar_gates_;
    length_type g = guard_cells_;
    length_type gates_used = 0;

    // The number of range gates must be sufficiently greater than the sum
    // of CFAR gates and guard cells.  If not, the radar signal processing 
    // parameters are flawed!
    length_type gates = cube.size(2);
    test_assert( 2 * (c + g) < gates );


    // Compute the square of all values in the data cube.  This is 
    // done in advance once, as the values are needed many times
    // (approximately twice as many times as the number of guard cells)
    // cpow = sq(cube);

    // Clear scratch space used to hold sums of squares and counts for 
    // targets found per gate.
    index_type k;
    for ( k = 0; k < gates; ++k )
      count[k] = 0;

    float* vsum;

    dda::Data<Block, dda::inout> data_cube(cube.block());
    dda::Data<Block, dda::inout> data_cpow(cpow.block());

    float* p_cube = data_cube.ptr();
    float* p_cpow = data_cpow.ptr();

    length_type l_beams = cube.size(0);
    length_type l_dbins = cube.size(1);
    for ( index_type i = 0; i < l_beams; ++i )
    {
      for ( index_type j = 0; j < l_dbins; j += 4 )
      {
	__m128 sum = *(__m128*)zero;

	for (index_type aa=0; aa<gates; ++aa)
	{
	  p_cpow[4*aa+0] = p_cube[0*gates+aa] * p_cube[0*gates+aa];
	  p_cpow[4*aa+1] = p_cube[1*gates+aa] * p_cube[1*gates+aa];
	  p_cpow[4*aa+2] = p_cube[2*gates+aa] * p_cube[2*gates+aa];
	  p_cpow[4*aa+3] = p_cube[3*gates+aa] * p_cube[3*gates+aa];
	}

	p_cube += 4*gates;
	  

        for ( k = 0; k < (g + c + 1); ++k )
        {
          // Case 0: Initialize
          if ( k == 0 )
          {
            gates_used = c;
            for ( length_type lnd = g; lnd < g + c; ++lnd )
	      sum = _mm_add_ps(sum, *(__m128*)(p_cpow + 4*(1 + lnd)));
	      // sum += p_cpow[1 + lnd];
          }
          // Case 1: No cell included on left side of CFAR; 
          // very close to left boundary 
          else if ( k < (g + 1) )
          {
            gates_used = c;
            // sum += p_cpow[k+g+c]   - p_cpow[k+g];
	    sum = _mm_add_ps(sum, *(__m128*)(p_cpow + 4*(k+g+c)));
	    sum = _mm_sub_ps(sum, *(__m128*)(p_cpow + 4*(k+g)));
          }
          // Case 2: Some cells included on left side of CFAR;
          // close to left boundary 
          else
          {
            gates_used = c + k - (g + 1);
            // sum += p_cpow[k+g+c]  - p_cpow[k+g] + p_cpow[k-(g+1)];
	    sum = _mm_add_ps(sum, *(__m128*)(p_cpow + 4*(k+g+c)));
	    sum = _mm_sub_ps(sum, *(__m128*)(p_cpow + 4*(k+g)));
	    sum = _mm_add_ps(sum, *(__m128*)(p_cpow + 4*(k-(g+1))));
          }

          T inv_gates = (1.0 / gates_used);
	  T eps = Precision_traits<T>::eps;
	  __m128 factor = _mm_load1_ps(&inv_gates);
	  __m128 v_eps  = _mm_load1_ps(&eps);
	  T fmu = this->mu_;
	  __m128 v_mu   = _mm_load1_ps(&fmu);
	  __m128 tmp    = _mm_mul_ps(sum, factor);
	  tmp           = _mm_max_ps(tmp, v_eps);
	  tmp           = _mm_div_ps(*(__m128*)(p_cpow + 4*k), tmp);
	  tmp           = _mm_cmpgt_ps(tmp, v_mu);

	  vsum = (float*)& tmp;
	  for (int aa=0; aa<4; ++aa)
	  {
	    if (vsum[aa])
	      located.row(k).put(count[k]++, Index<2>(i, j+aa));
	  }
        }

        gates_used = 2 * c;
        T inv_gates = (1.0 / gates_used);
	T eps = Precision_traits<T>::eps;
	T fmu = this->mu_;
	__m128 factor = _mm_load1_ps(&inv_gates);
	__m128 v_eps  = _mm_load1_ps(&eps);
	__m128 v_mu   = _mm_load1_ps(&fmu);
        for ( k = (g + c + 1); (k + (g + c)) < gates; ++k )
        {
          // Case 3: All cells included on left and right side of CFAR;
          // somewhere in the middle of the range vector
          // sum += p_cpow[k+g+c]     - p_cpow[k+g] 
          //      + p_cpow[k-(g+1)]   - p_cpow[k-(c+g+1)];
	  __m128 v1 = _mm_load_ps(p_cpow + 4*(k+g+c));
	  __m128 v2 = _mm_load_ps(p_cpow + 4*(k+g));
	  __m128 v3 = _mm_load_ps(p_cpow + 4*(k-(g+1)));
	  __m128 v4 = _mm_load_ps(p_cpow + 4*(k-(c+g+1)));

	  sum = _mm_add_ps(sum, v1);
	  sum = _mm_sub_ps(sum, v2);
	  sum = _mm_add_ps(sum, v3);
	  sum = _mm_sub_ps(sum, v4);

	  __m128 tmp    = _mm_mul_ps(sum, factor);
	  tmp           = _mm_max_ps(tmp, v_eps);
	  tmp           = _mm_div_ps(*(__m128*)(p_cpow + 4*k), tmp);
	  tmp           = _mm_cmpgt_ps(tmp, v_mu);

	  int hit = _mm_movemask_ps(tmp);
	  if (hit)
	  {
	    vsum = (float*)& tmp;
	    for (int aa=0; aa<4; ++aa)
	    {
	      if (vsum[aa])
		located.row(k).put(count[k]++, Index<2>(i, j+aa));
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
            // sum +=                             - p_cpow[k+g] 
            //      + p_cpow[k-(g+1)] - p_cpow[k-(c+g+1)];
	    sum = _mm_sub_ps(sum, *(__m128*)(p_cpow + 4*(k+g)));
	    sum = _mm_add_ps(sum, *(__m128*)(p_cpow + 4*(k-(g+1))));
	    sum = _mm_sub_ps(sum, *(__m128*)(p_cpow + 4*(k-(c+g+1))));
          }
          // Case 5: No cell included on right side of CFAR; 
          // very close to right boundary 
          else
          {
            gates_used = c;
            // sum += p_cpow[k-(g+1)] - p_cpow[k-(c+g+1)];
	    sum = _mm_add_ps(sum, *(__m128*)(p_cpow + 4*(k-(g+1))));
	    sum = _mm_sub_ps(sum, *(__m128*)(p_cpow + 4*(k-(c+g+1))));
          }
          T inv_gates = (1.0 / gates_used);
	  vsum = (float*)&sum;
	  for (int aa=0; aa<4; ++aa)
	  {
	    if (p_cpow[4*k+aa] / max((vsum[aa] * inv_gates), Precision_traits<T>::eps) >
		 this->mu_ )
	      located.row(k).put(count[k]++, Index<2>(i, j+aa));
	  }
        }    
      }
    }
  }


  void operator()(length_type size, length_type loop, float& time)
  {
    length_type beams = this->beams_;
    length_type dbins = this->dbins_;
    length_type gates = size;
    
    // Create a "root" view for each that will give the first
    // processor access to all of the data.  
    root_view_type root(beams, dbins, gates);
    initialize_cube(root);
    
#if PARALLEL_CFAR
    // Create the distributed views that will give each processor a 
    // subset of the data
    typedef Map<Block_dist, Block_dist, Whole_dist>  map_type;
    typedef Dense<3, T, row3_type, map_type>         block_type;
    typedef Tensor<T, block_type>                    view_type;
    typedef typename view_type::local_type           local_type;

    processor_type np = num_processors();
    map_type map = map_type(Block_dist(np), Block_dist(1), Whole_dist());

    view_type cube(beams, dbins, gates, map);
    cube = root;

    // Create temporary to hold squared values
    view_type cpow(beams, dbins, gates, map);
#else
    typedef Dense<3, T, col1_type>  block_type;
    typedef Tensor<T, block_type>   view_type;
    typedef view_type local_type;

    view_type& cube = root;

    view_type cpow(beams, dbins, gates);
#endif

    local_type l_cube = LOCAL(cube);
    local_type l_cpow = LOCAL(cpow);
    // length_type l_beams  = l_cube.size(0);

    // And a place to hold found targets
    Matrix<Index<2> > located(gates, this->ntargets_, Index<2>());
    length_type *count = new length_type[gates];

    
    // Run the test and time it
    vsip_csl::profile::Timer t1;
    t1.start();
    for (index_type l=0; l<loop; ++l)
    {
      cfar_detect(l_cube, l_cpow, located, count);
    }
    t1.stop();
    time = t1.delta();


    // Verify targets detected
    cfar_verify(l_cube, located, count);

    delete[] count;
  }


  t_cfar_by_vector_csimd(length_type beams, length_type bins,
                     length_type cfar_gates, length_type guard_cells)
   : t_cfar_base<T>(beams, bins),
     cfar_gates_(cfar_gates), guard_cells_(guard_cells), 
     mu_(100)
  {}

public:
  // Member data
  length_type cfar_gates_;    // Number of ranges gates to consider
  length_type guard_cells_;   // Number of cells to skip near target
  length_type mu_;            // Threshold for determining targets
};

#endif // #if defined(__SSE__)


/***********************************************************************
  Benchmark driver defintions.
***********************************************************************/

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
Name     Description                     Set 0  Set 1 Set 2 Set 3 Units
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

  switch (what)
  {
  // parameters are number of: beams, doppler bins, CFAR range gates and 
  // CFAR guard cells respectively

  case 21: loop(t_cfar_by_vector_c<float>(16,  24,  5,  4)); break;
  case 22: loop(t_cfar_by_vector_c<float>(48, 128, 10,  8)); break;
  case 23: loop(t_cfar_by_vector_c<float>(48,  64, 10,  8)); break;
  case 24: loop(t_cfar_by_vector_c<float>(16,  16, 20, 16)); break;

#if defined(__SSE__)
  case 41: loop(t_cfar_by_vector_csimd<float>(16,  24,  5,  4)); break;
  case 42: loop(t_cfar_by_vector_csimd<float>(48, 128, 10,  8)); break;
  case 43: loop(t_cfar_by_vector_csimd<float>(48,  64, 10,  8)); break;
  case 44: loop(t_cfar_by_vector_csimd<float>(16,  16, 20, 16)); break;
#endif

  case  0:
    std::cout
      << "cfar_c -- Constant False Alarm Rate Detection (C version)\n"
      << "\n"
      << "                      CFAR  CFAR\n"
      << "              doppler range guard\n"
      << "        beams   bins  gates cells\n"
      << "        ----- ------- ----- -----\n"
      << "  -21:    16      24     5     4   \n"
      << "  -22:    48     128    10     8   \n"
      << "  -23:    48      64    10     8   \n"
      << "  -24:    16      16    20    16  \n"
      << "\n"
      << "  -41:    16      24     5     4   simd (SSE)\n"
      << "  -42:    48     128    10     8   simd (SSE)\n"
      << "  -43:    48      64    10     8   simd (SSE)\n"
      << "  -44:    16      16    20    16   simd (SSE)\n"
      ;
  default: 
    return 0;
  }
  return 1;
}
 
