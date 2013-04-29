/* Copyright (c) 2005, 2006 by CodeSourcery, LLC.  All rights reserved. */

/** @file    support.hpp
    @author  Jules Bergmann
    @date    2005-01-18
    @brief   VSIPL++ Library: [support] basic macros, types, exceptions,
             and support functions.

    This file declares the basic macros, types, exceptions, and support
    functions used throughout the VSIPL++ library as specified in the
    [support] section of the VSIPL++ specification.
*/

#ifndef VSIP_SUPPORT_HPP
#define VSIP_SUPPORT_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/core/config.hpp>
#include <vsip/core/complex_decl.hpp>
#include <cassert>
#include <stdexcept>
#include <complex>
#include <string>
#include <cstdlib>



/***********************************************************************
  Macros
***********************************************************************/

/// The maximum number of dimensions supported by this implementation.
#define VSIP_MAX_DIMENSION 3

/// If the compiler provides a way to annotate functions with an
/// indication that they never return (except possibly by throwing an
/// exception), then VSIP_NORETURN is defined to that annotation,
/// otherwise it is defined to nothing.
/// GCC supports it, as does Green Hills, as well as Intel.
/// The latter defines __GNUC__, too.
#if __GNUC__ >= 2 || defined(__ghs__)
#  define VSIP_IMPL_NORETURN __attribute__ ((__noreturn__))
#else
#  define VSIP_IMPL_NORETURN
#endif



/// If the compiler provides a way to annotate functions with an
/// indication that they should not be inlined, then VSIP_IMPL_NOINLINE
/// is defined to that annotation, otherwise it is defined to
/// nothing. 
///
/// Greenhills does not use this attribute.
#if __GNUC__ >= 2
#  define VSIP_IMPL_NOINLINE __attribute__ ((__noinline__))
#else
#  define VSIP_IMPL_NOINLINE
#endif



/// If the compiler provides a way to annotate functions that every
/// call inside the function should be inlined, then VSIP_IMPL_FLATTEN
/// is defined to that annotation, otherwise it is defined to nothing.
#if __GNUC__ >= 4
#  define VSIP_IMPL_FLATTEN __attribute__ ((__flatten__))
#else
#  define VSIP_IMPL_FLATTEN
#endif



/// Loop vectorization pragmas.
#if __INTEL_COMPILER && !__ICL
#  define PRAGMA_IVDEP _Pragma("ivdep")
#  define PRAGMA_VECTOR_ALWAYS _Pragma("vector always")
#else
#  define PRAGMA_IVDEP
#  define PRAGMA_VECTOR_ALWAYS
#endif


/// Controls whether this VSIPL++ implementation throws exceptions.

/// This implementation supports exceptions, but their use can be
/// disabled by setting this flag to zero.  The library must then
/// be recompiled.

/// When set to non-zero, the implementation will throw exceptions as
/// required by the specification.  When set to zero, exceptions will
/// not be thrown.  Instead, whenever the implementation would have
/// thrown an exception, it will instead print a diagnostic to cerr
/// and call abort().
/// FUTURE: Mode in which exceptional conditions are not detected at all.

#ifndef VSIP_HAS_EXCEPTIONS
// If the Intel compiler on windows is used without exception handling (-GX)
#  if defined(__ICL) && !__EXCEPTIONS
#    define VSIP_HAS_EXCEPTIONS 0
#  else
#    define VSIP_HAS_EXCEPTIONS 1
#  endif
#endif

#if VSIP_HAS_EXCEPTIONS
#  define VSIP_NOTHROW throw()
#  define VSIP_THROW(x) throw x         ///< Wraps throw-specifications
#  define VSIP_IMPL_THROW(x) throw x    ///< Wraps throw statements
#else
#  define VSIP_NOTHROW
#  define VSIP_THROW(x)
#  define VSIP_IMPL_THROW(x) vsip::impl::fatal_exception(__FILE__, __LINE__, x)
#endif

/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{

/// Value Types [support.types]

typedef float scalar_f;
typedef int scalar_i;
typedef complex<scalar_f> cscalar_f;
typedef complex<scalar_i> cscalar_i;

// Note: complex.hpp imports std::complex into the vsip namespace.


/// Default value type for Vectors, Matrixes, and Tensors.
typedef scalar_f VSIP_DEFAULT_VALUE_TYPE;

typedef unsigned int dimension_type;

enum whole_domain_type { whole_domain};


/// Domain types [support.types.domain]

namespace impl 
{

typedef std::ptrdiff_t signed_index_type;
typedef std::size_t unsigned_index_type;

} // namespace vsip::impl

typedef impl::unsigned_index_type index_type;
typedef impl::signed_index_type index_difference_type;
typedef impl::signed_index_type stride_type;
typedef impl::unsigned_index_type stride_scalar_type;

typedef index_type	length_type;


/// Dimension Ordering [support.types.dimorder]

/// Class to define dimension order for Dense blocks.
///
/// The template parameters to tuple specify the order in which
/// logical dimensions are used to address memory, starting from
/// major-dimension to minor-dimension.  For a contiguous data-layout,
/// elements in the minor-dimension are unit-stride.
///
/// Logical dimensions are numbered from left-to-right, starting with zero.
///
/// For a tensor 'ten' element reference 'ten(x, y, z)',
/// 'x' is the 0th dimension, 'y' is the 1st dimension, and 'z' is the
/// 2nd dimension.
///
/// For a Matrix 'mat' element reference 'mat(r, c)',
/// 'r' is the 0th dimension (also the row) and 'c' is the 1st dimension
/// (also the column).
template <dimension_type dim0 = 0,
	  dimension_type dim1 = 1,
	  dimension_type dim2 = 2>
class tuple
{
public:
  static dimension_type const impl_dim0 = dim0;
  static dimension_type const impl_dim1 = dim1;
  static dimension_type const impl_dim2 = dim2;
};



// row-major dimension orders.
typedef tuple<0, 1, 2>	row1_type;  ///< Convenience type for 1-dim row-major.
typedef tuple<0, 1, 2>	row2_type;  ///< Convenience type for 2-dim row-major.
typedef tuple<0, 1, 2>	row3_type;  ///< Convenience type for 3-dim row-major.

// column-major dimension orders.
typedef tuple<0, 1, 2>	col1_type;  ///< Convenience type for 1-dim col-major.
typedef tuple<1, 0, 2>	col2_type;  ///< Convenience type for 2-dim col-major.
typedef tuple<2, 1, 0>	col3_type;  ///< Convenience type for 3-dim col-major.



// Define convenience template classes Row_major and Col_major.  They
// are used by Dense template class to choose the default dimension
// order for a choosen dimension.
namespace impl
{

template <dimension_type Dim>
struct Row_major;

template <> struct Row_major<1> { typedef row1_type type; };
template <> struct Row_major<2> { typedef row2_type type; };
template <> struct Row_major<3> { typedef row3_type type; };


template <dimension_type Dim>
struct Col_major;

template <> struct Col_major<1> { typedef col1_type type; };
template <> struct Col_major<2> { typedef col2_type type; };
template <> struct Col_major<3> { typedef col3_type type; };


} // namespace vsip::impl

/// Map and multiprocessor types [support.types.map].

template <dimension_type Dim>
class Replicated_map;

class Local_map;

#if VSIP_IMPL_PAR_SERVICE == 2
typedef long processor_type;
typedef long processor_difference_type;
#else
typedef unsigned int processor_type;
typedef signed int processor_difference_type;
#endif

index_type     const no_index     = static_cast<index_type>(-1);
index_type     const no_subblock  = static_cast<index_type>(-1);
index_type     const no_rank      = static_cast<index_type>(-1);
processor_type const no_processor = static_cast<processor_type>(-1);



/// Enumeration to indicate parallel data distribution.
enum distribution_type
{
  whole,
  block,
  cyclic,
  other
};

/// Enumeration to indicate how function object returns result.
enum return_mechanism_type
{
  by_value,
  by_reference
};



/// Exceptions [support.exceptions.comput].

/// Class for VSIPL++ computation error exceptions.
class computation_error : public std::runtime_error
{
public:
  /// Constructor.
  explicit	computation_error(const std::string& wstr)
    : runtime_error(wstr)
  {}
   
};

namespace impl
{
/// This function is called instead of throwing an exception
/// when VSIP_HAS_EXCEPTIONS is 0.
extern void fatal_exception(char const * file, unsigned int line,
                            std::exception const& E) VSIP_IMPL_NORETURN;


/// Exception to mark unimplemented VSIPL++ features and functionality.

/// During development, unimplemented functions will have stubs that
/// throw this exception.
class unimplemented : public std::runtime_error
{
public:
  /// Constructor.
  explicit unimplemented(const std::string& str)
    : runtime_error(str)
  {}
   
};

} // namespace vsip::impl

/// Synonyms for dimension_type [support.constants].

dimension_type const row = 0;		///< Row dimension of a Matrix.
dimension_type const col = 1;		///< Column dimension of a Matrix.

dimension_type const dim0 = 0;		///< First dimension of a Tensor.
dimension_type const dim1 = 1;		///< Second dimension of a Tensor.
dimension_type const dim2 = 2;		///< Third dimension of a Tensor.

/// Support functions [support.functions].

/// Return the total number of processors executing the program.
length_type num_processors() VSIP_NOTHROW;
// processor_set() defined in parallel.hpp
processor_type local_processor() VSIP_NOTHROW;

} // namespace vsip

namespace vsip_csl
{
// import all vsip symbols into vsip_csl
using namespace vsip;
// Instead of aliasing the vsip::impl namespace,
// we create a new one, so we can add to it directly.
namespace impl
{
using namespace vsip::impl;
}

}

#endif
