//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#ifndef storage_hpp_
#define storage_hpp_

#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/tensor.hpp>
#include <vsip/dense.hpp>
#include <vsip/parallel.hpp>


/* -------------------------------------------------------------------- *
 * The Storage class templates define storage views that simplify
 * the creation of general testcases.
 *
 * The following class templates exist:
 *   Vector_storage	- standard vector views
 *   Matrix_storage	- standard matrix views
 *   Vector12Storage	- vector view of a 1,2-dimension block
 *   Sub_vector_storage	- vector subview of a vector
 *   Col_vector_storage	- column vector subview of a matrix
 *
 * The following class templates exist but rely on functionality
 * not required by the VSIPL++ specification.
 *   Vector12_subview_storage
 *			- vector view of a 1,2-dimension matrix
 *			  subview block.
 *
 * Example: a general test case to test addition of two views might
 * look like:
 *
 *    template <typename                  T,
 *	        template <typename> class ViewStorage>
 *    void
 *    test_add()
 *    {
 *       using namespace vsip;
 * 
 *       int const	N = 7;
 * 
 *       ViewStorage<T> ptr(N);
 * 
 *       scalar_f	alpha = 0.25;
 * 
 *       data.viewR = data.view1 + data.view2;
 * 
 *       insist(equal(get_origin(data.viewR),
 *                      Test_rig<T>::test_value1()
 *                    + Test_rig<T>::test_value2() ));
 *    }
 *
 * The following calls would test standard vector, standard matrix, and
 * matrix column sub-vector additions:
 *
 *    test_add<scalar_f, Vector_storage>();
 *    test_add<scalar_f, Matrix_storage>();
 *    test_add<scalar_f, Col_vector_storage>();
 * -------------------------------------------------------------------- */


template <vsip::dimension_type Dim,
	  typename             MapT>
struct Create_map {};

template <vsip::dimension_type Dim>
struct Create_map<Dim, vsip::Local_map>
{
  typedef vsip::Local_map type;
  static type exec() { return type(); }
};

template <vsip::dimension_type Dim>
struct Create_map<Dim, vsip::Replicated_map<Dim> >
{
  typedef vsip::Replicated_map<Dim> type;
  static type exec() { return type(); }
};

template <typename Dist0, typename Dist1, typename Dist2>
struct Create_map<1, vsip::Map<Dist0, Dist1, Dist2> >
{
  typedef vsip::Map<Dist0, Dist1, Dist2> type;
  static type exec() { return type(vsip::num_processors()); }
};

template <typename Dist0, typename Dist1, typename Dist2>
struct Create_map<2, vsip::Map<Dist0, Dist1, Dist2> >
{
  typedef vsip::Map<Dist0, Dist1, Dist2> type;

  static type exec()
  {
    using vsip::processor_type;

    processor_type np = vsip::num_processors();
    processor_type nr = (processor_type)floor(sqrt((double)np));
    processor_type nc = (processor_type)floor((double)np/nr);

    return type(nr, nc);
  }
};

template <vsip::dimension_type Dim,
	  typename             MapT>
MapT
create_map()
{
  return Create_map<Dim, MapT>::exec();
}




// -------------------------------------------------------------------- //
// Scalar_storage -- provide default vector storage.
template <typename T>
struct Scalar_storage {
   static vsip::dimension_type const
		dim = 0;
   
   typedef T	value_type;
   
   typedef T	view_type;

   view_type	view;

   Scalar_storage(int N)
    : view(T())
   {}

   Scalar_storage(int , T val)
    : view(val)
   {}
};

template <vsip::dimension_type Dim>
struct Default_order;

template <> struct Default_order<0> { typedef vsip::row1_type type; };
template <> struct Default_order<1> { typedef vsip::row1_type type; };
template <> struct Default_order<2> { typedef vsip::row2_type type; };
template <> struct Default_order<3> { typedef vsip::row3_type type; };



// Indicate that "default" complex format should be used.

// When using Storage, specifying Cmplx_default_fmt causes
// a Dense block to be used.  Specifying a specific format,
// CmplInterFmt or CmplxSplitFmt, will cause a Strided to
// be used with an explicit layout policy.

template <vsip::dimension_type Dim,
	  typename             T,
	  typename             Order      = typename Default_order<Dim>::type,
          typename             MapT       = vsip::Local_map,
	  vsip::storage_format_type C = ovxx::default_storage_format<T>::value>
class Storage;

template <vsip::dimension_type Dim,
	  typename             T,
	  typename             Order = typename Default_order<Dim>::type>
class Const_storage;

template <vsip::dimension_type D,
	  typename             T,
	  typename             O,
          typename             M,
	  vsip::storage_format_type F>
struct Storage_block
{
  typedef vsip::Layout<D, O, vsip::dense, F> L;
  typedef ovxx::Strided<D, T, L, M> strided_type;
  typedef vsip::Dense<D, T, O, M> dense_type;
  typedef typename ovxx::conditional<dense_type::storage_format == F,
				     dense_type, strided_type>::type type;
};



// -------------------------------------------------------------------- //
// Scalar_storage -- provide default scalar storage.
template <typename T,
	  typename Order,
	  typename MapT,
	  vsip::storage_format_type F>
class Storage<0, T, Order, MapT, F>
{
public:
   static vsip::dimension_type const
		dim = 0;
   
   typedef T	value_type;
   
   typedef T	view_type;

  // Constructors.
public:
  Storage() : view(T()) {}

  template <vsip::dimension_type Dim>
  Storage(vsip::Domain<Dim> const&)
    : view(T())
  {}

  template <vsip::dimension_type Dim>
  Storage(vsip::Domain<Dim> const&, T val)
    : view(val)
  {}

  // Public member data.
public:
  view_type	view;
};



template <typename T,
	  typename Order>
class Const_storage<0, T, Order> {
public:
   static vsip::dimension_type const
		dim = 0;
   
   typedef T	value_type;
   
   typedef T const	view_type;

  // Constructors.
public:
  Const_storage(int N)
   : view(T())
  {}

  Const_storage(int , T val)
    : view(val)
  {}

  // Public member data.
public:
  view_type	view;
};



// -------------------------------------------------------------------- //
// Storage<1, ...> -- provide default vector storage.

template <typename T,
	  typename Order,
	  typename MapT,
	  vsip::storage_format_type C>
class Storage<1, T, Order, MapT, C>
{
  // Compile-time values and typedefs.
public:
  static vsip::dimension_type const
		dim = 1;
   
  typedef T	                                            value_type;
  typedef MapT                                              map_type;
  typedef typename Storage_block<dim, T, Order, MapT, C>::type
                                                            block_type;
  typedef vsip::Vector<T, block_type>                       view_type;


  // Constructors.
public:
  Storage()
    : map(create_map<1, map_type>()), view(5, map)
  {}

  Storage(vsip::Domain<dim> const& dom)
    : map(create_map<1, map_type>()), view(dom.length(), map)
  {}

  Storage(vsip::Domain<dim> const& dom, T val)
    : map(create_map<1, map_type>()), view(dom.length(), val, map)
  {}


  // Accessor.
public:
  block_type& block()
   { return view.block(); }

  // Public member data.
public:
  map_type      map;
  view_type	view;
};



// -------------------------------------------------------------------- //
// Cnst_vector_storage -- provide default const_Vector storage.
template <typename T,
	  typename Order>
class Const_storage<1, T, Order>
{
  // Compile-time values and typedefs.
public:
  static vsip::dimension_type const
		dim = 1;
   
  typedef T	value_type;

  typedef vsip::Dense<dim, T, Order>
		block_type;

  typedef vsip::const_Vector<T, block_type>
		view_type;


  // Constructors and destructor.
public:
  Const_storage(vsip::Domain<dim> const& dom)
    : block_	(new block_type(dom))
    , view	(*block_)
    {}

  Const_storage(vsip::Domain<dim> const& dom, T val)
    : block_	(new block_type(dom, val))
    , view	(*block_)
    {}

  ~Const_storage()
    { block_->decrement_count(); }


  // Accessor.
public:
  block_type& block()
    { return *block_; }


  // Member data.
private:
   block_type*	block_;

public:
   view_type	view;
};



// -------------------------------------------------------------------- //
// Matrix_storage -- provide default vector storage.
template <typename T,
	  typename Order,
	  typename MapT,
	  vsip::storage_format_type C>
class Storage<2, T, Order, MapT, C>
{
  // Compile-time values and typedefs.
public:
  static vsip::dimension_type const
		dim = 2;

  typedef T	                                            value_type;
  typedef MapT                                              map_type;
  typedef typename Storage_block<dim, T, Order, MapT, C>::type
                                                            block_type;
  typedef vsip::Matrix<T, block_type>                       view_type;


  // Constructors.
public:
  Storage() : view(5, 7) {}

  Storage(vsip::Domain<dim> const& dom)
    : view(dom[0].length(), dom[1].length())
  {}

  Storage(vsip::Domain<dim> const& dom, T val)
    : view(dom[0].length(), dom[1].length(), val)
  {}

  // Accessor.
public:
  block_type& block()
    { return view.block(); }

  // Public member data.
public:
  view_type	view;

};



// -------------------------------------------------------------------- //
// Const_matrix_storage -- provide default const_Vector storage.
template <typename T,
	  typename Order>
class Const_storage<2, T, Order>
{
  // Compile-time values and typedefs.
public:
  static vsip::dimension_type const
		dim = 2;

  typedef T	value_type;

  typedef vsip::Dense<dim, T, Order>
		block_type;

  typedef vsip::const_Matrix<T, block_type>
		view_type;


  // Constructors and destructor.
public:
  Const_storage(vsip::Domain<dim> const& dom)
    : block_	(new block_type(dom))
    , view	(*block_)
  {}

  Const_storage(vsip::Domain<dim> const& dom, T val)
    : block_	(new block_type(dom, val))
    , view	(*block_)
  {}

  ~Const_storage()
    { block_->decrement_count(); }


  // Accessor.
public:
  block_type& block()
    { return *block_; }


  // Member data.
private:
  block_type*	block_;

public:
  view_type	view;
};



/// Storage specialization for Tensors.

template <typename T,
	  typename Order,
	  typename MapT,
	  vsip::storage_format_type C>
class Storage<3, T, Order, MapT, C>
{
  // Compile-time values and typedefs.
public:
  static vsip::dimension_type const dim = 3;

  typedef T	                                            value_type;
  typedef MapT                                              map_type;
  typedef typename Storage_block<dim, T, Order, MapT, C>::type
                                                            block_type;
  typedef vsip::Tensor<T, block_type>                       view_type;

  // Constructors.
public:
  Storage() : view(5, 7, 3) {}

  Storage(vsip::Domain<dim> const& dom)
    : view(dom[0].length(), dom[1].length(), dom[2].length())
  {}

  Storage(vsip::Domain<dim> const& dom, T val)
    : view(dom[0].length(), dom[1].length(), dom[2].length(), val)
  {}

  // Accessor.
public:
  block_type& block()
    { return view.block(); }

  // Public member data.
public:
  view_type	view;

};



// -------------------------------------------------------------------- //
// Const_storage -- provide default const_Tensor storage.
template <typename T,
	  typename Order>
class Const_storage<3, T, Order>
{
  // Compile-time values and typedefs.
public:
  static vsip::dimension_type const dim = 3;

  typedef T	                            value_type;
  typedef vsip::Dense<dim, T, Order>        block_type;
  typedef vsip::const_Tensor<T, block_type> view_type;

  // Constructors and destructor.
public:
  Const_storage(vsip::Domain<dim> const& dom)
    : block_	(new block_type(dom))
    , view	(*block_)
  {}

  Const_storage(vsip::Domain<dim> const& dom, T val)
    : block_	(new block_type(dom, val))
    , view	(*block_)
  {}

  ~Const_storage()
    { block_->decrement_count(); }


  // Accessor.
public:
  block_type& block()
    { return *block_; }


  // Member data.
private:
  block_type*	block_;

public:
  view_type	view;
};



// -------------------------------------------------------------------- //
// Additional storage types.

template <typename T,
	  typename Order>
class Row_vector
{
  // Compile-time values and typedefs.
public:
  static vsip::dimension_type const
		dim = 1;
   
  typedef T value_type;

  typedef vsip::Matrix<T, vsip::Dense<2, T, Order> > parent_type;
  typedef typename parent_type::row_type             view_type;
  typedef typename view_type::block_type             block_type;

  // Constructors.
public:
  // 5 rows and 1st row are arbitrary
  Row_vector(vsip::Domain<dim> const& dom)
    : parent_view(5, dom.length()),
      view       (parent_view.row(1))
  {}

  Row_vector(vsip::Domain<dim> const& dom, T val)
    : parent_view(5, dom.length(), val),
      view       (parent_view.row(1))
  {}

  // Accessor.
public:
  block_type& block()
  { return view.block(); }

  // Member data.
private:
  parent_type parent_view;
public:
  view_type   view;
};



template <typename T,
	  typename Order>
class Diag_vector
{
  // Compile-time values and typedefs.
public:
  static vsip::dimension_type const
		dim = 1;
   
  typedef T value_type;

  typedef vsip::Matrix<T, vsip::Dense<2, T, Order> > parent_type;
  typedef typename parent_type::diag_type            view_type;
  typedef typename view_type::block_type             block_type;

  // Constructors.
public:
  Diag_vector(vsip::Domain<dim> const& dom)
    : parent_view(dom.length(), dom.length()),
      view       (parent_view.diag(0))
  {}

  Diag_vector(vsip::Domain<dim> const& dom, T val)
    : parent_view(dom.length(), dom.length(), val),
      view       (parent_view.diag(0))
  {}

  // Accessor.
public:
  block_type& block()
  { return view.block(); }

  // Member data.
private:
  parent_type parent_view;
public:
  view_type   view;
};



template <typename T,
	  typename Order>
class Transpose_matrix
{
  // Compile-time values and typedefs.
public:
  static vsip::dimension_type const
		dim = 2;

  typedef T	value_type;

  typedef vsip::Matrix<T, vsip::Dense<2, T, Order> > parent_type;
  typedef typename parent_type::transpose_type       view_type;
  typedef typename view_type::block_type             block_type;

  // Constructors.
public:
  Transpose_matrix(vsip::Domain<dim> const& dom)
    : parent_view(dom[1].length(), dom[0].length()),
      view       (parent_view.transpose())
  {}

  Transpose_matrix(vsip::Domain<dim> const& dom, T val)
    : parent_view(dom[1].length(), dom[0].length(), val),
      view       (parent_view.transpose())
  {}

  // Accessor.
public:
  block_type& block()
    { return view.block(); }

  // Member data.
private:
  parent_type	parent_view;
public:
  view_type	view;
};



// get_size -- get size of a view.

template <typename T>
inline vsip::length_type
get_size(T const&)
{
  return 1;
}

template <typename T,
	  typename Block>
inline vsip::length_type
get_size(
   vsip::Vector<T, Block> view)
{
   return view.size();
}

template <typename T,
	  typename Block>
inline vsip::length_type
get_size(
   vsip::Matrix<T, Block> view)
{
  return view.size();
}

template <typename T,
	  typename Block>
inline vsip::length_type
get_size(
   vsip::Tensor<T, Block> view)
{
  return view.size();
}


// -------------------------------------------------------------------- //
// get_nth -- get n-th element of a scalar.
template <typename T>
inline T
get_nth(
  T&  view,
  vsip::index_type)
{
  return view;
}



// -------------------------------------------------------------------- //
// get_nth -- get n-th element of a vector.
template <typename T,
	  typename Block>
inline T
get_nth(
   vsip::Vector<T, Block> view,
   vsip::index_type	  n)
{
   return view.get(n);
}



// -------------------------------------------------------------------- //
// get_nth -- get n-th element of a matrix.
template <typename T,
	  typename Block>
inline T
get_nth(
   vsip::Matrix<T, Block> view,
   vsip::index_type	  n)
{
   return view.get(n/view.size(1), n%view.size(1));
}



// Get the n'th element of a tensor.

template <typename T,
	  typename Block>
inline T
get_nth(
   vsip::Tensor<T, Block> view,
   vsip::index_type	  n)
{
#ifndef NDEBUG
  vsip::index_type orig = n;
#endif
  vsip::index_type k = n % view.size(2); n = (n-k) / view.size(2);
  vsip::index_type j = n % view.size(1); n = (n-j) / view.size(1);
  vsip::index_type i = n;

  assert((i * view.size(1) + j) * view.size(2) + k == orig);

  return view.get(i, j, k);
}



// -------------------------------------------------------------------- //
// put_nth -- put n-th element of a scalar.
template <typename T>
inline void
put_nth(
  T&               view,
  vsip::index_type /*n*/,
  T const&         value)
{
  view = value;
}



// -------------------------------------------------------------------- //
// put_nth -- put n-th element of a vector.
template <typename T,
	  typename Block>
inline void
put_nth(
  vsip::Vector<T, Block> view,
  vsip::index_type       n,
  T const&		 value)
{
   view.put(n, value);
}



// -------------------------------------------------------------------- //
// put_nth -- put n-th element of a matrix.
template <typename T,
	  typename Block>
inline void
put_nth(
  vsip::Matrix<T, Block> view,
  vsip::index_type       n,
  T const&		 value)
{
   view.put(n/view.size(1), n%view.size(1), value);
}



// put_nth -- put n-th element of a tensor.

template <typename T,
	  typename Block>
inline void
put_nth(
  vsip::Tensor<T, Block> view,
  vsip::index_type       n,
  T const&		 value)
{
#ifndef NDEBUG
  vsip::index_type orig = n;
#endif
  vsip::index_type k = n % view.size(2); n = (n-k) / view.size(2);
  vsip::index_type j = n % view.size(1); n = (n-j) / view.size(1);
  vsip::index_type i = n;

  assert((i * view.size(1) + j) * view.size(2) + k == orig);

  view.put(i, j, k, value);
}



// -------------------------------------------------------------------- //
// put_nth -- put nth element 0,0 of a Dense<2,T> block.
template <typename T,
	  typename Order,
	  typename Map>
inline void
put_nth(
  vsip::Dense<2, T, Order, Map>& block,
  vsip::index_type               n,
  T const&		         value)
{
   block->put(n/block->size(2, 1), n%block->size(2, 1), value);
}


template <typename T,
	  typename Block>
inline vsip::index_type
nth_from_index(
   vsip::const_Vector<T, Block>,
   vsip::Index<1> const&        idx)
{
  return idx[0];
}



template <typename T,
	  typename Block>
inline vsip::index_type
nth_from_index(
   vsip::const_Matrix<T, Block> view,
   vsip::Index<2> const&        idx)
{
  return idx[0] * view.size(1) + idx[1];
}



template <typename T,
	  typename Block>
inline vsip::index_type
nth_from_index(
   vsip::const_Tensor<T, Block> view,
   vsip::Index<3> const&        idx)
{
  return (idx[0] * view.size(1) + idx[1]) * view.size(2) + idx[2];
}


template <typename T>
struct is_scalar
{
  static bool const value = true;
};

template <typename T,
	  typename Block>
struct is_scalar<vsip::Vector<T, Block> >
{
  static bool const value = false;
};

template <typename T,
	  typename Block>
struct is_scalar<vsip::Matrix<T, Block> >
{
  static bool const value = false;
};

template <typename T,
	  typename Block>
struct is_scalar<vsip::Tensor<T, Block> >
{
  static bool const value = false;
};



template <typename T>
struct Value_type_of
{
  typedef T type;
};

template <typename T,
	  typename Block>
struct Value_type_of<vsip::Vector<T, Block> >
{
  typedef T type;
};

template <typename T,
	  typename Block>
struct Value_type_of<vsip::Matrix<T, Block> >
{
  typedef T type;
};

template <typename T,
	  typename Block>
struct Value_type_of<vsip::Tensor<T, Block> >
{
  typedef T type;
};

#endif // VSIP_TESTS_TEST_STORAGE_HPP
