/***********************************************************************

  File:   test-util.hpp
  Author: Jules Bergmann, CodeSourcery, LLC.
  Date:   11/12/2004

  Contents: Utilities for VSIPL++ tests.

Copyright 2005 Georgia Tech Research Corporation, all rights reserved.

A non-exclusive, non-royalty bearing license is hereby granted to all
Persons to copy, distribute and produce derivative works for any
purpose, provided that this copyright notice and following disclaimer
appear on All copies: THIS LICENSE INCLUDES NO WARRANTIES, EXPRESSED
OR IMPLIED, WHETHER ORAL OR WRITTEN, WITH RESPECT TO THE SOFTWARE OR
OTHER MATERIAL INCLUDING, BUT NOT LIMITED TO, ANY IMPLIED WARRANTIES
OF MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE, OR ARISING
FROM A COURSE OF PERFORMANCE OR DEALING, OR FROM USAGE OR TRADE, OR OF
NON-INFRINGEMENT OF ANY PATENTS OF THIRD PARTIES. THE INFORMATION IN
THIS DOCUMENT SHOULD NOT BE CONSTRUED AS A COMMITMENT OF DEVELOPMENT
BY ANY OF THE ABOVE PARTIES.

The US Government has a license under these copyrights, and this
Material may be reproduced by or for the US Government.

***********************************************************************/

#ifndef _VSIPLPP_TEST_UTIL_H_
#define _VSIPLPP_TEST_UTIL_H_

/***********************************************************************
  Class Declarations
***********************************************************************/

template <typename T>
struct TestRig;

template <typename T>
class DefaultValueGenerator;

template <typename T,
	  typename ValueGen = DefaultValueGenerator<T> >
class TestSeqIterator;



/***********************************************************************
  Class Definitions
***********************************************************************/

/* -------------------------------------------------------------------- *
 * TestRig is a traits class that simplifies writing tests that work
 * for different data types.
 *
 * Currently, it has value traits for test values.
 * -------------------------------------------------------------------- */


template <>
struct TestRig<vsip::scalar_f> {
   typedef vsip::scalar_f T;
   typedef TestSeqIterator<T> seq_iter_t;

   static T test_value1() { return  0.1f; }
   static T test_value2() { return -2.1f; }
   static T test_value3() { return  4.0f; }

   static seq_iter_t test_seq1(); // { return seq_iter_t(); }
};

template <>
struct TestRig<vsip::scalar_i> {
   typedef vsip::scalar_i T;
   static T test_value1() { return 42; }
   static T test_value2() { return 16; }
   static T test_value3() { return -5; }
};

template <>
struct TestRig<vsip::cscalar_f> {
   typedef vsip::cscalar_f T;
   static T test_value1() { return std::complex<float>(1.5f, 2.5f); }
   static T test_value2() { return std::complex<float>(-1.f, 1.0f); }
   static T test_value3() { return std::complex<float>( 2.f,-3.1f); }
};

template <>
struct TestRig<bool> {
   typedef bool T;
   static T test_value1() { return true; }
   static T test_value2() { return false; }
   static T test_value3() { return true; }
};

template <>
struct TestRig<vsip::index_type> {
   typedef vsip::index_type T;
   static T test_value1() { return 5; }
   static T test_value2() { return 7; }
   static T test_value3() { return 3; }
};



/* -------------------------------------------------------------------- *
 * The Storage class templates define storage views that simplify
 * the creation of general testcases.
 *
 * The following class templates exist:
 *   VectorStorage	- standard vector views
 *   MatrixStorage	- standard matrix views
 *   Vector12Storage	- vector view of a 1,2-dimension block
 *   SubVectorStorage	- vector subview of a vector
 *   ColVectorStorage	- column vector subview of a matrix
 *
 * The following class templates exist but rely on functionality
 * not required by the VSIPL++ specification.
 *   Vector12SubviewStorage
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
 *                      TestRig<T>::test_value1()
 *                    + TestRig<T>::test_value2() ));
 *    }
 *
 * The following calls would test standard vector, standard matrix, and
 * matrix column sub-vector additions:
 *
 *    test_add<scalar_f, VectorStorage>();
 *    test_add<scalar_f, MatrixStorage>();
 *    test_add<scalar_f, ColVectorStorage>();
 * -------------------------------------------------------------------- */



// -------------------------------------------------------------------- //
// ScalarStorage -- provide default vector storage.
template <typename T>
struct ScalarStorage {
   static vsip::dimension_type const
		dim = 0;
   
   typedef T	value_type;
   
   typedef T	view_type;

   view_type	view;

   ScalarStorage(int N)
    : view(T())
   {}

   ScalarStorage(int , T val)
    : view(val)
   {}
};



// -------------------------------------------------------------------- //
// VectorStorage -- provide default vector storage.
template <typename T>
struct VectorStorage {
   static vsip::dimension_type const
		dim = 1;
   
   typedef T	value_type;
   
   typedef vsip::Vector<T>
		view_type;

   typedef typename view_type::block_type
		block_type;

   view_type	view;

   VectorStorage(int N)
    : view(N)
   {}

   VectorStorage(int N, T val)
    : view(N, val)
   {}

   block_type& block()
   { return view.block(); }
};



// -------------------------------------------------------------------- //
// CnstVectorStorage -- provide default const_Vector storage.
template <typename T>
struct ConstVectorStorage {
   static vsip::dimension_type const
		dim = 1;
   
   typedef T	value_type;

   typedef vsip::const_Vector<T>
		view_type;

   typedef typename view_type::block_type
		block_type;


   ConstVectorStorage(int N)
      : block_	(new block_type(N))
      , view	(*block_)
   {}

   ConstVectorStorage(int N, T val)
      : block_	(new block_type(N, val))
      , view	(*block_)
   {}

   ~ConstVectorStorage()
   { block_->decrement_count(); }

   block_type& block()
   { return *block_; }

   // Member data.
   block_type*	block_;
   view_type	view;
};



// -------------------------------------------------------------------- //
// MatrixStorage -- provide default vector storage.
template <typename T>
struct MatrixStorage {
   static vsip::dimension_type const
		dim = 2;

   typedef T	value_type;

   typedef vsip::Matrix<T>
		view_type;

   typedef typename view_type::block_type
		block_type;

   view_type	view;

   MatrixStorage(int N)
    : view(N, N)
   {}

   MatrixStorage(int N, T val)
    : view(N, N, val)
   {}

   block_type& block()
   { return view.block(); }
};



// -------------------------------------------------------------------- //
// ConstMatrixStorage -- provide default const_Vector storage.
template <typename T>
struct ConstMatrixStorage {
   static vsip::dimension_type const
		dim = 2;

   typedef T	value_type;

   typedef vsip::const_Matrix<T>
		view_type;

   typedef typename view_type::block_type
		block_type;


   ConstMatrixStorage(int N)
      : block_	(new block_type(vsip::Domain<2>(N,N)))
      , view	(*block_)
   {}

   ConstMatrixStorage(int N, T val)
      : block_	(new block_type(vsip::Domain<2>(N,N), val))
      , view	(*block_)
   {}

   ~ConstMatrixStorage()
   { block_->decrement_count(); }

   block_type& block()
   { return *block_; }

   // Member data.
   block_type*	block_;
   view_type	view;
};



/* -------------------------------------------------------------------- *
 * The MStorage class templates define storage views that simplify
 * the creation of general testcases.
 */


// -------------------------------------------------------------------- //
// MVectorStorage -- provide default vector views.
template <typename T>
struct MVectorStorage {
   typedef vsip::Vector<T>
		view_type;

   view_type	viewR;
   view_type	view1;
   view_type	view2;
   view_type	view3;

   MVectorStorage(int N)
    : viewR(N),
      view1(N, TestRig<T>::test_value1()),
      view2(N, TestRig<T>::test_value2()),
      view3(N, TestRig<T>::test_value3())
   {}
};



// -------------------------------------------------------------------- //
// Vector12Storage -- provide vector views of 1,2-dimensional block.
template <typename T>
struct MVector12Storage {
   typedef vsip::Matrix<T>
		matrix_type;

   typedef typename matrix_type::block_type
		block_type;

   typedef vsip::Vector<T, block_type>
		view_type;

   matrix_type	matR;
   matrix_type	mat1;
   matrix_type	mat2;
   matrix_type	mat3;

   view_type	viewR;
   view_type	view1;
   view_type	view2;
   view_type	view3;

   MVector12Storage(int N)
    : matR(N, 1),
      mat1(N, 1),
      mat2(N, 1),
      mat3(N, 1),
      viewR(matR.block()),
      view1(mat1.block()),
      view2(mat2.block()),
      view3(mat3.block())
   {
      view1 = TestRig<T>::test_value1();
      view2 = TestRig<T>::test_value2();
      view3 = TestRig<T>::test_value3();
   }
};



// -------------------------------------------------------------------- //
// Vector12altStorage -- provides same views as Vector12Storage
template <typename T>
struct MVector12altStorage {
   typedef vsip::Dense<2, T>
		block_type;

   typedef vsip::Vector<T, block_type>
		view_type;

   block_type*	blockR;
   block_type*	block1;
   block_type*	block2;
   block_type*	block3;

   view_type	viewR;
   view_type	view1;
   view_type	view2;
   view_type	view3;

   MVector12altStorage(int N)
    : blockR(new block_type(vsip::Domain<2>(N, 1))),
      block1(new block_type(vsip::Domain<2>(N, 1))),
      block2(new block_type(vsip::Domain<2>(N, 1))),
      block3(new block_type(vsip::Domain<2>(N, 1))),
      viewR(*blockR),
      view1(*block1),
      view2(*block2),
      view3(*block3)
   {
      view1 = TestRig<T>::test_value1();
      view2 = TestRig<T>::test_value2();
      view3 = TestRig<T>::test_value3();
   }

   ~MVector12altStorage()
   {
      blockR->decrement_count();
      block1->decrement_count();
      block2->decrement_count();
      block3->decrement_count();
   }
};



// -------------------------------------------------------------------- //
// Vector12SubviewStorage -- provides same vector views of
//                           implementation subblock used for matrix
//                           subviews.
//
// Note: The specification does require that such blocks support
//       1-dimensional access.  the reference implementation does
//       not implement 1-dimensional access.
template <typename T>
struct MVector12SubviewStorage {
   typedef vsip::Matrix<T>
		matrix_type;

   typedef typename matrix_type::subview_type::block_type
		block_type;

   typedef vsip::Vector<T, block_type>
		view_type;

   matrix_type	mat;

   view_type	viewR;
   view_type	view1;
   view_type	view2;
   view_type	view3;

   MVector12SubviewStorage(int N)
    : mat(N, 4),
      viewR(mat(vsip::Domain<2>(vsip::Domain<1>(0,1,N), 0)).block()),
      view1(mat(vsip::Domain<2>(vsip::Domain<1>(0,1,N), 1)).block()),
      view2(mat(vsip::Domain<2>(vsip::Domain<1>(0,1,N), 2)).block()),
      view3(mat(vsip::Domain<2>(vsip::Domain<1>(0,1,N), 3)).block())
   {
      view1 = TestRig<T>::test_value1();
      view2 = TestRig<T>::test_value2();
      view3 = TestRig<T>::test_value3();
   }
};



// -------------------------------------------------------------------- //
// SubVectorStorage -- provide vector subviews of a vector
template <typename T>
struct MSubVectorStorage {
   typedef vsip::Vector<T>
		parent_t;

   typedef typename parent_t::subview_type
		view_type;

   parent_t	parent_view;

   view_type	viewR;
   view_type	view1;
   view_type	view2;
   view_type	view3;

   MSubVectorStorage(int N)
    : parent_view(4*N),
      viewR(parent_view(vsip::Domain<1>(  0,1,N))),
      view1(parent_view(vsip::Domain<1>(  N,1,N))),
      view2(parent_view(vsip::Domain<1>(2*N,1,N))),
      view3(parent_view(vsip::Domain<1>(3*N,1,N)))
   {
      view1 = TestRig<T>::test_value1();
      view2 = TestRig<T>::test_value2();
      view3 = TestRig<T>::test_value3();
   }
};



// -------------------------------------------------------------------- //
// SubVectorStorage -- provide column vector subviews of a matrix
template <typename T>
struct MColVectorStorage {
   typedef vsip::Matrix<T>
		parent_t;

   typedef typename parent_t::col_type
		view_type;

   parent_t	parent_view;

   view_type	viewR;
   view_type	view1;
   view_type	view2;
   view_type	view3;

   MColVectorStorage(int N)
    : parent_view(N, 4),
      viewR(parent_view.col(0)),
      view1(parent_view.col(1)),
      view2(parent_view.col(2)),
      view3(parent_view.col(3))
   {
      view1 = TestRig<T>::test_value1();
      view2 = TestRig<T>::test_value2();
      view3 = TestRig<T>::test_value3();
   }
};



// -------------------------------------------------------------------- //
// MatrixStorage -- provide default matrix views.
template <typename T>
struct MMatrixStorage {
   typedef vsip::Matrix<T>
		view_type;

   view_type	viewR;
   view_type	view1;
   view_type	view2;
   view_type	view3;

   MMatrixStorage(int N)
      : viewR(N, N+3),
	view1(N, N+3, TestRig<T>::test_value1()),
	view2(N, N+3, TestRig<T>::test_value2()),
	view3(N, N+3, TestRig<T>::test_value3())
      {}
};


template <typename T>
class DefaultValueGenerator {
public:
   T operator()(int i) const { return (T)(i); }
};



template <>
class DefaultValueGenerator<bool> {
public:
   bool operator()(int i) const
      { return (bool)(((i & 0x1) & ((i >> 2) & 0x1)) ^ ((i >> 4) & 0x1)); }
};


// -------------------------------------------------------------------- //
// TestSeqIterator

template <typename T,
	  typename ValueGen>
class TestSeqIterator {

// typedefs
public:
   typedef T	value_type;


// constructors
public:
   TestSeqIterator()
      : index_	  (0),
	value_gen_()
   {}

   TestSeqIterator(TestSeqIterator const& rhs)
      : index_	  (rhs.index_),
	value_gen_(value_gen_)
   {}

// self-assignment
public:
   TestSeqIterator& operator=(TestSeqIterator const& rhs)
   {
      index_     = rhs.index_;
      value_gen_ = rhs.value_gen_;
      return *this;
   }

// operators
   void operator++()       { ++index_; }
   void operator++(int)    { ++index_; }
   void operator--()       { --index_; }
   void operator--(int)    { --index_; }
   void operator+=(int dx) { index_ += dx; }
   void operator-=(int dx) { index_ -= dx; }

   bool operator==(TestSeqIterator const& rhs) const
   { return index_ == rhs.index_; }

   bool operator!=(TestSeqIterator const& rhs) const
   { return index_ != rhs.index_; }

   bool operator<(TestSeqIterator const& rhs) const
   { return index_ < rhs.index_; }

   int operator-(TestSeqIterator const& rhs) const
   { return index_ - rhs.index_; }

   TestSeqIterator operator+(int dx) const
   {
      TestSeqIterator res(index_);
      res += dx;
      return res;
   }

   TestSeqIterator operator-(int dx) const
   {
      TestSeqIterator res(index_);
      res -= dx;
      return res;
   }

//   T operator()() const
//   { return current_; }
//
//   T operator()(int d) const
//   { return current_ + d; }

   value_type operator*() const
      { return value_gen_(index_); }
        
private:
   int		index_;
   ValueGen	value_gen_;

};




// -------------------------------------------------------------------- //
template <typename View>
class FakeRef {

// typedefs
public:

   typedef typename View::value_type
		value_type;

// constructor
public:
   FakeRef(
      View	view,
      int	index
      ) :
      view_	(view),
      index_	(index)
   {}


// conversions
public:
   operator value_type() const { return get_nth(view_, index_); }


// Assignment operators.
public:
   FakeRef& operator=(value_type value)
   {
      put_nth(view_, index_, value);
      return *this;
   }

   FakeRef& operator=(FakeRef const& rhs)
   {
      put_nth(view_, index_, rhs.get_value());
      return *this;
   }
   

public:
   value_type get_value() const { return get_nth(view_, index_); }

private:
   View		view_;
   int		index_;
};



template <typename View>
bool operator==(
   FakeRef<View> const& lhs,
   typename FakeRef<View>::value_type rhs)
{
   return lhs.get_value() == rhs;
}



template <typename View>
bool operator==(
   typename FakeRef<View>::value_type lhs,
   FakeRef<View> const&		   rhs)
{
   return lhs == rhs.get_value();
}


// -------------------------------------------------------------------- //
// ViewIterator

template <typename View>
class ViewIterator {

// typedefs
public:
   typedef typename View::value_type
		value_type;

// constructors
public:
   ViewIterator(
      View	view,
      int	offset=0
      ) :
      view_	(view),
      index_	(offset)
   {}

   ViewIterator(ViewIterator const& rhs)
      : index_	(rhs.index_),
	view_	(rhs.view_)
   {}

// self-assignment
public:
   ViewIterator& operator=(ViewIterator const& rhs)
   {
      view_	= rhs.view_;
      index_	= rhs.index_;
      return *this;
   }

// operators
public:
   void operator++()       { ++index_; }
   void operator++(int)    { ++index_; }
   void operator--()       { --index_; }
   void operator--(int)    { --index_; }
   void operator+=(int dx) { index_ += dx; }
   void operator-=(int dx) { index_ -= dx; }

   bool operator==(ViewIterator const& rhs) const
   { return index_ == rhs.index_; }

   bool operator!=(ViewIterator const& rhs) const
   { return index_ != rhs.index_; }

   bool operator<(ViewIterator const& rhs) const
   { return index_ < rhs.index_; }

   int operator-(ViewIterator const& rhs) const
   { return index_ - rhs.index_; }

   ViewIterator operator+(int dx) const
   {
      ViewIterator res(index_);
      res += dx;
      return res;
   }

   ViewIterator operator-(int dx) const
   {
      ViewIterator res(index_);
      res -= dx;
      return res;
   }

//   T operator()() const
//   { return current_; }
//
//   T operator()(int d) const
//   { return current_ + d; }

   value_type operator*() const
      { return get_nth(view_, index_); }

   FakeRef<View> operator*()
      { return FakeRef<View>(view_, index_); }
        
// members:
private:
   View		view_;
   int		index_;
};


/***********************************************************************
  Function Definitions
***********************************************************************/

// -------------------------------------------------------------------- //
// get_origin -- access element 0 of a vector.
template <typename T,
	  typename Block>
inline T
get_origin(
   vsip::const_Vector<T, Block> view)
{
   return view.get(0);
}



// -------------------------------------------------------------------- //
// get_origin -- access element 0,0 of a matrix.
template <typename T,
	  typename Block>
inline T
get_origin(
   vsip::const_Matrix<T, Block> view)
{
   return view.get(0, 0);
}



// -------------------------------------------------------------------- //
// put_origin -- put element 0 of a vector.
template <typename T,
	  typename Block>
inline void
put_origin(
   vsip::Vector<T, Block> view,
   T const&		  value)
{
   view.put(0, value);
}



// -------------------------------------------------------------------- //
// put_origin -- put element 0,0 of a matrix.
template <typename T,
	  typename Block>
inline void
put_origin(
   vsip::Matrix<T, Block> view,
   T const&		  value)
{
   view.put(0, value);
}



// -------------------------------------------------------------------- //
// put_origin -- put element 0,0 of a Dense<2,T> block.
template <typename T,
	  typename Order,
	  typename Map>
inline void
put_origin(
   vsip::Dense<2, T, Order, Map>* block,
   T const&		          value)
{
   block->put(0, 0, value);
}



// -------------------------------------------------------------------- //
// get_nth -- get n-th element of a vector.
template <typename T,
	  typename Block>
inline T
get_nth(
   vsip::Vector<T, Block> view,
   int			  n)
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
   int			  n)
{
   return view.get(n/view.size(1), n%view.size(1));
}



// -------------------------------------------------------------------- //
// put_nth -- put n-th element of a vector.
template <typename T,
	  typename Block>
inline void
put_nth(
   vsip::Vector<T, Block> view,
   int			  n,
   T const&		  value)
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
   int			  n,
   T const&		  value)
{
   view.put(n/view.size(1), n%view.size(1), value);
}

// -------------------------------------------------------------------- //
// put_origin -- put element 0,0 of a Dense<2,T> block.
template <typename T,
	  typename Order,
	  typename Map>
inline void
put_nth(
   vsip::Dense<2, T, Order, Map>* block,
   int			  n,
   T const&		          value)
{
   block->put(n/block->size(2, 1), n%block->size(2, 1), value);
}



#endif // _VSIPLPP_TEST_UTIL_H_
