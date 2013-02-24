/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    benchmarks/vmul.hpp
    @author  Jules Bergmann
    @date    2005-07-11
    @brief   VSIPL++ Library: Benchmarks for vector multiply.

*/

#ifndef VSIP_BENCHMARKS_VMUL_HPP
#define VSIP_BENCHMARKS_VMUL_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <vsip/support.hpp>
#include <vsip/math.hpp>
#include <vsip/random.hpp>
#include <vsip/selgen.hpp>
#include <vsip_csl/assignment.hpp>
#include <vsip/opt/assign_diagnostics.hpp>

#include "../benchmarks.hpp"

using namespace vsip;



/***********************************************************************
  Declarations
***********************************************************************/

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
struct Create_map<Dim, vsip::Global_map<Dim> >
{
  typedef vsip::Global_map<Dim> type;
  static type exec() { return type(); }
};

template <typename Dist0, typename Dist1, typename Dist2>
struct Create_map<1, vsip::Map<Dist0, Dist1, Dist2> >
{
  typedef vsip::Map<Dist0, Dist1, Dist2> type;
  static type exec() { return type(vsip::num_processors()); }
};

template <vsip::dimension_type Dim,
	  typename             MapT>
MapT
create_map()
{
  return Create_map<Dim, MapT>::exec();
}


// Sync Policy: use barrier.

struct Barrier
{
  Barrier() : comm_(DEFAULT_COMMUNICATOR()) {}

  void sync() { BARRIER(comm_); }

  COMMUNICATOR_TYPE& comm_;
};



// Sync Policy: no barrier.

struct No_barrier
{
  No_barrier() {}

  void sync() {}
};





/***********************************************************************
  Definitions - vector element-wise multiply
***********************************************************************/

// Elementwise vector-multiply, non-distributed (explicit Local_map)
// This is equivalent to t_vmul1<T, Local_map>.

template <typename T>
struct t_vmul1_nonglobal : Benchmark_base
{
  char const* what() { return "t_vmul1_nonglobal"; }
  int ops_per_point(length_type)  { return vsip::impl::Ops_info<T>::mul; }
  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
    VSIP_IMPL_NOINLINE
  {
    typedef Dense<1, T, row1_type, Local_map> block_type;

    Vector<T, block_type> A(size, T());
    Vector<T, block_type> B(size, T());
    Vector<T, block_type> C(size);

    Rand<T> gen(0, 0);
    A = gen.randu(size);
    B = gen.randu(size);

    A.put(0, T(3));
    B.put(0, T(4));

    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      C = A * B;
    t1.stop();
    
    for (index_type i=0; i<size; ++i)
      test_assert(equal(C.get(i), A.get(i) * B.get(i)));
    
    time = t1.delta();
  }
};



// Element-wise vector-multiply.  Supports distributed views, using
// implicit data-parallelism.

template <typename T,
	  typename MapT = Local_map,
	  typename SP   = No_barrier>
struct t_vmul1 : Benchmark_base
{
  typedef Dense<1, T, row1_type, MapT> block_type;

  char const* what() { return "t_vmul1"; }
  int ops_per_point(length_type)  { return vsip::impl::Ops_info<T>::mul; }
  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
    VSIP_IMPL_NOINLINE
  {
    MapT map = create_map<1, MapT>();

    Vector<T, block_type> A(size, T(), map);
    Vector<T, block_type> B(size, T(), map);
    Vector<T, block_type> C(size,      map);

    Rand<T> gen(0, 0);
    // A, B, and C have the same map.
    for (index_type i=0; i<C.local().size(); ++i)
    {
      A.local().put(i, gen.randu());
      B.local().put(i, gen.randu());
    }
    A.put(0, T(3));
    B.put(0, T(4));

    vsip::impl::profile::Timer t1;
    SP sp;
    
    t1.start();
    sp.sync();
    for (index_type l=0; l<loop; ++l)
      C = A * B;
    sp.sync();
    t1.stop();
    
    for (index_type i=0; i<C.local().size(); ++i)
      test_assert(equal(C.local().get(i),
			A.local().get(i) * B.local().get(i)));
    
    time = t1.delta();
  }

  void diag()
  {
    length_type const size = 256;

    MapT map = create_map<1, MapT>();

    Vector<T, block_type> A(size, T(), map);
    Vector<T, block_type> B(size, T(), map);
    Vector<T, block_type> C(size,      map);

    impl::assign_diagnostics(C, A * B);
  }
};



// Element-wise vector-multiply (using mul() instead operator*()).
// Supports distributed views, using implicit data-parallelism.

template <typename T,
	  typename MapT = Local_map,
	  typename SP   = No_barrier>
struct t_vmul_func : Benchmark_base
{
  typedef Dense<1, T, row1_type, MapT> block_type;

  char const* what() { return "t_vmul1"; }
  int ops_per_point(length_type)  { return vsip::impl::Ops_info<T>::mul; }
  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
    VSIP_IMPL_NOINLINE
  {
    MapT map = create_map<1, MapT>();

    Vector<T, block_type> A(size, T(), map);
    Vector<T, block_type> B(size, T(), map);
    Vector<T, block_type> C(size,      map);

    Rand<T> gen(0, 0);
    // A, B, and C have the same map.
    for (index_type i=0; i<C.local().size(); ++i)
    {
      A.local().put(i, gen.randu());
      B.local().put(i, gen.randu());
    }
    A.put(0, T(3));
    B.put(0, T(4));

    vsip::impl::profile::Timer t1;
    SP sp;
    
    t1.start();
    sp.sync();
    for (index_type l=0; l<loop; ++l)
      C = mul(A, B);
    sp.sync();
    t1.stop();
    
    for (index_type i=0; i<C.local().size(); ++i)
      test_assert(equal(C.local().get(i),
			A.local().get(i) * B.local().get(i)));
    
    time = t1.delta();
  }

  void diag()
  {
    length_type const size = 256;

    MapT map = create_map<1, MapT>();

    Vector<T, block_type> A(size, T(), map);
    Vector<T, block_type> B(size, T(), map);
    Vector<T, block_type> C(size,      map);

    impl::assign_diagnostics(C, mul(A, B));
  }
};




// Element-wise vector-multiply.  Supports distributed views, using
// in-loop local views.

template <typename T,
	  typename MapT = Local_map,
	  typename SP   = No_barrier>
struct t_vmul1_local : Benchmark_base
{
  char const* what() { return "t_vmul1"; }
  int ops_per_point(length_type)  { return vsip::impl::Ops_info<T>::mul; }
  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
    VSIP_IMPL_NOINLINE
  {
    typedef Dense<1, T, row1_type, MapT> block_type;

    MapT map = create_map<1, MapT>();

    Vector<T, block_type> A(size, T(), map);
    Vector<T, block_type> B(size, T(), map);
    Vector<T, block_type> C(size,      map);

    Rand<T> gen(0, 0);
    A = gen.randu(size);
    B = gen.randu(size);

    A.put(0, T(3));
    B.put(0, T(4));

    vsip::impl::profile::Timer t1;
    SP sp;
    
    t1.start();
    sp.sync();
    for (index_type l=0; l<loop; ++l)
      C.local() = A.local() * B.local();
    sp.sync();
    t1.stop();
    
    for (index_type i=0; i<size; ++i)
      test_assert(equal(C.get(i), A.get(i) * B.get(i)));
    
    time = t1.delta();
  }
};



// Element-wise vector-multiply.  Supports distributed views, using
// early local views.

template <typename T,
	  typename MapT = Local_map,
	  typename SP   = No_barrier>
struct t_vmul1_early_local : Benchmark_base
{
  char const* what() { return "t_vmul1_early_local"; }
  int ops_per_point(length_type)  { return vsip::impl::Ops_info<T>::mul; }
  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
    VSIP_IMPL_NOINLINE
  {
    typedef Dense<1, T, row1_type, MapT> block_type;

    MapT map = create_map<1, MapT>();

    Vector<T, block_type> A(size, T(), map);
    Vector<T, block_type> B(size, T(), map);
    Vector<T, block_type> C(size,      map);

    Rand<T> gen(0, 0);
    A = gen.randu(size);
    B = gen.randu(size);

    A.put(0, T(3));
    B.put(0, T(4));

    vsip::impl::profile::Timer t1;
    SP sp;
    
    t1.start();
    typename Vector<T, block_type>::local_type A_local = A.local();
    typename Vector<T, block_type>::local_type B_local = B.local();
    typename Vector<T, block_type>::local_type C_local = C.local();
    sp.sync();
    for (index_type l=0; l<loop; ++l)
      C_local = A_local * B_local;
    sp.sync();
    t1.stop();
    
    for (index_type i=0; i<size; ++i)
      test_assert(equal(C.get(i), A.get(i) * B.get(i)));
    
    time = t1.delta();
  }
};



// Element-wise vector-multiply.  Supports distributed views, using Assignment.
template <typename T,
	  typename MapT = Local_map,
	  typename SP   = No_barrier>
struct t_vmul1_sa : Benchmark_base
{
  char const* what() { return "t_vmul1_sa"; }
  int ops_per_point(length_type)  { return vsip::impl::Ops_info<T>::mul; }
  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
    VSIP_IMPL_NOINLINE
  {
    typedef Dense<1, T, row1_type, MapT> block_type;

    MapT map = create_map<1, MapT>();

    Vector<T, block_type> A(size, T(), map);
    Vector<T, block_type> B(size, T(), map);
    Vector<T, block_type> C(size,      map);

    Rand<T> gen(0, 0);
    A = gen.randu(size);
    B = gen.randu(size);

    A.put(0, T(3));
    B.put(0, T(4));


    vsip::impl::profile::Timer t1;
    SP sp;
    
    vsip_csl::Assignment expr(C, A*B);
    t1.start();
    sp.sync();
    for (index_type l=0; l<loop; ++l)
      expr();
    sp.sync();
    t1.stop();

    for (index_type i=0; i<size; ++i)
      test_assert(equal(C.get(i), A.get(i) * B.get(i)));
    
    time = t1.delta();
  }
};



// Elementwise vector-multiply using get/put.
//
// Non-distributed (explicit Local_map).

template <typename T>
struct t_vmul_gp : Benchmark_base
{
  char const* what() { return "t_vmul_gp"; }
  int ops_per_point(length_type)  { return vsip::impl::Ops_info<T>::mul; }
  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
    VSIP_IMPL_NOINLINE
  {
    typedef Dense<1, T, row1_type, Local_map> block_type;

    Vector<T, block_type> A(size, T());
    Vector<T, block_type> B(size, T());
    Vector<T, block_type> C(size);

    Rand<T> gen(0, 0);
    A = gen.randu(size);
    B = gen.randu(size);

    A.put(0, T(3));
    B.put(0, T(4));

    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      for (index_type i=0; i<size; ++i)
	C.put(i, A.get(i) * B.get(i));
    t1.stop();
    
    for (index_type i=0; i<size; ++i)
      test_assert(equal(C.get(i), A.get(i) * B.get(i)));
    
    time = t1.delta();
  }
};



template <typename T>
struct t_vmul_ip1 : Benchmark_base
{
  char const* what() { return "t_vmul_ip1"; }
  int ops_per_point(length_type)  { return vsip::impl::Ops_info<T>::mul; }
  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
    VSIP_IMPL_NOINLINE
  {
    Vector<T>   A(size, T(1));
    Vector<T>   C(size);
    Vector<T>   chk(size);

    Rand<T> gen(0, 0);
    chk = gen.randu(size);
    C = chk;

    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      C *= A;
    t1.stop();
    
    for (index_type i=0; i<size; ++i)
      test_assert(equal(chk.get(i), C.get(i)));
    
    time = t1.delta();
  }

  void diag()
  {
    length_type const size = 256;

    Vector<T> A(size, T(1));
    Vector<T> C(size);

    impl::assign_diagnostics(C, C * A);
  }
};



template <typename T>
struct t_vmul_dom1 : Benchmark_base
{
  char const* what() { return "t_vmul_dom1"; }
  int ops_per_point(length_type)  { return vsip::impl::Ops_info<T>::mul; }
  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
    VSIP_IMPL_NOINLINE
  {
    Vector<T>   A(size, T());
    Vector<T>   B(size, T());
    Vector<T>   C(size);

    Rand<T> gen(0, 0);
    A = gen.randu(size);
    B = gen.randu(size);

    A.put(0, T(3));
    B.put(0, T(4));

    Domain<1> dom(size);
    
    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      C(dom) = A(dom) * B(dom);
    t1.stop();
    
    for (index_type i=0; i<size; ++i)
      test_assert(equal(C.get(i), A.get(i) * B.get(i)));
    
    time = t1.delta();
  }
};


#ifdef VSIP_IMPL_SOURCERY_VPP
template <typename T, typename ComplexFmt>
struct t_vmul2 : Benchmark_base
{
  // compile-time typedefs
  typedef impl::Layout<1, row1_type, impl::Stride_unit_dense, ComplexFmt>
		LP;
  typedef impl::Strided<1, T, LP> block_type;

  // benchmark attributes
  char const* what() { return "t_vmul2"; }
  int ops_per_point(length_type)  { return vsip::impl::Ops_info<T>::mul; }
  int riob_per_point(length_type) { return 2*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 3*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
    VSIP_IMPL_NOINLINE
  {
    Vector<T, block_type> A(size, T());
    Vector<T, block_type> B(size, T());
    Vector<T, block_type> C(size);

    A.put(0, T(3));
    B.put(0, T(4));
    
    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      C = A * B;
    t1.stop();
    
    test_assert(equal(C.get(0), T(12)));
    
    time = t1.delta();
  }

  void diag()
  {
    length_type const size = 256;

    Vector<T, block_type> A(size, T());
    Vector<T, block_type> B(size, T());
    Vector<T, block_type> C(size);

    impl::assign_diagnostics(C, A * B);
  }
};
#endif // VSIP_IMPL_SOURCERY_VPP


/***********************************************************************
  Definitions - real * complex vector element-wise multiply
***********************************************************************/

template <typename T>
struct t_rcvmul1 : Benchmark_base
{
  char const* what() { return "t_rcvmul1"; }
  int ops_per_point(length_type)  { return 2*vsip::impl::Ops_info<T>::mul; }
  int riob_per_point(length_type) { return sizeof(T) + sizeof(complex<T>); }
  int wiob_per_point(length_type) { return 1*sizeof(complex<T>); }
  int mem_per_point(length_type)  { return 1*sizeof(T)+2*sizeof(complex<T>); }

  void operator()(length_type size, length_type loop, float& time)
    VSIP_IMPL_NOINLINE
  {
    Vector<complex<T> > A(size);
    Vector<T>           B(size);
    Vector<complex<T> > C(size);

    Rand<complex<T> > cgen(0, 0);
    Rand<T>           sgen(0, 0);

    A = cgen.randu(size);
    B = sgen.randu(size);

    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      C = B * A;
    t1.stop();
    
    for (index_type i=0; i<size; ++i)
      test_assert(equal(C.get(i), A.get(i) * B.get(i)));
    
    time = t1.delta();
  }

  void diag()
  {
    length_type const size = 256;

    Vector<complex<T> > A(size);
    Vector<T>           B(size);
    Vector<complex<T> > C(size);

    impl::assign_diagnostics(C, B * A);
  }
};



// Benchmark scalar-view vector multiply (Scalar * View)

template <typename ScalarT,
	  typename T,
	  typename DestT = T>
struct t_svmul1 : Benchmark_base
{
  char const* what() { return "t_svmul1"; }
  int ops_per_point(length_type)
  { if (sizeof(ScalarT) == sizeof(T))
      return vsip::impl::Ops_info<T>::mul;
    else
      return 2*vsip::impl::Ops_info<ScalarT>::mul;
  }
  int riob_per_point(length_type) { return 1*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 2*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    Vector<T>     A(size, T());
    Vector<DestT> C(size);

    ScalarT alpha = ScalarT(3);

    Rand<T>     gen(0, 0);
    A = gen.randu(size);
    A.put(0, T(4));

    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      C = alpha * A;
    t1.stop();
    
    for (index_type i=0; i<size; ++i)
      test_assert(equal(C.get(i), alpha * A.get(i)));
    
    time = t1.delta();
  }

  void diag()
  {
    length_type const size = 256;

    Vector<T>   A(size, T());
    Vector<T>   C(size);

    ScalarT alpha = ScalarT(3);

    impl::assign_diagnostics(C, alpha * A);
  }
};



// Benchmark scalar-view vector multiply (Scalar * View), w/subdomain

template <typename ScalarT,
	  typename T>
struct t_svmul_dom : Benchmark_base
{
  char const* what() { return "t_svmul_dom"; }
  int ops_per_point(length_type)
  { if (sizeof(ScalarT) == sizeof(T))
      return vsip::impl::Ops_info<T>::mul;
    else
      return 2*vsip::impl::Ops_info<ScalarT>::mul;
  }
  int riob_per_point(length_type) { return 1*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 2*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    Vector<T>   A(size, T());
    Vector<T>   C(size);

    ScalarT alpha = ScalarT(3);

    Rand<T>     gen(0, 0);
    A = gen.randu(size);
    A.put(0, T(4));

    Domain<1> dom(size);

    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      C(dom) = alpha * A(dom);
    t1.stop();
    
    for (index_type i=0; i<size; ++i)
      test_assert(equal(C.get(i), alpha * A.get(i)));
    
    time = t1.delta();
  }

  void diag()
  {
    length_type const size = 256;

    Vector<T>   A(size, T());
    Vector<T>   C(size);

    ScalarT alpha = ScalarT(3);

    Domain<1> dom(size);

    impl::assign_diagnostics(C(dom), alpha * A(dom));
  }
};



// Benchmark scalar-view vector multiply (Scalar * View), w/cold cache.

template <typename ScalarT,
	  typename T>
struct t_svmul_cc : Benchmark_base
{
  char const* what() { return "t_svmul_cc (cold cache)"; }
  int ops_per_point(length_type)
  { if (sizeof(ScalarT) == sizeof(T))
      return vsip::impl::Ops_info<T>::mul;
    else
      return 2*vsip::impl::Ops_info<ScalarT>::mul;
  }
  int riob_per_point(length_type) { return 1*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 2*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    length_type rows = footprint_ / size;
    Matrix<T>   A(rows, size, T());
    Matrix<T>   C(rows, size);

    ScalarT alpha = ScalarT(3);

    Rand<T>     gen(0, 0);
    A = gen.randu(rows, size);
    A.put(0, 0, T(4));

    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      C.row(l%rows) = alpha * A.row(l%rows);
    t1.stop();
    
    for (index_type i=0; i<size; ++i)
      for (index_type r=0; r<rows && r<loop; ++r)
	test_assert(equal(C.get(r, i), alpha * A.get(r, i)));
    
    time = t1.delta();
  }

  void diag()
  {
    length_type const size = 256;
    length_type rows = footprint_ / size;

    Matrix<T>   A(rows, size, T());
    Matrix<T>   C(rows, size);

    ScalarT alpha = ScalarT(3);

    impl::assign_diagnostics(C.row(0), alpha * A.row(0));
  }

  t_svmul_cc(length_type footprint) : footprint_(footprint) {}

  length_type footprint_;
};



// Benchmark scalar-view vector multiply w/literal (Scalar * View)

template <typename T>
struct t_svmul3 : Benchmark_base
{
  char const* what() { return "t_svmul3"; }
  int ops_per_point(length_type)  { return vsip::impl::Ops_info<T>::mul; }
  int riob_per_point(length_type) { return 1*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 2*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    Vector<T>   A(size, T());
    Vector<T>   C(size);

    A.put(0, T(4));
    
    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
      C = 3.f * A;
    t1.stop();

    test_assert(equal(C.get(0), T(12)));
    
    time = t1.delta();
  }
};



// Benchmark scalar-view vector multiply w/literal (Scalar * View)

template <typename T,
	  typename DataMapT  = Local_map,
	  typename CoeffMapT = Local_map,
	  typename SP        = No_barrier>
struct t_svmul4 : Benchmark_base
{
  char const* what() { return "t_svmul4"; }
  int ops_per_point(length_type)  { return vsip::impl::Ops_info<T>::mul; }
  int riob_per_point(length_type) { return 1*sizeof(T); }
  int wiob_per_point(length_type) { return 1*sizeof(T); }
  int mem_per_point(length_type)  { return 2*sizeof(T); }

  void operator()(length_type size, length_type loop, float& time)
  {
    typedef Dense<1, T, row1_type, DataMapT>  block_type;
    typedef Dense<1, T, row1_type, CoeffMapT> coeff_block_type;

    DataMapT  map_data  = create_map<1, DataMapT>();
    CoeffMapT map_coeff = create_map<1, CoeffMapT>();

    Vector<T, block_type>       A(size, T(), map_data);
    Vector<T, block_type>       C(size,      map_data);
    Vector<T, coeff_block_type> K(size, T(), map_coeff);

    // ramp does not work for distributed assignments (060726)
    // A = cos(ramp(0.f, 0.15f*3.14159f, size));
    for (index_type i=0; i<A.local().size(); ++i)
    {
      index_type g_i = global_from_local_index(A, 0, i);
      A.local().put(i, cos(T(g_i)*0.15f*3.14159f));
    }

    // ramp does not work for distributed assignments (060726)
    // K = cos(ramp(0.f, 0.25f*3.14159f, size));
    for (index_type i=0; i<K.local().size(); ++i)
    {
      index_type g_i = global_from_local_index(K, 0, i);
      K.local().put(i, cos(T(g_i)*0.25f*3.14159f));
    }

    T alpha;

    vsip::impl::profile::Timer t1;
    
    t1.start();
    for (index_type l=0; l<loop; ++l)
    {
      alpha = K.get(1);
      C = alpha * A;
    }
    t1.stop();

    alpha = K.get(1);
    for (index_type i=0; i<C.local().size(); ++i)
      test_assert(equal(C.local().get(i), A.local().get(i) * alpha));
    
    time = t1.delta();
  }
};

#endif // VSIP_BENCHMARKS_VMUL_HPP
