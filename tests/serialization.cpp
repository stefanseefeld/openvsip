#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/dense.hpp>
#include <vsip/serialization.hpp>
#include <vsip_csl/strided.hpp>
#include <vsip/selgen.hpp>
#include <vsip_csl/test.hpp>
#include <vsip_csl/output.hpp>
#include "test-random.hpp"
#include <cassert>
#include <iostream>

using namespace vsip;
namespace s = serialization;
using vsip_csl::view_equal;
using vsip_csl::Strided;

template <typename T, storage_format_type S> 
struct allocator
{
  typedef T *ptr_type;
  static T *allocate(length_type N) { return new T[N];}
  static void deallocate(T *storage) { delete [] storage;}
};

template <typename T>
struct allocator<complex<T>, interleaved_complex>
{
  typedef T *ptr_type;
  static T *allocate(length_type N) { return new T[2*N];}
  static void deallocate(T *storage) { delete [] storage;}
};

template <typename T>
struct allocator<complex<T>, split_complex>
{
  typedef std::pair<T*,T*> ptr_type;
  static std::pair<T*,T*> allocate(length_type N)
  {
    T *real = new T[2*N];
    return std::make_pair(real, real + N);
  }
  static void deallocate(std::pair<T*,T*> storage) { delete [] storage.first;}
};

template <dimension_type D> struct make_domain;
template <> 
struct make_domain<1>
{
  static Domain<1> create(s::Descriptor const &info) { return info.size[0];}
};
template <> 
struct make_domain<2>
{
  static Domain<2> create(s::Descriptor const &info) 
  { return Domain<2>(info.size[0], info.size[1]);}
};
template <> 
struct make_domain<3>
{
  static Domain<3> create(s::Descriptor const &info) 
  { return Domain<3>(info.size[0], info.size[1], info.size[2]);}
};

// Take a raw pointer and a descriptor to reconstruct a block.
// Then validate against a reference block.
template <typename P, typename B>
void serialize_and_validate(P data, s::Descriptor const &info, B const &reference)
{
  typedef typename B::value_type value_type;
  typedef Strided<B::dim, value_type, typename get_block_layout<B>::type> block_type;
  Domain<B::dim> dom;
  block_type block(dom, P());
  typename impl::view_of<block_type>::type output(block);
  test_assert(s::is_compatible<block_type>(info));
  block.rebind(data, make_domain<B::dim>::create(info));
  block.admit();
  test_assert(view_equal(output, typename impl::view_of<B>::type(const_cast<B &>(reference))));
}

// Construct a block of the given type and size, then serialize it via DDA.
// Validate the serialized data against the original block.
template <typename B>
void test_serialize_data(length_type N)
{
  Vector<typename B::value_type, B> input(N);
  randomize(input);
  dda::Data<B, dda::inout> data(input.block());
  s::Descriptor info;
  s::describe_data(data, info);
  serialize_and_validate(data.ptr(), info, input.block());
}

template <typename B>
void test_serialize_data(length_type N, length_type M)
{
  Matrix<typename B::value_type, B> input(N, M);
  randomize(input);
  dda::Data<B, dda::inout> data(input.block());
  s::Descriptor info;
  s::describe_data(data, info);
  serialize_and_validate(data.ptr(), info, input.block());
}

// Construct a user-storage block of the given type and size,
// then serialize its user storage.
// Validate the serialized data against the original block.
template <typename B>
void test_serialize_user_storage(Domain<B::dim> const dom)
{
  typedef allocator<typename B::value_type, B::storage_format> alloc;
  typedef typename B::value_type value_type;
  typename alloc::ptr_type data = alloc::allocate(dom.size());
  // The block constructors unfortunately ignore the stride...
  B block(dom, data);
  block.admit();
  typename impl::view_of<B>::type input(block);
  randomize(input);
  block.release();
  s::Descriptor info;
  s::describe_user_storage(input.block(), info);
  serialize_and_validate(data, info, input.block());
  alloc::deallocate(data);
}

int
main(int argc, char** argv)
{
  typedef Layout<1, tuple<0,1,2>, dense, interleaved_complex> inter_layout;
  typedef Layout<1, tuple<0,1,2>, dense, split_complex> split_layout;
  typedef Layout<2, tuple<0,1,2>, dense, interleaved_complex> inter_row_layout;
  typedef Layout<2, tuple<1,0,2>, dense, interleaved_complex> inter_col_layout;
  typedef Layout<2, tuple<0,1,2>, dense, split_complex> split_row_layout;
  typedef Layout<2, tuple<1,0,2>, dense, split_complex> split_col_layout;

  typedef Strided<1, float> real_block_type;
  typedef Strided<1, complex<float>, inter_layout> inter_block_type;
  typedef Strided<1, complex<float>, split_layout> split_block_type;

  typedef Strided<2, float, inter_row_layout> real_row_block_type;
  typedef Strided<2, float, inter_col_layout> real_col_block_type;
  typedef Strided<2, complex<float>, inter_row_layout> inter_row_block_type;
  typedef Strided<2, complex<float>, inter_col_layout> inter_col_block_type;
  typedef Strided<2, complex<float>, split_row_layout> split_row_block_type;
  typedef Strided<2, complex<float>, split_col_layout> split_col_block_type;

  vsipl init(argc, argv);

  test_serialize_data<real_block_type>(8);
  test_serialize_data<inter_block_type>(8);
  test_serialize_data<split_block_type>(8);

  test_serialize_user_storage<real_block_type>(8);
  test_serialize_user_storage<inter_block_type>(8);
  test_serialize_user_storage<split_block_type>(8);

  test_serialize_data<real_row_block_type>(8, 8);
  test_serialize_data<real_col_block_type>(8, 8);
  test_serialize_data<inter_row_block_type>(8, 8);
  test_serialize_data<inter_col_block_type>(8, 8);
  test_serialize_data<split_row_block_type>(8, 8);
  test_serialize_data<split_col_block_type>(8, 8);

  test_serialize_user_storage<real_row_block_type>(Domain<2>(8, 8));
  test_serialize_user_storage<real_col_block_type>(Domain<2>(8, 8));
  test_serialize_user_storage<inter_row_block_type>(Domain<2>(8, 8));
  test_serialize_user_storage<inter_col_block_type>(Domain<2>(8, 8));
  test_serialize_user_storage<split_row_block_type>(Domain<2>(8, 8));
  test_serialize_user_storage<split_col_block_type>(Domain<2>(8, 8));
}
