#include <iostream>

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/dense.hpp>
#include <vsip_csl/strided.hpp>
#include <vsip_csl/block_marshal.hpp>
#include <vsip/selgen.hpp>
#include <vsip_csl/test.hpp>
#include <vsip_csl/output.hpp>
#include <cassert>

using namespace vsip;
using namespace vsip_csl;
using namespace vsip::impl;

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);
  // Marshal full dense block
  {
    Vector<complex<float> > input = ramp(0, 1, 8);
    vsip_csl::impl::Block_marshal marshal;
    // externalize 'input' to storage
    marshal.marshal(input.block());
    // Create a new vector with user-storage
    typedef Dense<1, complex<float> > dense_type;
#if VSIP_IMPL_PREFER_SPLIT_COMPLEX
    dense_type dense(Domain<1>(), 0, 0);
#else
    dense_type dense(Domain<1>(), (float*)0);
#endif
    Vector<complex<float> > output(dense);
    // Yes we can.
    test_assert(marshal.can_unmarshal<dense_type>());
    // internalize 'output' from storage
    marshal.unmarshal(output.block());
    // Make sure that both match
    std::cout << input << std::endl;
    std::cout << output << std::endl;
    test_assert(view_equal(input,output));
  }
  // Marshal user-storage dense block
  {
    using namespace vsip::impl;
    typedef Layout<1, tuple<0,1,2>, dense, interleaved_complex> inter_layout;
    typedef Layout<1, tuple<0,1,2>, dense, split_complex> split_layout;
    typedef Dense<1, complex<float> > dense_type;
    typedef Strided<1, complex<float>,  inter_layout> inter_type;
    typedef Strided<1, complex<float>,  split_layout> split_type;
    float data[] = {0., 1., 2., 3., 4., 5., 6., 7.,
		    8., 9., 10., 11., 12., 13., 14., 15.};
    inter_type input_block(8, data + 0, data + 8);
    input_block.admit();
    Vector<complex<float>, inter_type> input(input_block);
    vsip_csl::impl::Block_marshal marshal;
    // externalize 'input' to storage
    marshal.marshal(input.block());
    // Create a new vector with user-storage
    split_type output_block(Domain<1>(), 0, 0);
    Vector<complex<float>, split_type> output(output_block);
    // Yes we can.
    test_assert(marshal.can_unmarshal<split_type>());
    // internalize 'output' from storage
    marshal.unmarshal(output.block());
    // Make sure that both match
    test_assert(view_equal(input,output));
    
    // Recover 'output' into a new Block_marshal object.
    vsip_csl::impl::Block_marshal marshal2;
    marshal2.recover(output.block());
    // Confirm that the descriptors match.
    test_assert(marshal2.descriptor.value_type == marshal.descriptor.value_type);
    test_assert(marshal2.descriptor.dimensions == marshal.descriptor.dimensions);
    test_assert(marshal2.descriptor.storage_format == marshal.descriptor.storage_format);
    test_assert(marshal2.descriptor.size0 == marshal.descriptor.size0);
    test_assert(marshal2.descriptor.stride0 == marshal.descriptor.stride0);
    
    // Recover 'input'.
    marshal2.recover(input.block());
    // Confirm that the recovered descriptor is split-complex.
    test_assert(marshal2.descriptor.storage_format == vsip_csl::impl::block_marshal::SPLIT);
  }
  // Marshal dense subblock
  {
    Matrix<complex<float> > input(4, 4);
    for (index_type r = 0; r != input.size(0); ++r)
      input.row(r) = ramp<complex<float> >(r, 1, 4);
    vsip_csl::impl::Block_marshal marshal;
    // externalize subblock to storage
    marshal.marshal(input(Domain<2>(Domain<1>(1, 1, 2), 4)).block());
    // Create a new vector with user-storage
    typedef Dense<2, complex<float> > dense_type;
#if VSIP_IMPL_PREFER_SPLIT_COMPLEX
    dense_type dense(Domain<2>(), 0, 0);
#else
    dense_type dense(Domain<2>(), (float*)0);
#endif
    Matrix<complex<float> > output(dense);
    // Yes we can.
    test_assert(marshal.can_unmarshal<dense_type>());
    // internalize 'output' from storage
    marshal.unmarshal(output.block());
    // Make sure that both match
    test_assert(view_equal(input(Domain<2>(Domain<1>(1, 1, 2), 4)),output));
  }
}
