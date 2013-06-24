//
// Copyright (c) 2007 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.GPL file.

#define VERBOSE 0

#if VERBOSE
#  include <iostream>
#endif

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/domain.hpp>
#include <vsip/random.hpp>
#include <test.hpp>

using namespace ovxx;

template <typename T, storage_format_type C>
void
test_vmul(length_type len)
{
  typedef Layout<1, row1_type, dense, C> LP;
  typedef Strided<1, T, LP> block_type;

  Rand<T> gen(0, 0);

  Vector<T, block_type> A(len);
  Vector<T, block_type> B(len);
  Vector<T, block_type> Z(len);

  A = gen.randu(len);
  B = gen.randu(len);

  Z = A * B;

  for (index_type i=0; i<len; ++i)
  {
#if VERBOSE
    if (!equal(Z.get(i), A.get(i) * B.get(i)))
    {
      std::cout << "Z(" << i << ")        = " << Z(i) << std::endl;
      std::cout << "A(" << i << ") * B(" << i << ") = "
		<< A(i) * B(i) << std::endl;
    }
#endif
    test_assert(almost_equal(Z.get(i), A.get(i) * B.get(i)));
  }
}




template <typename T, storage_format_type C>
void
test_sweep()
{
  for (index_type i=1; i<=128; ++i)
    test_vmul<T, C>(i);
}

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  test_sweep<float,          array>();
  test_sweep<complex<float>, interleaved_complex>();
  test_sweep<complex<float>, split_complex>();
}
