//
// Copyright (c) 2005 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
//
// This file is made available under the GPL.
// See the accompanying file LICENSE.GPL for details.

#include <ovxx/support.hpp>
#include <ovxx/ct_assert.hpp>

using namespace ovxx;

// This test contains negative-compile test cases.  To test, enable
// one of the tests manually and check that the compilation fails.

#define ILLEGAL1 0
#define ILLEGAL2 0
#define ILLEGAL3 0
#define ILLEGAL4 0
#define ILLEGAL5 0

void 
test_ct_assert()
{
  using ovxx::scalar_f;
  using ovxx::cscalar_f;
  using ovxx::complex;

  OVXX_CT_ASSERT(true);
  OVXX_CT_ASSERT(sizeof(float)          == sizeof(scalar_f));
  OVXX_CT_ASSERT(sizeof(complex<float>) == sizeof(cscalar_f));

#if ILLEGAL1
  OVXX_CT_ASSERT(false);
#endif

#if ILLEGAL2
  OVXX_CT_ASSERT(sizeof(scalar_f) == sizeof(cscalar_f));
#endif
}

int
main()
{
  test_ct_assert();
}
