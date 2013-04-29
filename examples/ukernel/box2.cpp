/* Copyright (c) 2008 by CodeSourcery.  All rights reserved. */

#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/matrix.hpp>
#include <vsip/signal.hpp>
#include <vsip_csl/ukernel/host/ukernel.hpp>
#include <kernels/host/box2.hpp>

#include <vsip_csl/test.hpp>
#include <vsip_csl/output.hpp>

namespace vsip_csl
{
namespace ukernel
{

template <>
struct Task_map<example::Box2, void(float*,float*)>
{
  static char const *plugin() { return "box2.plg";}
};
} // namespace vsip_csl::ukernel
} // namespace vsip_csl

using namespace vsip;

template <typename ViewT>
typename ViewT::value_type
get(ViewT view, stride_type r, stride_type c, 
    stride_type rr, stride_type cc)
{
  if (r + rr >= 0 && r + rr < (stride_type)view.size(0) &&
      c + cc >= 0 && c + cc < (stride_type)view.size(1))
    return view.get(r + rr, c + cc);
  else
    return typename ViewT::value_type(0);
}

template <typename Kernel, typename T>
void
run_ukernel(length_type rows, length_type cols, length_type overlap)
{
  using namespace vsip_csl;

  Kernel kernel(overlap);

  ukernel::Ukernel<Kernel> uk(kernel);

  Matrix<T> in(rows, cols);
  Matrix<T> out(rows, cols);

  for (index_type r=0; r<rows; ++r)
    in.row(r) = ramp<T>(T(r), 0.1, cols);

  in = T(1);
  for (index_type r=0; r<rows; ++r)
    in(r, 0) = 100 + r;

  for (index_type r=0; r<rows; ++r)
    in(r, r) = 200 + r;

  uk(in, out);

  int misco = 0;
  for (index_type r=0; r<rows; ++r)
    for (index_type c=0; c<cols; ++c)
    {
      T exp = T(0);
      for (stride_type rr=-overlap; rr<=+(stride_type)overlap; ++rr)
	for (stride_type cc=-overlap; cc<=+(stride_type)overlap; ++cc)
	  exp += get(in, r, c, rr, cc);

      if (!equal(out.get(r, c), exp))
      {
#if VERBOSE >= 2
	std::cout << r << ", " << c << ": " << out.get(r, c) << " != "
		  << exp
		  << std::endl;
#endif
	misco++;
      }
    }
  std::cout << "box2: size " << rows << " x " << cols 
	    << "  overlap " << overlap
	    << "  misco " << misco << std::endl;
}

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  run_ukernel<example::Box2, float>(16, 16, 1);
}
