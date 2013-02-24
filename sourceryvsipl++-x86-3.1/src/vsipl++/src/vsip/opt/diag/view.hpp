/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/diag/view.hpp
    @author  Jules Bergmann
    @date    2007-08-22
    @brief   VSIPL++ Library: Diagnostics for views.
*/

#ifndef VSIP_OPT_DIAG_VIEW_HPP
#define VSIP_OPT_DIAG_VIEW_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>
#include <iomanip>
#include <sstream>

#include <vsip/opt/diag/class_name.hpp>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{
namespace impl
{

namespace diag_detail
{

template <dimension_type dim0,
	  dimension_type dim1,
	  dimension_type dim2>
struct Class_name<tuple<dim0, dim1, dim2> >
{
  static std::string name()
  {
    std::ostringstream n;
    n << "tuple<" << dim0 << ", " << dim1 << ", " << dim2 << ">";
    return n.str();
  }
};

VSIP_IMPL_CLASS_NAME(Block_dist);
VSIP_IMPL_CLASS_NAME(Whole_dist);

template <dimension_type Dim,
	  typename       T,
	  typename       OrderT,
	  typename       MapT>
struct Class_name<Dense<Dim, T, OrderT, MapT> >
{
  static std::string name()
  {
    std::ostringstream n;
    n << "Dense<" << Dim << ", "
      << Class_name<T>::name() << ", "
      << Class_name<OrderT>::name() << ", "
      << Class_name<MapT>::name() << ">";
    return n.str();
  }
};

template <typename BlockT>
struct Class_name<Subset_block<BlockT> >
{
  static std::string name()
  {
    std::ostringstream n;
    n << "Subset_block<"
      << Class_name<BlockT>::name() << ">";
    return n.str();
  }
};



template <typename Dist0,
	  typename Dist1,
	  typename Dist2>
struct Class_name<Map<Dist0, Dist1, Dist2> >
{
  static std::string name()
  {
    std::ostringstream n;
    n << "Map<"
      << Class_name<Dist0>::name() << ", "
      << Class_name<Dist1>::name() << ", "
      << Class_name<Dist2>::name() << ">";
    return n.str();
  }
};

template <dimension_type Dim>
struct Class_name<Subset_map<Dim> >
{
  static std::string name()
  {
    std::ostringstream n;
    n << "Subset_map<" << Dim << ">";
    return n.str();
  }
};



/// Write a vector to a stream.

template <typename T,
	  typename Block>
inline
std::ostream&
output_view(
  std::ostream&		       out,
  vsip::const_Vector<T, Block> vec)
{
  for (vsip::index_type i=0; i<vec.size(); ++i)
    out << "        index " << i << "    : " << vec.get(i) << "\n";
  return out;
}



/// Write a matrix to a stream.

template <typename T,
	  typename Block>
inline
std::ostream&
output_view(
  std::ostream&		       out,
  vsip::const_Matrix<T, Block> v)
{
  for (vsip::index_type r=0; r<v.size(0); ++r)
  {
    out << "        row " << r << "     :";
    for (vsip::index_type c=0; c<v.size(1); ++c)
      out << "  " << v.get(r, c);
    out << std::endl;
  }
  return out;
}

} // namespace diag_detail

template <typename ViewT>
void
diagnose_view(char const* str, ViewT view, bool display = false)
{
  using std::cout;
  using std::endl;
  using std::flush;
  using vsip::impl::diag_detail::Class_name;

  dimension_type const view_dim  = ViewT::dim;

  typedef typename ViewT::block_type    block_type;
  typedef typename block_type::map_type map_type;

  dimension_type const block_dim = get_block_layout<block_type>::dim;

  map_type const& map = view.block().map();

  impl::Communicator& comm = impl::default_communicator();

  comm.barrier();
  if (comm.rank() == 0)
  {
    cout << "diagnose_view(" << str << "):" << std::endl;

    cout << "  General\n";
    cout << "    view dim       : " << view_dim << "  (" << view.size(0);
    for (dimension_type i=1; i<view_dim; ++i)
      cout << ", " << view.size(i) ;
    cout << ")\n";
    
    cout << "    block dim      : " << block_dim << "  ("
	 << view.block().size(block_dim, 0);
    for (dimension_type i=1; i<view_dim; ++i)
      cout << ", " << view.block().size(block_dim, i) ;
    cout << ")\n";
    
    cout << "    block_type     : " << Class_name<block_type>::name() << endl
	 << "    map_type       : " << Class_name<map_type>::name() << endl
      // << "    map typeid     : " << typeid(map_type).name() << endl
      ;
    
    cout << "  Map info         : " << endl
	 << "    subblocks      : " << map.num_subblocks() << endl
	 << "    processors     : " << map.num_processors() << endl
      ;
    cout << flush;
  }
  comm.barrier();
  for (index_type proc=0; proc<num_processors(); ++proc)
  {
    comm.barrier();
    if (local_processor() == proc)
    {
      index_type sb = map.subblock(proc);
      cout << "    * processor    : " << proc << endl;
      if (sb != no_subblock)
      {
	cout << "      subblock     : " << sb << endl;
	cout << "      subblock_dom : "
	     << map.template impl_subblock_domain<block_dim>(sb) << endl;

	typename ViewT::local_type l_view = view.local(); 
	cout << "      local view   : " << view_dim << "  (" << l_view.size(0);
	for (dimension_type i=1; i<view_dim; ++i)
	  cout << ", " << l_view.size(i) ;
	cout << ")\n";


	cout << "      patches      : " << map.impl_num_patches(sb) << endl;
	for (index_type p=0; p<map.impl_num_patches(sb); ++p)
	{
	  cout << "      - patch      : " << p << endl;
	  cout << "        global_dom : "
	       << map.template impl_global_domain<block_dim>(sb, p) << endl;
	  cout << "        local_dom  : "
	       << map.template impl_local_domain<block_dim>(sb, p) << endl;
	}
	if (display)
	{
	  diag_detail::output_view(cout, l_view);
	}
      }
      else
	cout << "      subblock     : no subblock\n";
      cout << flush;
    }
  }
  comm.barrier();

}

} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_OPT_DIAG_VIEW_HPP
