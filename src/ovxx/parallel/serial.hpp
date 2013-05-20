//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_parallel_serial_hpp_
#define ovxx_parallel_serial_hpp_

#include <ovxx/support.hpp>
#include <vector>

namespace vsip
{
inline length_type num_processors() VSIP_NOTHROW { return 1;}
inline processor_type local_processor() VSIP_NOTHROW { return 0;}
inline index_type local_processor_index() VSIP_NOTHROW { return 0;}
}

namespace ovxx
{
namespace parallel
{

typedef int par_ll_pbuf_type;
typedef int par_ll_pset_type;

class Communicator 
{
public:
  typedef std::vector<vsip::processor_type> pvec_type;
  pvec_type const &pvec() const { return pvec_;}

  par_ll_pset_type impl_ll_pset() const VSIP_NOTHROW
  { return par_ll_pset_type();}
private:
  pvec_type pvec_;
};

inline void 
create_ll_pset(std::vector<vsip::processor_type> const&,
	       par_ll_pset_type&)
{}

inline void
destroy_ll_pset(par_ll_pset_type&)
{}

inline Communicator &default_communicator()
{
  static Communicator communicator;
  return communicator;
}

// template <reduction_type R, typename T>
// struct reduction_supported { static bool const value = true;};

} // namespace ovxx::parallel
} // namespace ovxx

#endif
