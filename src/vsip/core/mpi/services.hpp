//
// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef vsip_core_mpi_services_hpp_
#define vsip_core_mpi_services_hpp_

// Only one parallel/xxx.hpp header should be included
#ifdef VSIP_IMPL_PAR_SERVICES_UNIQUE
#  error "Only one parallel/xxx.hpp should be included"
#endif
#define VSIP_IMPL_PAR_SERVICES_UNIQUE

#include <vsip/support.hpp>
#include <vsip/core/reductions/types.hpp>
#include <vsip/core/profile.hpp>
#include <vsip/core/parallel/assign_fwd.hpp>
#include <vsip/core/mpi/communicator.hpp>
#include <vsip/core/mpi/chain_builder.hpp>
#include <vector>
#include <complex>

namespace vsip
{
namespace impl
{
namespace mpi
{

/// Service class for MPI.
class Service
{
public:
  typedef Communicator communicator_type;

  Service(int& argc, char**& argv);
  ~Service();

  /// Set new default communicator, and return the old one.
  static communicator_type set_default_communicator(communicator_type c)
  {
    communicator_type old = *default_communicator_;
    *default_communicator_ = c;
    return old;
  }

  static communicator_type& default_communicator()
  {
    return *default_communicator_;
  }

private:
  static length_type const BUFSIZE = 2*256*256*4;
#if VSIP_IMPL_ENABLE_THREADING
  // GCC 4.4 does not allow non-POD types to use thread-local storage.
  // Thus we store a pointer.
  static thread_local communicator_type *default_communicator_;
#else
  static communicator_type *default_communicator_;
#endif
  bool  initialized_;
  char *buf_;
};


inline Communicator
set_default_communicator(Communicator const &c)
{
  return Service::set_default_communicator(c);
}

} // namespace vsip::impl::mpi

typedef mpi::Service Par_service;
using mpi::Communicator;
using mpi::Chain_builder;
using mpi::free_chain;
typedef Chained_assign par_assign_impl_type;

typedef int par_ll_pbuf_type;
typedef int par_ll_pset_type;

inline void
create_ll_pset(std::vector<processor_type> const&, par_ll_pset_type&) {}

inline void
destroy_ll_pset(par_ll_pset_type&) {}


// supported reductions

template <reduction_type rtype,
	  typename       T>
struct Reduction_supported
{ static bool const value = false; };

template <> struct Reduction_supported<reduce_sum, int>
{ static bool const value = true; };
template <> struct Reduction_supported<reduce_sum, float>
{ static bool const value = true; };

template <> struct Reduction_supported<reduce_all_true, int>
{ static bool const value = true; };

template <> struct Reduction_supported<reduce_all_true_bool, bool>
{ static bool const value = true; };

template <> struct Reduction_supported<reduce_any_true, int>
{ static bool const value = true; };

template <> struct Reduction_supported<reduce_any_true_bool, bool>
{ static bool const value = true; };

} // namespace vsip::impl
} // namespace vsip

#endif
