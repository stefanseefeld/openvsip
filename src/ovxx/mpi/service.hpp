// Copyright (c) 2005, 2006 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_mpi_service_hpp_
#define ovxx_mpi_service_hpp_

#include <ovxx/support.hpp>
#include <ovxx/reductions/types.hpp>
#include <ovxx/parallel/assign_fwd.hpp>
#include <ovxx/mpi/communicator.hpp>
#include <ovxx/mpi/chain_builder.hpp>
#include <vector>

namespace vsip
{
length_type num_processors() VSIP_NOTHROW;
processor_type local_processor() VSIP_NOTHROW;
index_type local_processor_index() VSIP_NOTHROW;
}

namespace ovxx
{
namespace mpi
{

void initialize(int &, char **&);
void finalize(bool);

} // namespace mpi

namespace parallel
{
using mpi::Communicator;
using mpi::Group;
using mpi::Chain_builder;
using mpi::free_chain;

typedef int ll_pbuf_type;
typedef int ll_pset_type;

/// Set new default communicator, and return the old one.
Communicator set_default_communicator(Communicator c);
Communicator &default_communicator();

inline void
create_ll_pset(std::vector<processor_type> const&, ll_pset_type&) {}

inline void
destroy_ll_pset(ll_pset_type&) {}


// supported reductions

template <reduction_type rtype, typename T>
struct reduction_supported
{ static bool const value = false;};

template <> struct reduction_supported<reduce_sum, int>
{ static bool const value = true;};
template <> struct reduction_supported<reduce_sum, float>
{ static bool const value = true;};
template <> struct reduction_supported<reduce_all_true, int>
{ static bool const value = true;};
template <> struct reduction_supported<reduce_all_true_bool, bool>
{ static bool const value = true;};
template <> struct reduction_supported<reduce_any_true, int>
{ static bool const value = true;};
template <> struct reduction_supported<reduce_any_true_bool, bool>
{ static bool const value = true;};
} // namespace ovxx::parallel
} // namespace ovxx

#endif
