//
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

// These tests are inspired by boost.mpi

#include <ovxx/library.hpp>
#include <ovxx/mpi/communicator.hpp>
#include <test.hpp>

using namespace ovxx;

template<typename T>
void
broadcast_test(mpi::Communicator &comm, T *values, size_t length,
               char const *kind, int root = -1)
{
  if (root == -1) 
  {
    for (root = 0; root < comm.size(); ++root)
      broadcast_test(comm, values, length, kind, root);
  }
  else
  {
    T *local_values = new T[length];
    if (comm.rank() == root) 
      std::copy(values, values + length, local_values);

    comm.broadcast(root, local_values, length);
    test_assert(std::equal(local_values, local_values + length, values));
    delete [] local_values;
  }

  comm.barrier();
}

int main(int argc, char* argv[])
{
  library lib(argc, argv);
  mpi::Communicator comm = ovxx::parallel::default_communicator();
  if (comm.size() == 1) 
  {
    std::cerr << "ERROR: Must run the broadcast test with more than one "
              << "process." << std::endl;
    MPI_Abort(comm, -1);
  }

  int iarray[] = { 0, 1, 2, 3, 4, 5, 6, 7};
  broadcast_test(comm, iarray, 8, "int");
  float farray[] = { 0., 1., 2., 3., 4., 5., 6., 7.};
  broadcast_test(comm, farray, 8, "float");
  double darray[] = { 0., 1., 2., 3., 4., 5., 6., 7.};
  broadcast_test(comm, darray, 8, "float");

}
