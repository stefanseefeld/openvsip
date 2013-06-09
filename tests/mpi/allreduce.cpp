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
#include <numeric>

using namespace ovxx;

template <typename T>
struct generator
{
  generator(int base = 1) : base(base) {}
  T operator()(T p) const { return base + p;}

private:
  T base;
};

template<typename T>
struct sum_sq : public std::binary_function<T, T, T>
{
  T operator()(T const &x, T const &y) const { return x*x+y*y;}
};

template<typename T>
struct maximum : public std::binary_function<T, T, T>
{
  T const &operator()(T const &x, T const &y) const { return x < y? y : x;}
};

template<typename T>
struct minimum : public std::binary_function<T, T, T>
{
  T const &operator()(T const &x, T const &y) const { return x < y? x : y;}
};

template <typename T, reduction_type R> struct operation;
template <typename T> struct operation<T, reduce_sum> : std::plus<T> {};
template <typename T> struct operation<T, reduce_sum_sq> : sum_sq<T> {};
template <typename T> struct operation<T, reduce_max> : maximum<T> {};
template <typename T> struct operation<T, reduce_min> : minimum<T> {};


template<typename T, reduction_type R>
void
allreduce_test(mpi::Communicator &comm, T init)
{
  generator<T> gen;
  T value = gen(comm.rank());
  T result_value = comm.allreduce(R, value);

  // Compute expected result
  std::vector<T> generated_values;
  for (int p = 0; p < comm.size(); ++p)
    generated_values.push_back(gen(p));
  T expected_result = std::accumulate(generated_values.begin(),
				      generated_values.end(),
				      init, operation<T, R>());
  test_assert(result_value == expected_result);
  if (result_value == expected_result && comm.rank() == 0)
    std::cout << "OK." << std::endl;

  comm.barrier();
}

int main(int argc, char* argv[])
{
  library lib(argc, argv);
  mpi::Communicator comm = ovxx::parallel::default_communicator();

  allreduce_test<int, reduce_sum>(comm, 0);
  allreduce_test<float, reduce_max>(comm, 2);
  allreduce_test<double, reduce_min>(comm, 1);
}
