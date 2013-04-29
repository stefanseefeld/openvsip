/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#include <iostream>
#include <fstream>
#include <cerrno>
#include <string>

#include <vsip/initfin.hpp>
#include <vsip/math.hpp>
#include <vsip/signal.hpp>
#include <vsip_csl/output.hpp>
#include <vsip_csl/memory_pool.hpp>
#include <vsip_csl/profile.hpp>
#include <vsip_csl/parallel.hpp>

int loop = 1;
int nn = 1024;

void
process_vadd_options(int argc, char** argv)
{

  for (int i=1; i<argc; ++i)
  {
    if (!strcmp(argv[i], "-loop")) loop = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-N")) nn = atoi(argv[++i]);
    else
    {
      std::cerr << "Unknown arg: " << argv[i] << std::endl;
      std::cerr << "Usage: " << argv[0] << " [-loop n] [-N n] " << std::endl;
      exit(-1);
    }
  }

}


int
main(int argc, char** argv)
{

  vsip::vsipl init(argc, argv);

  process_vadd_options(argc, argv);

  vsip_csl::parallel::Communicator comm = vsip_csl::parallel::default_communicator();

  int r = comm.rank();

#if 0
  pid_t pid = getpid();

  std::cout << "rank: "   << r
	    << "  size: " << comm.size()
	    << "  pid: "  << pid 
	    << std::endl;

  // Enable this section for easier debugging.
  // Stop each process, allow debugger to be attached.
  char c;
  if (r == 0) std::cerr >> c;
#endif
  comm.barrier();

  vsip::length_type const N = nn;

//cout << "[" << r << "] start, N=" << N << endl;
  //
  // Declare a map type to control the partitioning of vectors
  // among several processors.  We don't know the exact number
  // until run-time.
  //
  typedef vsip::Map<vsip::Block_dist> vector_map_type;
  //
  // Declare an instance of that map type and initialize it
  // with the number of processors.
  //
  vector_map_type map = vector_map_type(vsip::num_processors());
  //
  // Declare a block type based on our map type.
  //
  typedef vsip::Dense<1, float, vsip::row2_type, vector_map_type> vector_block_type;
  //
  // Declare a vector view based on our block type.
  //
  typedef vsip::Vector<float, vector_block_type> vector_type;
  //
  // Finally, declare vectors of our view type, using the
  // run-time knowledge encapsulated in our map instance.
  //
  vector_type A(N,map);
  vector_type B(N,map);
  vector_type C(N,map);
  //
  // Initialize the vectors.  Since our algorithm is A=B+C, the
  // result in A(i) is 3*i.
  //
  A = 0.0f;
  B = vsip::ramp(0.0f,1.0f,N);
  C = vsip::ramp(0.0f,2.0f,N);
  //
  // Create "local" views of the full vectors.  Each one
  // will see only the partition distributed to the
  // processor on which it is created.
  //
  vector_type::local_type A_local = A.local();
  vector_type::local_type B_local = B.local();
  vector_type::local_type C_local = C.local();
  //
  // Note the number of elements in the distributed
  // partitions.
  //
  vsip::length_type asize = A_local.size(0);
  
  using vsip_csl::profile::Scope;
  using vsip_csl::profile::user;
  // 
  // Run the algorithm over and over and collect timings.
  // 
  vsip_csl::profile::Acc_timer t1;
  vsip::Vector<double> process_time(loop);
  comm.barrier();
  for (int l=0; l<loop; ++l)
  {
    t1.start();
    {
      Scope<user> scope("algorithm", asize*sizeof(float));
      //
      // Run our simple algorithm.  Add partitions from vectors
      // B and C and write the result into C's partition.
      //
      A_local = B_local + C_local;
    }
    t1.stop();
    process_time.put(l, t1.delta());
  }
  comm.barrier();
  if (r==0 && loop>0)
    printf("Mean time: %12.10f\n", vsip::meanval(process_time));
  //
  // Declare a globally mapped vector where each processor
  // can store its results.
  //
  vsip::Vector<float,
    vsip::Dense<1, float, vsip::row2_type, vsip::Replicated_map<1> > >
    R(N);
  //
  // Copy the distributed vector to the global vector.
  //
  R = A;
  //
  // Wait for all processors to arrive here.
  //
  comm.barrier();
//cout << "[" << r << "] finished" << endl;
  //
  // Processor 0 checks results.  It looks into the non-local
  // view to see all elements of A.
  //
  if (r == 0)
  {
//  cout << "[" << r << "] " << "checking ..." << endl;
    for (vsip::index_type i=0; i<N; ++i)
    {
      float x = 3.0f*i;
      if (R(i) != x)
      {
	std::cout << "[" << r << "] " << "A(" << i << ") should be " << x << " but is " << R(i) << std::endl;
        break;
      }
    }
  }
  comm.barrier();

#if 0
  // Display statistics
  if (loop > 0)
  {
    Index<1> idx;
    double mean = vsip::meanval(process_time);
    std::cout << "[" << r << "] " << "loops:   " << loop << std::endl;
    std::cout << "[" << r << "] " << "mean:    " << mean << std::endl;
    std::cout << "[" << r << "] " << "min:     " << vsip::minval(process_time, idx) << std::endl;
    std::cout << "[" << r << "] " << "max:     " << vsip::maxval(process_time, idx) << std::endl;
    std::cout << "[" << r << "] " << "std-dev: " << sqrt(vsip::meansqval(process_time - mean)) << std::endl;
  }
#endif

  return 0;
}
