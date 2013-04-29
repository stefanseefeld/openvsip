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

using namespace vsip;
namespace p = vsip_csl::parallel;

/// This task reduces the world that is seen by nested objects
/// to a subset of all available processes. Over its lifetime communication will
/// be confined to the processes participating in this task.
/// After the task has completed, the original communicator will be restored.
///
/// This allows to separate the processes into groups, such that different groups
/// run different operations (MPMD).
struct Task
{
public:
  // Reduces the "world" to the new group, over the lifetime of the task.
  Task(int id, int rank, p::Group const &g)
    : id_(id),
      rank_(rank),
      orig_comm_(p::set_default_communicator(p::Communicator(p::default_communicator(), g)))
  {
  };
  // Switch back to the original communicator.
  ~Task() { p::set_default_communicator(orig_comm_);}
  void wait()
  {
    p::Communicator comm = p::default_communicator();
    if (comm) // if the communicator is valid, do work
    {
      std::cout << "rank " << rank_ << " working on task " << id_ << std::endl;
      // Create a task-specific map, using all available processors.
      Map<> map(num_processors());
      std::cout << "task-specific map uses " << map.num_processors() << " processors" << std::endl;
    }
  };
private:
  int id_;
  int rank_;
  p::Communicator orig_comm_;
};

int
main(int argc, char** argv)
{
  vsipl init(argc, argv);

  p::Communicator comm = p::default_communicator();

  if (comm.size() < 2)
  {
    std::cerr << "Must run with `mpirun -np X` with X >= 2. Terminating." << std::endl;
    return -1;
  }
  p::Group group = comm.group();
  int group1[] = {0, 1};

  // Create a global map, using all available processors.
  Map<> map(num_processors());
  // let everyone in group1 work on task 1...
  {
    Task task1(1, comm.rank(), group.include(group1, group1 + 2));
    task1.wait();
  }
  // (...while pre-allocated objects continue using all processors...)
  std::cout << "global map uses " << map.num_processors() << " processors" << std::endl;
  // ...and everyone else on task 2.
  {
    Task task2(2, comm.rank(), group.exclude(group1, group1 + 2));
    task2.wait();
  }
}
