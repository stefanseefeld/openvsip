#include <vsip/initfin.hpp>
#include <vsip_csl/cuda.hpp>
#include <iostream>

void usage(char const *name)
{
  std::cout << "Usage: " << name << " [options]\n"
	    << "Options:\n"
	    << " --help               Display this information.\n"
	    << " --version            Display version information.\n"
	    << " --vsip-cuda-device N Set the active device to be N."
	    << std::endl;
};

void version(char const *name)
{
  std::cout << name << ' ' << VSIP_IMPL_MAJOR_VERSION_STRING 
	    << " (Sourcery VSIPL++ " << VSIP_IMPL_VERSION_STRING << ")\n"
	    << " Copyright (C) 2010 CodeSourcery, Inc." << std::endl;
}

int
main(int argc, char **argv)
{
  using namespace vsip_csl::cuda;

  for (int i = 1; i < argc; ++i)
  {
    std::string arg = argv[i];
    if (arg == "-h" || arg == "--help")
    {
      usage(argv[0]);
      return 0;
    }
    else if (arg == "-v" || arg == "--version")
    {
      version(argv[0]);
      return 0;
    }
    else if (arg[0] != '-')
    {
      usage(argv[0]);
      return -1;
    }
    else break;
  }

  vsip::vsipl library(argc, argv);

  int devices = num_devices();
  for (int i = 0; i != devices; ++i)
    std::cout << "CUDA device " << i << " : " << get_device(i).name() << std::endl;

  std::cout << "current device is " << get_device().name() << std::endl;
}
