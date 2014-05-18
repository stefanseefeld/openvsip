#include <vsip/initfin.hpp>
#include <ovxx/opencl.hpp>
#include <iostream>

void usage(char const *name)
{
  std::cout << "Usage: " << name << " [options]\n"
	    << "Options:\n"
	    << " --help               Display this information.\n"
	    << " --version            Display version information.\n"
	    << " -p N, --platform N   Platform index.\n"
	    << " -d N, --device N     Device index.\n"
	    << " -t <type>            Select device type.\n"
	    << "                      options: default, cpu, gpu, accelerator, all"
	    << std::endl;
};

void version(char const *name)
{
  std::cout << name << ' ' << PACKAGE_VERSION << " (" << PACKAGE_STRING << ")\n"
	    << " Copyright (C) 2014 Stefan Seefeld" << std::endl;
}

int
main(int argc, char **argv)
{
  namespace ocl = ovxx::opencl;
  int pid = -1;
  int did = -1;
  cl_device_type t = ocl::device::default_;

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
    else if (arg == "-p" || arg == "--platform")
    {
      pid = atoi(argv[++i]);
    }
    else if (arg == "-d" || arg == "--device")
    {
      did = atoi(argv[++i]);
    }
    else if (arg == "-t")
    {
      std::string type(argv[++i]);
      if (type == "cpu") t = ocl::device::cpu;
      else if (type == "gpu") t = ocl::device::gpu;
      else if (type == "accelerator") t = ocl::device::accelerator;
      else if (type == "all") t = ocl::device::all;
    }
    else if (arg[0] != '-')
    {
      usage(argv[0]);
      return -1;
    }
    else break;
  }

  vsip::vsipl library(argc, argv);

  std::vector<ocl::platform> platforms = ocl::platform::platforms();
  if (!platforms.size())
  {
    std::cout << "no OpenCL platforms found" << std::endl;
    return 0;
  }
  if (pid == -1)
  {
    std::cout << "Available platforms :\n";
    for (size_t p = 0; p != platforms.size(); ++p)
    {
      std::cout << "  " << p << ':'
		<< " Name : " << platforms[p].name() 
		<< " (" << platforms[p].version() 
		<< "), Vendor: " << platforms[p].vendor() << std::endl;
    }
    return 0;
  }
  else if (pid > platforms.size() - 1)
  {
    std::cerr << "invalid platform selected" << std::endl;
    return -1;
  }
  ocl::platform pl = platforms[pid];
  std::vector<ocl::device> devices = pl.devices(t);
  if (did == -1)
  {
    if (!devices.size())
    {
      std::cout << "no devices found" << std::endl;
      return 0;
    }
    else if (devices.size() == 1)
      std::cout << "1 device found :" << std::endl;
    else
      std::cout << devices.size() << " devices found :" << std::endl;
    // If no device is selected print a summary and exit
    for (size_t d = 0; d != devices.size(); ++d)
    {
      std::cout << "  " << d << ':'
		<< " Name : " << devices[d].name()
		<< " (" << (devices[d].available() ? "available" : "not available") << ")\n";
    }
    return 0;
  }
  else if (did >= devices.size())
  {
    std::cerr << "invalid device selected" << std::endl;
    return -1;
  }
  ocl::device dev = devices[did];
  std::cout << "Name : " << dev.name() << '\n'
	    << "Profile : " << dev.profile() << '\n'
	    << "available : " << (dev.available() ? "yes" : "no") << '\n'
	    << "compiler available : " << (dev.compiler_available() ? "yes" : "no") << '\n'
	    << "extensions : \n";
  std::vector<std::string> extensions = dev.extensions();
  for (int e = 0; e != extensions.size(); ++e)
    std::cout << "  " << extensions[e] << '\n';
}
