//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_opencl_device_hpp_
#define ovxx_opencl_device_hpp_

#include <ovxx/opencl/exception.hpp>
#include <vector>
#include <string>
#include <iterator>

namespace ovxx
{
namespace opencl
{
namespace detail
{
template <typename T>
T info(cl_device_id id, cl_device_info i)
{
  T param;
  OVXX_OPENCL_CHECK_RESULT(clGetDeviceInfo, (id, i, sizeof(param), &param, 0));
  return param;
}
template <>
std::string info(cl_device_id id, cl_device_info i)
{
  char param[1024];
  OVXX_OPENCL_CHECK_RESULT(clGetDeviceInfo, (id, i, sizeof(param), param, 0));
  return std::string(param);
}
} // namespace ovxx::opencl::detail

class device
{
public:
  enum type 
  {
    default_ = CL_DEVICE_TYPE_DEFAULT,
    cpu = CL_DEVICE_TYPE_CPU,
    gpu = CL_DEVICE_TYPE_GPU,
    accelerator = CL_DEVICE_TYPE_ACCELERATOR,
    all = CL_DEVICE_TYPE_ALL
  };

  device(cl_device_id id) : id_(id) {}

  std::string profile() const { return info<std::string>(CL_DEVICE_PROFILE);}
  std::string name() const { return info<std::string>(CL_DEVICE_NAME);}

  cl_uint address_bits() const { return info<cl_uint>(CL_DEVICE_ADDRESS_BITS);}
  bool available() const { return info<cl_bool>(CL_DEVICE_AVAILABLE);}
  bool compiler_available() const { return info<cl_bool>(CL_DEVICE_COMPILER_AVAILABLE);}
  bool little_endian() const { return info<cl_bool>(CL_DEVICE_ENDIAN_LITTLE);}
  std::vector<std::string> extensions() const
  {
    std::string ext = info<std::string>(CL_DEVICE_EXTENSIONS);
    std::istringstream buf(ext);
    std::istream_iterator<std::string> beg(buf), end;
    return std::vector<std::string>(beg, end);
  }
  cl_ulong global_mem_cache_size() const
  { return info<cl_ulong>(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE);}
  cl_ulong global_mem_cacheline_size() const
  { return info<cl_ulong>(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE);}
  cl_ulong global_mem_size() const
  { return info<cl_ulong>(CL_DEVICE_GLOBAL_MEM_SIZE);}
  cl_ulong local_mem_size() const
  { return info<cl_ulong>(CL_DEVICE_LOCAL_MEM_SIZE);}
  bool image_support() const { return info<cl_bool>(CL_DEVICE_IMAGE_SUPPORT);}
  cl_uint max_clock_frequency() const
  { return info<cl_uint>(CL_DEVICE_MAX_CLOCK_FREQUENCY);}
  cl_uint max_compute_units() const
  { return info<cl_uint>(CL_DEVICE_MAX_COMPUTE_UNITS);}
  cl_uint max_constant_args() const
  { return info<cl_uint>(CL_DEVICE_MAX_CONSTANT_ARGS);}
  cl_ulong max_constant_buffer_size() const
  { return info<cl_ulong>(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE);}
  cl_ulong max_mem_alloc_size() const
  { return info<cl_ulong>(CL_DEVICE_MAX_MEM_ALLOC_SIZE);}
  size_t max_parameter_size() const
  { return info<size_t>(CL_DEVICE_MAX_PARAMETER_SIZE);}

  operator cl_device_id() const { return id_;}
private:
  template <typename T>
  T info(cl_device_info i) const { return detail::info<T>(id_, i);}

  cl_device_id id_;
};
} // namespace ovxx::opencl
} // namespace ovxx

#endif
