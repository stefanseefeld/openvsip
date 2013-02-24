/* Copyright (c) 2010 by CodeSourcery, Inc.  All rights reserved. */

#ifndef vsip_opt_cuda_exception_hpp_
#define vsip_opt_cuda_exception_hpp_

#include <vsip/support.hpp>
#include <string>
#include <stdexcept>
#include <cuda.h>

namespace vsip
{
namespace impl
{
namespace cuda
{

class exception : public std::runtime_error
{
 public:
  exception(char const *routine, CUresult code)
    : std::runtime_error(make_message(routine, code)),
      routine_(routine), code_(code)
  {}

  virtual ~exception() throw() {}
  char const *routine() const { return routine_;}
  CUresult result_code() const { return code_;}

 private:
  static std::string make_message(char const *routine, CUresult c)
  {
    std::string result = routine;
    result += " failed: ";
    result += curesult_to_str(c);
    return result;
  }
  static char const *curesult_to_str(CUresult e)
  {
    switch (e)
    {
    case CUDA_SUCCESS: return "success";
    case CUDA_ERROR_INVALID_VALUE: return "invalid value";
    case CUDA_ERROR_OUT_OF_MEMORY: return "out of memory";
    case CUDA_ERROR_NOT_INITIALIZED: return "not initialized";
    case CUDA_ERROR_DEINITIALIZED: return "deinitialized";
    case CUDA_ERROR_NO_DEVICE: return "no device";
    case CUDA_ERROR_INVALID_DEVICE: return "invalid device";

    case CUDA_ERROR_INVALID_IMAGE: return "invalid image";
    case CUDA_ERROR_INVALID_CONTEXT: return "invalid context";
    case CUDA_ERROR_CONTEXT_ALREADY_CURRENT: return "context already current";
    case CUDA_ERROR_MAP_FAILED: return "map failed";
    case CUDA_ERROR_UNMAP_FAILED: return "unmap failed";
    case CUDA_ERROR_ARRAY_IS_MAPPED: return "array is mapped";
    case CUDA_ERROR_ALREADY_MAPPED: return "already mapped";
    case CUDA_ERROR_NO_BINARY_FOR_GPU: return "no binary for gpu";
    case CUDA_ERROR_ALREADY_ACQUIRED: return "already acquired";
    case CUDA_ERROR_NOT_MAPPED: return "not mapped";
    case CUDA_ERROR_NOT_MAPPED_AS_ARRAY: return "not mapped as array";
    case CUDA_ERROR_NOT_MAPPED_AS_POINTER: return "not mapped as pointer";
    case CUDA_ERROR_ECC_UNCORRECTABLE: return "ECC uncorrectable";
    case CUDA_ERROR_INVALID_SOURCE: return "invalid source";
    case CUDA_ERROR_FILE_NOT_FOUND: return "file not found";

    case CUDA_ERROR_INVALID_HANDLE: return "invalid handle";

    case CUDA_ERROR_NOT_FOUND: return "not found";

    case CUDA_ERROR_NOT_READY: return "not ready";

    case CUDA_ERROR_LAUNCH_FAILED: return "launch failed";
    case CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES: return "launch out of resources";
    case CUDA_ERROR_LAUNCH_TIMEOUT: return "launch timeout";
    case CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING: return "launch incompatible texturing";

    case CUDA_ERROR_POINTER_IS_64BIT: return "pointer is 64-bit";
    case CUDA_ERROR_SIZE_IS_64BIT: return "size is 64-bit";

    case CUDA_ERROR_UNKNOWN: return "unknown";

    default: return "invalid error code";
    }
  }

  /// The CUDA routine that triggered the error
  char const *routine_;

  /// The failed result code reported by CUDA.
  CUresult code_;
};

#define VSIP_IMPL_CUDA_CHECK_RESULT(Func, Args)			\
{								\
  CUresult check_result_ = Func Args;				\
  if (check_result_ != CUDA_SUCCESS)				\
    VSIP_IMPL_THROW(exception(#Func, check_result_));		\
}

} // namespace vsip::impl::mpi
} // namespace vsip::impl
} // namespace vsip

#endif
