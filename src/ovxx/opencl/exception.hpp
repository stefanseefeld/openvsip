//
// Copyright (c) 2014 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_opencl_exception_hpp_
#define ovxx_opencl_exception_hpp_

#include <ovxx/support.hpp>
#include <string>
#include <CL/cl.h>

namespace ovxx
{
namespace opencl
{

class exception : public std::exception
{
 public:
  exception(char const *routine, int code)
  : routine_(routine), code_(code)
  {
    msg_.append(routine_);
    msg_.append(" : ");
    msg_.append(error_string(code_));
  }

  virtual ~exception() throw() {}
  virtual char const *what() const throw() { return this->msg_.c_str();}
  char const *routine() const { return routine_;}
  int result_code() const { return code_;}

 private:
  static std::string error_string(int c)
  {
    switch (c)
    {
      case CL_SUCCESS: return "success";
      case CL_DEVICE_NOT_FOUND: return "device not found";
      case CL_DEVICE_NOT_AVAILABLE: return "device not available";
      case CL_COMPILER_NOT_AVAILABLE: return "compiler not available";
      case CL_INVALID_VALUE: return "invalid value";
      case CL_INVALID_DEVICE_TYPE: return "invalid device type";
      case CL_INVALID_PLATFORM: return "invalid platform";
      case CL_INVALID_DEVICE: return "invalid device";
      case CL_INVALID_CONTEXT: return "invalid context";
      case CL_INVALID_MEM_OBJECT: return "invalid memory object";
      case CL_INVALID_KERNEL_NAME: return "invalid kernel name";
      default:
      {
	std::ostringstream oss;
	oss << "unknown (" << c << ")";
	return oss.str();
      }
    }
  }
  /// The OpenCL routine that triggered the error
  char const *routine_;

  /// The failed result code reported by the OpenCL implementation.
  int code_;

  /// The formatted error message
  std::string msg_;
};

#define OVXX_OPENCL_CHECK_RESULT(Func, Args)	       \
{						       \
  int check_result_ = Func Args;		       \
  if (check_result_ != CL_SUCCESS)		       \
    OVXX_DO_THROW(exception(#Func, check_result_));    \
}

} // namespace ovxx::mpi
} // namespace ovxx

#endif
