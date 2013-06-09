//
// Copyright (c) 2010 CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is part of OpenVSIP. It is made available under the
// license contained in the accompanying LICENSE.BSD file.

#ifndef ovxx_mpi_exception_hpp_
#define ovxx_mpi_exception_hpp_

#include <ovxx/support.hpp>
#include <string>
#include <mpi.h>

namespace ovxx
{
namespace mpi
{

class exception : public std::exception
{
 public:
  exception(char const *routine, int code)
  : routine_(routine), code_(code)
  {
    // Query the MPI implementation for its reason for failure
    char buffer[MPI_MAX_ERROR_STRING];
    int len;
    MPI_Error_string(code, buffer, &len);

    // Construct the complete error message
    msg_.append(routine_);
    msg_.append(": ");
    msg_.append(buffer, len);
  }

  virtual ~exception() throw() {}
  virtual char const *what() { return this->msg_.c_str();}
  char const *routine() const { return routine_;}
  int result_code() const { return code_;}
  int error_class() const 
  { 
    int result;
    MPI_Error_class(code_, &result);
    return result;
  }

 protected:
  /// The MPI routine that triggered the error
  char const *routine_;

  /// The failed result code reported by the MPI implementation.
  int code_;

  /// The formatted error message
  std::string msg_;
};

#define OVXX_MPI_CHECK_RESULT(MPIFunc, Args)           \
{						       \
  int check_result_ = MPIFunc Args;		       \
  if (check_result_ != MPI_SUCCESS)		       \
    OVXX_DO_THROW(exception(#MPIFunc, check_result_)); \
}

} // namespace ovxx::mpi
} // namespace ovxx

#endif
