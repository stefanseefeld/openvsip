/* Copyright (c) 2009 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/

#include <cublas.h>
#include <vsip/opt/cuda/library.hpp>
#include <vsip/opt/cuda/device.hpp>
#include <vsip/opt/cuda/blas.hpp>
#include <vsip/core/argv_utils.hpp>
#include <sstream>
#include <cstring>

namespace vsip
{
namespace impl
{
namespace cuda
{

/// This function must be called prior to any other CUDA-related function.
/// Presently it only initializes the CUDA BLAS library.
void
initialize(int &argc, char **&argv)
{
  // See whether any device id was specified
  char** value = argv;
  int device = -1;
  for (int i=1; i < argc; ++i)
  {
    ++value;
    if (!strcmp(*value, "--vsip-cuda-device"))
    {
      std::istringstream arg(argv[i+1]);
      arg >> device;
      if (!arg)
	VSIP_IMPL_THROW(std::runtime_error("Invalid argument to --vsip-cuda-device"));
      shift_argv(argc, argv, i, 2);
    }
  }
  if (device >= 0)
    set_device(device);

  cuInit(0);

  cublasStatus status = cublasInit();
  ASSERT_CUBLAS_OK();

  if (status != CUBLAS_STATUS_SUCCESS)
  {
    std::ostringstream message;
    message << "CUDA Library failed to initialize (error "
	    << status << ")" << std::endl;
    VSIP_IMPL_THROW(std::runtime_error(message.str()));
  }
}

/// Called to shut down and free all resources
void
finalize()
{
  cublasShutdown();
  ASSERT_CUBLAS_OK();
}

} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip
