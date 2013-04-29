/* Copyright (c) 2010 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/// Description
///   CUDA module access API

#ifndef vsip_opt_cuda_module_hpp_
#define vsip_opt_cuda_module_hpp_

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

#include <vsip/opt/cuda/exception.hpp>
#include <vsip/core/noncopyable.hpp>
#include <cuda.h>

namespace vsip
{
namespace impl
{
namespace cuda
{

class Function
{
public:
  struct Block
  {
    Block(int xx, int yy, int zz) : x(xx), y(yy), z(zz) {}
    int x, y, z;
  };
  struct Grid
  {
    Grid(int xx, int yy) : x(xx), y(yy) {}
    int x, y;
  };

  Function(CUfunction func, std::string const &symbol)
    : function_(func), symbol_(symbol) {}

  template <typename A1>
  void operator()(Block const &b, A1 a1)
  {
    set_block_shape(b.x, b.y, b.z);
    param_set(0, a1);
    param_set_size(sizeof(A1));
    launch();
  }
  template <typename A1, typename A2>
  void operator()(Block const &b, A1 a1, A2 a2)
  {
    set_block_shape(b.x, b.y, b.z);
    unsigned size = 0;
    param_set(size, a1);
    size += sizeof(A1);
    param_set(size, a2);
    size += sizeof(A2);
    param_set_size(size);
    launch();
  }
  template <typename A1, typename A2, typename A3>
  void operator()(Block const &b, A1 a1, A2 a2, A3 a3)
  {
    set_block_shape(b.x, b.y, b.z);
    unsigned size = 0;
    param_set(size, a1);
    size += sizeof(A1);
    param_set(size, a2);
    size += sizeof(A2);
    param_set(size, a3);
    size += sizeof(A3);
    param_set_size(size);
    launch();
  }
  template <typename A1, typename A2, typename A3, typename A4>
  void operator()(Block const &b, A1 a1, A2 a2, A3 a3, A4 a4)
  {
    set_block_shape(b.x, b.y, b.z);
    unsigned size = 0;
    param_set(size, a1);
    size += sizeof(A1);
    param_set(size, a2);
    size += sizeof(A2);
    param_set(size, a3);
    size += sizeof(A3);
    param_set(size, a4);
    size += sizeof(A4);
    param_set_size(size);
    launch();
  }
  template <typename A1, typename A2, typename A3, typename A4, typename A5>
  void operator()(Block const &b, A1 a1, A2 a2, A3 a3, A4 a4, A5 a5)
  {
    set_block_shape(b.x, b.y, b.z);
    unsigned size = 0;
    param_set(size, a1);
    size += sizeof(A1);
    param_set(size, a2);
    size += sizeof(A2);
    param_set(size, a3);
    size += sizeof(A3);
    param_set(size, a4);
    size += sizeof(A4);
    param_set(size, a5);
    size += sizeof(A5);
    param_set_size(size);
    launch();
  }
  template <typename A1, typename A2, typename A3, typename A4, typename A5, typename A6>
  void operator()(Block const &b, A1 a1, A2 a2, A3 a3, A4 a4, A5 a5, A6 a6)
  {
    set_block_shape(b.x, b.y, b.z);
    unsigned size = 0;
    param_set(size, a1);
    size += sizeof(A1);
    param_set(size, a2);
    size += sizeof(A2);
    param_set(size, a3);
    size += sizeof(A3);
    param_set(size, a4);
    size += sizeof(A4);
    param_set(size, a5);
    size += sizeof(A5);
    param_set(size, a6);
    size += sizeof(A6);
    param_set_size(size);
    launch();
  }


  void set_block_shape(int x, int y, int z)
  { 
    VSIP_IMPL_CUDA_CHECK_RESULT(cuFuncSetBlockShape, (function_, x, y, z)); 
  }
  void set_shared_size(unsigned int bytes)
  { 
    VSIP_IMPL_CUDA_CHECK_RESULT(cuFuncSetSharedSize, (function_, bytes)); 
  }

  void param_set_size(unsigned int bytes)
  { 
    VSIP_IMPL_CUDA_CHECK_RESULT(cuParamSetSize, (function_, bytes)); 
  }
  void param_set(unsigned int offset, unsigned int value)
  { 
    VSIP_IMPL_CUDA_CHECK_RESULT(cuParamSeti, (function_, offset, value)); 
  }
  void param_set(unsigned int offset, unsigned long value)
  { 
    VSIP_IMPL_CUDA_CHECK_RESULT(cuParamSeti, (function_, offset, value)); 
  }
  void param_set(unsigned int offset, float value)
  { 
    VSIP_IMPL_CUDA_CHECK_RESULT(cuParamSetf, (function_, offset, value)); 
  }
  void param_setv(unsigned int offset, void *buf, unsigned long len)
  { 
    VSIP_IMPL_CUDA_CHECK_RESULT(cuParamSetv, (function_, offset, buf, len)); 
  }
  template <typename T>
  void param_set(unsigned int offset, T *value)
  { 
    param_setv(offset, &value, sizeof(T*));
  }
  // void param_set_texref(const texture_reference &tr)
  // { 
  //   VSIP_IMPL_CUDA_CHECK_RESULT(cuParamSetTexRef, (function_, CU_PARAM_TR_DEFAULT, tr)); 
  // }

  void launch()
  { 
    VSIP_IMPL_CUDA_CHECK_RESULT(cuLaunch, (function_)); 
  }
  void launch_grid(int grid_width, int grid_height)
  { 
    VSIP_IMPL_CUDA_CHECK_RESULT(cuLaunchGrid, (function_, grid_width, grid_height)); 
  }
  // void launch_grid_async(int grid_width, int grid_height, stream const &s)
  // { 
  //   VSIP_IMPL_CUDA_CHECK_RESULT(cuLaunchGridAsync, (function_, grid_width, grid_height, s));
  // }

  int get_attribute(CUfunction_attribute attr) const
  {
    int result;
    VSIP_IMPL_CUDA_CHECK_RESULT(cuFuncGetAttribute, (&result, attr, function_));
    return result;
  }

  void set_cache_config(CUfunc_cache fc)
  {
    VSIP_IMPL_CUDA_CHECK_RESULT(cuFuncSetCacheConfig, (function_, fc));
  }
private:
  CUfunction function_;
  std::string symbol_;
};

class Module : Non_copyable
{
public:
  Module(char const *filename)
  {
    VSIP_IMPL_CUDA_CHECK_RESULT(cuModuleLoad, (&module_, filename));
  }
  Module(void const *image)
  {
    VSIP_IMPL_CUDA_CHECK_RESULT(cuModuleLoadData, (&module_, image));
  }
  ~Module()
  {
    VSIP_IMPL_CUDA_CHECK_RESULT(cuModuleUnload, (module_));
  }
  operator CUmodule () const { return module_;}
  
  Function get_function(char const *name)
  {
    CUfunction func;
    VSIP_IMPL_CUDA_CHECK_RESULT(cuModuleGetFunction, (&func, module_, name));
    return Function(func, name);
  }

  void get_global(char const *name, CUdeviceptr &ptr, unsigned int &bytes)
  {
    VSIP_IMPL_CUDA_CHECK_RESULT(cuModuleGetGlobal, (&ptr, &bytes, module_, name));
  }
  
private:
  CUmodule module_;
};

} // namespace vsip::impl::cuda
} // namespace vsip::impl
} // namespace vsip


#endif
