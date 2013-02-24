/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    benchmarks/alloc_block.cpp
    @author  Jules Bergmann
    @date    2007-02-28
    @brief   VSIPL++ Library: Helper routines and classes for creating
             blocks that use huge pages.

*/

#ifndef BENCHMARK_ALLOC_BLOCK_HPP
#define BENCHMARK_ALLOC_BLOCK_HPP

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <string.h>

#include <vsip/support.hpp>
#include <vsip/domain.hpp>
#include <vsip/dense.hpp>



/***********************************************************************
  Definitions
***********************************************************************/

// Helper class to allocate a block using a given pointer to
// memory, or from the heap.

template <vsip::dimension_type D,
	  typename T,
	  vsip::storage_format_type S,
	  typename M = vsip::Local_map>
struct Alloc_block
{
  typedef typename vsip::impl::Row_major<D>::type order_type;
  typedef vsip::Dense<D, T, order_type, M> block_type;

  static block_type*
  alloc(vsip::Domain<D> const& dom,
	char *addr,
	unsigned long offset,
	M const &map)
  {
    block_type* blk;
    if (addr == NULL)
      blk =  new block_type(dom, T(), map);
    else
    {
      blk = new block_type(dom, (T*)(addr + offset), map);
      blk->admit(false);
      // is_alloc is private // assert(!blk->is_alloc());
    }
    return blk;
  }
};



template <vsip::dimension_type Dim,
	  typename             T,
	  typename             MapT>
struct Alloc_block<Dim, std::complex<T>, vsip::split_complex, MapT>
{
  typedef typename vsip::impl::Row_major<Dim>::type order_type;
  typedef vsip::Dense<Dim, std::complex<T>, order_type, MapT> block_type;

  static block_type*
  alloc(
    vsip::Domain<Dim> const& dom,
    char*                    addr,
    unsigned long            offset,
    MapT const&              map)
  {
    block_type* blk;
    if (addr == NULL)
      blk =  new block_type(dom, T());
    else
    {
      blk = new block_type(dom, 0, 0, map);

      blk->rebind( (T*)(addr + offset) + 0,
		   (T*)(addr + offset) + dom.size());
      blk->admit(false);
      // is_alloc is private // assert(!blk->is_alloc());
    }
    return blk;
  }
};



template <typename T>
struct Alloc_block<1, T, vsip::interleaved_complex, vsip::Local_map>
{
  typedef typename vsip::impl::Row_major<1>::type order_type;
  typedef vsip::impl::Strided<1, T, 
    vsip::Layout<1, order_type, vsip::dense, vsip::interleaved_complex>,
			      vsip::Local_map>
          block_type;

  static block_type*
  alloc(
    vsip::Domain<1> const& dom,
    char*                  addr,
    unsigned long          offset,
    vsip::Local_map const& map)
  {
    block_type* blk;
    if (addr == NULL)
      blk =  new block_type(dom, T(), map);
    else
    {
      blk = new block_type(dom, (T*)(addr + offset), map);
      blk->admit(false);
      // is_alloc is private // assert(!blk->is_alloc());
    }
    return blk;
  }
};



template <typename T>
struct Alloc_block<1, std::complex<T>, vsip::split_complex,
		   vsip::Local_map>
{
  typedef typename vsip::impl::Row_major<1>::type order_type;
  typedef vsip::impl::Strided<1, std::complex<T>,
    vsip::Layout<1, order_type, vsip::dense, vsip::split_complex>,
			      vsip::Local_map>
          block_type;

  static block_type*
  alloc(
    vsip::Domain<1> const& dom,
    char*                    addr,
    unsigned long            offset,
    vsip::Local_map const&   map)
  {
    block_type* blk;
    if (addr == NULL)
      blk =  new block_type(dom, T());
    else
    {
      blk = new block_type(dom, 0, 0, map);

      blk->rebind( (T*)(addr + offset) + 0,
		   (T*)(addr + offset) + dom.size());
      blk->admit(false);
      // is_alloc is private // assert(!blk->is_alloc());
    }
    return blk;
  }
};



/// Create a block, optionally using pointer to memory.
///
/// Allocates a block.  If ADDR is non-zero, block will use memory
/// at ADDR+OFFSET.  Otherwise block will use default allocation.
///
/// Useful for optionally allocating a block using huge page memory.
///
/// Requires
///   DOM to be the domain of the block.
///   ADDR to be either a valid pointer large enough to hold DOM
///     starting at ADDR+OFFSET, or NULL.
///   OFFSET to be offset into ADDR.
///   MAP to be map used for distribution.
///
/// Returns a block pointer.
template <vsip::dimension_type D, typename T, vsip::storage_format_type S, typename M>
typename Alloc_block<D, T, S, M>::block_type*
alloc_block(vsip::Domain<D> const& dom,
	    char *addr, unsigned long offset, M const &map = M())
{
  return Alloc_block<D, T, S, M>::alloc(dom, addr, offset, map);
}



// Overload with default map type of Local_map.
template <vsip::dimension_type D, typename T, vsip::storage_format_type S>
typename Alloc_block<D, T, S, vsip::Local_map>::block_type*
alloc_block(vsip::Domain<D> const &dom, char *addr, unsigned long offset)
{
  return Alloc_block<D, T, S, vsip::Local_map>
    ::alloc(dom, addr, offset, vsip::Local_map());
}



/// Allocate memory in huge page space (that are freed on program
/// termination)
///
/// Requires
///   MEM_FILE to be a filename in the /huge pages directory.
///   PAGES to be the number of pages.
///
/// Returns a pointer to the start of the memory if successful,
///   NULL otherwise.
inline char*
open_huge_pages(char const* mem_file, int pages)
{
  int   fmem;
  char* mem_addr;

  if ((fmem = open(mem_file, O_CREAT | O_RDWR, 0755)) == -1)
  {
    std::cerr << "WARNING: unable to open file " << mem_file
	      << " (errno=" << errno << " " << strerror(errno) << ")\n";
    return 0;
  }

  // Delete file so that huge pages will get freed on program termination.
  remove(mem_file);
	
  mem_addr = (char *)mmap(0, pages * 0x1000000,
			   PROT_READ | PROT_WRITE, MAP_SHARED, fmem, 0);

  if (mem_addr == MAP_FAILED)
  {
    std::cerr << "ERROR: unable to mmap file " << mem_file
	      << " (errno=" << errno << " " << strerror(errno) << ")\n";
    close(fmem);
    return 0;
  }

    // Touch each of the large pages.
    for (int i=0; i<pages; ++i)
      mem_addr[i*0x1000000 + 0x0800000] = (char) 0;

  return mem_addr;
}

#endif // BENCHMARK_ALLOC_BLOCK_HPP
