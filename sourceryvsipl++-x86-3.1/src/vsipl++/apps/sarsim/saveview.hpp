/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    example/common/saveview.hpp
    @author  Jules Bergmann
    @date    03/02/2005
    @brief   Utilities for VSIPL++ class examples.
*/

#ifndef COMMON_SAVEVIEW_HPP
#define COMMON_SAVEVIEW_HPP

/***********************************************************************
  Includes
***********************************************************************/

#include <iostream>
#include <fstream>

#include <vsip/vector.hpp>
#include <vsip/matrix.hpp>
#include <vsip/complex.hpp>



/***********************************************************************
  Definitions
***********************************************************************/

// -------------------------------------------------------------------- //
template <typename T>
struct SaveViewTraits
{
   typedef T base_t;
   static unsigned const factor = 1;
};

template <typename T>
struct SaveViewTraits<vsip::complex<T> >
{
   typedef T base_t;
   static unsigned const factor = 2;
};



// -------------------------------------------------------------------- //
template <vsip::dimension_type Dim,
	  typename             T>
class SaveView
{
public:
   typedef typename SaveViewTraits<T>::base_t base_t;
   static unsigned const factor = SaveViewTraits<T>::factor;

   typedef vsip::Dense<Dim, T> block_t;
   typedef typename vsip::impl::view_of<block_t>::type view_t;

public:
   static void save(char*  filename,
		    view_t view)
   {
      vsip::Domain<Dim> dom(get_domain(view));
      base_t*           data(new base_t[factor*dom.size()]);

      block_t           block(dom, data);
      view_t            store(block);

      FILE*  fd;
      size_t size = dom.size();

      if (!(fd = fopen(filename,"w"))) {
	 fprintf(stderr,"SaveView: error opening '%s'.\n", filename);
	 exit(1);
	 }

      block.admit(false);
      store = view;
      block.release(true);

      if (size != fwrite(data, sizeof(T), size, fd)) {
	 fprintf(stderr, "SaveView: Error writing.\n");
	 exit(1);
	 }

      fclose(fd);
   }

private:
   template <typename T1,
	     typename Block1>
   static vsip::Domain<1> get_domain(vsip::const_Vector<T1, Block1> view)
   { return vsip::Domain<1>(view.size()); }

   template <typename T1,
	     typename Block1>
   static vsip::Domain<2> get_domain(vsip::const_Matrix<T1, Block1> view)
   { return vsip::Domain<2>(view.size(0), view.size(1)); }
};


template <typename T,
	  typename Block>
void
save_view(
   char*                  filename,
   vsip::const_Vector<T, Block> view)
{
   SaveView<1, T>::save(filename, view);
}



template <typename T,
	  typename Block>
void
save_view(
   char*                  filename,
   vsip::const_Matrix<T, Block> view)
{
   SaveView<2, T>::save(filename, view);
}

#endif
