/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    loadview.hpp
    @author  Jules Bergmann
    @date    16 Jun 2005
    @brief   VSIPL++ Library: Load views from file.
*/


// -------------------------------------------------------------------- //
template <typename T>
struct LoadViewTraits
{
   typedef T base_t;
   static unsigned const factor = 1;
};

template <typename T>
struct LoadViewTraits<vsip::complex<T> >
{
   typedef T base_t;
   static unsigned const factor = 2;
};


// -------------------------------------------------------------------- //
template <vsip::dimension_type Dim,
	  typename             T>
class LoadView
{
public:
   typedef typename LoadViewTraits<T>::base_t base_t;
   static unsigned const factor = LoadViewTraits<T>::factor;

   typedef vsip::Dense<Dim, T> block_t;
   typedef typename vsip::impl::view_of<block_t>::type view_t;

public:
   LoadView(char*                    filename,
	    vsip::Domain<Dim> const& dom,
	    bool                     twizzle = true)
    : data_  (new base_t[factor*dom.size()]),
      block_ (dom, data_),
      view_  (block_)
   {
      FILE*  fd;
      size_t size = dom.size();

      if (!(fd = fopen(filename,"r"))) {
	 fprintf(stderr,"LoadView: error opening '%s'.\n", filename);
	 exit(1);
	 }

      if (twizzle) {
	 if (size != fread_bo(data_, sizeof(T), size, fd, sizeof(base_t))) {
	    fprintf(stderr, "Error reading I/Q even coefficients.\n");
	    exit(1);
	    }
	 }
      else {
	 if (size != fread(data_, sizeof(T), size, fd)) {
	    fprintf(stderr, "Error reading I/Q even coefficients.\n");
	    exit(1);
	    }
	 }

      fclose(fd);

      block_.admit(true);
   }
   LoadView(FILE*              fd,
	    vsip::Domain<Dim> const& dom)
    : data_  (new base_t[factor*dom.size()]),
      block_ (dom, data_),
      view_  (block_)
   {
      size_t size = dom.size();

      if (size != fread_bo(data_, sizeof(T), size, fd, sizeof(base_t))) {
	 fprintf(stderr, "Error reading I/Q even coefficients.\n");
	 exit(1);
	 }

      block_.admit(true);
   }

   ~LoadView()
   {
      delete[] data_;
   }

   view_t view() { return view_; }

private:
   base_t*       data_;

   block_t       block_;
   view_t        view_;
};
