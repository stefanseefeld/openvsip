/* Copyright (c) 2007, 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    vsip/opt/diag/fft.hpp
    @author  Jules Bergmann
    @date    2007-03-27
    @brief   VSIPL++ Library: Diagnostics for Fft.
*/

#ifndef VSIP_OPT_DIAG_FFT_HPP
#define VSIP_OPT_DIAG_FFT_HPP

#if VSIP_IMPL_REF_IMPL
# error "vsip/opt files cannot be used as part of the reference impl."
#endif

/***********************************************************************
  Included Files
***********************************************************************/

#include <iostream>
#include <iomanip>



/***********************************************************************
  Declarations
***********************************************************************/

namespace vsip
{
namespace impl
{



namespace diag_detail
{

// Helper class to return the name corresponding to a dispatch tag.

template <typename T> 
struct Class_name
{
  static std::string name() { return "unknown"; }
};

#define VSIP_IMPL_CLASS_NAME(TYPE)				\
  template <>							\
  struct Class_name<TYPE> {					\
    static std::string name() { return "" # TYPE; }		\
  };

template <typename Fft>
struct Fft_traits;

template <template <typename, typename> class V,
	  typename                            I,
	  typename                            O,
	  int                                 S,
	  return_mechanism_type               R,
	  unsigned                            N,
	  alg_hint_type                       H>
struct Fft_traits<Fft<V, I, O, S, R, N, H> >
{
  static dimension_type        const dim = impl::Dim_of_view<V>::dim;
  static return_mechanism_type const rm  = R;
};



template <typename FftmT>
struct Fftm_traits;

template <typename              I,
	  typename              O,
	  int                   A,
	  int                   D,
	  return_mechanism_type R,
	  unsigned              N,
	  alg_hint_type         H>
struct Fftm_traits<Fftm<I, O, A, D, R, N, H> >
{
  static dimension_type        const dim  = 2;
  static int                   const axis = A;
  static int                   const dir  = D;
  static return_mechanism_type const rm   = R;
};



struct Diagnose_fft
{
  template <typename FftT>
  static void diag(std::string const& name, FftT const& fft)
  {
    using diag_detail::Class_name;
    using std::cout;
    using std::endl;

    typedef Fft_traits<FftT> traits;

    bool inter_fp_ok, ifp_pack, ifp_complex, ifp_req_copy;
    bool split_fp_ok, sfp_pack, sfp_complex, sfp_req_copy;

    // Check if Backend supports interleaved unit-stride fastpath:
    {
      Rt_layout<1> ref;
      ref.packing = dense;
      ref.order = Rt_tuple(row1_type());
      ref.storage_format = interleaved_complex;
      ref.alignment = 0;
      Rt_layout<1> rtl_in(ref);
      Rt_layout<1> rtl_out(ref);
      fft.backend_.get()->query_layout(rtl_in, rtl_out);

      ifp_pack = rtl_in.packing == ref.packing && rtl_out.packing == ref.packing;
      ifp_complex = rtl_in.storage_format == ref.storage_format &&
	            rtl_out.storage_format == ref.storage_format;
      ifp_req_copy = !fft.backend_.get()->requires_copy(rtl_in);

      inter_fp_ok = ifp_pack &&
	// rtl_in.order == ref.order     && rtl_out.order == ref.order &&
	ifp_complex && ifp_req_copy;
    }
    // Check if Backend supports split unit-stride fastpath:
    {
      Rt_layout<1> ref;
      ref.packing = dense;
      ref.order = Rt_tuple(row1_type());
      ref.storage_format = split_complex;
      ref.alignment = 0;
      Rt_layout<1> rtl_in(ref);
      Rt_layout<1> rtl_out(ref);
      fft.backend_.get()->query_layout(rtl_in, rtl_out);

      sfp_pack = rtl_in.packing == ref.packing && rtl_out.packing == ref.packing;
      sfp_complex = rtl_in.storage_format == ref.storage_format &&
	            rtl_out.storage_format == ref.storage_format;
      sfp_req_copy = !fft.backend_.get()->requires_copy(rtl_in);

      split_fp_ok = sfp_pack &&
	// rtl_in.order == ref.order     && rtl_out.order == ref.order &&
	sfp_complex && sfp_req_copy;
    }

    cout << "diagnose_fft(" << name << ")" << endl
	 << "  dim: " << traits::dim << endl
	 << "  rm : " << (traits::rm == by_value ? "val" : "ref") << endl
	 << "  be : " << fft.backend_.get()->name() << endl;
    cout << "  inter_fastpath_ok : " << (inter_fp_ok ? "yes" : "no")
	 << "  (pack: " << (ifp_pack ? "yes" : "no")
	 << "  cmplx: " << (ifp_complex ? "yes" : "no")
	 << "  rqcp: " << (ifp_req_copy ? "yes" : "no")
	 << ")" << endl;
    cout << "  split_fastpath_ok : " << (split_fp_ok ? "yes" : "no")
	 << "  (pack: " << (sfp_pack ? "yes" : "no")
	 << "  cmplx: " << (sfp_complex ? "yes" : "no")
	 << "  rqcp: " << (sfp_req_copy ? "yes" : "no")
	 << ")" << endl;
  }

  template <typename FftT,
	    typename T,
	    typename Block0,
	    typename Block1>
  static void diag_call(
    std::string const&                     name,
    FftT const&                            fft,
    const_Vector<std::complex<T>, Block0>& /*in*/,
    Vector<std::complex<T>, Block1>&       /*out*/)
  {
    using diag_detail::Class_name;
    using std::cout;
    using std::endl;

    typedef Fft_traits<FftT> traits;

    static storage_format_type const storage_format = get_block_layout<Block0>::storage_format;
    typedef Layout<1, row1_type, unit_stride, storage_format> LP;

    diag(name, fft);
    cout << " dda::Data<Block0, LP>::ct_cost : " 
	 <<   dda::Data<Block0, dda::in, LP>::ct_cost << endl
	 << " dda::Data<Block1, LP>::ct_cost : " 
	 <<   dda::Data<Block1, dda::out, LP>::ct_cost << endl
      ;
  }
};



struct Diagnose_fftm
{
  template <typename FftmT>
  static void diag(std::string const& name, FftmT const& fftm)
  {
    using diag_detail::Class_name;
    using std::cout;
    using std::endl;

    typedef Fftm_traits<FftmT> traits;

    cout << "diagnose_fftm(" << name << ")" << endl
	 << "  dim : " << traits::dim << endl
	 << "  i sz: " << fftm.input_size()[0].size() << " x "
	               << fftm.input_size()[1].size() << endl
	 << "  o sz: " << fftm.output_size()[0].size() << " x "
	               << fftm.output_size()[1].size() << endl
	 << "  dir : " << (traits::dir == fft_fwd ? "fwd" : "inv") << endl
	 << "  axis: " << ((dimension_type)traits::axis == row ? "row" : "col") << endl
	 << "  rm  : " << (traits::rm == by_value ? "val" : "ref") << endl
	 << "  be  : " << fftm.backend_.get()->name() << endl;
  }
};

} // namespace vsip::impl::diag_detail



template <typename FftT>
void
diagnose_fft(std::string const& name, FftT const& fft)
{
  diag_detail::Diagnose_fft::diag<FftT>(name, fft);
}



template <typename FftT,
	  typename InViewT,
	  typename OutViewT>
void
diagnose_fft_call(
  std::string const& name,
  FftT const&        fft,
  InViewT&           in,
  OutViewT&          out)
{
  typedef typename InViewT::value_type T;
  diag_detail::Diagnose_fft::diag_call<FftT>(name, fft, in, out);
}



template <typename FftmT>
void
diagnose_fftm(std::string const& name, FftmT const& fftm)
{
  diag_detail::Diagnose_fftm::diag<FftmT>(name, fftm);
}

} // namespace vsip::impl
} // namespace vsip

#endif // VSIP_OPT_DIAG_FFT_HPP
