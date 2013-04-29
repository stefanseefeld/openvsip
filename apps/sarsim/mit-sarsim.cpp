/* Copyright (c) 2005, 2006 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license and under the GPL.  It is not part of the VSIPL++
   reference implementation and is not available under the BSD license.
*/
/** @file    sarsim.cpp
    @author  Jules Bergmann
    @date    03/02/2005
    @brief   VSIPL++ implementation of RASSP benchmark 0.
*/

#define PARALLEL 1

#include <cstdio>
#include <cstring>
#include <errno.h>

extern "C" {
#include "sarx.h"
#include "read_adts.h"
#include "read_tbv.h"
#include "util_io.h"
}

#include "sarsim.hpp"
#include "saveview.hpp"
#include "loadview.hpp"

using vsip::index_type;
using vsip::Vector;
using vsip::Matrix;
using vsip::Domain;
using vsip::complex;
using vsip::impl::cast_view;

#define ENABLE_DOUBLE 0

template <typename T>
class MITSarSim : public SarSim<T> {
public:

  typedef typename SarSim<T>::cval_type cval_type;
  typedef typename SarSim<T>::io_type io_type;
  typedef typename SarSim<T>::cio_type cio_type;

  MITSarSim(int nrange,
	    int npulse,
	    int ncsamples,
	    unsigned niq,
	    io_type swath,
	    Matrix<cio_type> w_eq,
	    Vector<io_type> rcs,
	    Vector<io_type> i_coef,
	    Vector<io_type> q_coef,
	    Matrix<cio_type> cphase,
#if PARALLEL
	    vsip::Map<> map_in,
	    vsip::Map<> map_rg,
	    vsip::Map<> map_az,
	    vsip::Map<> map_out,
#endif
	    int pmode,
	    unsigned itype,
	    FILE *fpin,
	    FILE **fpout) : 
    SarSim<T>(nrange, npulse, ncsamples, niq, swath,
	      w_eq, rcs, i_coef, q_coef, cphase, 
#if PARALLEL
	      map_in, map_rg, map_az, map_out,
#endif
	      pmode),
    itype_ (itype),
    fpin_ (fpin),
    fpout_ (fpout)
  {
    // FIXME: Should use autoptr-style allocation.
    rod_ = new short[2*this->ncsamples_];
    aux_ = new short[NAUX];
    aux_save_ = new short[NAUX];
  }

  ~MITSarSim () {
    delete[] aux_save_;
    delete[] aux_;
    delete[] rod_;
  }
  
protected:

  io_type read_pulse(int pol, index_type p) {
    io_type range = io_type();

    // Read one polarization according to its input format.
    if (itype_) {
      if (read_adts(fpin_, rod_, aux_, &range, pol, this->ncsamples_) < 0) {
	fprintf(stderr, "Error reading PRI #%d.\n", p);
	exit(1);
      }
    } else {
      if (read_tbv(fpin_, rod_, aux_, &range, pol, this->ncsamples_) < 0) {
	fprintf(stderr, "Error reading PRI #%d.\n", p);
	exit(1);
      }
    }

    // Get the input odd/even pairs for this PRI. Move to base band.
    for (index_type i = 0; i < this->ncsamples_; i += 2) {
      this->vec_iq_.put(i + 0, cval_type(rod_[2 * i + 0], rod_[2 * i + 1]));
      this->vec_iq_.put(i + 1, cval_type(-rod_[2 * i + 2], -rod_[2 * i + 3]));
    }

    if (pol == HH && p == this->npulse_ / 2 - 1) {
      // Save auxiliary info from HH polarization.
      for(int i=0; i<NAUX; i++)
	aux_save_[i] = aux_[i];
    }

    return range;
  }

  void write_output_header(int pol) {
    // Output header information for this frame and polarization.
    if (write_frm_hdr(fpout_[pol], aux_save_, pol) < 0)
    {
      fprintf(stderr,"Error writing frame header.\n");
      exit(1);
    }

  }

  void write_output(int pol) {
    const index_type npulse = this->npulse_;
    Vector<cval_type> v(npulse);
    v = this->azbuf_ (Domain<1>(npulse, 1, npulse));

    io_type* io_buf =  vsip::impl::alloc_align<io_type>(32, 2 * npulse);
    vsip::Dense<1, vsip::complex<io_type> > io_block(Domain<1>(npulse), io_buf);
    vsip::Vector<vsip::complex<io_type> > io_vec(io_block);
    
    io_vec.block().admit(false);
    io_vec = cast_view<cio_type>(v);
    io_vec.block().release(true);
    if (fwrite_bo(io_buf, sizeof(io_type), 2*npulse, fpout_[pol], 4) 
	!= 2*npulse)
      {
	fprintf(stderr,"Error writing output data.\n");
	exit(1);
      }
    
    vsip::impl::free_align((void*)io_buf);
  }
  
private:
  unsigned itype_;
  short *rod_;
  short *aux_;
  short *aux_save_;
  FILE* fpin_;
  FILE** fpout_;
};


int
main(
   int	argc,
   char	**argv)
{
  vsip::vsipl init(argc, argv);

  FILE		*fpin, *fpeq;
  FILE		*fpout[4];
  int		i, j;
  float		swath;
  char		fnin[128], fnkrn[128], fneq[128], fnrcs[128],
		fniqe[128], fniqo[128],
		fnouthh[128], fnouthv[128], fnoutvh[128], fnoutvv[128];

  int		nrange = MAXRANGE;
  int		npulse = MAXPULSE;
  int		ncsamples; /* defined later = nrange - 16 */

/**
Parameters that might be varied.
**/
  unsigned	nframe = 4,	/* Number of frames to process.	*/
		niq = 8,	/* Number of I/Q coef pairs	*/
		itype = 0;	/* Input type: 0=.tb, 1=.adts	*/

  bool	        pol_on[4];	/* Process XX polarization?     */

  int		use_single = 1; // using single floating-point precision.
  bool		profile = false;
  int		pmode = 1;

  pol_on[HH] = 1;
  pol_on[HV] = 0;
  pol_on[VH] = 0;
  pol_on[VV] = 0;

/**
Default input and output filenames.
**/
  strcpy(fnin, "radar.bin");
  strcpy(fniqe, "iqe.bin");
  strcpy(fniqo, "iqo.bin");
  strcpy(fnkrn, "krn.bin");
  strcpy(fneq, "equ.bin");
  strcpy(fnrcs, "rcs.bin");
  strcpy(fnouthh, "hhimg.bin");
  strcpy(fnouthv, "hvimg.bin");
  strcpy(fnoutvh, "vhimg.bin");
  strcpy(fnoutvv, "vvimg.bin");
/**
Get input parameters from the command line.
**/
  for (i=1; i<argc; i++)
  {
    if (!strcmp(argv[i], "-help"))	/* Help Request */
    {
      fprintf(stderr, "Usage: \n\
		-help Print this message \n\
		-nrange  [# of range cells]         (2048)\n\
		-npulse  [# of pulses]              (512)\n\
		-nframe  [# of frames to process]   (4)\n\
		-niq     [Number of I/Q coef pairs] (8) \n\
		-itype   [Input type: 0=tb, 1=adts] (0) \n\
		-hhon    [Process HH polarization?] (1) \n\
		-hvon    [Process HV polarization?] (0) \n\
		-vhon    [Process VH polarization?] (0) \n\
		-vvon    [Process VV polarization?] (0) \n\
		-i       [Input radar data file]   (radar.bin)      \n\
		-iqe     [Input I/Q-e filter file] (iqe.bin) \n\
		-iqo     [Input I/Q-o filter file] (iqo.bin) \n\
		-eq      [Input equalization file] (equ.bin) \n\
		-rcs     [Input rcs file]          (rcs.bin) \n\
		-krn     [Input kernel file]       (krn.bin) \n\
		-ohh     [Output HH file]          (hhimg.bin)      \n\
		-ohv     [Output HV file]          (hvimg.bin)      \n\
		-ovh     [Output VH file]          (vhimg.bin)      \n\
		-ovv     [Output VV file]          (vvimg.bin)      \n");
      exit(1);
    }
    else if (!strcmp(argv[i], "-i"))	/* Input File */
    {
      if (++i < argc)
      {
	if (!(fpin=fopen(argv[i],"r")))
	{
	  fprintf(stderr,"Can't find input file %s\n", argv[i]);
	  exit(1);
	}
	else strcpy(fnin, argv[i]);
      }
      else fprintf(stderr, "Invalid value for option %s\n",argv[i-1]);
    }
/**
Output filenames.
**/
    else if (!strcmp(argv[i], "-ohh"))	/* Output HH File */
    {
      if (++i < argc)
      {
	strcpy(fnouthh, argv[i]);
      }
      else fprintf(stderr, "Invalid value for option %s\n",argv[i-1]);
    }
    else if (!strcmp(argv[i], "-ohv"))	/* Output HV File */
    {
      if (++i < argc)
      {
	strcpy(fnouthv, argv[i]);
      }
      else fprintf(stderr, "Invalid value for option %s\n",argv[i-1]);
    }
    else if (!strcmp(argv[i], "-ovh"))	/* Output VH File */
    {
      if (++i < argc)
      {
	strcpy(fnoutvh, argv[i]);
      }
      else fprintf(stderr, "Invalid value for option %s\n",argv[i-1]);
    }
    else if (!strcmp(argv[i], "-ovv"))	/* Output VV File */
    {
      if (++i < argc)
      {
	strcpy(fnoutvv, argv[i]);
      }
      else fprintf(stderr, "Invalid value for option %s\n",argv[i-1]);
    }
/**
Input data filenames.
**/
    else if (!strcmp(argv[i], "-eq"))	/* Input .eq File */
    {
      if (++i < argc)
      {
	if (!(fpeq=fopen(argv[i],"r")))
	{
	  fprintf(stderr,"Can't find input file %s\n",argv[i]);
	  exit(1);
	}
	else strcpy(fneq, argv[i]);
      }
      else fprintf(stderr, "Invalid value for option %s\n",argv[i-1]);
    }
    else if (!strcmp(argv[i], "-iqe"))	/* Input .iqe File */
    {
      if (++i < argc)
      {
	strcpy(fniqe, argv[i]);
      }
      else fprintf(stderr, "Invalid value for option %s\n",argv[i-1]);
    }
    else if (!strcmp(argv[i], "-iqo"))	/* Input .iqo File */
    {
      if (++i < argc)
      {
	strcpy(fniqo, argv[i]);
      }
      else fprintf(stderr, "Invalid value for option %s\n",argv[i-1]);
    }
    else if (!strcmp(argv[i], "-krn"))	/* Input .krn File */
    {
      if (++i < argc)
      {
	strcpy(fnkrn, argv[i]);
      }
      else fprintf(stderr, "Invalid value for option %s\n",argv[i-1]);
    }
    else if (!strcmp(argv[i], "-rcs"))	/* Input .rcs File */
    {
      if (++i < argc)
      {
	strcpy(fnrcs, argv[i]);
      }
      else fprintf(stderr, "Invalid value for option %s\n",argv[i-1]);
    }
/**
Constants.
**/
    else if (!strcmp(argv[i], "-nrange"))     /* Num. of range cells */
    {
      if (1 == sscanf(argv[++i], "%d", &j))
	nrange = j;
      else
	fprintf(stderr, "Invalid value for option %s\n",argv[i-1]);
    }
    else if (!strcmp(argv[i], "-npulse"))	/* Num. of pulses */
    {
      if (1 == sscanf(argv[++i], "%d", &j))
	npulse = j;
      else
	fprintf(stderr, "Invalid value for option %s\n",argv[i-1]);
    }
    else if (!strcmp(argv[i], "-nframe"))	/* Num. of frames */
    {
      if (1 == sscanf(argv[++i], "%d", &j))
	nframe = j;
      else
	fprintf(stderr, "Invalid value for option %s\n",argv[i-1]);
    }
    else if (!strcmp(argv[i], "-niq"))		/* Num. of I/Q coefs */
    {
      if (1 == sscanf(argv[++i], "%d", &j))
	niq = j;
      else
	fprintf(stderr, "Invalid value for option %s\n",argv[i-1]);
    }
    else if (!strcmp(argv[i], "-itype"))	/* Input Format */
    {
      if (1 == sscanf(argv[++i], "%d", &j))
	itype = j;
      else
	fprintf(stderr, "Invalid value for option %s\n",argv[i-1]);
    }
    else if (!strcmp(argv[i], "-hhon"))		/* Process HH? */
    {
      if (1 == sscanf(argv[++i], "%d", &j))
	pol_on[HH] = j;
      else
	fprintf(stderr, "Invalid value for option %s\n",argv[i-1]);
    }
    else if (!strcmp(argv[i], "-hvon"))		/* Process HV? */
    {
      if (1 == sscanf(argv[++i], "%d", &j))
	pol_on[HV] = j;
      else
	fprintf(stderr, "Invalid value for option %s\n",argv[i-1]);
    }
    else if (!strcmp(argv[i], "-vhon"))		/* Process VH? */
    {
      if (1 == sscanf(argv[++i], "%d", &j))
	pol_on[VH] = j;
      else
	fprintf(stderr, "Invalid value for option %s\n",argv[i-1]);
    }
    else if (!strcmp(argv[i], "-vvon"))		/* Process VV? */
    {
      if (1 == sscanf(argv[++i], "%d", &j))
	pol_on[VV] = j;
      else
	fprintf(stderr, "Invalid value for option %s\n",argv[i-1]);
    }
    else if (!strcmp(argv[i], "-single"))    use_single = 1;
    else if (!strcmp(argv[i], "-double"))    use_single = 0;
    else if (!strcmp(argv[i], "-profile"))   profile    = true;
    else if (!strcmp(argv[i], "-noprofile")) profile    = false;
    else if (!strcmp(argv[i], "-par"))       pmode      = atoi(argv[++i]);
    else
    {fprintf(stderr, "Invalid option: %s Do `sarsim -help' for option list\n",argv[i]);
      exit(1);
    }
  }

  ncsamples = nrange-16;

  printf("VPP (%d, %d) %d.\n", nrange, npulse, pmode);

/**
Define other parameters.
**/
  swath = (float) (nrange*0.2287); /* Size of the range window in m.*/

  if (pol_on[HH])
  {   
    if (!(fpout[HH] = fopen(fnouthh,"w")))
    {
      fprintf(stderr,"Error opening HH output file '%s'.\n", fnouthh);
      exit(1);
    }
  }
  if (pol_on[HV])
  {   
    if (!(fpout[HV] = fopen(fnouthv,"w")))
    {
      fprintf(stderr,"Error opening HV output file '%s'.\n", fnouthv);
      exit(1);
    }
  }
  if (pol_on[VH])
  {   
    if (!(fpout[VH] = fopen(fnoutvh,"w")))
    {
      fprintf(stderr,"Error opening VH output file '%s'.\n", fnoutvh);
      exit(1);
    }
  }
  if (pol_on[VV])
  {   
    if (!(fpout[VV] = fopen(fnoutvv,"w")))
    {
      fprintf(stderr, "Error opening VV output file '%s': %s\n",
	      fnoutvv, strerror(errno));
      exit(1);
    }
  }
/**
Obtain I/Q coefficient odd/even weights.
**/
  printf("Reading I/Q coefficient odd/even weights.\n");

  LoadView<1, float> load_icoef(fniqe, Domain<1>(niq));
  LoadView<1, float> load_qcoef(fniqo, Domain<1>(niq));

  Vector<float> Icoef(niq); /* FIR coefficients for input I array    */
  Vector<float> Qcoef(niq); /* FIR coefficients for input Q array    */

  for (index_type ii=0; ii<(unsigned)niq; ++ii)
  {
    Icoef.put(ii, load_icoef.view().get(niq-ii-1));
    Qcoef.put(ii, load_qcoef.view().get(niq-ii-1));
  }
  save_view("conv-coeff.raw", Icoef);
/**
Obtain convolution kernels.
**/
  printf("Reading convolution kernels.\n");

  /* Holds transformed convolution kernels. */
  LoadView<2, complex<float> > cphase_file(fnkrn, Domain<2>(NREF,2*npulse));
  save_view("cphase.raw", cphase_file.view());

/**
Obtain Equalization and Taylor weighting.
**/
  printf("Reading EQ/Taylor data.\n");

  if (!(fpeq = fopen(fneq,"r")))
  {
    fprintf(stderr,"Error opening EQ/Taylor weighting file.\n");
    exit(1);
  }

  LoadView<1, complex<float> > whh_file(fpeq, nrange); // HH EQ Weights.
  LoadView<1, complex<float> > whv_file(fpeq, nrange); // HV EQ Weights.
  LoadView<1, complex<float> > wvh_file(fpeq, nrange); // VH EQ Weights.
  LoadView<1, complex<float> > wvv_file(fpeq, nrange); // VV EQ Weights.

  fclose(fpeq);

  // Pack EQ weights into Matrix.
  Matrix<complex<float> > w_eq(4, nrange);
  w_eq.row(HH) = whh_file.view();
  w_eq.row(HV) = whh_file.view();
  w_eq.row(VH) = whh_file.view();
  w_eq.row(VV) = whh_file.view();

/**
Obtain RCS weighting.
**/
  printf("Reading RCS data.\n");
  LoadView<1, float> rcs_file(fnrcs, Domain<1>(nrange));


/**
Open the input 32-bit disk file.
**/
  printf("Opening the input data.\n");
  if (!(fpin = fopen(fnin,"r")))
  {
    fprintf(stderr,"Error opening input.\n");
    exit(1);
  }

#if PARALLEL
  using vsip::Map;
  using vsip::Block_dist;
  using vsip::Vector;
  using vsip::processor_type;

  vsip::processor_type np = vsip::num_processors();

  Map<> map_in;
  Map<> map_rg;
  Map<> map_az;
  Map<> map_out;

  Map<> root_map(Block_dist(1),  Block_dist(1),  Block_dist(1));
  Map<> map_pols(Block_dist(np), Block_dist(1),  Block_dist(1));

//  Map<> map_pols_core(pset_core, Block_dist(pset_core.size()), Block_dist(1),  Block_dist(1));
//  Map<> map_pulse(Block_dist(1),  Block_dist(np), Block_dist(1));
//  Map<> map_range(Block_dist(1),  Block_dist(1),  Block_dist(np));
//  Map<> map_proc0(pset0, Block_dist(1),  Block_dist(1),  Block_dist(1));
//  Map<> map_proc1(pset1, Block_dist(1),  Block_dist(1),  Block_dist(1));
//  Map<> map_procN(psetN, Block_dist(1),  Block_dist(1),  Block_dist(1));

  if (pmode == 1)
  {
    map_in  = root_map;
    map_rg  = map_pols;
    map_az  = map_pols;
    map_out = root_map;
  }
  else if (pmode == 2)
  {
    Map<> map_pulse(Block_dist(1),  Block_dist(np), Block_dist(1));
    Map<> map_range(Block_dist(1),  Block_dist(1),  Block_dist(np));

    map_in  = root_map;
    map_rg  = map_pulse;
    map_az  = map_range;
    map_out = root_map;
  }
  else if (pmode == 3)
  {
    assert(np > 1 && np % 2 == 0);

    Vector<processor_type> pset_pulse(np/2);
    Vector<processor_type> pset_range(np/2);

    for (processor_type i=0; i<np/2; ++i)
    {
      pset_pulse(i) = i;
      pset_range(i) = np/2 + i;
    }

    Map<> map_pulse(pset_pulse,
		    Block_dist(1), Block_dist(np/2), Block_dist(1) );
    Map<> map_range(pset_range,
		    Block_dist(1), Block_dist(1), Block_dist(np/2));

    map_in  = root_map;
    map_rg  = map_pulse;
    map_az  = map_range;
    map_out = root_map;
  }
  else if (pmode == 4)
  {
    assert(np > 1 && np % 2 == 0);

    Vector<processor_type> pset_pulse(np/2);
    Vector<processor_type> pset_range(np/2);

    for (processor_type i=0; i<np/2; ++i)
    {
      pset_pulse(i) = i;
      pset_range(i) = np/2 + i;
    }

    Map<> map_rg(pset_pulse,
		 Block_dist(np/2), Block_dist(1), Block_dist(1) );
    Map<> map_az(pset_range,
		 Block_dist(np/2), Block_dist(1), Block_dist(1));

    map_in  = root_map;
    map_rg  = map_rg;
    map_az  = map_az;
    map_out = root_map;
  }
  else { assert(0); }
#else
  vsip::Local_map map_in;
  vsip::Local_map map_rg;
  vsip::Local_map map_az;
  vsip::Local_map map_out;
#endif

  if (profile)
  {
    vsip::impl::profile::prof->set_mode(vsip::impl::profile::pm_trace);
  }

  printf("Processing.\n");
  if (use_single) {
    MITSarSim<float> mss(nrange, npulse, ncsamples, niq, swath,
			 w_eq,
			 rcs_file.view(), Icoef, Qcoef,
			 cphase_file.view (),
#if PARALLEL
			 map_in, map_rg,map_az, map_out,
#endif
			 pmode,
			 itype, fpin, fpout);
    
    mss.process(nframe, pol_on);
    mss.report_performance();
  } else {
#if ENABLE_DOUBLE
    MITSarSim<double> mss(nrange, npulse, ncsamples, niq, swath,
			  w_eq,
			  rcs_file.view(), Icoef, Qcoef,
			  cphase_file.view (),
			  map_in, map_rg,map_az, map_out, pmode,
			  itype, fpin, fpout);
    
    mss.process(nframe, pol_on);
    mss.report_performance();
#else
    VSIP_IMPL_THROW(vsip::impl::unimplemented(
	     "Support for double precision not enabled"));
#endif
  }

  fclose(fpin);

  if (pol_on[HH]) fclose(fpout[HH]);
  if (pol_on[HV]) fclose(fpout[HV]);
  if (pol_on[VH]) fclose(fpout[VH]);
  if (pol_on[VV]) fclose(fpout[VV]);

  if (profile)
  {
    vsip::processor_type rank =
      vsip::impl::Par_service::default_communicator().rank();

    char name[256];
    sprintf(name, "vprof.%d.out", rank);

    vsip::impl::profile::prof->dump(name);
  }

  exit(0);
}


