######################################################### -*-Makefile-*-
#
# Contents: Standalone Makefile for VSIPL++ benchmarks
#
########################################################################

########################################################################
# Variables
########################################################################

HAVE_LAPACK := 1
HAVE_IPP := 
HAVE_SAL := 
HAVE_FFTW3 := 1
HAVE_MPI := 1
HAVE_CBE_SDK := 
HAVE_CUDA := 

benchmarks := $(patsubst %.cpp, %, $(filter-out main.cpp, $(wildcard *.cpp)))

expr := $(patsubst %.cpp, %, $(wildcard expr/*.cpp))
ifdef HAVE_LAPACK
lapack := $(patsubst %.cpp, %, $(wildcard lapack/*.cpp))
endif
ifdef HAVE_IPP
ipp := $(patsubst %.cpp, %, $(wildcard ipp/*.cpp)) 
endif
ifdef HAVE_SAL
sal := $(patsubst %.cpp, %, $(wildcard sal/*.cpp))
endif
ifdef HAVE_FFTW3
fftw3 := $(patsubst %.cpp, %, $(wildcard fftw3/*.cpp)) 
endif
ifdef HAVE_MPI
mpi := $(patsubst %.cpp, %, $(wildcard mpi/*.cpp))
endif
ifdef HAVE_CBE_SDK
cell := $(patsubst %.cpp, %, $(wildcard cell/*.cpp))
endif
ifdef HAVE_CUDA
cuda := $(patsubst %.cpp, %, $(wildcard cuda/*.cpp))
endif
hpec := $(patsubst %.cpp, %, $(wildcard hpec_kernel/*.cpp))

benchmarks += $(expr) $(lapack) $(ipp) $(sal) $(fftw3) $(mpi) $(cell) $(cuda)
benchmarks += $(hpec)

targets := $(benchmarks)

include ../common.mk

########################################################################
# Targets
########################################################################

.PHONY: expr lapack ipp sal fftw3 mpi cell cuda hpec

expr: $(expr)
lapack: $(lapack)
ipp: $(ipp)
sal: $(sal)
fftw3: $(fftw3)
mpi: $(mpi)
cell: $(cell)
cuda: $(cuda)
hpec: $(hpec)

########################################################################
# Rules
########################################################################

$(benchmarks): % : main.o %.o
	$(CXX) $(LDFLAGS) -o $@ $^ $(LIBS)

%.o: %.cpp
	$(CXX) -I . $(CXXFLAGS) -c -o $@ $<

clean::
	rm -f *.o */*.o
