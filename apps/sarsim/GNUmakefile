########################################################################
#
# File:   GNUmakefile
# Author: Jules Bergmann
# Date:   2005-06-08
#
# Contents: Makefile for VSIPL++ SarSim example application
#
########################################################################

PGM = sarsim
PKG = vsipl++

CXX      := $(shell pkg-config --variable=cxx $(PKG))

IOPT =  --param max-inline-insns-single=2000	\
	--param large-function-insns=6000	\
	--param large-function-growth=800	\
	--param inline-unit-growth=300

OPT1	:= -g -W -Wall
OPT2	:= -O2 -funswitch-loops -fgcse-after-reload -DNDEBUG
OPT3	:= -O2 -funswitch-loops -fgcse-after-reload $(IOPT) -DNDEBUG

OPT     := $(shell pkg-config --variable=cxxflags $(PKG))
ifeq ($(O),1)
  OPT := $(OPT1)
endif
ifeq ($(O),2)
  OPT := $(OPT2)
endif
ifeq ($(O),3)
  OPT := $(OPT3)
endif

CXXFLAGS := $(shell pkg-config --cflags $(PKG)) $(OPT)
CFLAGS   := $(CXXFLAGS)

LIBS     := $(shell pkg-config --libs   $(PKG))

OBJEXT   := o

# cdft.o misc.o
OBJ = read_adts.o read_tbv.o frm_hdr.o util_io.o

cxx_sources := $(wildcard *.cpp)
c_sources   := $(wildcard *.c)

objects     := $(patsubst %.cpp, %.$(OBJEXT), $(cxx_sources))	\
	       $(patsubst %.c, %.$(OBJEXT), $(c_sources))
deps        := $(patsubst %.cpp, %.d, $(cxx_sources))



########################################################################
# Standard Targets
########################################################################

all::

depend:: $(deps)

clean::

ifeq (,$(filter $(MAKECMDGOALS), depend doc clean))
include $(deps)
endif

ifneq (,$(findstring depend, $(MAKECMDGOALS)))
$(deps): %.d:	.FORCE
endif



########################################################################
# Application Targets
########################################################################

all:: sarsim histcmp dat2xv

sarsim: sarsim.o
	$(CXX) $(OPT) -o $@ $^ $(LIBS)

mit-sarsim: mit-sarsim.o $(OBJ)
	$(CXX) $(OPT) -o $@ $^ $(LIBS)

histcmp: histcmp.o frm_hdr.o read_adts.o util_io.o
	$(CXX) $(OPT) -o $@ $^ -lm

dat2xv: dat2xv.o frm_hdr.o read_adts.o misc.o util_io.o
	$(CXX) $(OPT) -o $@ $^ -lm

clean::
	rm -f $(objects)
	rm -f $(deps)
	rm -f *.raw



########################################################################
# Implicit Rules
########################################################################

# Generate a dependency Makefile fragment for a C++ source file.
# (This recipe is taken from the GNU Make manual.)
%.d: %.cpp
	$(SHELL) -ec '$(CXX) -M $(CXXFLAGS) \
		      $(call dir_var,$(dir $<),CXXFLAGS) $< \
		      | sed "s|$(*F)\\.$(OBJEXT)[ :]*|$*\\.d $*\\.$(OBJEXT) : |g" > $@'

