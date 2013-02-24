######################################################### -*-Makefile-*-
#
# Contents: build-system functionality for VSIPL++ example programs.
#
########################################################################

########################################################################
# Variables
########################################################################

# This should point to the directory where Sourcery VSIPL++ is installed.
prefix = /opt/codesourcery/sourceryvsipl++-2.3

pkg_config_path := $(prefix)/lib/pkgconfig

# This selects the desired library variants.
# Please see the Getting Started manual for a complete list 
# of variants appropriate for your system.
#
# You may also set it on the command line when invoking make.
# For example:
#
#   $ make variant=em64t-ser-builtin
#
# would select the x86 64-bit serial-builtin configuration listed in the 
# pkgconfig/ directory as vsipl++-em64t-ser-builtin.pc. 
#
variant :=
suffix := $(if $(variant),-$(variant))
pkg ?= vsipl++$(suffix)
variables :=
variables_opt := $(patsubst %, --define-variable=%, $(variables))
pkgcommand := PKG_CONFIG_PATH=$(pkg_config_path) pkg-config $(pkg) \
                --define-variable=prefix=$(prefix) $(variables_opt)

CC       := $(shell ${pkgcommand} --variable=cc)
CXX      := $(shell ${pkgcommand} --variable=cxx)
CPPFLAGS := $(shell ${pkgcommand} --cflags)
CFLAGS   := $(CPPFLAGS) $(shell ${pkgcommand} --variable=cflags)
CXXFLAGS := $(CPPFLAGS) $(shell ${pkgcommand} --variable=cxxflags)
LIBS     := $(shell ${pkgcommand} --libs)

# We can't use the '+=' operator as that inserts a whitespace.
ld_library_path:=$(shell ${pkgcommand} --variable=libdir):$(shell ${pkgcommand} --variable=builtin_libdir)

# By default, the 'check' target will run all targets.
check_targets ?= $(targets)

# For check targets we need to prepare the environment
%.check: export LD_LIBRARY_PATH:=$(ld_library_path):$(LD_LIBRARY_PATH)
%.check: export ALF_LIBRARY_PATH:=.:$(ld_library_path):$(ALF_LIBRARY_PATH)

########################################################################
# Targets
########################################################################

all: $(targets)

check: $(patsubst %, %.check, $(check_targets))

show-library-path:
	@echo $(ld_library_path)

show-info:
	@echo "info for variant '$(variant)', using package '$(pkg)'"
	@echo "  CC=$(CC)"
	@echo "  CXX=$(CXX)"
	@echo "  CPPFLAGS=$(CPPFLAGS)"
	@echo "  CFLAGS=$(CFLAGS)"
	@echo "  CXXFLAGS=$(CXXFLAGS)"
	@echo "  LIBS=$(LIBS)"

clean::
	rm -rf $(targets)

########################################################################
# Rules
########################################################################

%.check: %
	@(./$< > /dev/null && echo "PASS: $<") || echo "FAIL: $<"

%: %.cpp
	$(CXX) -I . $(CXXFLAGS) -o $@ $^ $(LIBS)
