######################################################### -*-Makefile-*-
#
# Contents: Makefile for VSIPL++ example DSP programs.
#
########################################################################

########################################################################
#
# Copyright (c) 2005-2011 CodeSourcery, Inc.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials
#       provided with the distribution.
#     * Neither the name of CodeSourcery nor the names of its
#       contributors may be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY CODESOURCERY, INC. "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL CODESOURCERY BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

targets := high_pass low_pass moving_average resample

include ../../common.mk

run_examples:
	@echo "Running DSP examples demonstrating simple filtering"
	./moving_average
	./low_pass
	./high_pass
	@echo "Done.  Data saved as .view files"
	@echo "Running resampling example using polyphase filtering"
	./resample

make_plots:
	@echo "Creating graphs using Octave"
	octave moving_average.m &> octave_err1.log
	octave low_pass.m &> octave_err2.log
	octave high_pass.m &> octave_err3.log
	@echo "Done.  Plots saved as .svg files"
