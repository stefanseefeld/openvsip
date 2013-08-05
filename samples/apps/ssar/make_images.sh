#! /bin/sh

########################################################################
#
# Contents: VSIPL++ implementation of SSCA #3: Kernel 1, Image Formation
#
########################################################################

########################################################################
#
# Copyright (c) 2008, 2011 CodeSourcery, Inc.  All rights reserved.
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


# This script creates images from the input and output raw data files produced 
# during Kernel 1 processing.  It also creates intermediate images from the
# VSIPL++ views that are saved when VERBOSE is defined in kernel1.hpp (these
# are helpful in diagnosing problems and/or providing visual feedback as to
# what is occuring during each stage of processing).

# Parameters
#   result The result view
#   input  The directory where the input and reference image files are located
#   output The directory where the output (png) images should be stored.
#   n      Input image rows
#   mc     Input image columns
#   m      Output image rows
#   nx     Output image columns

# Usage
#   ./make_images.sh RESULT INPUT OUTPUT N MC M NX

result=$1
input=$2
output=$3
n=$4
mc=$5
m=$6
nx=$7

echo "Converting to greyscale png..."
mkdir -p $output
./viewtopng $input/sar.view $output/p00_sar.png $n $mc
if [ -f p62_s_filt.view ]; then 
    ./viewtopng -r p62_s_filt.view $output/p62_s_filt.png $n $mc
fi
if [ -f p76_fs_ref.view ]; then 
    ./viewtopng -r p76_fs_ref.view $output/p76_fs_ref.png $n $m
fi
if [ -f p77_fsm_half_fc.view ]; then 
    ./viewtopng p77_fsm_half_fc.view $output/p77_fsm_half_fc.png $n $m
fi
if [ -f p77_fsm_row.view ]; then 
    ./viewtopng p77_fsm_row.view $output/p77_fsm_row.png $n $m
fi
if [ -f p92_F.view ]; then 
    ./viewtopng p92_F.view $output/p92_F.png $nx $m
fi
./viewtopng -s $result $output/p95_image.png $m $nx

