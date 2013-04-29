#! /bin/sh

# Copyright (c) 2008 by CodeSourcery.  All rights reserved.
#
#  This file is available for license from CodeSourcery, Inc. under the terms
#  of a commercial license and under the GPL.  It is not part of the VSIPL++
#   reference implementation and is not available under the BSD license.
#
#   @file    make_images.sh
#   @author  Don McCoy
#   @date    2008-08-19
#   @brief   VSIPL++ implementation of SSCA #3: Kernel 1, Image Formation

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
if [ -f $input/p62_s_filt.view ]; then 
    ./viewtopng -r $input/p62_s_filt.view $output/p62_s_filt.png $n $mc
fi
if [ -f $input/p76_fs_ref.view ]; then 
    ./viewtopng -r $input/p76_fs_ref.view $output/p76_fs_ref.png $n $m
fi
if [ -f $input/p77_fsm_half_fc.view ]; then 
    ./viewtopng $input/p77_fsm_half_fc.view $output/p77_fsm_half_fc.png $n $m
fi
if [ -f $input/p77_fsm_row.view ]; then 
    ./viewtopng $input/p77_fsm_row.view $output/p77_fsm_row.png $n $m
fi
if [ -f $input/p92_F.view ]; then 
    ./viewtopng $input/p92_F.view $output/p92_F.png $nx $m
fi
./viewtopng -s $result $output/p95_image.png $m $nx

