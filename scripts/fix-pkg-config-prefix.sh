#! /bin/sh

########################################################################
#
# File:   fix-pkg-config-prefix.sh
# Author: Jules Bergmann
# Date:   2006-04-28
#
# Contents:
#   Edit pkg-config files to put install prefixes for libraries such
#   as IPP, MKL, and MPI into pkg-config variables.
#
########################################################################

# SYNOPSIS
#   fix-pkg-config-prefix.sh -p PCFILE -k VAR -v VALUE [-d]
#

drop_arch="no"

# .pc file
pcfile=""

while getopts "p:v:k:d" arg; do
    case $arg in
	p)
	    pcfile=$OPTARG
	    ;;
	v)
	    prefix=$OPTARG
	    ;;
	k)
	    key=$OPTARG
	    ;;
	d)
	    drop_arch="yes";
	    ;;
	\?)
            error "usage: fix-pkg-config-prefix.sh -p PCFILE [-i IPPDIR] [-m MKLDIR]"
	    ;;
    esac
done

if test ! -f "$pcfile"; then
  error "error: fix-intel-pkg-config-prefix.sh -p PCFILE option required"
fi

if test "$drop_arch" = "yes"; then
  prefix=`dirname $prefix`
fi

echo "$key=$prefix" >  $pcfile.new

cat $pcfile | sed -e "s|$prefix/|\${$key}/|g" >> $pcfile.new

if test -f "$pcfile.new"; then
  mv $pcfile.new $pcfile
fi
