#! /bin/sh

########################################################################
#
# File:   set-prefix.sh
# Author: Jules Bergmann
# Date:   2005-12-20
#
# Contents:
#   Changes install prefixes for a Sourcery VSIPL++ binary package.
#
########################################################################

########################################################################
# Notes
########################################################################

# SYNOPSIS
#   set-prefix.sh [-p PREFIX]
#                 [-l PKGCONFIG_DIR]
#                 [-i IPP_PREFIX]
#                 [-m IPP_PREFIX]
#                 -v
#                 PRE1:PATH1 [PRE2:PATH2 ...]
#                 
#
# DESCRIPTION
#   Sets the prefix variables in the a Sourcery VSIPL++ binary package's
#   .pc files.
#
#   PREFIX is the library installation prefix that will be inserted
#   into pkg-config .pc files.  It can either be specified with the
#   -p option, or guessed (using the location of the set-prefix.sh
#   script).
#
#   PKGCONFIG_DIR is the directory containing the library's .pc
#   files.  It can either be specified with the -l option, or
#   if it is in the standard location ($prefix/lib/pkgconfig),
#   derived from the library PREFIX.  
#
#   Arguments of the form 'PRE:PATH' indicate the value of
#   variable 'PRE_prefix' should be set to 'PATH'.
#
#   For backwards compatibility, the "-i" and "-m" options can
#   be used to substitute IPP_PREFIX and MKL_PREFIX are the IPP and
#   MKL prefixes.  The option '-i IPP_PREFIX' is equivalent to the
#   argument 'ipp:IPP_PREFIX'.  The option '-m MKL_PREFIX' is equivalent
#   to the argument 'mkl:MKL_PREFIX'.
#
#   It is always necessary to specify or guess the library prefix,
#   since this determines the location of the pkg-config files.
#   The library prefix will always be updated, even if the old and
#   new prefix are the same. However, if the IPP or MKL prefix are
#   not set, they will not be substituted.
#
#   The -v option turns on verbose output.
#
# EXAMPLES
#


########################################################################
# Subroutines
########################################################################

prefix=`dirname $0`		# directory where set-prefix.sh is located.
				#   in src package: scripts/set-prefix.sh
				#   in bin package: sbin/set-prefix.sh
prefix="`(cd $prefix; echo \"$PWD\")`"
prefix=`dirname $prefix`	# Go up 1 level to package root.

pcdir='*use-default*'
verbose="no"
pairs=''

while getopts "xp:l:i:m:v" arg; do
    case $arg in
	p)
	    prefix=$OPTARG
	    ;;
	l)
	    pcdir=$OPTARG
	    ;;
	i)
	    # ipp_prefix=$OPTARG
	    pairs="ipp:$OPTARG $pairs"
	    ;;
	m)
	    # mkl_prefix=$OPTARG
	    pairs="mkl:$OPTARG $pairs"
	    ;;
	v)
	    verbose="yes"
	    ;;
    esac
done

# put remaing args into $pairs

i=$(( $OPTIND - 1 ))
for arg in "$@"; do
  if test $i -gt 0; then
    i=$(( $i - 1 ))
  else
    pairs="$pairs $arg"
  fi
done

if test "$pcdir" == '*use-default*'; then
  pcdir="$prefix/lib/pkgconfig"
fi

if test "$verbose" == "yes"; then
  echo "VSIPL++ prefix  : " $prefix
  echo "  pkgconfig dir : " $pcdir

  for pair in $pairs; do
    old_IFS=$IFS
    IFS=":"
    i=0
    for x in $pair; do
      if test $i = 0; then
        key=$x
        i=1
      else
        value=$x
      fi
    done
    IFS=$old_IFS
    echo "$key prefix      : " $value
  done
fi

for file in `ls $pcdir/*.pc`; do
  if test "x$prefix" != "x"; then
    cat $file | sed -e "s|^prefix=.*$|prefix=$prefix|" > $file.tmp
  else
    cp $file $file.tmp
  fi

  for pair in $pairs; do
    old_IFS=$IFS
    IFS=":"
    i=0
    for x in $pair; do
      if test $i = 0; then
        key=$x
        i=1
      else
        value=$x
      fi
    done
    IFS=$old_IFS
    cat $file.tmp | sed -e "s|^${key}_prefix=.*$|${key}_prefix=$value|" \
		> $file.tmp2
    mv $file.tmp2 $file.tmp
  done

  if test -f "$file.tmp"; then
    mv $file.tmp $file
  else
    echo "set-prefix.sh: error processing '" $file "'"
  fi
done
