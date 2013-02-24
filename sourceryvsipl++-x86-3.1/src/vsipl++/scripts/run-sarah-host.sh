#! /bin/sh
# set -x
pgm=$1
shift
scp -q $(dirname "$0")/run-sarah-target.sh sarah:/tmp/run-sarah-target-${USER}.sh
scp -q $pgm sarah:/tmp/svxx-temp-pgm-${USER}
ssh sarah SYSROOTRUN=/net/henry4/scratch/brooks/te600/usr/bin/sysroot-run \
  RUNSHSIMPLE=1 /tmp/run-sarah-target-${USER}.sh /tmp/svxx-temp-pgm-${USER} $*
