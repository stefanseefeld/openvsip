#!/bin/bash

# set -x

# To get some tracing out of this script, set the environment
# variable DBGRUNSH to any non-null value.
#
dbgecho() {
  if [ -n "$DBGRUNSH" ] ; then
    echo $*
  fi
}

# Record the process id of this script.
#
export PID=$$
dbgecho "PID is $PID"

# If not specified, time out after one minute.  Set it to 0 (zero)
# if you don't want to time out.
#
if [ -z "$TIMEOUT" ] ; then
  TIMEOUT=60
fi
export TIMEOUT

# If you don't like Mike's, set your own in SYSROOTRUN.

if [ -z "$SYSROOTRUN" ] ; then
  SYSROOTRUN=/net/install/install/gnu/sourceryg++/4.3-53/i686-mingw32/powerpc-linux-gnu/libc/te600/usr/bin/sysroot-run
fi

# Start the production process in the background

export CMD="$@"

if [ -n "$RUNSHSIMPLE" ] ; then
  $SYSROOTRUN $CMD
  exit $?
fi

# define exit function
exit_timeout() {
  dbgecho ".. arrive in exit_timeout"
  PRODPID=`cat .prodpid`
  dbgecho kill_process -TERM $PRODPID "from exit_timeout"
  kill -TERM $PRODPID
  # timeout exit
  exit
}

# Handler for signal USR1 for the timer
trap exit_timeout SIGUSR1

# starting timer in subshell. It sends a SIGUSR1 to the father if it timeouts.
timer_process() {
  exit_timer() {
    dbgecho ".. arrive in exit_timer"
    kill $timerpid
    exit
  }
  trap exit_timer SIGUSR1
  touch .timer
  dbgecho ".. enter timer, pid=$$"
  sleep $TIMEOUT &
  timerpid=$!
  wait $timerpid
  dbgecho "... done sleep loop, try to kill"
  kill -SIGUSR1 $PID
  rm .timer
  touch .timeout
  (
    date
    echo $CMD
  ) > .timeout 2>&1
}

if [ $TIMEOUT -gt 0 ] ; then
  timer_process &
  echo $! > .timer
fi

$SYSROOTRUN $CMD &
echo $! > .prodpid
dbgecho "Process running."
#echo "Normal termination: $SYSROOTRUN $CMD" >&2

sleep 2

# wait for the production process to finish
PRODPID=`cat .prodpid`
#echo "		Waiting for $PRODPID ..." >&2
wait $PRODPID
#echo "		Done waiting ..." >&2

# Normal exit.  If the timer still exists, kill it.

# kill timer
if [ -f .timer ] ; then
  TPID=`cat .timer`
  rm .timer
  kill -SIGUSR1 $TPID 2> /dev/null
fi
