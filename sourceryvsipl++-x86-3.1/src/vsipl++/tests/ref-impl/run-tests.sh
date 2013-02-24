#! /bin/sh

TESTS="view-math admitrelease complex dense math math-matvec math-reductions \
	     math-scalarview matrix matrix-math matrix-const \
	     random selgen \
	     signal signal-correlation \
	     signal-fir signal-histogram signal-windows \
	     fft-coverage \
	     solvers-covsol solvers-chol solvers-lu solvers-qr \
	     vector vector-math vector-const \
	     signal-convolution signal-fft"


for test in $TESTS; do
   echo $test
   $test
done
