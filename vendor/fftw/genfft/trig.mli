(*
 * Copyright (c) 1997-1999 Massachusetts Institute of Technology
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 *)

val rdft : int -> int -> Complex.signal -> Complex.signal
val hdft : int -> int -> Complex.signal -> Complex.signal
val dft_via_rdft : int -> int -> Complex.signal -> Complex.signal
val dht : int -> int -> Complex.signal -> Complex.signal

val dctI : int -> Complex.signal -> Complex.signal
val dctII : int -> Complex.signal -> Complex.signal
val dctIII : int -> Complex.signal -> Complex.signal
val dctIV : int -> Complex.signal -> Complex.signal

val dstI : int -> Complex.signal -> Complex.signal
val dstII : int -> Complex.signal -> Complex.signal
val dstIII : int -> Complex.signal -> Complex.signal
val dstIV : int -> Complex.signal -> Complex.signal
