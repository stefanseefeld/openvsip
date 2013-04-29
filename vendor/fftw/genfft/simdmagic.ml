(*
 * Copyright (c) 1997-1999 Massachusetts Institute of Technology
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 *)

(* SIMD magic parameters *)
let simd_mode = ref false
let store_multiple = ref 1

open Magic

let speclist = [
  "-simd", set_bool simd_mode, undocumented;
  "-store-multiple", set_int store_multiple, undocumented;
]
