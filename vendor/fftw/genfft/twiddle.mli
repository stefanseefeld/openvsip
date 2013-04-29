(*
 * Copyright (c) 1997-1999 Massachusetts Institute of Technology
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 *)

val speclist : (string * Arg.spec * string) list

type twinstr

val twiddle_policy :
  int -> bool ->
  (int -> int -> (int -> Complex.expr) -> (int -> Complex.expr) ->
     int -> Complex.expr) *(int -> int) * (int -> twinstr list)

val twinstr_to_c_string : twinstr list -> string
val twinstr_to_simd_string : string -> twinstr list -> string
