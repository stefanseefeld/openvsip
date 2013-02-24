(*
 * Copyright (c) 1997-1999 Massachusetts Institute of Technology
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 *)

type variable

val hash : variable -> int
val same : variable -> variable -> bool
val is_constant : variable -> bool
val is_temporary : variable -> bool
val is_locative : variable -> bool
val same_location : variable -> variable -> bool
val same_class : variable -> variable -> bool
val make_temporary : unit -> variable
val make_constant : Unique.unique -> string -> variable
val make_locative :
  Unique.unique -> Unique.unique -> (int -> string) -> 
  int -> string -> variable
val unparse : variable -> string
val unparse_for_alignment : int -> variable -> string
val vstride_of_locative : variable -> string
