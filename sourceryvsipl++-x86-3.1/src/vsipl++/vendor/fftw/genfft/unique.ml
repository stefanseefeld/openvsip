(*
 * Copyright (c) 1997-1999 Massachusetts Institute of Technology
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 *)

(* repository of unique tokens *)

type unique = Unique of unit

(* this depends on the compiler not being too smart *)
let make () =
  let make_aux x = Unique x in
  make_aux ()

(* note that the obvious definition

      let make () = Unique ()

   fails *)

let same (a : unique) (b : unique) =
  (a == b)
