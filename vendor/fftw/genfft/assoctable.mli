(*
 * Copyright (c) 1997-1999 Massachusetts Institute of Technology
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 *)

type ('a, 'b) elem =
  | Leaf
  | Node of int * ('a, 'b) elem * ('a, 'b) elem * ('a * 'b) list
val empty : ('a, 'b) elem
val lookup :
    ('a -> int) -> ('a -> 'b -> bool) -> 'a -> ('b, 'c) elem -> 'c option
val insert :
    ('a -> int) -> 'a -> 'c -> ('a, 'c) elem -> ('a, 'c) elem
