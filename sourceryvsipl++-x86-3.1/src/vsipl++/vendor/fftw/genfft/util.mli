(*
 * Copyright (c) 1997-1999 Massachusetts Institute of Technology
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 *)

val invmod : int -> int -> int
val gcd : int -> int -> int
val lowest_terms : int -> int -> int * int
val find_generator : int -> int
val pow_mod : int -> int -> int -> int
val forall : 'a -> ('b -> 'a -> 'a) -> int -> int -> (int -> 'b) -> 'a
val sum_list : int list -> int
val max_list : int list -> int
val min_list : int list -> int
val count : ('a -> bool) -> 'a list -> int
val remove : 'a -> 'a list -> 'a list
val for_list : 'a list -> ('a -> unit) -> unit
val rmap : 'a list -> ('a -> 'b) -> 'b list
val cons : 'a -> 'a list -> 'a list
val null : 'a list -> bool
val (@@) : ('a -> 'b) -> ('c -> 'a) -> 'c -> 'b
val forall_flat : int -> int -> (int -> 'a list) -> 'a list
val identity : 'a -> 'a
val minimize : ('a -> 'b) -> 'a list -> 'a option
val find_elem : ('a -> bool) -> 'a list -> 'a option
val suchthat : int -> (int -> bool) -> int
val info : string -> unit
val iota : int -> int list
val interval : int -> int -> int list
val array : int -> (int -> 'a) -> int -> 'a
val take : int -> 'a list -> 'a list
val drop : int -> 'a list -> 'a list
val either : 'a option -> 'a -> 'a
