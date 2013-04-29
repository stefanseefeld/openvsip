(*
 * Copyright (c) 1997-1999 Massachusetts Institute of Technology
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 *)

type transcendent = I | MULTI_A | MULTI_B | CONJ

type expr =
  | Num of Number.number
  | NaN of transcendent
  | Plus of expr list
  | Times of expr * expr
  | CTimes of expr * expr
  | CTimesJ of expr * expr
  | Uminus of expr
  | Load of Variable.variable
  | Store of Variable.variable * expr

type assignment = Assign of Variable.variable * expr

val hash_float : float -> int
val hash : expr -> int
val to_string : expr -> string
val assignment_to_string : assignment -> string
val transcendent_to_float : transcendent -> float
val string_of_transcendent : transcendent -> string

val find_vars : expr -> Variable.variable list
val is_constant : expr -> bool
val is_known_constant : expr -> bool

val dump : (string -> unit) -> assignment list -> unit

val expr_to_constants : expr -> Number.number list
val unique_constants : Number.number list -> Number.number list
