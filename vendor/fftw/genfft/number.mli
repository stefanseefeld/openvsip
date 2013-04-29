(*
 * Copyright (c) 1997-1999 Massachusetts Institute of Technology
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 *)

type number

val equal : number -> number -> bool
val of_int : int -> number
val zero : number
val one : number
val two : number
val mone : number
val is_zero : number -> bool
val is_one : number -> bool
val is_mone : number -> bool
val is_two : number -> bool
val mul : number -> number -> number
val div : number -> number -> number
val add : number -> number -> number
val sub : number -> number -> number
val negative : number -> bool
val greater : number -> number -> bool
val negate : number -> number
val sqrt : number -> number

(* cexp n i = (cos (2 * pi * i / n), sin (2 * pi * i / n)) *)
val cexp : int -> int -> (number * number)

val to_konst : number -> string
val to_string : number -> string
val to_float : number -> float

