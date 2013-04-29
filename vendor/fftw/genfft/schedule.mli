(*
 * Copyright (c) 1997-1999 Massachusetts Institute of Technology
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 *)

type schedule =
  | Done
  | Instr of Expr.assignment
  | Seq of (schedule * schedule)
  | Par of schedule list

val schedule : Expr.assignment list -> schedule
val sequentially : Expr.assignment list -> schedule
val isolate_precomputations_and_schedule : Expr.assignment list -> schedule
