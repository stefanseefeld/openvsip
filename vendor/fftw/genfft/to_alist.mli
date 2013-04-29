(*
 * Copyright (c) 1997-1999 Massachusetts Institute of Technology
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 *)

val to_assignments : Expr.expr list -> Expr.assignment list
val dump : (string -> unit) -> Expr.assignment list -> unit
val good_for_fma : Expr.expr * Expr.expr -> bool
