(*
 * Copyright (c) 1997-1999 Massachusetts Institute of Technology
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 *)

val makeNum : Number.number -> Expr.expr
val makeUminus : Expr.expr -> Expr.expr
val makeTimes : Expr.expr * Expr.expr -> Expr.expr
val makePlus : Expr.expr list -> Expr.expr
