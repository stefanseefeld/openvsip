(*
 * Copyright (c) 1997-1999 Massachusetts Institute of Technology
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 *)

val should_flip_sign : Expr.expr -> bool
val likely_equal : Expr.expr -> Expr.expr -> bool
val hash : Expr.expr -> int
