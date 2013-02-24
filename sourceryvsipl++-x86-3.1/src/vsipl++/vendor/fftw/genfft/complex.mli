(*
 * Copyright (c) 1997-1999 Massachusetts Institute of Technology
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 *)

type expr
val make : (Expr.expr * Expr.expr) -> expr
val two : expr
val one : expr
val i : expr
val zero : expr
val half : expr
val inverse_int : int -> expr
val inverse_int_sqrt : int -> expr
val int_sqrt : int -> expr
val times : expr -> expr -> expr
val ctimes : expr -> expr -> expr
val ctimesj : expr -> expr -> expr
val uminus : expr -> expr
val exp : int -> int -> expr
val sec : int -> int -> expr
val csc : int -> int -> expr
val tan : int -> int -> expr
val cot : int -> int -> expr
val plus : expr list -> expr
val real : expr -> expr
val imag : expr -> expr
val conj : expr -> expr
val nan : Expr.transcendent -> expr
val sigma : int -> int -> (int -> expr) -> expr

val (@*) : expr -> expr -> expr
val (@+) : expr -> expr -> expr
val (@-) : expr -> expr -> expr

(* a signal is a map from integers to expressions *)
type signal = int -> expr
val infinite : int -> signal -> signal

val store_real : Variable.variable -> expr -> Expr.expr
val store_imag : Variable.variable -> expr -> Expr.expr
val store :
  Variable.variable * Variable.variable -> expr -> Expr.expr * Expr.expr

val assign_real : Variable.variable -> expr -> Expr.assignment
val assign_imag : Variable.variable -> expr -> Expr.assignment
val assign :
  Variable.variable * Variable.variable ->
  expr -> Expr.assignment * Expr.assignment

val hermitian : int -> (int -> expr) -> int -> expr
val antihermitian : int -> (int -> expr) -> int -> expr
