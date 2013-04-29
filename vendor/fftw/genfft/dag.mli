(*
 * Copyright (c) 1997-1999 Massachusetts Institute of Technology
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 *)

open Util

type color = | RED | BLUE | BLACK | YELLOW

type dagnode = 
    { assigned: Variable.variable;
      mutable expression: Expr.expr;
      input_variables: Variable.variable list;
      mutable successors: dagnode list;
      mutable predecessors: dagnode list;
      mutable label: int;
      mutable color: color}

type dag

val makedag : (Variable.variable * Expr.expr) list -> dag

val map : (dagnode -> dagnode) -> dag -> dag
val for_all : dag -> (dagnode -> unit) -> unit
val to_list : dag -> (dagnode list)
val bfs : dag -> dagnode -> int -> unit
val find_node : (dagnode -> bool) -> dag -> dagnode option
