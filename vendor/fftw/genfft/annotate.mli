(*
 * Copyright (c) 1997-1999 Massachusetts Institute of Technology
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 *)

open Variable
open Expr

type annotated_schedule = 
    Annotate of variable list * variable list * variable list *
	int * aschedule
and aschedule = 
    ADone
  | AInstr of assignment
  | ASeq of (annotated_schedule * annotated_schedule)

val annotate :
  variable list list -> Schedule.schedule -> annotated_schedule

val dump : (string -> unit) -> annotated_schedule -> unit
