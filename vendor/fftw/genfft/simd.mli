(*
 * Copyright (c) 1997-1999 Massachusetts Institute of Technology
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 *)

val unparse_function : C.c_fcn -> string
val extract_constants : C.c_ast -> C.c_decl list
val realtype : string
val realtypep : string
val constrealtype : string
val constrealtypep : string

