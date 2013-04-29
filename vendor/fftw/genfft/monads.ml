(*
 * Copyright (c) 1997-1999 Massachusetts Institute of Technology
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 *)

(*************************************************************
 *   Monads
 *************************************************************)

(*
 * Phil Wadler has many well written papers about monads.  See
 * http://cm.bell-labs.com/cm/cs/who/wadler/ 
 *)
(* vanilla state monad *)
module StateMonad = struct
  let returnM x = fun s -> (x, s)

  let (>>=) = fun m k -> 
    fun s ->
      let (a', s') = m s
      in let (a'', s'') = k a' s'
      in (a'', s'')

  let (>>) = fun m k ->
    m >>= fun _ -> k

  let rec mapM f = function
      [] -> returnM []
    | a :: b ->
	f a >>= fun a' ->
	  mapM f b >>= fun b' ->
	    returnM (a' :: b')

  let runM m x initial_state =
    let (a, _) = m x initial_state
    in a

  let fetchState =
    fun s -> s, s

  let storeState newState =
    fun _ -> (), newState
end

(* monad with built-in memoizing capabilities *)
module MemoMonad =
  struct
    open StateMonad

    let memoizing lookupM insertM f k =
      lookupM k >>= fun vMaybe ->
	match vMaybe with
	  Some value -> returnM value
	| None ->
	    f k >>= fun value ->
	      insertM k value >> returnM value

    let runM initial_state m x  = StateMonad.runM m x initial_state
end
