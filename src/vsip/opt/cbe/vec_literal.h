/* Copyright (c) 2007 by CodeSourcery.  All rights reserved.
*/
/** @file    vsip/opt/cbe/vec_literal.h
    @author  Brooks Moses
    @date    2007-10-16
    @brief   VSIPL++ Library: Compatibility file for IBM SDK 2.1 vec_literal.h.
*/

#define VEC_LITERAL(TYPE, ...) ((TYPE){__VA_ARGS__})
