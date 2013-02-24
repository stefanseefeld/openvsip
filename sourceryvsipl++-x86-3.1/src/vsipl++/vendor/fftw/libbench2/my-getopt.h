/*
 * Copyright (c) 2003, 2007-8 Matteo Frigo
 * Copyright (c) 2003, 2007-8 Massachusetts Institute of Technology
 *
 * See the file COPYING for license information.
 *
 */

#ifndef __MY_GETOPT_H__
#define __MY_GETOPT_H__

#ifdef __cplusplus
extern "C" {
#endif                          /* __cplusplus */

enum { REQARG, OPTARG, NOARG };

struct my_option {
     const char *long_name;
     int argtype;
     int short_name;
};

extern int my_optind;
extern const char *my_optarg;

extern void my_usage(const char *progname, const struct my_option *opt);
extern int my_getopt(int argc, char *argv[], const struct my_option *optarray);

#ifdef __cplusplus
}                               /* extern "C" */
#endif                          /* __cplusplus */

#endif /* __MY_GETOPT_H__ */
