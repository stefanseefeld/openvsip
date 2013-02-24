/* Copyright (c) 2008 by CodeSourcery.  All rights reserved.

   This file is available for license from CodeSourcery, Inc. under the terms
   of a commercial license.  It is not part of the VSIPL++
   reference implementation and is not available under the GPL or BSD licenses.
*/
/** @file    tests/cvsip/output.h
    @author  Stefan Seefeld
    @date    2008-05-02
    @brief   view output support functions.
*/

#ifndef output_h_
#define output_h_

void voutput_f(vsip_vview_f const *v)
{
  size_t i;
  for (i = 0; i != vsip_vgetlength_f(v); ++i)
    printf("%f ", vsip_vget_f(v, i));
}

void voutput_d(vsip_vview_d const *v)
{
  size_t i;
  for (i = 0; i != vsip_vgetlength_d(v); ++i)
    printf("%f ", vsip_vget_d(v, i));
}

void cvoutput_f(vsip_cvview_f const *v)
{
  size_t i;
  for (i = 0; i != vsip_cvgetlength_f(v); ++i)
  {
    vsip_cscalar_f value = vsip_cvget_f(v, i);
    printf("(%f %f) ", value.r, value.i);
  }
}

void cvoutput_d(vsip_cvview_d const *v)
{
  size_t i;
  for (i = 0; i != vsip_cvgetlength_d(v); ++i)
  {
    vsip_cscalar_d value = vsip_cvget_d(v, i);
    printf("(%f %f) ", value.r, value.i);
  }
}

void moutput_f(vsip_mview_f const *m)
{
  size_t i, j;
  for (i = 0; i != vsip_mgetcollength_f(m); ++i)
  {
    for (j = 0; j != vsip_mgetrowlength_f(m); ++j)
      printf("%f ", vsip_mget_f(m, i, j));
    printf("\n");
  }
}

void moutput_d(vsip_mview_d const *m)
{
  size_t i, j;
  for (i = 0; i != vsip_mgetcollength_d(m); ++i)
  {
    for (j = 0; j != vsip_mgetrowlength_d(m); ++j)
      printf("%f ", vsip_mget_d(m, i, j));
    printf("\n");
  }
}

void cmoutput_f(vsip_cmview_f const *m)
{
  size_t i, j;
  for (i = 0; i != vsip_cmgetcollength_f(m); ++i)
  {
    for (j = 0; j != vsip_cmgetrowlength_f(m); ++j)
    {
      vsip_cscalar_f tmp = vsip_cmget_f(m, i, j);
      printf("(%f %f) ", tmp.r, tmp.i);
    }
    printf("\n");
  }
}

void cmoutput_d(vsip_cmview_d const *m)
{
  size_t i, j;
  for (i = 0; i != vsip_cmgetcollength_d(m); ++i)
  {
    for (j = 0; j != vsip_cmgetrowlength_d(m); ++j)
    {
      vsip_cscalar_d tmp = vsip_cmget_d(m, i, j);
      printf("(%f %f) ", tmp.r, tmp.i);
    }
    printf("\n");
  }
}

void toutput_f(vsip_tview_f const *t)
{
  size_t h, i, j;
  for (h = 0; h != vsip_tgetzlength_f(t); ++h)
  {
    printf("plane %d :\n", (int)h);
    for (i = 0; i != vsip_tgetylength_f(t); ++i)
    {
      for (j = 0; j != vsip_tgetxlength_f(t); ++j)
        printf("%f ", vsip_tget_f(t, h, i, j));
      printf("\n");
    }
  }
}

void toutput_d(vsip_tview_d const *t)
{
  size_t h, i, j;
  for (h = 0; h != vsip_tgetzlength_d(t); ++h)
  {
    printf("plane %d :\n", (int)h);
    for (i = 0; i != vsip_tgetylength_d(t); ++i)
    {
      for (j = 0; j != vsip_tgetxlength_d(t); ++j)
        printf("%f ", vsip_tget_d(t, h, i, j));
      printf("\n");
    }
  }
}

void ctoutput_f(vsip_ctview_f const *t)
{
  size_t h, i, j;
  for (h = 0; h != vsip_ctgetzlength_f(t); ++h)
  {
    printf("plane %d :\n", (int)h);
    for (i = 0; i != vsip_ctgetylength_f(t); ++i)
    {
      for (j = 0; j != vsip_ctgetxlength_f(t); ++j)
      {
        vsip_cscalar_f tmp = vsip_ctget_f(t, h, i, j);
        printf("(%f %f) ", tmp.r, tmp.i);
      }
      printf("\n");
    }
  }
}

void ctoutput_d(vsip_ctview_d const *t)
{
  size_t h, i, j;
  for (h = 0; h != vsip_ctgetzlength_d(t); ++h)
  {
    printf("plane %d :\n", (int)h);
    for (i = 0; i != vsip_ctgetylength_d(t); ++i)
    {
      for (j = 0; j != vsip_ctgetxlength_d(t); ++j)
      {
        vsip_cscalar_d tmp = vsip_ctget_d(t, h, i, j);
        printf("(%f %f) ", tmp.r, tmp.i);
      }
      printf("\n");
    }
  }
}

#endif
