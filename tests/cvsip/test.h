//
// Copyright (c) 2008 by CodeSourcery
// Copyright (c) 2013 Stefan Seefeld
// All rights reserved.
//
// This file is made available under the GPL.
// See the accompanying file LICENSE.GPL for details.

#include <math.h>

inline float sq_f(float f) { return f*f; }
inline double sq_d(double f) { return f*f; }

inline float
verror_db_f(vsip_vview_f const *v1, vsip_vview_f const *v2)
{
  size_t i;
  float refmax = 0;
  size_t size = vsip_vgetlength_f(v1);

  for (i=0; i<size; ++i)
  {
    if (isnan(vsip_vget_f(v1, i) || isnan(vsip_vget_f(v2, i))))
      return 201.0;

    float magsq = sq_f(vsip_vget_f(v1, i));
    if (magsq > refmax) refmax = magsq;
    magsq = sq_f(vsip_vget_f(v2, i));
    if (magsq > refmax) refmax = magsq;
  }

  float maxsum = -201;
  for (i=0; i<size; ++i)
  {
    float diff = sq_f(vsip_vget_f(v1, i)-vsip_vget_f(v2, i));
    if (diff > 1.e-20)
    {
      float sum = 10.0 * log10(diff/(2.0*refmax));
      if (sum > maxsum)
	maxsum = sum;
    }
  }
  return maxsum;
}

inline double
verror_db_d(vsip_vview_d const *v1, vsip_vview_d const *v2)
{
  size_t i;
  double refmax = 0;
  size_t size = vsip_vgetlength_d(v1);

  for (i=0; i<size; ++i)
  {
    if (isnan(vsip_vget_d(v1, i) || isnan(vsip_vget_d(v2, i))))
      return 201.0;

    double magsq = sq_d(vsip_vget_d(v1, i));
    if (magsq > refmax) refmax = magsq;
    magsq = sq_d(vsip_vget_d(v2, i));
    if (magsq > refmax) refmax = magsq;
  }

  double maxsum = -201;
  for (i=0; i<size; ++i)
  {
    double diff = sq_d(vsip_vget_d(v1, i)-vsip_vget_d(v2, i));
    if (diff > 1.e-20)
    {
      double sum = 10.0 * log10(diff/(2.0*refmax));
      if (sum > maxsum)
	maxsum = sum;
    }
  }
  return maxsum;
}

inline int
vequal_f(vsip_vview_f const *v1, vsip_vview_f const *v2)
{
  printf("vequal %f\n", verror_db_f(v1, v2));
  return verror_db_f(v1, v2) < -100.;
}

inline int
vequal_d(vsip_vview_d const *v1, vsip_vview_d const *v2)
{
  printf("vequal %f\n", verror_db_d(v1, v2));
  return verror_db_d(v1, v2) < -100.;
}

inline float
cverror_db_f(vsip_cvview_f const *v1, vsip_cvview_f const *v2)
{
  size_t i;
  float refmax = 0;
  size_t size = vsip_cvgetlength_f(v1);
  for (i=0; i<size; ++i)
  {
    vsip_cscalar_f s1 = vsip_cvget_f(v1, i);
    vsip_cscalar_f s2 = vsip_cvget_f(v2, i);
    if (isnan(s1.r) || isnan(s1.i) || isnan(s2.r) || isnan(s2.i))
      return 201.0;

    float magsq = sq_f(s1.r) + sq_f(s1.i);
    if (magsq > refmax) refmax = magsq;
    magsq = sq_f(s2.r) + sq_f(s2.i);
    if (magsq > refmax) refmax = magsq;
  }

  float maxsum = -201;
  for (i=0; i<size; ++i)
  {
    vsip_cscalar_f s1 = vsip_cvget_f(v1, i);
    vsip_cscalar_f s2 = vsip_cvget_f(v2, i);
    float diff = sq_f(s1.r-s2.r) + sq_f(s1.i-s2.i);
    if (diff > 1.e-20)
    {
      float sum = 10.0 * log10(diff/(2.0*refmax));
      if (sum > maxsum)
	maxsum = sum;
    }
  }
  return maxsum;
}

inline double
cverror_db_d(vsip_cvview_d const *v1, vsip_cvview_d const *v2)
{
  size_t i;
  double refmax = 0;
  size_t size = vsip_cvgetlength_d(v1);
  for (i=0; i<size; ++i)
  {
    vsip_cscalar_d s1 = vsip_cvget_d(v1, i);
    vsip_cscalar_d s2 = vsip_cvget_d(v2, i);
    if (isnan(s1.r) || isnan(s1.i) || isnan(s2.r) || isnan(s2.i))
      return 201.0;

    double magsq = sq_d(s1.r) + sq_d(s1.i);
    if (magsq > refmax) refmax = magsq;
    magsq = sq_d(s2.r) + sq_d(s2.i);
    if (magsq > refmax) refmax = magsq;
  }

  double maxsum = -201;
  for (i=0; i<size; ++i)
  {
    vsip_cscalar_d s1 = vsip_cvget_d(v1, i);
    vsip_cscalar_d s2 = vsip_cvget_d(v2, i);
    double diff = sq_d(s1.r-s2.r) + sq_d(s1.i-s2.i);
    if (diff > 1.e-20)
    {
      double sum = 10.0 * log10(diff/(2.0*refmax));
      if (sum > maxsum)
	maxsum = sum;
    }
  }
  return maxsum;
}

inline int
cvequal_f(vsip_cvview_f const *v1, vsip_cvview_f const *v2)
{ return cverror_db_f(v1, v2) < -200.;}

inline int
cvequal_d(vsip_cvview_d const *v1, vsip_cvview_d const *v2)
{ return cverror_db_d(v1, v2) < -200.;}

inline float
merror_db_f(vsip_mview_f const *m1, vsip_mview_f const *m2)
{
  size_t r, c;
  float refmax = 0;
  size_t rows = vsip_mgetcollength_f(m1);
  size_t cols = vsip_mgetrowlength_f(m1);
  for (r=0; r<rows; ++r)
    for (c=0; c<cols; ++c)
    {
      if (isnan(vsip_mget_f(m1, r, c)) || isnan(vsip_mget_f(m2, r, c)))
        return 201.0;

      if (vsip_mget_f(m1, r, c)*vsip_mget_f(m1, r, c) > refmax)
	refmax = vsip_mget_f(m1, r, c)*vsip_mget_f(m1, r, c);
      if (vsip_mget_f(m2, r, c)*vsip_mget_f(m2, r, c) > refmax)
	refmax = vsip_mget_f(m2, r, c)*vsip_mget_f(m2, r, c);
    }

  float maxsum = -201;
  for (r=0; r<rows; ++r)
    for (c=0; c<cols; ++c)
    {
      float diff = fabs(vsip_mget_f(m1, r, c) - vsip_mget_f(m2, r, c));
      if (diff > 1.e-20)
      {
	float sum = 10.0 * log10(diff/(2.0*refmax));
	if (sum > maxsum)
	  maxsum = sum;
      }
    }
  return maxsum;
}

inline double
merror_db_d(vsip_mview_d const *m1, vsip_mview_d const *m2)
{
  size_t r, c;
  double refmax = 0;
  size_t rows = vsip_mgetcollength_d(m1);
  size_t cols = vsip_mgetrowlength_d(m1);
  for (r=0; r<rows; ++r)
    for (c=0; c<cols; ++c)
    {
      if (isnan(vsip_mget_d(m1, r, c)) || isnan(vsip_mget_d(m2, r, c)))
        return 201.0;

      if (vsip_mget_d(m1, r, c)*vsip_mget_d(m1, r, c) > refmax)
	refmax = vsip_mget_d(m1, r, c)*vsip_mget_d(m1, r, c);
      if (vsip_mget_d(m2, r, c)*vsip_mget_d(m2, r, c) > refmax)
	refmax = vsip_mget_d(m2, r, c)*vsip_mget_d(m2, r, c);
    }

  double maxsum = -201;
  for (r=0; r<rows; ++r)
    for (c=0; c<cols; ++c)
    {
      double diff = fabs(vsip_mget_d(m1, r, c) - vsip_mget_d(m2, r, c));
      if (diff > 1.e-20)
      {
	double sum = 10.0 * log10(diff/(2.0*refmax));
	if (sum > maxsum)
	  maxsum = sum;
      }
    }
  return maxsum;
}

inline int
mequal_f(vsip_mview_f const *m1, vsip_mview_f const *m2)
{ return merror_db_f(m1, m2) < -200.;}

inline int
mequal_d(vsip_mview_d const *m1, vsip_mview_d const *m2)
{ return merror_db_d(m1, m2) < -200.;}

inline float
cmerror_db_f(vsip_cmview_f const *m1, vsip_cmview_f const *m2)
{
  size_t r, c;
  float refmax = 0;
  size_t rows = vsip_cmgetcollength_f(m1);
  size_t cols = vsip_cmgetrowlength_f(m1);
  for (r=0; r<rows; ++r)
    for (c=0; c<cols; ++c)
    {
      vsip_cscalar_f s1 = vsip_cmget_f(m1, r, c);
      vsip_cscalar_f s2 = vsip_cmget_f(m2, r, c);
      if (isnan(s1.r) || isnan(s1.i) || isnan(s2.r) || isnan(s2.i))
        return 201.0;

      float magsq = sq_f(s1.r) + sq_f(s1.i);
      if (magsq > refmax) refmax = magsq;
      magsq = sq_f(s2.r) + sq_f(s2.i);
      if (magsq > refmax) refmax = magsq;
    }

  float maxsum = -201;
  for (r=0; r<rows; ++r)
    for (c=0; c<cols; ++c)
    {
      vsip_cscalar_f s1 = vsip_cmget_f(m1, r, c);
      vsip_cscalar_f s2 = vsip_cmget_f(m2, r, c);
      float diff = sq_f(s1.r-s2.r) + sq_f(s1.i-s2.i);
      if (diff > 1.e-20)
      {
	float sum = 10.0 * log10(diff/(2.0*refmax));
	if (sum > maxsum)
	  maxsum = sum;
      }
    }
  return maxsum;
}

inline double
cmerror_db_d(vsip_cmview_d const *m1, vsip_cmview_d const *m2)
{
  size_t r, c;
  double refmax = 0;
  size_t rows = vsip_cmgetcollength_d(m1);
  size_t cols = vsip_cmgetrowlength_d(m1);
  for (r=0; r<rows; ++r)
    for (c=0; c<cols; ++c)
    {
      vsip_cscalar_d s1 = vsip_cmget_d(m1, r, c);
      vsip_cscalar_d s2 = vsip_cmget_d(m2, r, c);
      if (isnan(s1.r) || isnan(s1.i) || isnan(s2.r) || isnan(s2.i))
        return 201.0;

      double magsq = sq_d(s1.r) + sq_d(s1.i);
      if (magsq > refmax) refmax = magsq;
      magsq = sq_d(s2.r) + sq_d(s2.i);
      if (magsq > refmax) refmax = magsq;
    }

  double maxsum = -201;
  for (r=0; r<rows; ++r)
    for (c=0; c<cols; ++c)
    {
      vsip_cscalar_d s1 = vsip_cmget_d(m1, r, c);
      vsip_cscalar_d s2 = vsip_cmget_d(m2, r, c);
      double diff = sq_d(s1.r-s2.r) + sq_d(s1.i-s2.i);
      if (diff > 1.e-20)
      {
	double sum = 10.0 * log10(diff/(2.0*refmax));
	if (sum > maxsum)
	  maxsum = sum;
      }
    }
  return maxsum;
}

inline int
cmequal_f(vsip_cmview_f const *m1, vsip_cmview_f const *m2)
{ return cmerror_db_f(m1, m2) < -200.;}

inline int
cmequal_d(vsip_cmview_d const *m1, vsip_cmview_d const *m2)
{ return cmerror_db_d(m1, m2) < -200.;}

inline float
terror_db_f(vsip_tview_f const *t1, vsip_tview_f const *t2)
{
  size_t h, i, j;
  float refmax = 0;
  size_t z_length = vsip_tgetzlength_f(t1);
  size_t y_length = vsip_tgetylength_f(t1);
  size_t x_length = vsip_tgetxlength_f(t1);
  for (h=0; h<z_length; ++h)
    for (i=0; i<y_length; ++i)
      for (j=0; j<x_length; ++j)
      {
        if (isnan(vsip_tget_f(t1, h, i, j)) || isnan(vsip_tget_f(t2, h, i, j)))
          return 201.0;

        if (vsip_tget_f(t1, h, i, j)*vsip_tget_f(t1, h, i, j) > refmax)
          refmax = vsip_tget_f(t1, h, i, j)*vsip_tget_f(t1, h, i, j);
        if (vsip_tget_f(t2, h, i, j)*vsip_tget_f(t2, h, i, j) > refmax)
          refmax = vsip_tget_f(t2, h, i, j)*vsip_tget_f(t2, h, i, j);
      }

  float maxsum = -201;
  for (h=0; h<z_length; ++h)
    for (i=0; i<y_length; ++i)
      for (j=0; j<x_length; ++j)
      {
        float diff = fabs(vsip_tget_f(t1, h, i, j) - vsip_tget_f(t2, h, i, j));
        if (diff > 1.e-20)
        {
          float sum = 10.0 * log10(diff/(2.0*refmax));
          if (sum > maxsum)
            maxsum = sum;
        }
      }
  return maxsum;
}

inline double
terror_db_d(vsip_tview_d const *t1, vsip_tview_d const *t2)
{
  size_t h, i, j;
  double refmax = 0;
  size_t z_length = vsip_tgetzlength_d(t1);
  size_t y_length = vsip_tgetylength_d(t1);
  size_t x_length = vsip_tgetxlength_d(t1);
  for (h=0; h<z_length; ++h)
    for (i=0; i<y_length; ++i)
      for (j=0; j<x_length; ++j)
      {
        if (isnan(vsip_tget_d(t1, h, i, j)) || isnan(vsip_tget_d(t2, h, i, j)))
          return 201.0;

        if (vsip_tget_d(t1, h, i, j)*vsip_tget_d(t1, h, i, j) > refmax)
          refmax = vsip_tget_d(t1, h, i, j)*vsip_tget_d(t1, h, i, j);
        if (vsip_tget_d(t2, h, i, j)*vsip_tget_d(t2, h, i, j) > refmax)
          refmax = vsip_tget_d(t2, h, i, j)*vsip_tget_d(t2, h, i, j);
      }

  double maxsum = -201;
  for (h=0; h<z_length; ++h)
    for (i=0; i<y_length; ++i)
      for (j=0; j<x_length; ++j)
      {
        double diff = fabs(vsip_tget_d(t1, h, i, j) - vsip_tget_d(t2, h, i, j));
        if (diff > 1.e-20)
        {
          double sum = 10.0 * log10(diff/(2.0*refmax));
          if (sum > maxsum)
            maxsum = sum;
        }
      }
  return maxsum;
}

inline int
tequal_f(vsip_tview_f const *t1, vsip_tview_f const *t2)
{ return terror_db_f(t1, t2) < -200.;}

inline int
tequal_d(vsip_tview_d const *t1, vsip_tview_d const *t2)
{ return terror_db_d(t1, t2) < -200.;}

inline float
cterror_db_f(vsip_ctview_f const *t1, vsip_ctview_f const *t2)
{
  size_t h, i, j;
  float refmax = 0;
  size_t z_length = vsip_ctgetzlength_f(t1);
  size_t y_length = vsip_ctgetylength_f(t1);
  size_t x_length = vsip_ctgetxlength_f(t1);
  for (h=0; h<z_length; ++h)
    for (i=0; i<y_length; ++i)
      for (j=0; j<x_length; ++j)
      {
        vsip_cscalar_f s1 = vsip_ctget_f(t1, h, i, j);
        vsip_cscalar_f s2 = vsip_ctget_f(t2, h, i, j);
        if (isnan(s1.r) || isnan(s1.i) || isnan(s2.r) || isnan(s2.i))
          return 201.0;

        float magsq = sq_f(s1.r) + sq_f(s1.i);
        if (magsq > refmax) refmax = magsq;
        magsq = sq_f(s2.r) + sq_f(s2.i);
        if (magsq > refmax) refmax = magsq;
      }

  float maxsum = -201;
  for (h=0; h<z_length; ++h)
    for (i=0; i<y_length; ++i)
      for (j=0; j<x_length; ++j)
      {
        vsip_cscalar_f s1 = vsip_ctget_f(t1, h, i, j);
        vsip_cscalar_f s2 = vsip_ctget_f(t2, h, i, j);
        float diff = sq_f(s1.r-s2.r) + sq_f(s1.i-s2.i);
        if (diff > 1.e-20)
        {
          float sum = 10.0 * log10(diff/(2.0*refmax));
          if (sum > maxsum)
            maxsum = sum;
        }
      }
  return maxsum;
}

inline double
cterror_db_d(vsip_ctview_d const *t1, vsip_ctview_d const *t2)
{
  size_t h, i, j;
  double refmax = 0;
  size_t z_length = vsip_ctgetzlength_d(t1);
  size_t y_length = vsip_ctgetylength_d(t1);
  size_t x_length = vsip_ctgetxlength_d(t1);
  for (h=0; h<z_length; ++h)
    for (i=0; i<y_length; ++i)
      for (j=0; j<x_length; ++j)
      {
        vsip_cscalar_d s1 = vsip_ctget_d(t1, h, i, j);
        vsip_cscalar_d s2 = vsip_ctget_d(t2, h, i, j);
        if (isnan(s1.r) || isnan(s1.i) || isnan(s2.r) || isnan(s2.i))
          return 201.0;

        double magsq = sq_d(s1.r) + sq_d(s1.i);
        if (magsq > refmax) refmax = magsq;
        magsq = sq_d(s2.r) + sq_d(s2.i);
        if (magsq > refmax) refmax = magsq;
      }

  double maxsum = -201;
  for (h=0; h<z_length; ++h)
    for (i=0; i<y_length; ++i)
      for (j=0; j<x_length; ++j)
      {
        vsip_cscalar_d s1 = vsip_ctget_d(t1, h, i, j);
        vsip_cscalar_d s2 = vsip_ctget_d(t2, h, i, j);
        double diff = sq_d(s1.r-s2.r) + sq_d(s1.i-s2.i);
        if (diff > 1.e-20)
        {
          double sum = 10.0 * log10(diff/(2.0*refmax));
          if (sum > maxsum)
            maxsum = sum;
        }
      }
  return maxsum;
}

inline int
ctequal_f(vsip_ctview_f const *t1, vsip_ctview_f const *t2)
{ return cterror_db_f(t1, t2) < -200.;}

inline int
ctequal_d(vsip_ctview_d const *t1, vsip_ctview_d const *t2)
{ return cterror_db_d(t1, t2) < -200.;}

inline void
test_assert_(int expr, char const *assertion, char const *file, unsigned int line)
{
  if (!expr)
  {
    fprintf(stderr, "TEST ASSERT FAIL: %s %s %d\n", assertion, file, line);
    abort();
  }
}

#define test_assert(expr) test_assert_(expr, #expr, __FILE__, __LINE__)

