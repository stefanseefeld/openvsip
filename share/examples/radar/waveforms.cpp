/* Copyright (c) 2010, 2011 CodeSourcery, Inc.  All rights reserved. */

/* Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

       * Redistributions of source code must retain the above copyright
         notice, this list of conditions and the following disclaimer.
       * Redistributions in binary form must reproduce the above
         copyright notice, this list of conditions and the following
         disclaimer in the documentation and/or other materials
         provided with the distribution.
       * Neither the name of CodeSourcery nor the names of its
         contributors may be used to endorse or promote products
         derived from this software without specific prior written
         permission.

   THIS SOFTWARE IS PROVIDED BY CODESOURCERY, INC. "AS IS" AND ANY
   EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
   PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL CODESOURCERY BE LIABLE FOR
   ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
   CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
   SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
   BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
   WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
   OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
   EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  */

/// Description
///   Use SV++ to generate some common signal waveforms: a rectangular
///   pulse with a given amplitude and width and a linear FM chirp with
///   a specified frequency rate and duration.

#include <cassert>

#include <vsip/vector.hpp>
#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/signal.hpp>
#include <vsip/selgen.hpp>
#include <vsip/math.hpp>

using namespace vsip;

// Function objects to generate waveforms of a rectangular pulse and a linear
//  FM chirp.  The objects are constructed with the parameters of the signal
//  shape while realizations of a given signal with a specified time
//  vector are returned from the function call.

// Pulse Waveform
template <typename T>
class
Pulse
{
public:
  Pulse(T amplitude, scalar_f duration)
  : amplitude_(amplitude), duration_(duration) {}

  Vector<T>
  operator()(const_Vector<scalar_f> time, scalar_f start = 0.0)
  {
    length_type N = time.size();
    Vector<T> out(N, T());

    for (index_type i = 0; i < N; ++i)
      out(i) = (time(i) < (duration_ + start) && time(i) >= start) ? T(amplitude_) : T();

    return out;
  }
private:
  T amplitude_;       // Pulse amplitude
  scalar_f duration_; // Pulse duration
};

// Chirp Waveform
template <typename T>
class
Chirp
{
public:
  Chirp(scalar_f frequency_rate): frequency_rate_(frequency_rate) {}

  Vector<T>
  operator()(const_Vector<scalar_f> time, scalar_f start = 0.0)
  {
    length_type N = time.size();
    Vector<T> out(N, T());
    Vector<T> phase(N, T());

    phase.imag() = -2.f * OVXX_PI * frequency_rate_ * (time - start) * (time - start);
    out = exp(phase);

    return out;
  }

  // Overload operator() to allow the use of a weighting function
  Vector<T>
  operator()(const_Vector<scalar_f> time,
             const_Vector<scalar_f> weights,
             scalar_f start = 0.0)
  {
    length_type N = time.size();
    Vector<T> out(N, T());
    Vector<T> phase(N, T());

    phase.imag() = -2.f * OVXX_PI * frequency_rate_ * (time - start) * (time - start);
    out = weights * exp(phase);

    return out;
  }

private:
  scalar_f frequency_rate_; // Frequency sweep rate
};

// Specialization for generating a real chirp signal without retaining
//  the phase information.
template <>
class
Chirp<scalar_f>
{
public:
  Chirp(scalar_f frequency_rate): frequency_rate_(frequency_rate) {}

  Vector<scalar_f>
  operator()(const_Vector<scalar_f> time, scalar_f start = 0.0)
  {
    length_type N = time.size();
    Vector<cscalar_f> out(N, cscalar_f());
    Vector<cscalar_f> phase(N, cscalar_f());

    phase.imag() = -2.f * OVXX_PI * frequency_rate_ * (time - start) * (time - start);
    out = exp(phase);

    return out.real();
  }

  // Overload operator() to allow the use of a weighting function
  Vector<scalar_f>
  operator()(const_Vector<scalar_f> time,
             const_Vector<scalar_f> weights,
             scalar_f start = 0.0)
  {
    length_type N = time.size();
    Vector<cscalar_f> out(N, cscalar_f());
    Vector<cscalar_f> phase(N, cscalar_f());

    phase.imag() = -2.f * OVXX_PI * frequency_rate_ * (time - start) * (time - start);
    out = weights * exp(phase);

    return out.real();
  }

private:
  scalar_f frequency_rate_; // Frequency sweep rate
};

int
main(int argc, char **argv)
{
  vsipl init(argc, argv);

  // Initialize a vector of times
  length_type N = 16;                   // Signal length
  scalar_f t_max = 10;                  // End time in seconds
  scalar_f secs_per_sample = t_max / N; // Sample duration

  Vector<scalar_f> t(N);                // Time vector
  t = ramp(0.0f, secs_per_sample, N);   // Populate the time vector

  // First generate a unit rectangular pulse of duration 2 seconds
  Vector<scalar_f> pulse_signal(N);

  scalar_f amp = 1.0;          // Amplitude
  scalar_f dur = 2.0;          // Pulse duration in seconds

  // Set the pulse parameters and ...
  Pulse<scalar_f> pulse(amp, dur);

  // ... generate the pulse.
  pulse_signal = pulse(t);

  // Now generate a pulse that starts 3 seconds after the initial time.
  scalar_f start = 3.0;       // Start time in seconds
  pulse_signal = pulse(t, start);

  // Generate a signal with a linear frequency sweep at a rate of 5 Hz/s.
  Vector<cscalar_f> chirp_signal(N);

  scalar_f rate = 5.0;

  // As before, set the parameters and then generate the signal.
  Chirp<cscalar_f> chirp(rate);

  chirp_signal = chirp(t);

  // Just as before, delay the signal starting time.
  chirp_signal = chirp(t, start);

  // Now, generate a chirp over a pulse in order to define the amplitude
  //  and duration of the signal, here use the pulse we have already
  //  constructed.
  chirp_signal = chirp(t, pulse(t));

  // This signal can be delayed arbitrarily as well
  chirp_signal = chirp(t, pulse(t, start), start);

  // Rather than a pulse, now generate the chirp using a different
  //  amplitude weighting.  For example, a Hanning window.
  chirp_signal = chirp(t, hanning(N));

  // It is also possible to generate only the real component of the chirp
  //  signal by specifying 'scalar_f' rather than 'cscalar_f' as the
  //  template argument to the chirp.
  Chirp<scalar_f> chirp_real(rate);

  // Store the signal in the previously used 'pulse_signal'.
  pulse_signal = chirp_real(t);

  return 0;
}
