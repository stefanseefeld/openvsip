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
///   Filter an incoming signal which has been delayed by 'delay'.  Use
///   a matched filter to detect and reconstruct the signal delay.
///
///   In this simulation two signals will be used: a rectangular pulse
///   and a "chirp" where the frequency is swept linearly.

#include <cassert>

#include <vsip/vector.hpp>
#include <vsip/initfin.hpp>
#include <vsip/support.hpp>
#include <vsip/signal.hpp>
#include <vsip/selgen.hpp>
#include <vsip/math.hpp>
#include <iostream>

using namespace vsip;

// Function objects for a pulse and chirp waveform.  By using a standard form
//  for the function signature they can be used to define a matched filter
//  object.

// Pulse Waveform
template <typename T>
class Pulse
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
  T amplitude_;        // Signal amplitude
  scalar_f duration_;  // Signal duration
};

// Chirp Waveform
template <typename T>
class Chirp
{
  typedef typename ovxx::scalar_of<T>::type scalar_type;
  typedef complex<scalar_type> complex_type;

public:
  Chirp(scalar_f frequency_rate, scalar_f duration)
  : frequency_rate_(frequency_rate), duration_(duration) {}

  Vector<T>
  operator()(const_Vector<scalar_f> time, scalar_f start = 0.0)
  {
    length_type N = time.size();
    Vector<T> out(N, T());
    Vector<T> phase(N, T());
    Vector<scalar_f> weights(N, scalar_f());

    phase.imag() = OVXX_PI * frequency_rate_ * (time - start) * (time - start);

    for (index_type i = 0; i < N; ++i)
      weights(i) = (time(i) < (duration_ + start) && time(i) >= start) ? scalar_f(1) : scalar_f();

    out = weights * exp(phase);

    return out;
  }

private:
  scalar_f frequency_rate_; // Frequency sweep rate Hz/s
  scalar_f duration_;       // Signal duration
};


// Matched filter class, the constructor takes a vector of times (delays) and
//  frequencies (doppler shifts) and a function specifying the time-domain
//  signal corresponding to the filter coefficients.
template <typename RefT, typename InT, typename SignalT>
class Matched_filter
{
  typedef Fftm<cscalar_f, cscalar_f, row, fft_fwd, by_value, 0> f_fftm_type;
  typedef Fftm<cscalar_f, cscalar_f, row, fft_inv, by_value, 0> i_fftm_type;
  typedef Fft<Vector, scalar_f, cscalar_f, 0> f_fft_rc_type;

public:
  Matched_filter(Vector<scalar_f> t,
                 Vector<scalar_f> freqs,
                 SignalT ref_signal)
  : Nt_(t.size()),
    Nf_(freqs.size()),
    f_fftm_(Domain<2>(Nf_, Nt_), 1.0f),
    i_fftm_(Domain<2>(Nf_, Nt_), 1.0f/Nt_),
    f_fft_rc_(Domain<1>(Nt_), 1.0f),
    frequencies_(freqs),
    times_(t),
    reference_(ref_signal(times_)),
    coeffs_(Nf_, Nt_)
    {
      Vector<cscalar_f> phase_offset(Nt_, cscalar_f());
      Matrix<cscalar_f> reference_matrix(Nf_, Nt_);

      for (index_type i = 0; i < Nf_; ++i)
      {
        phase_offset.imag() = 2*OVXX_PI*frequencies_(i)*times_;
        reference_matrix.row(i) = reference_ * exp(phase_offset);
      }
  
      // Construct the matched filter coefficients corresponding to the desired
      //  frequencies.
      coeffs_ = f_fftm_(reference_matrix);
    }

  template <typename Block1, 
            typename Block2>
  void 
  operator()(const_Vector<InT, Block1> in, Matrix<cscalar_f, Block2> out)
  {
    Vector<cscalar_f> temp(Nt_);

    temp(Domain<1>(Nt_ / 2 + 1)) = conj(f_fft_rc_(in));
    for (index_type i = 0; i < Nt_ / 2 - 1; ++i)
      temp(i + Nt_ / 2 + 1) = conj(temp(Nt_ / 2 - i - 1));

    out = i_fftm_(vmmul<0>(temp, coeffs_));
  }

private:
  length_type Nt_;                 // Length of the time vector
  length_type Nf_;                 // Length of the frequency vector
  f_fftm_type f_fftm_;            // Forward Fftm object
  i_fftm_type i_fftm_;            // Inverse Fftm object
  f_fft_rc_type f_fft_rc_;        // Forward Fft object
  Vector<scalar_f> frequencies_;   // Vector of frequencies
  Vector<scalar_f> times_;         // Vector of times
  Vector<RefT> reference_;     // Reference signal
  Matrix<cscalar_f> coeffs_;       // Filter coefficients
};

// Specialization for complex input signals
template <typename RefT, typename SignalT>
class Matched_filter<RefT, cscalar_f, SignalT>
{
  typedef Fftm<cscalar_f, cscalar_f, row, fft_fwd, by_value, 0> f_fftm_type;
  typedef Fftm<cscalar_f, cscalar_f, row, fft_inv, by_value, 0> i_fftm_type;
  typedef Fft<Vector, cscalar_f, cscalar_f, fft_fwd> f_fft_cc_type;

public:
  Matched_filter(Vector<scalar_f> t,
                 Vector<scalar_f> freqs,
                 SignalT ref_signal)
  : Nt_(t.size()),
    Nf_(freqs.size()),
    f_fftm_(Domain<2>(Nf_, Nt_), 1.0f),
    i_fftm_(Domain<2>(Nf_, Nt_), 1.0f/Nt_),
    f_fft_cc_(Domain<1>(Nt_), 1.0f),
    frequencies_(freqs),
    times_(t),
    reference_(ref_signal(times_)),
    coeffs_(Nf_, Nt_)
    {
      Vector<cscalar_f> phase_offset(Nt_, cscalar_f());
      Matrix<cscalar_f> reference_matrix(Nf_, Nt_);

      for (index_type i = 0; i < Nf_; ++i)
      {
        phase_offset.imag() = 2*OVXX_PI*frequencies_(i)*times_;
        reference_matrix.row(i) = reference_ * exp(phase_offset);
      }

      // Construct the matched filter coefficients corresponding to the desired
      //  frequencies.
      coeffs_ = f_fftm_(reference_matrix);
    }

  template <typename Block1, 
            typename Block2>
  void 
  operator()(const_Vector<cscalar_f, Block1> in, Matrix<cscalar_f, Block2> out)
  {
    Vector<cscalar_f> temp(Nt_);

    temp = conj(f_fft_cc_(in));

    out = i_fftm_(vmmul<0>(temp, coeffs_));
  }

private:
  length_type Nt_;                 // Length of the time vector
  length_type Nf_;                 // Length of the frequency vector
  f_fftm_type f_fftm_;            // Forward Fftm object
  i_fftm_type i_fftm_;            // Inverse Fftm object
  f_fft_cc_type f_fft_cc_;        // Forward Fft object
  Vector<scalar_f> frequencies_;   // Vector of frequencies
  Vector<scalar_f> times_;         // Vector of times
  Vector<RefT> reference_;     // Reference signal
  Matrix<cscalar_f> coeffs_;       // Filter coefficients
};

void
simulate_filter(length_type N,
                length_type Nf,
                scalar_f    Tmax,
                scalar_f    Fmax,
                scalar_f    input_delay)
{
  // Some typedefs for convenience
  typedef Chirp<cscalar_f> chirp_type;
  typedef Pulse<scalar_f> pulse_type;

  scalar_f pulse_amplitude = 1.0;       // Pulse amplitude
  scalar_f frequency_rate = -10.0;      // Frequency rate of chirp, Hz / s
  scalar_f signal_duration = 1.0;       // Duration of the signal, s

  chirp_type chirp(frequency_rate, signal_duration); // Chirp waveform
  pulse_type pulse(pulse_amplitude, signal_duration);// Pulse waveform

  Vector<scalar_f> t(N);                 // Time vector
  Vector<scalar_f> f(Nf);                // Vector of test frequencies
  Vector<scalar_f> in(N, scalar_f());    // Vector to hold a real input signal
  Vector<cscalar_f> cin(N, cscalar_f()); // Vector to hold a complex input signal

  Matrix<cscalar_f> out(Nf, N);          // Matrix to hold the complex output
  Matrix<scalar_f> mag_out(Nf, N);       // Magnitude of the output

  scalar_f secs_per_sample = Tmax / N;   // Distance between samples

  // Generate a vector of 'N' samples with a spacing of 'secs_per_sample'
  t = ramp(scalar_f(0.f), secs_per_sample, N);
  // Generate a length 'Nf' vector of frequencies from -Fmax to Fmax
  f = ramp(scalar_f(-Fmax), scalar_f(2 * Fmax / Nf), Nf);

  // Construct a filter matched to the chirp waveform to search for a target over
  //  the space of times and frequencies specified by 't' and 'f'.
  Matched_filter<cscalar_f, cscalar_f, chirp_type> chirp_filter(t, f, chirp);

  // Similarly for the pulse waveform.
  Matched_filter<scalar_f, scalar_f, pulse_type> pulse_filter(t, f, pulse);

  // Here use the reference functions to simulate an incoming signal
  //  of each shape delayed by 'input_delay' seconds.
  cin = chirp(t, input_delay);
  in = pulse(t, input_delay);


  // First, attempt to reconstruct the target delay from the incoming FM chirp
  //  signal.  Filter the signal through the chirp matched filter to
  //  get the output amplitude over a range of frequencies.  Then find the maximum
  //  filter output power by taking the squared magnitude of the output to
  //  estimate the delay.

  // Filter the chirp
  chirp_filter(cin, out);

  // Take the magnitude of the output and swap the delays to center the incoming
  //  post-correlation power relative to zero delay.
  for (index_type i = 0; i < N / 2; ++i)
  {
    mag_out.col(i + N / 2) = mag(out.col(i));
    mag_out.col(i) = mag(out.col(i + N / 2));
  }

  Index<2> idx;         // Store the indices of the maximum value
  maxval(mag_out, idx); // Determine those indices

  index_type output_delay_idx = N / 2 - idx[1];                      // Estimated delay in samples
  scalar_f output_delay = float(output_delay_idx) * secs_per_sample; // Estimated delay in seconds

  std::cout << "Waveform             Computed Delay [sec]" << std::endl;
  std::cout << "==========================================" << std::endl;
  std::cout << " Chirp:                    " << output_delay << std::endl;

  // Now repeat the process for the pulse waveform.

  // Filter the pulse
  pulse_filter(in, out);

  // Take the magnitude and swap the delays
  for (index_type i = 0; i < N / 2; ++i)
  {
    mag_out.col(i + N / 2) = mag(out.col(i));
    mag_out.col(i) = mag(out.col(i + N / 2));
  }

  maxval(mag_out, idx);                                     // Get the maximum
  output_delay_idx = N / 2 - idx[1];                        // Estimated delay in samples
  output_delay = float(output_delay_idx) * secs_per_sample; // Estimated delay in seconds

  std::cout << " Pulse:                    " << output_delay << std::endl;
}  

int
main(int argc, char **argv)
{
  vsipl init(argc, argv);

  length_type num_samples_time = 1000; // Number of samples in the time vector
  length_type num_samples_freq = 100;  // Number of frequency steps
  scalar_f    max_time = 10;           // Sec
  scalar_f    freq_range = 50;         // Hz
  scalar_f    delay = 2.19;            // Simulated delay in seconds

  std::cout << "Simulated Signal Delay: " << delay << " seconds " << std::endl;

  simulate_filter(num_samples_time, num_samples_freq, max_time, freq_range, delay);
}
