/* Copyright (c) 2011 CodeSourcery, Inc.  All rights reserved. */

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
///   Audio Data I/O, 32-bit monaural floating point
///
/// This example utilizes the direct data access extension to read/write 
/// the view's data (see the dda/ subdirectory for more information).


#include <iomanip>
#include <iostream>
#include <fstream>

#include <vsip/initfin.hpp>
#include <vsip/matrix.hpp>
#include <vsip_csl/dda.hpp>
#include <vsip_csl/test.hpp>

#include "wav.hpp"

using namespace vsip;



////////////////////////////////////////////////////////////////////////

void
fill_sine(float frequency, size_t sample_rate, Vector<float> sample)
{
  // Create a simple sine wave at the desired frequency
  float w = frequency / sample_rate;
  size_t N = sample.size();
  for (index_type i = 0; i < N; ++i)
    sample.put(i, sin(2 * M_PI * w * i));
}

void
wav_output(char const* filename)
{
  // Define the attributes of the audio sample we will create
  wav::Info mono;
  mono.channels = 1;
  mono.samples = 12340;
  mono.samples_per_second = 48000;
  mono.bytes_per_sample = sizeof(float);
  mono.buffer_size = mono.channels * mono.samples * mono.bytes_per_sample;
  mono.format_tag = WAVE_FORMAT_IEEE_FLOAT;

  // Create a tone at 440 Hz (A)
  Vector<float> tone(mono.samples);
  fill_sine(440.0f, mono.samples_per_second, tone);

  // Open the wave file and pass the stream to the output Encoder
  std::ofstream ofs(filename);
  wav::Encoder output(ofs.rdbuf(), mono);

  // Obtain a pointer to the data block and write to file
  vsip_csl::dda::Data<Vector<float>::block_type, vsip_csl::dda::out> 
    data(tone.block());
  output(data.ptr());
}


void
wav_input(char const* filename)
{
  // Open wave audio file
  std::ifstream ifs(filename);

  // Use the Decoder to extract information from the header
  wav::Info mono;
  wav::Decoder input(ifs.rdbuf(), mono);

  // Display these attributes
  float duration = (float)mono.samples / mono.samples_per_second;
  std::cout << std::fixed << std::setprecision(3)
            << "Reading wav file : " << filename << std::endl
            << "  Channels     : " << mono.channels << std::endl
            << "  Duration     : " << duration << " s" << std::endl
            << "  Samples/s    : " << mono.samples_per_second << std::endl
            << "  Bytes/sample : " << mono.bytes_per_sample * 8 << std::endl
            << "  Buffer size  : " << mono.buffer_size << std::endl
            << "  Format       : " << mono.format_tag << std::endl;

  // Verify the type passed in is suitable for the format found
  if ((mono.bytes_per_sample != sizeof(float)) ||
      (mono.format_tag != WAVE_FORMAT_IEEE_FLOAT))
  {
    std::cerr << "Incorrect sample size or format when reading WAV file" 
              << std::endl;
    return;
  }

  // Create a vector to hold the samples read from the file
  Vector<float> samples(mono.samples);

  // Read the samples, destroying the direct data access object
  // as soon as this is done (this allows us to safely access the 
  // data through the view)
  {
    vsip_csl::dda::Data<Vector<float>::block_type, vsip_csl::dda::in> 
      data(samples.block());
    input(data.ptr());
  }

  // Create a tone at 440 Hz (A)
  Vector<float> tone(mono.samples);
  fill_sine(440.0f, mono.samples_per_second, tone);

  // Compare the data read from the file with the data just created
  test_assert(vsip_csl::view_equal(tone, samples));
  std::cout << "Audio I/O check PASS" << std::endl;
}



////////////////////////////////////////////////////////////////////////

int main (int argc, char **argv)
{
  vsipl init(argc, argv);

  char const filename[] = "mono.wav";

  // Creates a view containing audio data and writes it to a file
  wav_output(filename);

  // Reads the audio file header, creates a view to hold the data and
  // then reads the audio data into the view
  wav_input(filename);
}

