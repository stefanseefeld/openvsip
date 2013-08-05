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
///   Read and write WAV audio files
/// 
/// Note: This presently only works with the byte-ordering expected
/// on x86 systems.

#ifndef WAV_HPP
#define WAV_HPP

#include <fstream>
#include <streambuf>
#include <stdexcept>
#include <stdint.h>
#include <vsip_csl/endian.hpp>

namespace wav
{

// Usage: Create Encoder/Decoder, giving it an appropriate stream and a
// struct containing desired meta-information (like bitrate, number of
// channels etc...).  Then call it giving it a pointer to the data buffer
// you wish to read or write.
//
// Examples:
//   wav::Encoder output(std::ofstream *, wav::Info const);
//   output(buffer);
// 
//   wav::Decoder input(std::ifstream *, wav::Info &);
//   input(buffer);


// WAV-file specific definitions
#define SHORT_FORMAT_SIZE  16
#define WAVE_FORMAT_PCM         0x0001
#define WAVE_FORMAT_IEEE_FLOAT  0x0003

#define CHUNK_ID(A, B, C, D)                                    \
  (((uint32_t)(char)(A)) <<  0 | ((uint32_t)(char)(B)) <<  8 |  \
   ((uint32_t)(char)(C)) << 16 | ((uint32_t)(char)(D)) << 24)

#define RIFF_ID  CHUNK_ID('R', 'I', 'F', 'F')
#define WAVE_ID  CHUNK_ID('W', 'A', 'V', 'E')
#define FMT_ID   CHUNK_ID('f', 'm', 't', ' ')
#define DATA_ID  CHUNK_ID('d', 'a', 't', 'a')

// The Format chunk specifies the format of the data. There are 3 variants 
// of the Format chunk for sampled data. These differ in the extensions to 
// the basic Formant chunk.
struct Wave_format
{                                 // ---------------------- 16-byte version
  uint16_t tag;                   // 2 Format code
  uint16_t channels;              // 2 Number of interleaved channels
  uint32_t samples_per_sec;       // 4 Sampling rate (blocks per second)
  uint32_t avg_bytes_per_sec;     // 4 Data rate
  uint16_t block_align;           // 2 Data block size (bytes)
  uint16_t bits_per_sample;       // 2 Bits per sample
                                  // ---------------------- 18-byte version
  uint16_t cb_size;               // 2 Size of the extension (0 or 22)
                                  // ---------------------- 40-byte version
  uint16_t valid_bits_per_sample; // 2 Number of valid bits
  uint32_t channel_mask;          // 4 Speaker position mask
  char     sub_format[16];        // 16 GUID, including the data format code
};

// Wave files are divided into chunks with a common 8-byte header
struct Wave_chunk
{
  uint32_t  id;    // Chunk identifier
  uint32_t  size;  // Size in bytes
};

// The File header is what can be reliably read from every wave file.  The
// first 20 bytes determine the size of the remaining bytes, likely 16.
struct Wave_file_header
{
  uint32_t file_id;       // 4 Chunk ID: "RIFF"
  uint32_t file_size;     // 4 Chunk size: 12 + chunk sizes
  uint32_t wave_id;       // 4 WAVE ID: "WAVE"
  uint32_t format_id;     // 4 Chunk ID: "fmt "
  uint32_t format_size;   // 4 Chunk size: 16 or 18 or 40
  Wave_format format;     //   Format chunk
};



/// Holds information necessary for creating a wav file, or for holding
/// the information read from a wav file header.
struct Info
{
  unsigned int channels;
  unsigned int samples;
  unsigned int samples_per_second;
  unsigned int bytes_per_sample;
  unsigned int buffer_size;
  unsigned short format_tag;
};

/// Reads a wav file and returns information derived from the header.
/// Once a buffer of suitable size is created, it may be read.
class Decoder 
{
public:
  Decoder(std::streambuf*, Info&);
  ~Decoder() {}
  void operator()(void const*);

private:
  std::streambuf *input_;
  std::streampos data_start_;
  Wave_file_header header_;
  Wave_chunk data_;
};

/// Writes a wav file based on the information given.
class Encoder 
{
public:
  Encoder(std::streambuf*, Info const);
  ~Encoder() {}
  void operator()(void*);

private:
  std::streambuf *output_;
  std::size_t header_size_;
  Wave_file_header header_;
  Wave_chunk data_;
};



Decoder::Decoder(std::streambuf *sbuf, Info& file_info)
  : input_(sbuf),
    data_start_(),
    header_(),
    data_()
{
  using vsip_csl::bswap;

  // Because the size of the header can vary, read the fixed-sized part
  // first, then read the remainder once its size is known.
  std::size_t preamble_size = sizeof(Wave_file_header) - sizeof(Wave_format);

  input_->sgetn((char*)&header_, preamble_size);
  bswap(header_.file_id, VSIP_BIG_ENDIAN);
  bswap(header_.file_size, VSIP_BIG_ENDIAN);
  bswap(header_.wave_id, VSIP_BIG_ENDIAN);
  bswap(header_.format_id, VSIP_BIG_ENDIAN);
  bswap(header_.format_size, VSIP_BIG_ENDIAN);
  if (header_.file_id != RIFF_ID || header_.wave_id != WAVE_ID)
    throw std::runtime_error("Not a WAV file");

  input_->sgetn((char*)&header_.format, header_.format_size);
  bswap(header_.format.tag, VSIP_BIG_ENDIAN);
  bswap(header_.format.channels, VSIP_BIG_ENDIAN);
  bswap(header_.format.samples_per_sec, VSIP_BIG_ENDIAN);
  bswap(header_.format.avg_bytes_per_sec, VSIP_BIG_ENDIAN);
  bswap(header_.format.block_align, VSIP_BIG_ENDIAN);
  bswap(header_.format.bits_per_sample, VSIP_BIG_ENDIAN);
  bswap(header_.format.cb_size, VSIP_BIG_ENDIAN);
  bswap(header_.format.valid_bits_per_sample, VSIP_BIG_ENDIAN);
  bswap(header_.format.channel_mask, VSIP_BIG_ENDIAN);
  if (header_.format.tag != WAVE_FORMAT_PCM && 
      header_.format.tag != WAVE_FORMAT_IEEE_FLOAT)
    throw std::runtime_error("Only PCM and IEEE-float formats are supported");
  if (header_.format_id != FMT_ID)
    throw std::runtime_error("Format identifier not found in WAV header");
  if (header_.format.channels < 1 || header_.format.channels > 4)
    throw std::runtime_error("Number of channels must be 1->4");

  // Read the next chunk header and stop when the data segment is found
  Wave_chunk next;
  input_->sgetn((char*)&next, sizeof(Wave_chunk));
  bswap(next.id, VSIP_BIG_ENDIAN);
  bswap(next.size, VSIP_BIG_ENDIAN);
  while (next.id != DATA_ID)
  {
    // skip over this chunk
    input_->pubseekoff(next.size, std::ios_base::cur, std::ios_base::in);
    input_->sgetn((char*)&next, sizeof(Wave_chunk));
    bswap(next.id, VSIP_BIG_ENDIAN);
    bswap(next.size, VSIP_BIG_ENDIAN);
  }

  // Save the current chunk header and the current position in the file
  data_ = next;
  data_start_ = input_->pubseekoff(0, std::ios_base::cur, std::ios_base::in);

  // Return header information into a simplified form
  file_info.channels = header_.format.channels;
  file_info.samples = data_.size / file_info.channels / 
    (header_.format.bits_per_sample / 8);
  file_info.samples_per_second = header_.format.samples_per_sec;
  file_info.bytes_per_sample = header_.format.bits_per_sample / 8;
  file_info.buffer_size = file_info.channels * file_info.samples * 
    file_info.bytes_per_sample;
  file_info.format_tag = header_.format.tag;
}

void 
Decoder::operator()(void const* buffer)
{
  // Reset the position to the data chunk and read the data
  input_->pubseekoff(data_start_, std::ios_base::beg, std::ios_base::in);
  input_->sgetn(const_cast<char *>(static_cast<char const*>(buffer)), data_.size);

#if VSIP_BIG_ENDIAN
  // Swap values when there are either 2 or 4 bytes per samples
  using vsip_csl::bswap;
  if (header_.format.bits_per_sample == 16)
  {
    std::size_t size = data_.size / 2;

    uint16_t* ptr = const_cast<uint16_t*>(static_cast<uint16_t const*>(buffer));
    for (std::size_t i = 0; i < size; ++i)
      bswap(*ptr++);
  }
  if (header_.format.bits_per_sample == 32)
  {
    std::size_t size = data_.size / 4;

    uint32_t* ptr = const_cast<uint32_t*>(static_cast<uint32_t const*>(buffer));
    for (std::size_t i = 0; i < size; ++i)
      bswap(*ptr++);
  }
#endif // VSIP_BIG_ENDIAN
}


Encoder::Encoder(std::streambuf *sbuf, Info const file_info)
  : output_(sbuf),
    header_size_(),
    header_(),
    data_()
{
  // Verify the format is supported
  if (file_info.format_tag != WAVE_FORMAT_PCM &&
      file_info.format_tag != WAVE_FORMAT_IEEE_FLOAT)
    throw std::runtime_error("Only PCM and IEEE-float formats are supported");

  // Create the data chunk
  data_.id = DATA_ID;
  data_.size = file_info.channels * file_info.samples * 
    file_info.bytes_per_sample;

  // Create the file header
  header_size_ = sizeof(Wave_file_header) - sizeof(Wave_format) + 
    SHORT_FORMAT_SIZE;

  header_.file_id = RIFF_ID;
  header_.file_size = header_size_ + sizeof(Wave_chunk) + data_.size;
  header_.wave_id = WAVE_ID;
  header_.format_id = FMT_ID;
  header_.format_size = SHORT_FORMAT_SIZE;
  header_.format.channels = file_info.channels;
  header_.format.samples_per_sec = file_info.samples_per_second;
  header_.format.avg_bytes_per_sec = file_info.channels *
    file_info.bytes_per_sample * file_info.samples_per_second;
  header_.format.block_align = file_info.channels * file_info.bytes_per_sample;
  header_.format.bits_per_sample = file_info.bytes_per_sample * 8;
  header_.format.tag = file_info.format_tag;
}


void 
Encoder::operator()(void* buffer)
{
#if !VSIP_BIG_ENDIAN
  // Write the header first
  output_->sputn((char*)&header_, header_size_);

  // Then the data chunk
  output_->sputn((char*)&data_, sizeof(Wave_chunk));
  output_->sputn((char*)buffer, data_.size);

#else
  using vsip_csl::bswap;
  Wave_file_header tmp = header_;
  bswap(tmp.file_id);
  bswap(tmp.file_size);
  bswap(tmp.wave_id);
  bswap(tmp.format_id);
  bswap(tmp.format_size);
  bswap(tmp.format.tag);
  bswap(tmp.format.channels);
  bswap(tmp.format.samples_per_sec);
  bswap(tmp.format.avg_bytes_per_sec);
  bswap(tmp.format.block_align);
  bswap(tmp.format.bits_per_sample);
  bswap(tmp.format.cb_size);
  bswap(tmp.format.valid_bits_per_sample);
  bswap(tmp.format.channel_mask);
  // tmp.format.sub_format is not swapped.
  output_->sputn((char*)&tmp, header_size_);

  Wave_chunk tmpd = data_;
  bswap(tmpd.id);
  bswap(tmpd.size);
  output_->sputn((char*)&tmpd, sizeof(Wave_chunk));

  if (header_.format.bits_per_sample == 16)
  {
    std::size_t size = data_.size / 2;

    // Swap values in place
    uint16_t* ptr = static_cast<uint16_t*>(buffer);
    for (std::size_t i = 0; i < size; ++i)
      bswap(*ptr++);

    // Write them out
    output_->sputn(static_cast<char*>(buffer), data_.size);

    // Swap values back to avoid having changed them (the 
    // alternative would be to allocate a temporary buffer
    // of the same size)
    ptr = static_cast<uint16_t*>(buffer);
    for (std::size_t i = 0; i < size; ++i)
      bswap(*ptr++);
  }
  if (header_.format.bits_per_sample == 32)
  {
    std::size_t size = data_.size / 4;

    uint32_t* ptr = static_cast<uint32_t*>(buffer);
    for (std::size_t i = 0; i < size; ++i)
      bswap(*ptr++);

    output_->sputn(static_cast<char*>(buffer), data_.size);

    ptr = static_cast<uint32_t*>(buffer);
    for (std::size_t i = 0; i < size; ++i)
      bswap(*ptr++);
  }
  else
    output_->sputn(static_cast<char*>(buffer), data_.size);
#endif // VSIP_BIG_ENDIAN
}

} // namespace wav

#endif // WAV_HPP
