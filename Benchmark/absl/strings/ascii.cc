// Copyright 2017 The Abseil Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "../strings/ascii.h"

#include <climits>
#include <cstddef>
#include <cstring>
#include <string>

#include "../base/config.h"
#include "../base/nullability.h"

namespace absl {
ABSL_NAMESPACE_BEGIN
namespace ascii_internal {

// # Table generated by this Python code (bit 0x02 is currently unused):
// TODO(mbar) Move Python code for generation of table to BUILD and link here.

// NOTE: The kAsciiPropertyBits table used within this code was generated by
// Python code of the following form. (Bit 0x02 is currently unused and
// available.)
//
// def Hex2(n):
//   return '0x' + hex(n/16)[2:] + hex(n%16)[2:]
// def IsPunct(ch):
//   return (ord(ch) >= 32 and ord(ch) < 127 and
//           not ch.isspace() and not ch.isalnum())
// def IsBlank(ch):
//   return ch in ' \t'
// def IsCntrl(ch):
//   return ord(ch) < 32 or ord(ch) == 127
// def IsXDigit(ch):
//   return ch.isdigit() or ch.lower() in 'abcdef'
// for i in range(128):
//   ch = chr(i)
//   mask = ((ch.isalpha() and 0x01 or 0) |
//           (ch.isalnum() and 0x04 or 0) |
//           (ch.isspace() and 0x08 or 0) |
//           (IsPunct(ch) and 0x10 or 0) |
//           (IsBlank(ch) and 0x20 or 0) |
//           (IsCntrl(ch) and 0x40 or 0) |
//           (IsXDigit(ch) and 0x80 or 0))
//   print Hex2(mask) + ',',
//   if i % 16 == 7:
//     print ' //', Hex2(i & 0x78)
//   elif i % 16 == 15:
//     print

// clang-format off
// Array of bitfields holding character information. Each bit value corresponds
// to a particular character feature. For readability, and because the value
// of these bits is tightly coupled to this implementation, the individual bits
// are not named. Note that bitfields for all characters above ASCII 127 are
// zero-initialized.
ABSL_DLL const unsigned char kPropertyBits[256] = {
    0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  // 0x00
    0x40, 0x68, 0x48, 0x48, 0x48, 0x48, 0x40, 0x40,
    0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,  // 0x10
    0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40, 0x40,
    0x28, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10,  // 0x20
    0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10,
    0x84, 0x84, 0x84, 0x84, 0x84, 0x84, 0x84, 0x84,  // 0x30
    0x84, 0x84, 0x10, 0x10, 0x10, 0x10, 0x10, 0x10,
    0x10, 0x85, 0x85, 0x85, 0x85, 0x85, 0x85, 0x05,  // 0x40
    0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05,
    0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05,  // 0x50
    0x05, 0x05, 0x05, 0x10, 0x10, 0x10, 0x10, 0x10,
    0x10, 0x85, 0x85, 0x85, 0x85, 0x85, 0x85, 0x05,  // 0x60
    0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05,
    0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05, 0x05,  // 0x70
    0x05, 0x05, 0x05, 0x10, 0x10, 0x10, 0x10, 0x40,
};

// Array of characters for the ascii_tolower() function. For values 'A'
// through 'Z', return the lower-case character; otherwise, return the
// identity of the passed character.
ABSL_DLL const char kToLower[256] = {
  '\x00', '\x01', '\x02', '\x03', '\x04', '\x05', '\x06', '\x07',
  '\x08', '\x09', '\x0a', '\x0b', '\x0c', '\x0d', '\x0e', '\x0f',
  '\x10', '\x11', '\x12', '\x13', '\x14', '\x15', '\x16', '\x17',
  '\x18', '\x19', '\x1a', '\x1b', '\x1c', '\x1d', '\x1e', '\x1f',
  '\x20', '\x21', '\x22', '\x23', '\x24', '\x25', '\x26', '\x27',
  '\x28', '\x29', '\x2a', '\x2b', '\x2c', '\x2d', '\x2e', '\x2f',
  '\x30', '\x31', '\x32', '\x33', '\x34', '\x35', '\x36', '\x37',
  '\x38', '\x39', '\x3a', '\x3b', '\x3c', '\x3d', '\x3e', '\x3f',
  '\x40',    'a',    'b',    'c',    'd',    'e',    'f',    'g',
     'h',    'i',    'j',    'k',    'l',    'm',    'n',    'o',
     'p',    'q',    'r',    's',    't',    'u',    'v',    'w',
     'x',    'y',    'z', '\x5b', '\x5c', '\x5d', '\x5e', '\x5f',
  '\x60', '\x61', '\x62', '\x63', '\x64', '\x65', '\x66', '\x67',
  '\x68', '\x69', '\x6a', '\x6b', '\x6c', '\x6d', '\x6e', '\x6f',
  '\x70', '\x71', '\x72', '\x73', '\x74', '\x75', '\x76', '\x77',
  '\x78', '\x79', '\x7a', '\x7b', '\x7c', '\x7d', '\x7e', '\x7f',
  '\x80', '\x81', '\x82', '\x83', '\x84', '\x85', '\x86', '\x87',
  '\x88', '\x89', '\x8a', '\x8b', '\x8c', '\x8d', '\x8e', '\x8f',
  '\x90', '\x91', '\x92', '\x93', '\x94', '\x95', '\x96', '\x97',
  '\x98', '\x99', '\x9a', '\x9b', '\x9c', '\x9d', '\x9e', '\x9f',
  '\xa0', '\xa1', '\xa2', '\xa3', '\xa4', '\xa5', '\xa6', '\xa7',
  '\xa8', '\xa9', '\xaa', '\xab', '\xac', '\xad', '\xae', '\xaf',
  '\xb0', '\xb1', '\xb2', '\xb3', '\xb4', '\xb5', '\xb6', '\xb7',
  '\xb8', '\xb9', '\xba', '\xbb', '\xbc', '\xbd', '\xbe', '\xbf',
  '\xc0', '\xc1', '\xc2', '\xc3', '\xc4', '\xc5', '\xc6', '\xc7',
  '\xc8', '\xc9', '\xca', '\xcb', '\xcc', '\xcd', '\xce', '\xcf',
  '\xd0', '\xd1', '\xd2', '\xd3', '\xd4', '\xd5', '\xd6', '\xd7',
  '\xd8', '\xd9', '\xda', '\xdb', '\xdc', '\xdd', '\xde', '\xdf',
  '\xe0', '\xe1', '\xe2', '\xe3', '\xe4', '\xe5', '\xe6', '\xe7',
  '\xe8', '\xe9', '\xea', '\xeb', '\xec', '\xed', '\xee', '\xef',
  '\xf0', '\xf1', '\xf2', '\xf3', '\xf4', '\xf5', '\xf6', '\xf7',
  '\xf8', '\xf9', '\xfa', '\xfb', '\xfc', '\xfd', '\xfe', '\xff',
};

// Array of characters for the ascii_toupper() function. For values 'a'
// through 'z', return the upper-case character; otherwise, return the
// identity of the passed character.
ABSL_DLL const char kToUpper[256] = {
  '\x00', '\x01', '\x02', '\x03', '\x04', '\x05', '\x06', '\x07',
  '\x08', '\x09', '\x0a', '\x0b', '\x0c', '\x0d', '\x0e', '\x0f',
  '\x10', '\x11', '\x12', '\x13', '\x14', '\x15', '\x16', '\x17',
  '\x18', '\x19', '\x1a', '\x1b', '\x1c', '\x1d', '\x1e', '\x1f',
  '\x20', '\x21', '\x22', '\x23', '\x24', '\x25', '\x26', '\x27',
  '\x28', '\x29', '\x2a', '\x2b', '\x2c', '\x2d', '\x2e', '\x2f',
  '\x30', '\x31', '\x32', '\x33', '\x34', '\x35', '\x36', '\x37',
  '\x38', '\x39', '\x3a', '\x3b', '\x3c', '\x3d', '\x3e', '\x3f',
  '\x40', '\x41', '\x42', '\x43', '\x44', '\x45', '\x46', '\x47',
  '\x48', '\x49', '\x4a', '\x4b', '\x4c', '\x4d', '\x4e', '\x4f',
  '\x50', '\x51', '\x52', '\x53', '\x54', '\x55', '\x56', '\x57',
  '\x58', '\x59', '\x5a', '\x5b', '\x5c', '\x5d', '\x5e', '\x5f',
  '\x60',    'A',    'B',    'C',    'D',    'E',    'F',    'G',
     'H',    'I',    'J',    'K',    'L',    'M',    'N',    'O',
     'P',    'Q',    'R',    'S',    'T',    'U',    'V',    'W',
     'X',    'Y',    'Z', '\x7b', '\x7c', '\x7d', '\x7e', '\x7f',
  '\x80', '\x81', '\x82', '\x83', '\x84', '\x85', '\x86', '\x87',
  '\x88', '\x89', '\x8a', '\x8b', '\x8c', '\x8d', '\x8e', '\x8f',
  '\x90', '\x91', '\x92', '\x93', '\x94', '\x95', '\x96', '\x97',
  '\x98', '\x99', '\x9a', '\x9b', '\x9c', '\x9d', '\x9e', '\x9f',
  '\xa0', '\xa1', '\xa2', '\xa3', '\xa4', '\xa5', '\xa6', '\xa7',
  '\xa8', '\xa9', '\xaa', '\xab', '\xac', '\xad', '\xae', '\xaf',
  '\xb0', '\xb1', '\xb2', '\xb3', '\xb4', '\xb5', '\xb6', '\xb7',
  '\xb8', '\xb9', '\xba', '\xbb', '\xbc', '\xbd', '\xbe', '\xbf',
  '\xc0', '\xc1', '\xc2', '\xc3', '\xc4', '\xc5', '\xc6', '\xc7',
  '\xc8', '\xc9', '\xca', '\xcb', '\xcc', '\xcd', '\xce', '\xcf',
  '\xd0', '\xd1', '\xd2', '\xd3', '\xd4', '\xd5', '\xd6', '\xd7',
  '\xd8', '\xd9', '\xda', '\xdb', '\xdc', '\xdd', '\xde', '\xdf',
  '\xe0', '\xe1', '\xe2', '\xe3', '\xe4', '\xe5', '\xe6', '\xe7',
  '\xe8', '\xe9', '\xea', '\xeb', '\xec', '\xed', '\xee', '\xef',
  '\xf0', '\xf1', '\xf2', '\xf3', '\xf4', '\xf5', '\xf6', '\xf7',
  '\xf8', '\xf9', '\xfa', '\xfb', '\xfc', '\xfd', '\xfe', '\xff',
};
// clang-format on

// Returns whether `c` is in the a-z/A-Z range (w.r.t. `ToUpper`).
// Implemented by:
//  1. Pushing the a-z/A-Z range to [SCHAR_MIN, SCHAR_MIN + 26).
//  2. Comparing to SCHAR_MIN + 26.
template <bool ToUpper>
constexpr bool AsciiInAZRange(unsigned char c) {
  constexpr unsigned char sub = (ToUpper ? 'a' : 'A') - SCHAR_MIN;
  constexpr signed char threshold = SCHAR_MIN + 26;  // 26 = alphabet size.
  // Using unsigned arithmetic as overflows/underflows are well defined.
  unsigned char u = c - sub;
  // Using signed cmp, as SIMD unsigned cmp isn't available in many platforms.
  return static_cast<signed char>(u) < threshold;
}

template <bool ToUpper>
constexpr bool AsciiInAZRangeNaive(unsigned char c) {
  constexpr unsigned char a = (ToUpper ? 'a' : 'A');
  constexpr unsigned char z = (ToUpper ? 'z' : 'Z');
  return a <= c && c <= z;
}

template <bool ToUpper, bool Naive>
constexpr void AsciiStrCaseFoldImpl(absl::Nonnull<char*> dst,
                                    absl::Nullable<const char*> src,
                                    size_t size) {
  // The upper- and lowercase versions of ASCII characters differ by only 1 bit.
  // When we need to flip the case, we can xor with this bit to achieve the
  // desired result. Note that the choice of 'a' and 'A' here is arbitrary. We
  // could have chosen 'z' and 'Z', or any other pair of characters as they all
  // have the same single bit difference.
  constexpr unsigned char kAsciiCaseBitFlip = 'a' ^ 'A';

  for (size_t i = 0; i < size; ++i) {
    unsigned char v = static_cast<unsigned char>(src[i]);
    if ABSL_INTERNAL_CONSTEXPR_SINCE_CXX17 (Naive) {
      v ^= AsciiInAZRangeNaive<ToUpper>(v) ? kAsciiCaseBitFlip : 0;
    } else {
      v ^= AsciiInAZRange<ToUpper>(v) ? kAsciiCaseBitFlip : 0;
    }
    dst[i] = static_cast<char>(v);
  }
}

// Splitting to short and long strings to allow vectorization decisions
// to be made separately in the long and short cases.
// Using slightly different implementations so the compiler won't optimize them
// into the same code (the non-naive version is needed for SIMD, so for short
// strings it's not important).
// `src` may be null iff `size` is zero.
template <bool ToUpper>
constexpr void AsciiStrCaseFold(absl::Nonnull<char*> dst,
                                absl::Nullable<const char*> src, size_t size) {
  size < 16 ? AsciiStrCaseFoldImpl<ToUpper, /*Naive=*/true>(dst, src, size)
            : AsciiStrCaseFoldImpl<ToUpper, /*Naive=*/false>(dst, src, size);
}

void AsciiStrToLower(absl::Nonnull<char*> dst, absl::Nullable<const char*> src,
                     size_t n) {
  return AsciiStrCaseFold<false>(dst, src, n);
}

void AsciiStrToUpper(absl::Nonnull<char*> dst, absl::Nullable<const char*> src,
                     size_t n) {
  return AsciiStrCaseFold<true>(dst, src, n);
}

static constexpr size_t ValidateAsciiCasefold() {
  constexpr size_t num_chars = 1 + CHAR_MAX - CHAR_MIN;
  size_t incorrect_index = 0;
  char lowered[num_chars] = {};
  char uppered[num_chars] = {};
  for (unsigned int i = 0; i < num_chars; ++i) {
    uppered[i] = lowered[i] = static_cast<char>(i);
  }
  AsciiStrCaseFold<false>(&lowered[0], &lowered[0], num_chars);
  AsciiStrCaseFold<true>(&uppered[0], &uppered[0], num_chars);
  for (size_t i = 0; i < num_chars; ++i) {
    const char ch = static_cast<char>(i),
               ch_upper = ('a' <= ch && ch <= 'z' ? 'A' + (ch - 'a') : ch),
               ch_lower = ('A' <= ch && ch <= 'Z' ? 'a' + (ch - 'A') : ch);
    if (uppered[i] != ch_upper || lowered[i] != ch_lower) {
      incorrect_index = i > 0 ? i : num_chars;
      break;
    }
  }
  return incorrect_index;
}

static_assert(ValidateAsciiCasefold() == 0, "error in case conversion");

}  // namespace ascii_internal

void AsciiStrToLower(absl::Nonnull<std::string*> s) {
  char* p = &(*s)[0];
  return ascii_internal::AsciiStrCaseFold<false>(p, p, s->size());
}

void AsciiStrToUpper(absl::Nonnull<std::string*> s) {
  char* p = &(*s)[0];
  return ascii_internal::AsciiStrCaseFold<true>(p, p, s->size());
}

void RemoveExtraAsciiWhitespace(absl::Nonnull<std::string*> str) {
  auto stripped = StripAsciiWhitespace(*str);

  if (stripped.empty()) {
    str->clear();
    return;
  }

  auto input_it = stripped.begin();
  auto input_end = stripped.end();
  auto output_it = &(*str)[0];
  bool is_ws = false;

  for (; input_it < input_end; ++input_it) {
    if (is_ws) {
      // Consecutive whitespace?  Keep only the last.
      is_ws = absl::ascii_isspace(static_cast<unsigned char>(*input_it));
      if (is_ws) --output_it;
    } else {
      is_ws = absl::ascii_isspace(static_cast<unsigned char>(*input_it));
    }

    *output_it = *input_it;
    ++output_it;
  }

  str->erase(static_cast<size_t>(output_it - &(*str)[0]));
}

ABSL_NAMESPACE_END
}  // namespace absl