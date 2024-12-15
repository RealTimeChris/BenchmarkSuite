/* From
https://github.com/endorno/pytorch/blob/master/torch/lib/TH/generic/simd/simd.h
Highly modified.

Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou,
Iain Melvin, Jason Weston) Copyright (c) 2006      Idiap Research Institute
(Samy Bengio) Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert,
Samy Bengio, Johnny Mariethoz)

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Facebook, Deepmind Technologies, NYU, NEC Laboratories
America and IDIAP Research Institute nor the names of its contributors may be
   used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/
#pragma once

#include "config.hpp"
#include <cstdint>
#include <array>
#include <cstdlib>
#if defined(_MSC_VER)
	#include <intrin.h>
#elif defined(HAVE_GCC_GET_CPUID) && defined(USE_GCC_GET_CPUID)
	#include <cpuid.hpp>
#endif

namespace oiml {

	enum class instruction_set : uint64_t {
		DEFAULT		= 0x0,
		NEON		= 0x1,
		SVE			= 0x2,
		AVX2		= 0x4,
		SSE42		= 0x8,
		PCLMULQDQ	= 0x10,
		BMI1		= 0x20,
		BMI2		= 0x40,
		AVX512F		= 0x80,
		AVX512DQ	= 0x100,
		AVX512IFMA	= 0x200,
		AVX512PF	= 0x400,
		AVX512ER	= 0x800,
		AVX512CD	= 0x1000,
		AVX512BW	= 0x2000,
		AVX512VL	= 0x4000,
		AVX512VBMI2 = 0x8000,
	};

#if defined(__aarch64__) || defined(_M_ARM64) || defined(_M_ARM64EC)

	#include <sys/types.h>
	#include <sys/sysctl.h>
	#include <cstring>

	static instruction_set detect_supported_architectures() {
		uint64_t host_isa = 0x0;

		int neon_supported = 0;
		size_t size		   = sizeof(neon_supported);
		if (sysctlbyname("hw.optional.neon", &neon_supported, &size, NULL, 0) == 0 && neon_supported) {
			host_isa |= static_cast<uint64_t>(instruction_set::NEON);
		}

		int sve_supported = 0;
		size			  = sizeof(sve_supported);
		if (sysctlbyname("hw.optional.sve", &sve_supported, &size, NULL, 0) == 0 && sve_supported) {
			host_isa |= static_cast<uint64_t>(instruction_set::SVE);
		}

		return static_cast<instruction_set>(host_isa);
	}

#elif defined(__x86_64__) || defined(_M_AMD64)

	namespace {
		inline static constexpr uint64_t cpuid_avx2_bit		   = 1 << 5;
		inline static constexpr uint64_t cpuid_bmi1_bit		   = 1 << 3;
		inline static constexpr uint64_t cpuid_bmi2_bit		   = 1 << 8;
		inline static constexpr uint64_t cpuid_avx512f_bit	   = 1 << 16;
		inline static constexpr uint64_t cpuid_avx512dq_bit	   = 1 << 17;
		inline static constexpr uint64_t cpuid_avx512ifma_bit  = 1 << 21;
		inline static constexpr uint64_t cpuid_avx512pf_bit	   = 1 << 26;
		inline static constexpr uint64_t cpuid_avx512er_bit	   = 1 << 27;
		inline static constexpr uint64_t cpuid_avx512cd_bit	   = 1 << 28;
		inline static constexpr uint64_t cpuid_avx512bw_bit	   = 1 << 30;
		inline static constexpr uint64_t cpuid_avx512vl_bit	   = 1U << 31;
		inline static constexpr uint64_t cpuid_avx512vbmi2_bit = 1 << 6;
		inline static constexpr uint64_t cpuid_avx256_saved	   = uint64_t(1) << 2;
		inline static constexpr uint64_t cpuid_avx512_saved	   = uint64_t(7) << 5;
		inline static constexpr uint64_t cpuid_sse42_bit	   = 1 << 20;
		inline static constexpr uint64_t cpuid_osxsave		   = (uint64_t(1) << 26) | (uint64_t(1) << 27);
		inline static constexpr uint64_t cpuid_pclmulqdq_bit   = 1 << 1;
	}

	BNCH_SWT_INLINE static void cpuid(int32_t* eax, int32_t* ebx, int32_t* ecx, int32_t* edx) {
	#if defined(_MSC_VER)
		int32_t cpu_info[4];
		__cpuidex(cpu_info, *eax, *ecx);
		*eax = cpu_info[0];
		*ebx = cpu_info[1];
		*ecx = cpu_info[2];
		*edx = cpu_info[3];
	#elif defined(HAVE_GCC_GET_CPUID) && defined(USE_GCC_GET_CPUID)
		__get_cpuid(*eax, eax, ebx, ecx, edx);
	#else
		uint64_t a = *eax, b, c = *ecx, d;
		asm volatile("cpuid\n\t" : "+a"(a), "=b"(b), "+c"(c), "=d"(d));
		*eax = a;
		*ebx = b;
		*ecx = c;
		*edx = d;
	#endif
	}

	BNCH_SWT_INLINE static uint64_t xgetbv() {
	#if defined(_MSC_VER)
		return _xgetbv(0);
	#else
		uint64_t xcr0_lo, xcr0_hi;
		asm volatile("xgetbv\n\t" : "=a"(xcr0_lo), "=d"(xcr0_hi) : "c"(0));
		return xcr0_lo | (uint64_t(xcr0_hi) << 32);
	#endif
	}

	BNCH_SWT_INLINE static instruction_set detect_supported_architectures() {
		int32_t eax, ebx, ecx, edx;
		uint64_t host_isa = 0x0;

		eax = 0x1;
		ecx = 0x0;
		cpuid(&eax, &ebx, &ecx, &edx);

		if (ecx & cpuid_sse42_bit) {
			host_isa |= static_cast<uint64_t>(instruction_set::SSE42);
		} else {
			return static_cast<instruction_set>(host_isa);
		}

		if (ecx & cpuid_pclmulqdq_bit) {
			host_isa |= static_cast<uint64_t>(instruction_set::PCLMULQDQ);
		}


		if ((ecx & cpuid_osxsave) != cpuid_osxsave) {
			return static_cast<instruction_set>(host_isa);
		}

		uint64_t xcr0 = xgetbv();

		if ((xcr0 & cpuid_avx256_saved) == 0) {
			return static_cast<instruction_set>(host_isa);
		}

		eax = 0x7;
		ecx = 0x0;
		cpuid(&eax, &ebx, &ecx, &edx);
		if (ebx & cpuid_avx2_bit) {
			host_isa |= static_cast<uint64_t>(instruction_set::AVX2);
		}
		if (ebx & cpuid_bmi1_bit) {
			host_isa |= static_cast<uint64_t>(instruction_set::BMI1);
		}

		if (ebx & cpuid_bmi2_bit) {
			host_isa |= static_cast<uint64_t>(instruction_set::BMI2);
		}

		if (!((xcr0 & cpuid_avx512_saved) == cpuid_avx512_saved)) {
			return static_cast<instruction_set>(host_isa);
		}

		if (ebx & cpuid_avx512f_bit) {
			host_isa |= static_cast<uint64_t>(instruction_set::AVX512F);
		}

		if (ebx & cpuid_avx512dq_bit) {
			host_isa |= static_cast<uint64_t>(instruction_set::AVX512DQ);
		}

		if (ebx & cpuid_avx512ifma_bit) {
			host_isa |= static_cast<uint64_t>(instruction_set::AVX512IFMA);
		}

		if (ebx & cpuid_avx512pf_bit) {
			host_isa |= static_cast<uint64_t>(instruction_set::AVX512PF);
		}

		if (ebx & cpuid_avx512er_bit) {
			host_isa |= static_cast<uint64_t>(instruction_set::AVX512ER);
		}

		if (ebx & cpuid_avx512cd_bit) {
			host_isa |= static_cast<uint64_t>(instruction_set::AVX512CD);
		}

		if (ebx & cpuid_avx512bw_bit) {
			host_isa |= static_cast<uint64_t>(instruction_set::AVX512BW);
		}

		if (ebx & cpuid_avx512vl_bit) {
			host_isa |= static_cast<uint64_t>(instruction_set::AVX512VL);
		}

		if (ecx & cpuid_avx512vbmi2_bit) {
			host_isa |= static_cast<uint64_t>(instruction_set::AVX512VBMI2);
		}

		return static_cast<instruction_set>(host_isa);
	}

#else

	BNCH_SWT_INLINE static uint64_t detect_supported_architectures() {
		return static_cast<uint64_t>(instruction_set::DEFAULT);
	}

#endif

	inline static const instruction_set cpu_arch{ detect_supported_architectures() };

	inline uint64_t get_cpu_arch_index(instruction_set set) {
		if (static_cast<uint64_t>(set) & static_cast<uint64_t>(instruction_set::AVX512F)) {
			return 2;
		} else if (static_cast<uint64_t>(set) & static_cast<uint64_t>(instruction_set::AVX2)) {
			return 1;
		} else if (static_cast<uint64_t>(set) & static_cast<uint64_t>(instruction_set::SSE42)) {
			return 0;
		} else if (static_cast<uint64_t>(set) & static_cast<uint64_t>(instruction_set::SVE)) {
			return 1;
		} else if (static_cast<uint64_t>(set) & static_cast<uint64_t>(instruction_set::NEON)) {
			return 0;
		} else {
			return 0;
		}
	}

	inline static const auto cpu_arch_index{ get_cpu_arch_index(cpu_arch) };

}
