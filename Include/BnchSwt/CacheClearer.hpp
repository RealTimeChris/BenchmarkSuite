/*
	MIT License

	Copyright (c) 2024 RealTimeChris

	Permission is hereby granted, free of charge, to any person obtaining a copy of this
	software and associated documentation files (the "Software"), to deal in the Software
	without restriction, including without limitation the rights to use, copy, modify, merge,
	publish, distribute, sublicense, and/or sell copies of the Software, and to permit
	persons to whom the Software is furnished to do so, subject to the following conditions:

	The above copyright notice and this permission notice shall be included in all copies or
	substantial portions of the Software.

	THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
	INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
	PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
	FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
	OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
	DEALINGS IN THE SOFTWARE.
*/
/// https://github.com/RealTimeChris/benchmarksuite
/// Sep 1, 2024
#pragma once

#include <BnchSwt/Config.hpp>
#include <iostream>

#if defined(BNCH_SWT_WIN)
	#include <Windows.h>

#elif defined(BNCH_SWT_LINUX)
	#include <xmmintrin.h>
	#include <fstream>
	#include <string>

#elif defined(BNCH_SWT_MAC)
	#include <mach/mach_time.h>
	#include <sys/sysctl.h>
#endif

namespace bnch_swt {

#if defined(BNCH_SWT_WIN)

	BNCH_SWT_ALWAYS_INLINE size_t getL1CacheSize() {
		DWORD bufferSize = 0;
		std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer{};

		GetLogicalProcessorInformation(nullptr, &bufferSize);
		buffer.resize(bufferSize / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));

		if (!GetLogicalProcessorInformation(buffer.data(), &bufferSize)) {
			std::cerr << "Failed to retrieve processor information!" << std::endl;
			return 0;
		}

		size_t l1CacheSize	 = 0;
		const auto infoCount = bufferSize / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
		for (size_t i = 0; i < infoCount; ++i) {
			if (buffer[i].Relationship == RelationCache && buffer[i].Cache.Level == 1 && buffer[i].Cache.Type == CacheData) {
				l1CacheSize = buffer[i].Cache.Size;
				break;
			}
		}

		return l1CacheSize;
	}

#elif defined(BNCH_SWT_LINUX)

	BNCH_SWT_ALWAYS_INLINE size_t getL1CacheSize() {
		const std::string cache_file = "/sys/devices/system/cpu/cpu0/cache/index0/size";
		std::ifstream file(cache_file);
		if (!file.is_open()) {
			std::cerr << "Failed to open cache info file: " << cache_file << std::endl;
			return 0;
		}

		std::string size_str;
		file >> size_str;
		file.close();

		if (size_str.back() == 'K') {
			return std::stoi(size_str) * 1024;
		} else if (size_str.back() == 'M') {
			return std::stoi(size_str) * 1024 * 1024;
		}

		return std::stoi(size_str);
	}

#elif defined(BNCH_SWT_MAC)

	BNCH_SWT_ALWAYS_INLINE size_t getL1CacheSize() {
		size_t l1CacheSize = 0;
		size_t size		   = sizeof(l1CacheSize);

		if (sysctlbyname("hw.l1dcachesize", &l1CacheSize, &size, nullptr, 0) != 0) {
			std::cerr << "Failed to retrieve L1 cache size using sysctl!" << std::endl;
			return 0;
		}

		return l1CacheSize;
	}

#else

	BNCH_SWT_ALWAYS_INLINE size_t getL1CacheSize() {
		std::cerr << "L1 cache size detection is not supported on this platform!" << std::endl;
		return 0;
	}
#endif

	struct cache_clearer {
		inline static size_t cacheLineSize = 64;
		inline static size_t l1CacheSize{ getL1CacheSize() };

#if defined(BNCH_SWT_WIN)
		BNCH_SWT_ALWAYS_INLINE static void flushCache(void* ptr, size_t size) {
			char* buffer = static_cast<char*>(ptr);
			for (size_t i = 0; i < size; i += cacheLineSize) {
				_mm_clflush(buffer + i);
			}
			_mm_sfence();
		}

#elif defined(BNCH_SWT_LINUX) || defined(BNCH_SWT_MAC)
		BNCH_SWT_ALWAYS_INLINE static void flushCache(void* ptr, size_t size) {
			char* buffer = static_cast<char*>(ptr);
	#if defined(__x86_64__) || defined(__i386__)
			for (size_t i = 0; i < size; i += cacheLineSize) {
				__builtin_ia32_clflush(buffer + i);
			}
			_mm_sfence();
	#elif defined(__arm__) || defined(__aarch64__)
			for (size_t i = 0; i < size; i += cacheLineSize) {
				asm volatile("dc cvac, %0" ::"r"(buffer + i) : "memory");
			}
			asm volatile("dsb sy" ::: "memory");
	#else
			std::cerr << "Flush cache is not supported on this architecture!" << std::endl;
	#endif
		}

#else
		BNCH_SWT_ALWAYS_INLINE static void flushCache(void* ptr, size_t size) {
			( void )ptr;
			( void )size;
			std::cerr << "Flush cache is not supported on this platform!" << std::endl;
		}

#endif

		BNCH_SWT_ALWAYS_INLINE static void evictL1Cache() {
			std::vector<char> evict_buffer(l1CacheSize + cacheLineSize);
			for (size_t i = 0; i < evict_buffer.size(); i += cacheLineSize) {
				evict_buffer[i] = static_cast<char>(i);
			}
			flushCache(evict_buffer.data(), evict_buffer.size());
		}
	};

}