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

#include <bnch_swt/config.hpp>
#include <iostream>
#include <fstream>

#if BNCH_SWT_PLATFORM_WINDOWS
	#include <Windows.h>
	#include <intrin.h>
#elif BNCH_SWT_PLATFORM_LINUX
	#include <unistd.h>	
	#include <vector>
	#include <string>
	#if defined(__i386__) || defined(__x86_64__)
		#include <immintrin.h>
	#endif

#elif BNCH_SWT_PLATFORM_MAC
	#include <libkern/OSCacheControl.h>
	#include <sys/sysctl.h>
	#include <unistd.h>
	#include <vector>
#endif

namespace bnch_swt::internal {

	enum class cache_level {
		one	  = 1,
		two	  = 2,
		three = 3,
	};

	BNCH_SWT_HOST size_t get_cache_line_size() {
#if BNCH_SWT_PLATFORM_WINDOWS
		DWORD buffer_size = 0;
		GetLogicalProcessorInformation(nullptr, &buffer_size);
		std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(buffer_size / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
		if (!GetLogicalProcessorInformation(buffer.data(), &buffer_size)) {
			std::cerr << "Failed to retrieve processor information!" << std::endl;
			return 0;
		}

		for (const auto& info: buffer) {
			if (info.Relationship == RelationCache && info.Cache.Level == 1) {
				return info.Cache.LineSize;
			}
		}
#elif BNCH_SWT_PLATFORM_LINUX
		long line_size = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
		if (line_size <= 0) {
			std::cerr << "Failed to retrieve cache line size using sysconf!" << std::endl;
			return 0;
		}
		return static_cast<size_t>(line_size);
#elif BNCH_SWT_PLATFORM_MAC
		size_t line_size = 0;
		size_t size		 = sizeof(line_size);
		if (sysctlbyname("hw.cachelinesize", &line_size, &size, nullptr, 0) != 0) {
			std::cerr << "Failed to retrieve cache line size using sysctl!" << std::endl;
			return 0;
		}
		return line_size;
#else
		std::cerr << "Unsupported platform!" << std::endl;
		return 0;
#endif
		return 0;
	}

	BNCH_SWT_HOST size_t get_cache_size(cache_level level) {
#if BNCH_SWT_PLATFORM_WINDOWS
		DWORD buffer_size = 0;
		cache_level cache_level_val{ level };
		PROCESSOR_CACHE_TYPE cache_type{ level == cache_level::one ? PROCESSOR_CACHE_TYPE::CacheInstruction : PROCESSOR_CACHE_TYPE::CacheUnified };
		std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer{};

		GetLogicalProcessorInformation(nullptr, &buffer_size);
		buffer.resize(buffer_size / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));

		if (!GetLogicalProcessorInformation(buffer.data(), &buffer_size)) {
			std::cerr << "Failed to retrieve processor information!" << std::endl;
			return 0;
		}

		size_t cache_size = 0;
		auto collect_size = [&](auto cache_level_new, auto cache_type_new) {
			size_t cache_size_new{};
			const auto info_count = buffer_size / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
			for (size_t i = 0; i < info_count; ++i) {
				if (buffer[i].Relationship == RelationCache && buffer[i].Cache.Level == static_cast<int32_t>(cache_level_new) && buffer[i].Cache.Type == cache_type_new) {
					cache_size_new = buffer[i].Cache.Size;
					break;
				}
			}
			return cache_size_new;
		};
		if (level == cache_level::one) {
			cache_size += collect_size(cache_level_val, PROCESSOR_CACHE_TYPE::CacheData);
		}
		return cache_size + collect_size(cache_level_val, cache_type);
#elif BNCH_SWT_PLATFORM_LINUX || BNCH_SWT_PLATFORM_ANDROID
		size_t cache_size = 0;

		auto get_cache_size_from_file = [](const std::string& cache_type) {
			const std::string cache_file_path = "/sys/devices/system/cpu/cpu0/cache/index" + cache_type + "/size";
			std::ifstream file(cache_file_path);
			if (!file.is_open()) {
				std::cerr << "Failed to open cache info file: " << cache_file_path << std::endl;
				return static_cast<size_t>(0);
			}

			std::string size_str;
			file >> size_str;
			file.close();

			size_t size = 0;
			if (size_str.back() == 'K') {
				size = std::stoul(size_str) * 1024;
			} else if (size_str.back() == 'M') {
				size = std::stoul(size_str) * 1024 * 1024;
			} else {
				size = std::stoul(size_str);
			}
			return size;
		};

		if (level == cache_level::one) {
			cache_size += get_cache_size_from_file("0");
			cache_size += get_cache_size_from_file("1");
		} else {
			std::string index = (level == cache_level::two) ? "2" : "3";
			cache_size		  = get_cache_size_from_file(index);
		}

		return cache_size;
#elif BNCH_SWT_PLATFORM_MAC
		auto get_cache_size = [](const std::string& cache_type) {
			size_t cache_size_new = 0;
			size_t size			  = sizeof(cache_size_new);

			std::string sysctl_query = "hw." + cache_type + "cachesize";
			if (sysctlbyname(sysctl_query.c_str(), &cache_size_new, &size, nullptr, 0) != 0) {
				return size_t{ 0 };
			}
			return cache_size_new;
		};

		if (level == cache_level::one) {
			return get_cache_size("l1d") + get_cache_size("l1i");
		} else if (level == cache_level::two) {
			return get_cache_size("l2");
		} else {
			return get_cache_size("l3");
		}
#endif
		return 0;
	}


#if BNCH_SWT_PLATFORM_WINDOWS
	BNCH_SWT_HOST static void flush_cache(void* ptr, size_t size, size_t cache_line_size, bool clear_instruction_cache = false) {
		char* buffer = static_cast<char*>(ptr);
		for (size_t i = 0; i < size; i += cache_line_size) {
			_mm_clflush(buffer + i);
		}
		_mm_sfence();

		if (clear_instruction_cache) {
			if (!FlushInstructionCache(GetCurrentProcess(), buffer, size)) {
				std::cerr << "Failed to flush instruction cache!" << std::endl;
			}
		}
#elif BNCH_SWT_PLATFORM_LINUX
	BNCH_SWT_HOST static void flush_cache(void* ptr, size_t size, size_t, bool clear_instruction_cache = false) {
		char* buffer = static_cast<char*>(ptr);
	#if BNCH_SWT_X86_64
		for (size_t i = 0; i < size; i += cache_line_size) {
			__builtin_ia32_clflush(buffer + i);
		}
	#else
	#endif

		if (clear_instruction_cache) {
			__builtin___clear_cache(buffer, buffer + size);
		}
#elif BNCH_SWT_PLATFORM_ANDROID
	BNCH_SWT_HOST static void flush_cache(void* ptr, size_t size, size_t, bool clear_instruction_cache = false) {
		char* buffer = static_cast<char*>(ptr);
		if (clear_instruction_cache) {
			__builtin___clear_cache(buffer, buffer + size);
		}
#elif BNCH_SWT_PLATFORM_MAC
	BNCH_SWT_HOST static void flush_cache(void* ptr, size_t size, size_t, bool clear_instruction_cache = false) {
		if (clear_instruction_cache) {
			sys_icache_invalidate(ptr, size);
		} else {
			sys_dcache_flush(ptr, size);
		}
#else
#endif
	}

	template<benchmark_types benchmark_type> class cache_clearer {
		size_t cache_line_size{ get_cache_line_size() };
		std::array<size_t, 3> cache_sizes{ { get_cache_size(cache_level::one), get_cache_size(cache_level::two), get_cache_size(cache_level::three) } };
		size_t top_level_cache{ [&] {
			if (cache_sizes[2] > cache_sizes[1]) {
				return 2ull;
			} else if (cache_sizes[1] > cache_sizes[0]) {
				return 1ull;
			} else {
				return 0ull;
			}
		}() };
		std::vector<char> evict_buffer{ [&] {
			std::vector<char> return_values{};
			return_values.resize(top_level_cache < 3 ? (cache_sizes[top_level_cache] + cache_line_size) : 0);
			return return_values;
		}() };

		BNCH_SWT_HOST void evict_cache(size_t cache_level) {
			if (cache_level >= 1 && cache_level <= 3 && cache_sizes[cache_level - 1] > 0) {
				for (size_t i = 0; i < cache_sizes[cache_level - 1] + cache_line_size; i += cache_line_size) {
					if (i < evict_buffer.size()) {
						evict_buffer[i] = static_cast<char>(i);
					}
				}

				flush_cache(evict_buffer.data(), evict_buffer.size(), cache_line_size);
				if (cache_level == 1) {
					flush_cache(evict_buffer.data(), evict_buffer.size(), cache_line_size, true);
				}
			}
		}

	  public:
		BNCH_SWT_HOST void evict_caches() {
			evict_cache(3);
			evict_cache(2);
			evict_cache(1);
		}
	};

}
