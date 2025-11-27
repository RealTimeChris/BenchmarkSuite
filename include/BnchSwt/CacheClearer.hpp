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

#if BNCH_SWT_PLATFORM_WINDOWS
	#include <Windows.h>
	#include <intrin.h>
#elif BNCH_SWT_PLATFORM_LINUX
	#include <unistd.h>
	#include <fstream>
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

	BNCH_SWT_HOST size_t getCacheLineSize() {
#if BNCH_SWT_PLATFORM_WINDOWS
		DWORD bufferSize = 0;
		GetLogicalProcessorInformation(nullptr, &bufferSize);
		std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(bufferSize / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
		if (!GetLogicalProcessorInformation(buffer.data(), &bufferSize)) {
			std::cerr << "Failed to retrieve processor information!" << std::endl;
			return 0;
		}

		for (const auto& info: buffer) {
			if (info.Relationship == RelationCache && info.Cache.Level == 1) {
				return info.Cache.LineSize;
			}
		}
#elif BNCH_SWT_PLATFORM_LINUX
		long lineSize = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
		if (lineSize <= 0) {
			std::cerr << "Failed to retrieve cache line size using sysconf!" << std::endl;
			return 0;
		}
		return static_cast<size_t>(lineSize);
#elif BNCH_SWT_PLATFORM_MAC
		size_t lineSize = 0;
		size_t size		= sizeof(lineSize);
		if (sysctlbyname("hw.cachelinesize", &lineSize, &size, nullptr, 0) != 0) {
			std::cerr << "Failed to retrieve cache line size using sysctl!" << std::endl;
			return 0;
		}
		return lineSize;
#else
		std::cerr << "Unsupported platform!" << std::endl;
		return 0;
#endif
		return 0;
	}

	BNCH_SWT_HOST size_t getCacheSize(cache_level level) {
#if BNCH_SWT_PLATFORM_WINDOWS
		DWORD bufferSize = 0;
		cache_level cacheLevel{ level };
		PROCESSOR_CACHE_TYPE cacheType{ level == cache_level::one ? PROCESSOR_CACHE_TYPE::CacheInstruction : PROCESSOR_CACHE_TYPE::CacheUnified };
		std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer{};

		GetLogicalProcessorInformation(nullptr, &bufferSize);
		buffer.resize(bufferSize / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));

		if (!GetLogicalProcessorInformation(buffer.data(), &bufferSize)) {
			std::cerr << "Failed to retrieve processor information!" << std::endl;
			return 0;
		}

		size_t cacheSize = 0;
		auto collectSize = [&](auto cacheLevelNew, auto cacheTypeNew) {
			size_t cacheSizeNew{};
			const auto infoCount = bufferSize / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
			for (size_t i = 0; i < infoCount; ++i) {
				if (buffer[i].Relationship == RelationCache && buffer[i].Cache.Level == static_cast<int32_t>(cacheLevelNew) && buffer[i].Cache.Type == cacheTypeNew) {
					cacheSizeNew = buffer[i].Cache.Size;
					break;
				}
			}
			return cacheSizeNew;
		};
		if (level == cache_level::one) {
			cacheSize += collectSize(cacheLevel, PROCESSOR_CACHE_TYPE::CacheData);
		}
		return cacheSize + collectSize(cacheLevel, cacheType);
#elif BNCH_SWT_PLATFORM_LINUX || defined(BNCH_SWT_ANDROID)
		size_t cacheSize = 0;

		auto getCacheSizeFromFile = [](const std::string& cacheType) {
			const std::string cacheFilePath = "/sys/devices/system/cpu/cpu0/cache/index" + cacheType + "/size";
			std::ifstream file(cacheFilePath);
			if (!file.is_open()) {
				std::cerr << "Failed to open cache info file: " << cacheFilePath << std::endl;
				return static_cast<size_t>(0);
			}

			std::string sizeStr;
			file >> sizeStr;
			file.close();

			size_t size = 0;
			if (sizeStr.back() == 'K') {
				size = std::stoul(sizeStr) * 1024;
			} else if (sizeStr.back() == 'M') {
				size = std::stoul(sizeStr) * 1024 * 1024;
			} else {
				size = std::stoul(sizeStr);
			}
			return size;
		};

		if (level == cache_level::one) {
			cacheSize += getCacheSizeFromFile("0");
			cacheSize += getCacheSizeFromFile("1");
		} else {
			std::string index = (level == cache_level::two) ? "2" : "3";
			cacheSize		  = getCacheSizeFromFile(index);
		}

		return cacheSize;
#elif BNCH_SWT_PLATFORM_MAC
		auto getCacheSize = [](const std::string& cacheType) {
			size_t cacheSizeNew = 0;
			size_t size			= sizeof(cacheSizeNew);

			std::string sysctlQuery = "hw." + cacheType + "cachesize";
			if (sysctlbyname(sysctlQuery.c_str(), &cacheSizeNew, &size, nullptr, 0) != 0) {
				return size_t{ 0 };
			}
			return cacheSizeNew;
		};

		if (level == cache_level::one) {
			return getCacheSize("l1d") + getCacheSize("l1i");
		} else if (level == cache_level::two) {
			return getCacheSize("l2");
		} else {
			return getCacheSize("l3");
		}
#endif
		return 0;
	}


#if BNCH_SWT_PLATFORM_WINDOWS
	BNCH_SWT_HOST static void flushCache(void* ptr, size_t size, size_t cacheLineSize, bool clearInstructionCache = false) {
		char* buffer = static_cast<char*>(ptr);
		for (size_t i = 0; i < size; i += cacheLineSize) {
			_mm_clflush(buffer + i);
		}
		_mm_sfence();

		if (clearInstructionCache) {
			if (!FlushInstructionCache(GetCurrentProcess(), buffer, size)) {
				std::cerr << "Failed to flush instruction cache!" << std::endl;
			}
		}
#elif BNCH_SWT_PLATFORM_LINUX
	BNCH_SWT_HOST static void flushCache(void* ptr, size_t size, size_t, bool clearInstructionCache = false) {
		char* buffer = static_cast<char*>(ptr);
	#if defined(BNCH_SWT_X86_64)
		for (size_t i = 0; i < size; i += cacheLineSize) {
			__builtin_ia32_clflush(buffer + i);
		}
	#else
	#endif

		if (clearInstructionCache) {
			__builtin___clear_cache(buffer, buffer + size);
		}
#elif defined(BNCH_SWT_ANDROID)
	BNCH_SWT_HOST static void flushCache(void* ptr, size_t size, size_t, bool clearInstructionCache = false) {
		char* buffer = static_cast<char*>(ptr);
		if (clearInstructionCache) {
			__builtin___clear_cache(buffer, buffer + size);
		}
#elif BNCH_SWT_PLATFORM_MAC
	BNCH_SWT_HOST static void flushCache(void* ptr, size_t size, size_t, bool clearInstructionCache = false) {
		if (clearInstructionCache) {
			sys_icache_invalidate(ptr, size);
		} else {
			sys_dcache_flush(ptr, size);
		}
#else
#endif
	}

	template<benchmark_types benchmark_type = benchmark_types::cpu> class cache_clearer {
		size_t cacheLineSize{ getCacheLineSize() };
		std::array<size_t, 3> cacheSizes{ { getCacheSize(cache_level::one), getCacheSize(cache_level::two), getCacheSize(cache_level::three) } };
		size_t topLevelCache{ [&] {
			if (cacheSizes[2] > cacheSizes[1]) {
				return 2ull;
			} else if (cacheSizes[1] > cacheSizes[0]) {
				return 1ull;
			} else {
				return 0ull;
			}
		}() };
		std::vector<char> evictBuffer{ [&] {
			std::vector<char> returnValues{};
			returnValues.resize(topLevelCache < 3 ? (cacheSizes[topLevelCache] + cacheLineSize) : 0);
			return returnValues;
		}() };

		BNCH_SWT_HOST void evictCache(size_t cacheLevel) {
			if (cacheLevel >= 1 && cacheLevel <= 3 && cacheSizes[cacheLevel - 1] > 0) {
				for (size_t i = 0; i < cacheSizes[cacheLevel - 1] + cacheLineSize; i += cacheLineSize) {
					if (i < evictBuffer.size()) {
						evictBuffer[i] = static_cast<char>(i);
					}
				}

				flushCache(evictBuffer.data(), evictBuffer.size(), cacheLineSize);
				if (cacheLevel == 1) {
					flushCache(evictBuffer.data(), evictBuffer.size(), cacheLineSize, true);
				}
			}
		}

	  public:
		BNCH_SWT_HOST void evictCaches() {
			evictCache(3);
			evictCache(2);
			evictCache(1);
		}
	};

}
