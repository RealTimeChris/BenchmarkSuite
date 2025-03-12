#pragma once

#include <cstdint>
#include <atomic>

#if defined(__clang__) || (defined(__GNUC__) && defined(__llvm__))
	#define OIML_CLANG 1
#elif defined(_MSC_VER)
	#pragma warning(disable : 4820)
	#pragma warning(disable : 4371)
	#pragma warning(disable : 4061)
	#define OIML_MSVC 1
#elif defined(__GNUC__) && !defined(__clang__)
	#define OIML_GNUCXX 1
#endif

#if defined(macintosh) || defined(Macintosh) || (defined(__APPLE__) && defined(__MACH__)) || defined(TARGET_OS_MAC)
	#define OIML_MAC 1
	#include <sys/types.h>
#elif defined(linux) || defined(__linux) || defined(__linux__) || defined(__gnu_linux__)
	#define OIML_LINUX 1
#elif defined(WIN32) || defined(_WIN32) || defined(_WIN64)
	#define OIML_WIN 1
#else
	#error "Undetected platform."
#endif

#if !defined(OIML_LIKELY)
	#define OIML_LIKELY(...) (__VA_ARGS__) [[likely]]
#endif

#if !defined(OIML_UNLIKELY)
	#define OIML_UNLIKELY(...) (__VA_ARGS__) [[unlikely]]
#endif

#if !defined(OIML_ELSE_UNLIKELY)
	#define OIML_ELSE_UNLIKELY(...) __VA_ARGS__ [[unlikely]]
#endif

#if defined(OIML_MSVC)
	#define OIML_INLINE inline
	#define OIML_FORCE_INLINE [[msvc::forceinline]] inline
#elif defined(OIML_CLANG)
	#define OIML_INLINE inline
	#define OIML_FORCE_INLINE inline __attribute__((always_inline))
#elif defined(OIML_GNUCXX)
	#define OIML_INLINE inline
	#define OIML_FORCE_INLINE inline __attribute__((always_inline))
#endif

#if !defined OIML_ALIGN
	#define OIML_ALIGN(b) alignas(b)
#endif
