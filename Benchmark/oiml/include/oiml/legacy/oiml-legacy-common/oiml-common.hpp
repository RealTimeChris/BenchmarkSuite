#pragma once
#if !defined(OIML_COMMON)
	#include <cstdint>

using oiml_half	 = uint16_t;
using oiml_half2 = uint32_t;

// QK = number of values after dequantization
// QK_K = super-block size

	#if defined(OIML_COMMON_DECL_CUDA) || defined(OIML_COMMON_DECL_HIP) || defined(OIML_COMMON_DECL_SYCL)
// QR = QK / number of values before dequantization
// QI = number of 32 bit integers before dequantization
inline static constexpr uint64_t QR4_0{ 2 };
inline static constexpr uint64_t QI4_0{ (oiml::Q_SIZE / (4 * QR4_0)) };

inline static constexpr uint64_t QR4_1{ 2 };
inline static constexpr uint64_t QI4_1{ (oiml::Q_SIZE / (4 * QR4_1)) };

inline static constexpr uint64_t QR5_0{ 2 };
inline static constexpr uint64_t QI5_0{ (oiml::Q_SIZE / (4 * QR5_0)) };

inline static constexpr uint64_t QR5_1{ 2 };
inline static constexpr uint64_t QI5_1{ (oiml::Q_SIZE / (4 * QR5_1)) };

inline static constexpr uint64_t QR8_0{ 1 };
inline static constexpr uint64_t QI8_0{ (oiml::Q_SIZE / (4 * QR8_0)) };

inline static constexpr uint64_t QR8_1{ 1 };
inline static constexpr uint64_t QI8_1{ (oiml::Q_SIZE / (4 * QR8_1)) };

inline static constexpr uint64_t QR2_K{ 4 };
inline static constexpr uint64_t QI2_K{ (oiml::QK_K / (4 * QR2_K)) };

inline static constexpr uint64_t QR3_K{ 4 };
inline static constexpr uint64_t QI3_K{ (oiml::QK_K / (4 * QR3_K)) };

inline static constexpr uint64_t QR4_K{ 2 };
inline static constexpr uint64_t QI4_K{ (oiml::QK_K / (4 * QR4_K)) };

inline static constexpr uint64_t QR5_K{ 2 };
inline static constexpr uint64_t QI5_K{ (oiml::QK_K / (4 * QR5_K)) };

inline static constexpr uint64_t QR6_K{ 2 };
inline static constexpr uint64_t QI6_K{ (oiml::QK_K / (4 * QR6_K)) };

inline static constexpr uint64_t QR2_XXS{ 4 };
inline static constexpr uint64_t QI2_XXS{ (oiml::QK_K / (4 * QR2_XXS)) };

inline static constexpr uint64_t QR2_XS{ 4 };
inline static constexpr uint64_t QI2_XS{ (oiml::QK_K / (4 * QR2_XS)) };

inline static constexpr uint64_t QR2_S{ 4 };
inline static constexpr uint64_t QI2_S{ (oiml::QK_K / (4 * QR2_S)) };

inline static constexpr uint64_t QR3_XXS{ 4 };
inline static constexpr uint64_t QI3_XXS{ (oiml::QK_K / (4 * QR3_XXS)) };

inline static constexpr uint64_t QR3_XS{ 4 };
inline static constexpr uint64_t QI3_XS{ (oiml::QK_K / (4 * QR3_XS)) };

inline static constexpr uint64_t QR1_S{ 8 };
inline static constexpr uint64_t QI1_S{ (oiml::QK_K / (4 * QR1_S)) };

inline static constexpr uint64_t QR1_M{ 8 };
inline static constexpr uint64_t QI1_M{ (oiml::QK_K / (4 * QR1_M)) };


	#endif// OIML_COMMON_DECL_CUDA || OIML_COMMON_DECL_HIP


#endif