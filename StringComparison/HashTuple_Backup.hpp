/*
	MIT License

	Copyright (c) 2023 RealTimeChris

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
/// https://github.com/RealTimeChris/jsonifier
/// Feb 20, 2023
#pragma once

#include <jsonifier/TypeEntities.hpp>
#include <jsonifier/Hash.hpp>
#include <jsonifier/StringView.hpp>
#include <jsonifier/Reflection.hpp>
#include <jsonifier/StringUtils.hpp>
#include <jsonifier/Tuple.hpp>
#include <algorithm>
#include <numeric>

namespace jsonifier_internal {

	struct tuple_reference {
		jsonifier::string_view key{};
		size_t oldIndex{};
	};

	struct index_count {
		const tuple_reference* currentPtr{};
		const tuple_reference* rootPtr{};
		size_t count{};
		char value{};
	};

	template<size_t maxIndex, size_t currentIndex = 0, typename tuple_type>
	constexpr auto collectTupleRefsInternal(const tuple_type& tuple, std::array<tuple_reference, maxIndex>& tupleRefs) {
		if constexpr (currentIndex < maxIndex) {
			tupleRefs[currentIndex].key		 = std::get<currentIndex>(tuple).view();
			tupleRefs[currentIndex].oldIndex = currentIndex;
			return collectTupleRefsInternal<maxIndex, currentIndex + 1>(tuple, tupleRefs);
		}
		return tupleRefs;
	}

	template<typename tuple_type> constexpr auto collectTupleRefs(const tuple_type& tuple) {
		constexpr auto tupleSize = std::tuple_size_v<tuple_type>;
		std::array<tuple_reference, tupleSize> tupleRefs{};
		return collectTupleRefsInternal<tupleSize>(tuple, tupleRefs);
	}

	template<size_t size> constexpr auto sortTupleRefsByFirstByte(const std::array<tuple_reference, size>& tupleRefs) {
		std::array<tuple_reference, size> returnValues{ tupleRefs };
		for (size_t x = 0; x < returnValues.size(); ++x) {
			for (size_t y = 0; y < returnValues.size(); ++y) {
				if (returnValues[x].key[0] < returnValues[y].key[0]) {
					std::swap(returnValues[x], returnValues[y]);
				}
			}
		}
		return returnValues;
	}

	template<size_t size> constexpr auto countUniqueFirstIndices(const std::array<tuple_reference, size>& tupleRefs) {
		size_t currentCount{};
		char currentValue{ -1 };
		for (size_t x = 0; x < size; ++x) {
			if (currentValue != tupleRefs[x].key[0]) {
				currentValue = tupleRefs[x].key[0];
				++currentCount;
			}
		}
		return currentCount;
	}

	template<size_t newSize, size_t size> constexpr auto collectFirstIndexCounts(const std::array<tuple_reference, size>& tupleRefs) {
		std::array<index_count, newSize> returnValues{};
		size_t currentIndex{ 0 };
		returnValues[0].currentPtr = &tupleRefs[0];
		returnValues[0].rootPtr	   = &tupleRefs[0];
		returnValues[0].value	   = tupleRefs[0].key[0];
		for (size_t x = 0; x < size; ++x) {
			if (returnValues[currentIndex].value != tupleRefs[x].key[0]) {
				++currentIndex;
				returnValues[currentIndex].currentPtr = &tupleRefs[x];
				returnValues[currentIndex].rootPtr	  = &tupleRefs[0];
				returnValues[currentIndex].value	  = tupleRefs[x].key[0];
				++returnValues[currentIndex].count;
			} else {
				++returnValues[currentIndex].count;
			}
		}
		return returnValues;
	}

	template<typename value_type> inline static constexpr auto tupleRefs{ collectTupleRefs(jsonifier::concepts::coreV<value_type>) };
	template<typename value_type> inline static constexpr auto sortedTupleReferences{ sortTupleRefsByFirstByte(tupleRefs<value_type>) };
	template<typename value_type> inline static constexpr auto uniqueStringLengthCount{ countUniqueFirstIndices(sortedTupleReferences<value_type>) };
	template<typename value_type>
	inline static constexpr auto uniqueStringLengths{ collectFirstIndexCounts<uniqueStringLengthCount<value_type>>(sortedTupleReferences<value_type>) };

	template<typename value_type, size_t... indices> JSONIFIER_ALWAYS_INLINE constexpr auto createNewTupleInternalSerialize(std::index_sequence<indices...>) noexcept {
		return std::make_tuple(std::get<sortedTupleReferences<value_type>[indices].oldIndex>(jsonifier::concepts::coreV<value_type>)...);
	}

	template<typename value_type> JSONIFIER_ALWAYS_INLINE constexpr auto createNewTupleSerialize() noexcept {
		constexpr auto& tupleRefs = sortedTupleReferences<value_type>;
		return createNewTupleInternalSerialize<value_type>(std::make_index_sequence<tupleRefs.size()>{});
	}

	template<typename value_type> inline static constexpr auto finalTupleStaticDataSerialize = createNewTupleSerialize<unwrap_t<value_type>>();

	template<typename value_type, size_t subTupleIndex, size_t index, size_t size>
	JSONIFIER_ALWAYS_INLINE constexpr const jsonifier::string_view& getKey(const std::array<index_count, size>& arrayOfIndexCounts) noexcept {
		return arrayOfIndexCounts[subTupleIndex].currentPtr[index].key;
	}

	template<typename value_type, size_t subTupleIndex, size_t maxIndex, size_t index> JSONIFIER_ALWAYS_INLINE constexpr auto keyStatsInternal(key_stats_t stats) noexcept {
		if constexpr (index < maxIndex) {
			constexpr const jsonifier::string_view& key{ getKey<value_type, subTupleIndex, index>(uniqueStringLengths<value_type>) };
			constexpr auto n{ key.size() };
			if (n < stats.minLength) {
				stats.minLength = n;
			}
			if (n > stats.maxLength) {
				stats.maxLength = n;
			}
			return keyStatsInternal<value_type, subTupleIndex, maxIndex, index + 1>(stats);
		} else {
			if constexpr (maxIndex > 0) {
				stats.lengthRange = stats.maxLength - stats.minLength;
			}
			return stats;
		}
	}

	constexpr std::array<size_t, 6> hashTupleMaxSizes{ 16, 32, 64, 128, 256, 512 };

	constexpr size_t getMaxSizeIndex(size_t currentSize) {
		for (uint64_t x = 0; x < hashTupleMaxSizes.size(); ++x) {
			if (currentSize <= hashTupleMaxSizes[x]) {
				return x;
			}
		}
		return std::numeric_limits<size_t>::max();
	}

	template<size_t newSize> constexpr auto collectLargestFirstIndexCount(const std::array<index_count, newSize>& values) {
		size_t returnValue{};
		for (const index_count& value: values) {
			if (value.count > returnValue) {
				returnValue = value.count;
			}
		}
		return returnValue;
	}

	template<typename value_type, size_t subTupleIndex> JSONIFIER_ALWAYS_INLINE constexpr auto keyStats() noexcept {
		constexpr auto N{ uniqueStringLengths<value_type>[subTupleIndex].count };
		key_stats_t returnValues{};
		returnValues.firstIndexWithLargestCount = collectLargestFirstIndexCount(uniqueStringLengths<value_type>);
		return keyStatsInternal<value_type, subTupleIndex, N, 0>(returnValues);
	}

	template<typename value_type01, typename value_type02>
	JSONIFIER_ALWAYS_INLINE constexpr bool contains(const value_type01* hashData, value_type02 byteToCheckFor, size_t size) noexcept {
		for (size_t x = 0; x < size; ++x) {
			if (hashData[x] == byteToCheckFor) {
				return true;
			}
		}
		return false;
	}

	template<size_t index = 0, size_t subTupleIndex, typename value_type> bool tupleSelectAndCompare(size_t runtimeIndex, const char* string) noexcept {
		if constexpr (index < uniqueStringLengths<value_type>[subTupleIndex].count) {
			if (runtimeIndex == index) {
				static constexpr auto currentKey = getKey<value_type, subTupleIndex, index>(uniqueStringLengths<value_type>);
				return compare<currentKey.size()>(currentKey.data(), string);
			} else {
				return tupleSelectAndCompare<index + 1, subTupleIndex, value_type>(runtimeIndex, string);
			}
		} else {
			return false;
		}
	}

	template<typename value_type> JSONIFIER_ALWAYS_INLINE constexpr size_t getActualSize() noexcept {
		using tuple_type = unwrap_t<decltype(finalTupleStaticDataSerialize)>;
		return std::tuple_size_v<tuple_type>;
	}

	template<typename value_type, size_t subTupleIndex> JSONIFIER_ALWAYS_INLINE constexpr size_t getActualSize() noexcept {
		return uniqueStringLengths<value_type>[subTupleIndex].count;
	}

	template<size_t length> struct tuple_simd {
		using type = std::conditional_t<length >= 64 && bytesPerStep >= 64, jsonifier_simd_int_512,
			std::conditional_t<length >= 32 && bytesPerStep >= 32, jsonifier_simd_int_256, jsonifier_simd_int_128>>;
	};

	template<size_t length> using tuple_simd_t = tuple_simd<length>::type;

	enum class sub_tuple_type {
		unset			  = 0,
		single_byte		  = 1,
		simd_unique_index = 2,
		simd_full_length  = 3,
	};

	template<typename key_type_new, typename value_type, size_t subTupleIndexNew, size_t maxSizeIndexNew> struct sub_tuple_construction_data : public key_hasher {
		using simd_type = tuple_simd_t<hashTupleMaxSizes[maxSizeIndexNew]>;
		JSONIFIER_ALIGN std::array<uint8_t, hashTupleMaxSizes[maxSizeIndexNew] + 1> controlBytes{};
		JSONIFIER_ALIGN std::array<size_t, hashTupleMaxSizes[maxSizeIndexNew] + 1> indices{};
		size_t storageSize{ hashTupleMaxSizes[maxSizeIndexNew] };
		size_t subTupleIndex{ subTupleIndexNew };
		sub_tuple_type type{};
		size_t uniqueIndex{};
		size_t bucketSize{};
		size_t numGroups{};

		JSONIFIER_ALWAYS_INLINE constexpr sub_tuple_construction_data() noexcept = default;
	};

	template<typename key_type, typename value_type, size_t subTupleIndex, template<typename, typename, size_t, size_t> typename sub_tuple_construction_data_type>
	using sub_tuple_construction_data_variant =
		std::variant<sub_tuple_construction_data_type<key_type, value_type, subTupleIndex, 0>, sub_tuple_construction_data_type<key_type, value_type, subTupleIndex, 1>,
			sub_tuple_construction_data_type<key_type, value_type, subTupleIndex, 2>, sub_tuple_construction_data_type<key_type, value_type, subTupleIndex, 3>,
			sub_tuple_construction_data_type<key_type, value_type, subTupleIndex, 4>, sub_tuple_construction_data_type<key_type, value_type, subTupleIndex, 5>>;

	template<size_t size>
	JSONIFIER_ALWAYS_INLINE constexpr size_t findUniqueColumnIndex(const std::array<jsonifier::string_view, size>& strings, const key_stats_t& keyStats) noexcept {
		constexpr size_t alphabetSize = 256;

		for (size_t index = 0; index < keyStats.minLength; ++index) {
			bool allDifferent = true;
			std::array<bool, alphabetSize> seen{};

			for (const auto& str: strings) {
				char c = str[index % str.size()];
				if (seen[static_cast<unsigned char>(c)]) {
					allDifferent = false;
					break;
				}
				seen[static_cast<unsigned char>(c)] = true;
			}

			if (allDifferent) {
				return index;
			}
		}

		return std::numeric_limits<size_t>::max();
	}

	constexpr uint64_t roundUpToPowerOfTwo(uint64_t value) noexcept {
		if (value <= 1) {
			return 1;
		}
		--value;
		value |= value >> 1;
		value |= value >> 2;
		value |= value >> 4;
		value |= value >> 8;
		value |= value >> 16;
		value |= value >> 32;
		return value + 1;
	}

	JSONIFIER_ALWAYS_INLINE constexpr size_t setSimdWidth(size_t length) noexcept {
		return length >= 64 && bytesPerStep >= 64 ? 64 : length >= 32 && bytesPerStep >= 32 ? 32 : 16;
	}

	JSONIFIER_ALWAYS_INLINE constexpr const jsonifier::string_view& getKey(const jsonifier_internal::index_count& arrayOfIndexCounts, size_t index) noexcept {
		return arrayOfIndexCounts.currentPtr[index].key;
	}

	JSONIFIER_ALWAYS_INLINE constexpr auto keyStats(const jsonifier_internal::index_count& tupleRefs) noexcept {
		jsonifier_internal::key_stats_t stats{};
		for (uint64_t x = 0; x < tupleRefs.count; ++x) {
			const jsonifier::string_view& key{ getKey(tupleRefs, x) };
			auto n{ key.size() };
			if (n < stats.minLength) {
				stats.minLength = n;
			}
			if (n > stats.maxLength) {
				stats.maxLength = n;
			}
		}
		if (tupleRefs.count > 0) {
			stats.lengthRange = stats.maxLength - stats.minLength;
		}
		return stats;
	}

	JSONIFIER_ALWAYS_INLINE constexpr size_t findUniqueColumnIndex(const jsonifier_internal::index_count& tupleRefs, const jsonifier_internal::key_stats_t& keyStats) noexcept {
		constexpr size_t alphabetSize = 256;

		for (size_t index = 0; index < keyStats.minLength; ++index) {
			bool allDifferent = true;
			std::array<bool, alphabetSize> seen{};

			for (uint64_t x = 0; x < tupleRefs.count; ++x) {
				char c = tupleRefs.rootPtr[x].key[index % tupleRefs.rootPtr[x].key.size()];
				if (seen[static_cast<unsigned char>(c)]) {
					allDifferent = false;
					break;
				}
				seen[static_cast<unsigned char>(c)] = true;
			}

			if (allDifferent) {
				return index;
			}
		}

		return std::numeric_limits<size_t>::max();
	}

	template<typename value_type, size_t maxSizeIndexNew> struct sub_tuple_construction_data_new {
		using simd_type = jsonifier_internal::tuple_simd_t<jsonifier_internal::hashTupleMaxSizes[maxSizeIndexNew]>;
		std::array<uint8_t, jsonifier_internal::hashTupleMaxSizes[maxSizeIndexNew] + 1> controlBytes{};
		std::array<size_t, jsonifier_internal::hashTupleMaxSizes[maxSizeIndexNew] + 1> indices{};
		size_t bucketSize{ jsonifier_internal::setSimdWidth(jsonifier_internal::hashTupleMaxSizes[maxSizeIndexNew]) };
		size_t numGroups{ jsonifier_internal::hashTupleMaxSizes[maxSizeIndexNew] / bucketSize };
		jsonifier_internal::sub_tuple_type type{};
		jsonifier_internal::key_hasher hasher{};
		size_t uniqueIndex{};

		JSONIFIER_ALWAYS_INLINE constexpr sub_tuple_construction_data_new() noexcept = default;
	};

	template<typename value_type, size_t numSubTuplesNew, size_t maxSizeIndexNew> struct tuple_construction_data {
		std::array<sub_tuple_construction_data_new<value_type, maxSizeIndexNew>, numSubTuplesNew> subTuples{};
	};

	template<typename value_type, size_t numSubTuplesNew, size_t maxSizeIndexNew, template<typename, size_t, size_t> typename sub_tuple_construction_data_new_type>
	using sub_tuple_construction_data_new_variant =
		std::variant<sub_tuple_construction_data_new_type<value_type, numSubTuplesNew, 0>, sub_tuple_construction_data_new_type<value_type, numSubTuplesNew, 1>,
			sub_tuple_construction_data_new_type<value_type, numSubTuplesNew, 2>, sub_tuple_construction_data_new_type<value_type, numSubTuplesNew, 3>,
			sub_tuple_construction_data_new_type<value_type, numSubTuplesNew, 4>, sub_tuple_construction_data_new_type<value_type, numSubTuplesNew, 5>>;

	template<typename value_type, size_t maxSizeIndexNew>
	JSONIFIER_ALWAYS_INLINE constexpr auto collectSimdFullLengthSubTupleData(const jsonifier_internal::index_count& pairsNew) noexcept {
		auto keyStatsVal = keyStats(pairsNew);
		jsonifier_internal::xoshiro256 prng{};
		auto constructForGivenStringLength = [&](const auto maxSizeIndex,
												 auto&& constructForGivenStringLength) mutable -> sub_tuple_construction_data_new<value_type, maxSizeIndex> {
			constexpr size_t storageSize = jsonifier_internal::hashTupleMaxSizes[maxSizeIndex];
			constexpr size_t bucketSize	 = jsonifier_internal::setSimdWidth(jsonifier_internal::hashTupleMaxSizes[maxSizeIndex]);
			constexpr size_t numGroups	 = storageSize > bucketSize ? storageSize / bucketSize : 1;
			sub_tuple_construction_data_new<value_type, maxSizeIndex> returnValues{};
			returnValues.numGroups	= numGroups;
			returnValues.bucketSize = bucketSize;
			bool collided{};
			for (size_t w = 0; w < keyStatsVal.maxLength; ++w) {
				returnValues.uniqueIndex = w;
				for (size_t x = 0; x < 2; ++x) {
					size_t bucketSizes[numGroups]{};
					std::fill(returnValues.controlBytes.begin(), returnValues.controlBytes.end(), std::numeric_limits<uint8_t>::max());
					std::fill(returnValues.indices.begin(), returnValues.indices.end(), returnValues.indices.size() - 1);
					returnValues.hasher.setSeedCt(prng());
					collided = false;
					for (size_t y = 0; y < pairsNew.count; ++y) {
						const auto keyLength	 = w > pairsNew.rootPtr[y].key.size() ? pairsNew.rootPtr[y].key.size() : w;
						const auto hash			 = returnValues.hasher.hashKeyCt(pairsNew.rootPtr[y].key.data(), keyLength);
						const auto groupPos		 = (hash >> 8) % numGroups;
						const auto ctrlByte		 = static_cast<uint8_t>(hash);
						const auto bucketSizeNew = bucketSizes[groupPos]++;
						const auto slot			 = ((groupPos * bucketSize) + bucketSizeNew);

						if (bucketSizeNew >= bucketSize || returnValues.indices[slot] != returnValues.indices.size() - 1 ||
							contains(returnValues.controlBytes.data() + groupPos * bucketSize, ctrlByte, bucketSize)) {
							std::fill(bucketSizes, bucketSizes + numGroups, 0);
							collided = true;
							break;
						}
						returnValues.controlBytes[slot] = ctrlByte;
						returnValues.indices[slot]		= y;
					}
					if (!collided) {
						break;
					}
				}
				if (!collided) {
					break;
				}
			}
			if (collided) {
				returnValues.uniqueIndex = std::numeric_limits<size_t>::max();
				return returnValues;
			}
			returnValues.type = jsonifier_internal::sub_tuple_type::simd_full_length;
			return returnValues;
		};
		return constructForGivenStringLength(std::integral_constant<size_t, maxSizeIndexNew>{}, constructForGivenStringLength);
	}

	template<typename value_type, size_t maxSizeIndexNew>
	JSONIFIER_ALWAYS_INLINE constexpr auto collectSimdUniqueIndexSubTupleData(const jsonifier_internal::index_count& pairsNew) noexcept {
		auto keyStatsVal = keyStats(pairsNew);
		jsonifier_internal::xoshiro256 prng{};

		const auto uniqueIndex = findUniqueColumnIndex(pairsNew, keyStatsVal);
		if (uniqueIndex != std::numeric_limits<size_t>::max()) {
			auto constructForGivenStringLength = [&](const auto maxSizeIndex,
													 auto&& constructForGivenStringLength) mutable -> sub_tuple_construction_data_new<value_type, maxSizeIndex> {
				constexpr size_t storageSize = jsonifier_internal::hashTupleMaxSizes[maxSizeIndex];
				constexpr size_t bucketSize	 = jsonifier_internal::setSimdWidth(jsonifier_internal::hashTupleMaxSizes[maxSizeIndex]);
				constexpr size_t numGroups	 = storageSize > bucketSize ? storageSize / bucketSize : 1;
				sub_tuple_construction_data_new<value_type, maxSizeIndex> returnValues{};
				returnValues.uniqueIndex = uniqueIndex;
				returnValues.numGroups	 = numGroups;
				returnValues.bucketSize	 = bucketSize;
				bool collided{};
				for (size_t w = 0; w < keyStatsVal.minLength; ++w) {
					returnValues.uniqueIndex = w;
					for (size_t x = 0; x < 2; ++x) {
						size_t bucketSizes[numGroups]{};
						std::fill(returnValues.controlBytes.begin(), returnValues.controlBytes.end(), std::numeric_limits<uint8_t>::max());
						std::fill(returnValues.indices.begin(), returnValues.indices.end(), returnValues.indices.size() - 1);
						returnValues.hasher.setSeedCt(prng());
						collided = false;
						for (size_t y = 0; y < pairsNew.count; ++y) {
							const auto hash			 = (returnValues.hasher.getSeed() ^ pairsNew.rootPtr[y].key.data()[uniqueIndex]);
							const auto groupPos		 = (hash >> 8) % numGroups;
							const auto ctrlByte		 = static_cast<uint8_t>(hash);
							const auto bucketSizeNew = bucketSizes[groupPos]++;
							const auto slot			 = ((groupPos * bucketSize) + bucketSizeNew);

							if (bucketSizeNew >= bucketSize || returnValues.indices[slot] != returnValues.indices.size() - 1 ||
								contains(returnValues.controlBytes.data() + groupPos * bucketSize, ctrlByte, bucketSize)) {
								std::fill(bucketSizes, bucketSizes + numGroups, 0);
								collided = true;
								break;
							}
							returnValues.controlBytes[slot] = ctrlByte;
							returnValues.indices[slot]		= y;
						}
						if (!collided) {
							break;
						}
					}
					if (!collided) {
						break;
					}
				}
				if (collided) {
					if constexpr (maxSizeIndex < std::size(jsonifier_internal::hashTupleMaxSizes) - 1) {
						returnValues.uniqueIndex = std::numeric_limits<size_t>::max();
						return returnValues;
					} else {
						return collectSimdFullLengthSubTupleData<value_type, maxSizeIndexNew>(pairsNew);
					}
				}
				returnValues.type = jsonifier_internal::sub_tuple_type::simd_unique_index;
				return returnValues;
			};
			return constructForGivenStringLength(std::integral_constant<size_t, maxSizeIndexNew>{}, constructForGivenStringLength);
		} else {
			return collectSimdFullLengthSubTupleData<value_type, maxSizeIndexNew>(pairsNew);
		}
		return sub_tuple_construction_data_new<value_type, maxSizeIndexNew>{};
	}

	template<typename value_type, size_t maxSizeIndexNew>
	JSONIFIER_ALWAYS_INLINE constexpr auto collectSingleByteSubTupleData(const jsonifier_internal::index_count& pairsNew) noexcept {
		auto keyStatsVal = keyStats(pairsNew);
		jsonifier_internal::xoshiro256 prng{};

		const auto uniqueIndex = findUniqueColumnIndex(pairsNew, keyStatsVal);
		if (uniqueIndex != std::numeric_limits<size_t>::max()) {
			auto constructForGivenStringLength = [&](const auto maxSizeIndex,
													 auto&& constructForGivenStringLength) mutable -> sub_tuple_construction_data_new<value_type, maxSizeIndex> {
				constexpr size_t storageSize = jsonifier_internal::hashTupleMaxSizes[maxSizeIndex];
				sub_tuple_construction_data_new<value_type, maxSizeIndex> returnValues{};
				returnValues.uniqueIndex = uniqueIndex;
				bool collided{};
				for (size_t x = 0; x < 2; ++x) {
					std::fill(returnValues.indices.begin(), returnValues.indices.end(), returnValues.indices.size() - 1);
					returnValues.hasher.setSeedCt(prng());
					collided = false;
					for (size_t y = 0; y < pairsNew.count; ++y) {
						const auto hash = (returnValues.hasher.getSeed() ^ pairsNew.rootPtr[y].key.data()[uniqueIndex]);
						const auto slot = hash % storageSize;
						if (returnValues.indices[slot] != returnValues.indices.size() - 1) {
							collided = true;
							break;
						}
						returnValues.indices[slot] = y;
					}
					if (!collided) {
						break;
					}
				}
				if (collided) {
					if constexpr (maxSizeIndex < std::size(jsonifier_internal::hashTupleMaxSizes) - 1) {
						returnValues.uniqueIndex = std::numeric_limits<size_t>::max();
						return returnValues;
					} else {
						return collectSimdUniqueIndexSubTupleData<value_type, maxSizeIndex>(pairsNew);
					}
				}
				returnValues.type = jsonifier_internal::sub_tuple_type::single_byte;
				return returnValues;
			};
			return constructForGivenStringLength(std::integral_constant<size_t, maxSizeIndexNew>{}, constructForGivenStringLength);
		} else {
			return collectSimdUniqueIndexSubTupleData<value_type, maxSizeIndexNew>(pairsNew);
		}
		return sub_tuple_construction_data_new<value_type, maxSizeIndexNew>{};
	}

	template<typename value_type> JSONIFIER_ALWAYS_INLINE constexpr auto collectTupleConstructionDataImpl() noexcept {
		jsonifier_internal::xoshiro256 prng{};
		constexpr std::array<jsonifier_internal::index_count, jsonifier_internal::uniqueStringLengths<value_type>.size()> uniqueStringLengthsNew =
			jsonifier_internal::uniqueStringLengths<value_type>;
		constexpr auto numSubTuples			= uniqueStringLengthsNew.size();
		constexpr auto maxGroupSize			= jsonifier_internal::collectLargestFirstIndexCount(uniqueStringLengthsNew);
		constexpr auto startingMaxSizeIndex = jsonifier_internal::getMaxSizeIndex(maxGroupSize);
		auto constuctForGivenMaxSizeIndex =
			[&](const auto maxSizeIndex,
				auto&& constructGivenMaxSizeIndexNew) -> sub_tuple_construction_data_new_variant<value_type, numSubTuples, maxSizeIndex, tuple_construction_data> {
			tuple_construction_data<value_type, numSubTuples, maxSizeIndex> returnValues{};
			for (uint64_t x = 0; x < numSubTuples; ++x) {
				returnValues.subTuples[x] = collectSingleByteSubTupleData<value_type, maxSizeIndex>(uniqueStringLengthsNew[x]);
				if (returnValues.subTuples[x].uniqueIndex == std::numeric_limits<size_t>::max()) {
					if constexpr (maxSizeIndex < jsonifier_internal::hashTupleMaxSizes.size() - 1) {
						return constructGivenMaxSizeIndexNew(std::integral_constant<size_t, maxSizeIndex + 1>{}, constructGivenMaxSizeIndexNew);
					}
				}
			}
			return returnValues;
		};
		return constuctForGivenMaxSizeIndex(std::integral_constant<size_t, startingMaxSizeIndex>{}, constuctForGivenMaxSizeIndex);
	}

	template<typename value_type> JSONIFIER_ALWAYS_INLINE constexpr auto collectTupleConstructionData() noexcept {
		constexpr auto newTuple = collectTupleConstructionDataImpl<value_type>();
		constexpr auto newIndex = newTuple.index();
		return std::get<newIndex>(newTuple);
	}

	template<typename key_type, typename value_type, size_t subTupleIndexNew>
	JSONIFIER_ALWAYS_INLINE constexpr auto collectSimdFullLengthConstructionData(const std::array<key_type, getActualSize<value_type, subTupleIndexNew>()>& pairsNew) noexcept {
		constexpr auto keyStatsVal = keyStats<value_type, subTupleIndexNew>();
		xoshiro256 prng{};
		auto constructForGivenStringLength =
			[&](const auto maxSizeIndex,
				auto&& constructForGivenStringLength) mutable -> sub_tuple_construction_data_variant<key_type, value_type, subTupleIndexNew, sub_tuple_construction_data> {
			bool collided{};
			constexpr size_t bucketSize	 = setSimdWidth(hashTupleMaxSizes[maxSizeIndex]);
			constexpr size_t storageSize = hashTupleMaxSizes[maxSizeIndex] + 1;
			constexpr size_t numGroups	 = storageSize > bucketSize ? storageSize / bucketSize : 1;
			sub_tuple_construction_data<key_type, value_type, subTupleIndexNew, maxSizeIndex> returnValues{};
			returnValues.numGroups	= numGroups;
			returnValues.bucketSize = bucketSize;
			for (size_t w = 0; w < keyStatsVal.maxLength; ++w) {
				returnValues.uniqueIndex = w;
				for (size_t x = 0; x < 2; ++x) {
					size_t bucketSizes[numGroups]{};
					std::fill(returnValues.controlBytes.begin(), returnValues.controlBytes.end(), std::numeric_limits<uint8_t>::max());
					std::fill(returnValues.indices.begin(), returnValues.indices.end(), returnValues.indices.size() - 1);
					returnValues.setSeedCt(prng());
					collided = false;
					for (size_t y = 0; y < getActualSize<value_type, subTupleIndexNew>(); ++y) {
						const auto keyLength	 = w > pairsNew[y].size() ? pairsNew[y].size() : w;
						const auto hash			 = returnValues.hashKeyCt(pairsNew[y].data(), keyLength);
						const auto groupPos		 = (hash >> 8) % numGroups;
						const auto ctrlByte		 = static_cast<uint8_t>(hash);
						const auto bucketSizeNew = bucketSizes[groupPos]++;
						const auto slot			 = ((groupPos * bucketSize) + bucketSizeNew);

						if (bucketSizeNew >= bucketSize || returnValues.indices[slot] != returnValues.indices.size() - 1 ||
							contains(returnValues.controlBytes.data() + groupPos * bucketSize, ctrlByte, bucketSize)) {
							std::fill(bucketSizes, bucketSizes + numGroups, 0);
							collided = true;
							break;
						}
						returnValues.controlBytes[slot] = ctrlByte;
						returnValues.indices[slot]		= y;
					}
					if (!collided) {
						break;
					}
				}
				if (!collided) {
					break;
				}
			}
			if (collided) {
				if constexpr (maxSizeIndex < std::size(hashTupleMaxSizes) - 1) {
					return sub_tuple_construction_data_variant<key_type, value_type, subTupleIndexNew, sub_tuple_construction_data>{ constructForGivenStringLength(
						std::integral_constant<size_t, maxSizeIndex + 1>{}, constructForGivenStringLength) };
				}
				returnValues.uniqueIndex = std::numeric_limits<size_t>::max();
				return returnValues;
			} else {
				returnValues.type = sub_tuple_type::simd_full_length;
				return returnValues;
			}
		};
		return constructForGivenStringLength(std::integral_constant<size_t, 0>{}, constructForGivenStringLength);
	}

	template<typename key_type, typename value_type, size_t subTupleIndexNew>
	JSONIFIER_ALWAYS_INLINE constexpr auto collectSimdUniqueIndexConstructionData(const std::array<key_type, getActualSize<value_type, subTupleIndexNew>()>& pairsNew) noexcept {
		constexpr auto keyStatsVal = keyStats<value_type, subTupleIndexNew>();
		xoshiro256 prng{};
		const auto uniqueIndex = findUniqueColumnIndex(pairsNew, keyStatsVal);
		if (uniqueIndex != std::numeric_limits<size_t>::max()) {
			auto constructForGivenStringLength =
				[&](const auto maxSizeIndex,
					auto&& constructForGivenStringLength) mutable -> sub_tuple_construction_data_variant<key_type, value_type, subTupleIndexNew, sub_tuple_construction_data> {
				bool collided{};
				constexpr size_t bucketSize	 = setSimdWidth(hashTupleMaxSizes[maxSizeIndex]);
				constexpr size_t storageSize = hashTupleMaxSizes[maxSizeIndex] + 1;
				constexpr size_t numGroups	 = storageSize > bucketSize ? storageSize / bucketSize : 1;
				sub_tuple_construction_data<key_type, value_type, subTupleIndexNew, maxSizeIndex> returnValues{};
				returnValues.numGroups	= numGroups;
				returnValues.bucketSize = bucketSize;
				for (size_t w = 0; w < keyStatsVal.minLength; ++w) {
					returnValues.uniqueIndex = w;
					for (size_t x = 0; x < 2; ++x) {
						size_t bucketSizes[numGroups]{};
						std::fill(returnValues.controlBytes.begin(), returnValues.controlBytes.end(), std::numeric_limits<uint8_t>::max());
						std::fill(returnValues.indices.begin(), returnValues.indices.end(), returnValues.indices.size() - 1);
						returnValues.setSeedCt(prng());
						collided = false;
						for (size_t y = 0; y < getActualSize<value_type, subTupleIndexNew>(); ++y) {
							const auto hash			 = (returnValues.getSeed() ^ pairsNew[y].data()[w]);
							const auto groupPos		 = (hash >> 8) % numGroups;
							const auto ctrlByte		 = static_cast<uint8_t>(hash);
							const auto bucketSizeNew = bucketSizes[groupPos]++;
							const auto slot			 = ((groupPos * bucketSize) + bucketSizeNew);

							if (bucketSizeNew >= bucketSize || returnValues.indices[slot] != returnValues.indices.size() - 1 ||
								contains(returnValues.controlBytes.data() + groupPos * bucketSize, ctrlByte, bucketSize)) {
								std::fill(bucketSizes, bucketSizes + numGroups, 0);
								collided = true;
								break;
							}
							returnValues.controlBytes[slot] = ctrlByte;
							returnValues.indices[slot]		= y;
						}
						if (!collided) {
							break;
						}
					}
					if (!collided) {
						break;
					}
				}
				if (collided) {
					if constexpr (maxSizeIndex < std::size(hashTupleMaxSizes) - 1) {
						return sub_tuple_construction_data_variant<key_type, value_type, subTupleIndexNew, sub_tuple_construction_data>{ constructForGivenStringLength(
							std::integral_constant<size_t, maxSizeIndex + 1>{}, constructForGivenStringLength) };
					} else {
						return collectSimdFullLengthConstructionData<key_type, value_type, subTupleIndexNew>(pairsNew);
					}
				} else {
					returnValues.type = sub_tuple_type::simd_unique_index;
					return returnValues;
				}
			};
			return constructForGivenStringLength(std::integral_constant<size_t, 0>{}, constructForGivenStringLength);
		} else {
			return collectSimdFullLengthConstructionData<key_type, value_type, subTupleIndexNew>(pairsNew);
		}
	}

	template<typename key_type, typename value_type, size_t subTupleIndexNew>
	JSONIFIER_ALWAYS_INLINE constexpr auto collectSingleByteSubTupleData(const std::array<key_type, getActualSize<value_type, subTupleIndexNew>()>& pairsNew) noexcept {
		constexpr auto keyStatsVal = keyStats<value_type, subTupleIndexNew>();
		xoshiro256 prng{};
		const auto uniqueIndex = findUniqueColumnIndex(pairsNew, keyStatsVal);
		if (uniqueIndex != std::numeric_limits<size_t>::max()) {
			auto constructForGivenStringLength =
				[&](const auto maxSizeIndex,
					auto&& constructForGivenStringLength) mutable -> sub_tuple_construction_data_variant<key_type, value_type, subTupleIndexNew, sub_tuple_construction_data> {
				constexpr size_t storageSize = hashTupleMaxSizes[maxSizeIndex];
				sub_tuple_construction_data<key_type, value_type, subTupleIndexNew, maxSizeIndex> returnValues{};
				returnValues.uniqueIndex = uniqueIndex;
				bool collided{};
				for (size_t x = 0; x < 2; ++x) {
					std::fill(returnValues.indices.begin(), returnValues.indices.end(), returnValues.indices.size() - 1);
					returnValues.setSeedCt(prng());
					collided = false;
					for (size_t y = 0; y < getActualSize<value_type, subTupleIndexNew>(); ++y) {
						const auto hash = (returnValues.getSeed() ^ pairsNew[y].data()[uniqueIndex]);
						const auto slot = hash % storageSize;
						if (returnValues.indices[slot] != returnValues.indices.size() - 1) {
							collided = true;
							break;
						}
						returnValues.indices[slot] = y;
					}
					if (!collided) {
						break;
					}
				}
				if (collided) {
					if constexpr (maxSizeIndex < std::size(hashTupleMaxSizes) - 1) {
						return sub_tuple_construction_data_variant<key_type, value_type, subTupleIndexNew, sub_tuple_construction_data>{ constructForGivenStringLength(
							std::integral_constant<size_t, maxSizeIndex + 1>{}, constructForGivenStringLength) };
					} else {
						return collectSimdUniqueIndexConstructionData<key_type, value_type, subTupleIndexNew>(pairsNew);
					}
				}
				returnValues.type = sub_tuple_type::single_byte;
				return returnValues;
			};
			return constructForGivenStringLength(std::integral_constant<size_t, 0>{}, constructForGivenStringLength);
		} else {
			return collectSimdUniqueIndexConstructionData<key_type, value_type, subTupleIndexNew>(pairsNew);
		}
	}

	template<typename value_type, size_t subTupleIndex, size_t... index>
	JSONIFIER_ALWAYS_INLINE constexpr auto generateSubTupleConstructionDataImpl(std::index_sequence<index...>) noexcept {
		constexpr auto tupleSize = uniqueStringLengths<value_type>[subTupleIndex].count;

		if constexpr (tupleSize == 0) {
			return nullptr;
		} else {
			return collectSingleByteSubTupleData<jsonifier::string_view, value_type, subTupleIndex>(
				{ getKey<value_type, subTupleIndex, index>(uniqueStringLengths<value_type>)... });
		}
	}

	template<typename value_type, size_t subTupleIndex> JSONIFIER_ALWAYS_INLINE constexpr auto generateSubTupleConstructionData() noexcept {
		constexpr auto tupleSize = uniqueStringLengths<value_type>[subTupleIndex].count;
		constexpr auto mapNew	 = generateSubTupleConstructionDataImpl<value_type, subTupleIndex>(std::make_index_sequence<tupleSize>{});
		constexpr auto newIndex	 = mapNew.index();
		return std::get<newIndex>(mapNew);
	}

	struct parse_options;

	template<typename derived_type> class parser;

	template<typename derived_type> struct parse_options_internal;

	template<typename derived_type, const parse_options_internal<derived_type>&, typename value_type> struct parse_impl {};

	template<size_t currentIndex, size_t maxIndex, size_t subTupleIndex, typename derived_type, const parse_options_internal<derived_type>& options, typename value_type,
		typename iterator>
	JSONIFIER_ALWAYS_INLINE bool checkAndParse(value_type& value, iterator& iter, iterator& end, size_t index) {
		if constexpr (currentIndex < uniqueStringLengths<value_type>[subTupleIndex].count) {
			if (currentIndex == index) {
				static constexpr auto& tuple = uniqueStringLengths<value_type>[subTupleIndex].currentPtr[currentIndex];
				if constexpr (tuple.oldIndex < std::tuple_size_v<jsonifier::concepts::core_t<value_type>>) {
					static constexpr auto& subTuple = std::get<tuple.oldIndex>(jsonifier::concepts::coreV<value_type>);
					static constexpr auto& key		= subTuple.view();
					static constexpr auto keySize	= key.size();
					if constexpr (jsonifier::concepts::is_double_ptr<iterator>) {
						if (!compare<keySize>(key.data(), (*iter) + 1)) {
							return false;
						}
						++iter;
						if (**iter == ':') [[likely]] {
							++iter;
						} else {
							static constexpr auto sourceLocation{ std::source_location::current() };
							options.parserPtr->getErrors().emplace_back(error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Missing_Colon>(
								getUnderlyingPtr(iter) - options.rootIter, getUnderlyingPtr(end) - options.rootIter, options.rootIter));
							skipToNextValue(iter, end);
							return false;
						}
					} else {
						if (!compare<keySize>(key.data(), iter + 1)) {
							return false;
						}
						iter += keySize + 2;
						if (*iter == ':') [[likely]] {
							++iter;
						} else {
							static constexpr auto sourceLocation{ std::source_location::current() };
							options.parserPtr->getErrors().emplace_back(error::constructError<sourceLocation, error_classes::Parsing, parse_errors::Missing_Colon>(
								getUnderlyingPtr(iter) - options.rootIter, getUnderlyingPtr(end) - options.rootIter, options.rootIter));
							skipToNextValue(iter, end);
							return false;
						}
					}

					static constexpr auto& ptr = subTuple.ptr();
					using member_type		   = unwrap_t<decltype(value.*ptr)>;
					parse_impl<derived_type, options, member_type>::impl(value.*ptr, iter, end);
					return true;
				} else {
					return checkAndParse<currentIndex + 1, maxIndex, subTupleIndex, derived_type, options, value_type, iterator>(value, iter, end, index);
				}
			} else {
				return checkAndParse<currentIndex + 1, maxIndex, subTupleIndex, derived_type, options, value_type, iterator>(value, iter, end, index);
			}
		}
		return false;
	}

	template<typename value_type, size_t subTupleIndex>
	inline static constexpr auto subTupleConstructionStaticData{ generateSubTupleConstructionData<unwrap_t<value_type>, subTupleIndex>() };

	template<typename value_type, typename iterator> struct sub_tuple_base {
		JSONIFIER_ALWAYS_INLINE virtual bool find(value_type& value, iterator& iter, iterator& end) const noexcept = 0;
	};

	template<typename derived_type, const parse_options_internal<derived_type>& options, typename value_type, typename iterator, size_t subTupleIndexNew>
	struct simd_sub_tuple_full_length : public sub_tuple_base<value_type, iterator> {
		inline static constexpr auto& constructionData{ subTupleConstructionStaticData<value_type, subTupleIndexNew> };
		using simd_type	   = typename decltype(subTupleConstructionStaticData<value_type, subTupleIndexNew>)::simd_type;
		using size_type	   = size_t;
		using control_type = uint8_t;

		JSONIFIER_ALIGN static constexpr std::array<control_type, constructionData.storageSize + 1> controlBytes{ constructionData.controlBytes };
		JSONIFIER_ALIGN static constexpr std::array<size_type, constructionData.storageSize + 1> indices{ constructionData.indices };
		static constexpr auto keyStatsVal{ keyStats<value_type, subTupleIndexNew>() };
		static constexpr size_t uniqueIndex{ constructionData.uniqueIndex };
		static constexpr size_t storageSize{ constructionData.storageSize };
		static constexpr size_t bucketSize{ constructionData.bucketSize };
		static constexpr key_hasher hasher{ constructionData.getSeed() };
		static constexpr size_t numGroups{ constructionData.numGroups };
		static constexpr sub_tuple_type type{ constructionData.type };
		static_assert(type != sub_tuple_type::unset, "Failed to build this tuple!");

		JSONIFIER_ALWAYS_INLINE constexpr simd_sub_tuple_full_length() noexcept = default;

		JSONIFIER_ALWAYS_INLINE bool find(value_type& value, iterator& iter, iterator& end) const noexcept {
			const char* newPtr{ iter + 1 + keyStatsVal.minLength };
			memchar<'"'>(newPtr, static_cast<size_t>(end - (iter + 1)));
			size_t length		   = static_cast<size_t>(newPtr - (iter + 1));
			length				   = uniqueIndex > length ? length : uniqueIndex;
			const auto hash		   = hasher.hashKeyRt(iter + 1, length);
			const auto resultIndex = (hash >> 8) & (numGroups - 1) * bucketSize;
			const auto finalIndex  = (simd_internal::tzcnt(simd_internal::opCmpEq(simd_internal::gatherValue<simd_type>(static_cast<control_type>(hash)),
										  simd_internal::gatherValues<simd_type>(controlBytes.data() + resultIndex))) +
				 resultIndex);
			return checkAndParse<0, uniqueStringLengths<value_type>.size(), subTupleIndexNew, derived_type, options, value_type, iterator>(value, iter, end, indices[finalIndex]);
		}

	  protected:
		JSONIFIER_ALWAYS_INLINE static constexpr size_type tzcnt(size_type value) noexcept {
			size_type count{};
			while ((value & 1) == 0 && value != 0) {
				value >>= 1;
				++count;
			}
			return count;
		}

		JSONIFIER_ALWAYS_INLINE static constexpr size_type constMatch(const control_type* hashData, control_type hash) noexcept {
			size_type mask = 0;
			for (size_type x = 0; x < constructionData.bucketSize; ++x) {
				if (hashData[x] == hash) {
					mask |= (1ull << x);
				}
			}
			return tzcnt(mask);
		}
	};

	template<typename derived_type, const parse_options_internal<derived_type>& options, typename value_type, jsonifier::concepts::is_double_ptr iterator, size_t subTupleIndexNew>
	struct simd_sub_tuple_full_length<derived_type, options, value_type, iterator, subTupleIndexNew> : public sub_tuple_base<value_type, iterator> {
		inline static constexpr auto& constructionData{ subTupleConstructionStaticData<value_type, subTupleIndexNew> };
		using simd_type	   = typename decltype(subTupleConstructionStaticData<value_type, subTupleIndexNew>)::simd_type;
		using size_type	   = size_t;
		using control_type = uint8_t;

		JSONIFIER_ALIGN static constexpr std::array<control_type, constructionData.storageSize + 1> controlBytes{ constructionData.controlBytes };
		JSONIFIER_ALIGN static constexpr std::array<size_type, constructionData.storageSize + 1> indices{ constructionData.indices };
		static constexpr auto keyStatsVal{ keyStats<value_type, subTupleIndexNew>() };
		static constexpr size_t uniqueIndex{ constructionData.uniqueIndex };
		static constexpr size_t storageSize{ constructionData.storageSize };
		static constexpr size_t bucketSize{ constructionData.bucketSize };
		static constexpr key_hasher hasher{ constructionData.getSeed() };
		static constexpr size_t numGroups{ constructionData.numGroups };
		static constexpr sub_tuple_type type{ constructionData.type };
		static_assert(type != sub_tuple_type::unset, "Failed to build this tuple!");

		JSONIFIER_ALWAYS_INLINE constexpr simd_sub_tuple_full_length() noexcept = default;

		JSONIFIER_ALWAYS_INLINE bool find(value_type& value, iterator& iter, iterator& end) const noexcept {
			const char* newPtr{ (*iter) + 1 + keyStatsVal.minLength };
			memchar<'"'>(newPtr, static_cast<size_t>((*end) - ((*iter) + 1)));
			size_t length		   = static_cast<size_t>(newPtr - ((*iter) + 1));
			length				   = uniqueIndex > length ? length : uniqueIndex;
			const auto hash		   = hasher.hashKeyRt((*iter) + 1, length);
			const auto resultIndex = (hash >> 8) & (numGroups - 1) * bucketSize;
			const auto finalIndex  = (simd_internal::tzcnt(simd_internal::opCmpEq(simd_internal::gatherValue<simd_type>(static_cast<control_type>(hash)),
										  simd_internal::gatherValues<simd_type>(controlBytes.data() + resultIndex))) +
				 resultIndex);
			return checkAndParse<0, uniqueStringLengths<value_type>.size(), subTupleIndexNew, derived_type, options, value_type, iterator>(value, iter, end, indices[finalIndex]);
		}

	  protected:
		JSONIFIER_ALWAYS_INLINE static constexpr size_type tzcnt(size_type value) noexcept {
			size_type count{};
			while ((value & 1) == 0 && value != 0) {
				value >>= 1;
				++count;
			}
			return count;
		}

		JSONIFIER_ALWAYS_INLINE static constexpr size_type constMatch(const control_type* hashData, control_type hash) noexcept {
			size_type mask = 0;
			for (size_type x = 0; x < constructionData.bucketSize; ++x) {
				if (hashData[x] == hash) {
					mask |= (1ull << x);
				}
			}
			return tzcnt(mask);
		}
	};

	template<typename derived_type, const parse_options_internal<derived_type>& options, typename value_type, typename iterator, size_t subTupleIndexNew>
	struct simd_sub_tuple_unique_index : public sub_tuple_base<value_type, iterator> {
		inline static constexpr auto& constructionData{ subTupleConstructionStaticData<value_type, subTupleIndexNew> };
		using simd_type	   = typename decltype(subTupleConstructionStaticData<value_type, subTupleIndexNew>)::simd_type;
		using size_type	   = size_t;
		using control_type = uint8_t;

		JSONIFIER_ALIGN static constexpr std::array<control_type, constructionData.storageSize + 1> controlBytes{ constructionData.controlBytes };
		JSONIFIER_ALIGN static constexpr std::array<size_type, constructionData.storageSize + 1> indices{ constructionData.indices };
		static constexpr size_t uniqueIndex{ constructionData.uniqueIndex };
		static constexpr size_t storageSize{ constructionData.storageSize };
		static constexpr size_t bucketSize{ constructionData.bucketSize };
		static constexpr key_hasher hasher{ constructionData.getSeed() };
		static constexpr size_t numGroups{ constructionData.numGroups };
		static constexpr sub_tuple_type type{ constructionData.type };
		static_assert(type != sub_tuple_type::unset, "Failed to build this tuple!");

		JSONIFIER_ALWAYS_INLINE constexpr simd_sub_tuple_unique_index() noexcept = default;

		JSONIFIER_ALWAYS_INLINE bool find(value_type& value, iterator& iter, iterator& end) const noexcept {
			const auto hash		   = (hasher.getSeed() ^ (iter)[uniqueIndex + 1]);
			const auto resultIndex = (hash >> 8) & (numGroups - 1) * bucketSize;
			const auto finalIndex  = (simd_internal::tzcnt(simd_internal::opCmpEq(simd_internal::gatherValue<simd_type>(static_cast<control_type>(hash)),
										  simd_internal::gatherValues<simd_type>(controlBytes.data() + resultIndex))) +
				 resultIndex);
			return checkAndParse<0, uniqueStringLengths<value_type>.size(), subTupleIndexNew, derived_type, options, value_type, iterator>(value, iter, end, indices[finalIndex]);
		}

	  protected:
		JSONIFIER_ALWAYS_INLINE static constexpr size_type tzcnt(size_type value) noexcept {
			size_type count{};
			while ((value & 1) == 0 && value != 0) {
				value >>= 1;
				++count;
			}
			return count;
		}

		JSONIFIER_ALWAYS_INLINE static constexpr size_type constMatch(const control_type* hashData, control_type hash) noexcept {
			size_type mask = 0;
			for (size_type x = 0; x < constructionData.bucketSize; ++x) {
				if (hashData[x] == hash) {
					mask |= (1ull << x);
				}
			}
			return tzcnt(mask);
		}
	};

	template<typename derived_type, const parse_options_internal<derived_type>& options, typename value_type, jsonifier::concepts::is_double_ptr iterator, size_t subTupleIndexNew>
	struct simd_sub_tuple_unique_index<derived_type, options, value_type, iterator, subTupleIndexNew> : public sub_tuple_base<value_type, iterator> {
		inline static constexpr auto& constructionData{ subTupleConstructionStaticData<value_type, subTupleIndexNew> };
		using simd_type	   = typename decltype(subTupleConstructionStaticData<value_type, subTupleIndexNew>)::simd_type;
		using size_type	   = size_t;
		using control_type = uint8_t;

		JSONIFIER_ALIGN static constexpr std::array<control_type, constructionData.storageSize + 1> controlBytes{ constructionData.controlBytes };
		JSONIFIER_ALIGN static constexpr std::array<size_type, constructionData.storageSize + 1> indices{ constructionData.indices };
		static constexpr size_t uniqueIndex{ constructionData.uniqueIndex };
		static constexpr size_t storageSize{ constructionData.storageSize };
		static constexpr size_t bucketSize{ constructionData.bucketSize };
		static constexpr key_hasher hasher{ constructionData.getSeed() };
		static constexpr size_t numGroups{ constructionData.numGroups };
		static constexpr sub_tuple_type type{ constructionData.type };
		static_assert(type != sub_tuple_type::unset, "Failed to build this tuple!");

		JSONIFIER_ALWAYS_INLINE constexpr simd_sub_tuple_unique_index() noexcept = default;

		JSONIFIER_ALWAYS_INLINE bool find(value_type& value, iterator& iter, iterator& end) const noexcept {
			const auto hash		   = (hasher.getSeed() ^ ((*iter))[uniqueIndex + 1]);
			const auto resultIndex = (hash >> 8) & (numGroups - 1) * bucketSize;
			const auto finalIndex  = (simd_internal::tzcnt(simd_internal::opCmpEq(simd_internal::gatherValue<simd_type>(static_cast<control_type>(hash)),
										  simd_internal::gatherValues<simd_type>(controlBytes.data() + resultIndex))) +
				 resultIndex);
			return checkAndParse<0, uniqueStringLengths<value_type>.size(), subTupleIndexNew, derived_type, options, value_type, iterator>(value, iter, end, indices[finalIndex]);
		}

	  protected:
		JSONIFIER_ALWAYS_INLINE static constexpr size_type tzcnt(size_type value) noexcept {
			size_type count{};
			while ((value & 1) == 0 && value != 0) {
				value >>= 1;
				++count;
			}
			return count;
		}

		JSONIFIER_ALWAYS_INLINE static constexpr size_type constMatch(const control_type* hashData, control_type hash) noexcept {
			size_type mask = 0;
			for (size_type x = 0; x < constructionData.bucketSize; ++x) {
				if (hashData[x] == hash) {
					mask |= (1ull << x);
				}
			}
			return tzcnt(mask);
		}
	};

	template<typename derived_type, const parse_options_internal<derived_type>& options, typename value_type, typename iterator, size_t subTupleIndexNew>
	struct single_byte_sub_tuple : public sub_tuple_base<value_type, iterator> {
		inline static constexpr auto& constructionData{ subTupleConstructionStaticData<value_type, subTupleIndexNew> };
		using size_type = size_t;

		static constexpr std::array<size_type, constructionData.storageSize + 1> indices{ constructionData.indices };
		static constexpr size_t uniqueIndex{ constructionData.uniqueIndex };
		static constexpr size_t storageSize{ constructionData.storageSize };
		static constexpr key_hasher hasher{ constructionData.getSeed() };
		static constexpr sub_tuple_type type{ constructionData.type };
		static_assert(type != sub_tuple_type::unset, "Failed to build this tuple!");

		JSONIFIER_ALWAYS_INLINE constexpr single_byte_sub_tuple() noexcept = default;

		JSONIFIER_ALWAYS_INLINE bool find(value_type& value, iterator& iter, iterator& end) const noexcept {
			const auto hash		  = (hasher.getSeed() ^ iter[uniqueIndex + 1]);
			const auto finalIndex = hash & (storageSize - 1);
			return checkAndParse<0, uniqueStringLengths<value_type>.size(), subTupleIndexNew, derived_type, options, value_type, iterator>(value, iter, end, indices[finalIndex]);
		}
	};

	template<typename derived_type, const parse_options_internal<derived_type>& options, typename value_type, jsonifier::concepts::is_double_ptr iterator, size_t subTupleIndexNew>
	struct single_byte_sub_tuple<derived_type, options, value_type, iterator, subTupleIndexNew> : public sub_tuple_base<value_type, iterator> {
		inline static constexpr auto& constructionData{ subTupleConstructionStaticData<value_type, subTupleIndexNew> };
		using size_type = size_t;

		static constexpr std::array<size_type, constructionData.storageSize + 1> indices{ constructionData.indices };
		static constexpr size_t uniqueIndex{ constructionData.uniqueIndex };
		static constexpr size_t storageSize{ constructionData.storageSize };
		static constexpr key_hasher hasher{ constructionData.getSeed() };
		static constexpr sub_tuple_type type{ constructionData.type };
		static_assert(type != sub_tuple_type::unset, "Failed to build this tuple!");

		JSONIFIER_ALWAYS_INLINE constexpr single_byte_sub_tuple() noexcept = default;

		JSONIFIER_ALWAYS_INLINE bool find(value_type& value, iterator& iter, iterator& end) const noexcept {
			const auto hash		  = (hasher.getSeed() ^ (*iter)[uniqueIndex + 1]);
			const auto finalIndex = hash & (storageSize - 1);
			return checkAndParse<0, uniqueStringLengths<value_type>.size(), subTupleIndexNew, derived_type, options, value_type, iterator>(value, iter, end, indices[finalIndex]);
		}
	};

	template<typename derived_type, const parse_options_internal<derived_type>& options, typename value_type, typename iterator, size_t index>
	JSONIFIER_ALWAYS_INLINE constexpr static auto getSubTuple() noexcept {
		if constexpr (subTupleConstructionStaticData<value_type, index>.type == sub_tuple_type::single_byte) {
			return single_byte_sub_tuple<derived_type, options, value_type, iterator, index>{};
		} else if constexpr (subTupleConstructionStaticData<value_type, index>.type == sub_tuple_type::simd_unique_index) {
			return simd_sub_tuple_unique_index<derived_type, options, value_type, iterator, index>{};
		} else {
			return simd_sub_tuple_full_length<derived_type, options, value_type, iterator, index>{};
		}
	}

	template<size_t size> constexpr std::array<uint8_t, 256> generateMappings(const std::array<index_count, size>& indexCounts) {
		std::array<uint8_t, 256> arrayOfIndices{};
		for (size_t x = 0; x < size; ++x) {
			arrayOfIndices[static_cast<size_t>(indexCounts[x].value)] = static_cast<uint8_t>(x);
		}
		return arrayOfIndices;
	}

	template<typename derived_type, const parse_options_internal<derived_type>& options, typename value_type, typename iterator, size_t subTupleIndexNew>
	static constexpr auto subTuple = getSubTuple<derived_type, options, value_type, iterator, subTupleIndexNew>();

}
