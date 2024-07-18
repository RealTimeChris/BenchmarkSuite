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
#include <jsonifier/StringView.hpp>
#include <jsonifier/Reflection.hpp>
#include <jsonifier/Derailleur.hpp>
#include <jsonifier/Tuple.hpp>
#include <algorithm>
#include <numeric>

namespace jsonifier_internal {

	struct tuple_reference {
		size_t stringLength{};
		size_t oldIndex{};
		size_t newIndex{};
	};

	struct string_length_counts {
		size_t length{};
		size_t count{};
	};

	struct hash_record {
		jsonifier::string_view hashedString{};
		uint64_t hashedStringLength{};
		uint64_t hashValue{};
		uint64_t seed{};
	};

	template<const auto& tuple, size_t currentIndex = 0, size_t maxIndex = std::tuple_size_v<jsonifier_internal::unwrap_t<decltype(tuple)>>>
	constexpr auto collectTupleReferences(std::array<tuple_reference, maxIndex> arrayOfRefs = std::array<tuple_reference, maxIndex>{}) {
		if constexpr (currentIndex < maxIndex) {
			tuple_reference values{};
			values.oldIndex			  = currentIndex;
			values.stringLength		  = std::get<currentIndex>(tuple).view().size();
			arrayOfRefs[currentIndex] = values;
			return collectTupleReferences<tuple, currentIndex + 1>(arrayOfRefs);
		} else {
			return arrayOfRefs;
		}
	}

	template<const auto& arrayOfRefs> constexpr auto bubbleSort() {
		std::array<tuple_reference, arrayOfRefs.size()> returnValues{ arrayOfRefs };
		for (size_t i = 0; i < arrayOfRefs.size() - 1; ++i) {
			for (size_t j = 0; j < arrayOfRefs.size() - i - 1; ++j) {
				if (returnValues[j].stringLength > returnValues[j + 1].stringLength) {
					std::swap(returnValues[j], returnValues[j + 1]);
				}
			}
		}
		for (size_t i = 0; i < arrayOfRefs.size(); ++i) {
			returnValues[i].newIndex = i;
		}
		return returnValues;
	}

	template<typename value_type> constexpr auto sortTupleReferences() {
		constexpr auto& tuple		 = jsonifier::core<unwrap_t<value_type>>::parseValue.parseValue;
		constexpr auto collectedRefs = collectTupleReferences<tuple>();
		constexpr auto newRefs		 = bubbleSort<make_static<collectedRefs>::value>();
		return newRefs;
	}

	template<size_t start, size_t end, size_t... indices> constexpr auto generateCustomIndexSequence(std::index_sequence<indices...>) {
		return std::index_sequence<(start + indices)...>{};
	}

	template<size_t start, size_t end> struct custom_index_sequence_generator;

	template<size_t start, size_t end> struct custom_index_sequence_generator {
		using type = decltype(generateCustomIndexSequence<start, end>(std::make_index_sequence<end - start + 1>{}));
	};

	template<size_t start, size_t end> using custom_index_sequence = typename custom_index_sequence_generator<start, end>::type;

	template<size_t size> constexpr auto countUniqueStringLengths(const std::array<tuple_reference, size>& arrayOfRefs) {
		size_t currentLength{};
		size_t currentCount{};
		for (size_t x = 0; x < size; ++x) {
			if (arrayOfRefs[x].stringLength != currentLength) {
				currentLength = arrayOfRefs[x].stringLength;
				++currentCount;
			}
		}
		return currentCount;
	}

	template<size_t counts, size_t size> constexpr auto collectUniqueStringLengths(const std::array<tuple_reference, size>& arrayOfRefs) {
		std::array<string_length_counts, counts> returnValues{};
		size_t currentIndex	 = 0;
		size_t currentLength = 0;

		for (size_t x = 0; x < arrayOfRefs.size(); ++x) {
			auto keyLength = arrayOfRefs[x].stringLength;
			if (currentLength != keyLength && currentIndex < counts) {
				currentLength					  = keyLength;
				returnValues[currentIndex].length = keyLength;
				returnValues[currentIndex].count  = 1;
				++currentIndex;
			} else if (currentIndex - 1 < returnValues.size()) {
				++returnValues[currentIndex - 1].count;
			}
		}

		return returnValues;
	}

	template<uint64_t inputIndex, const auto& stringLengths> constexpr auto getCurrentStringLength() {
		size_t currentStartingIndex{};
		for (size_t x = 0; x < inputIndex; ++x) {
			if (inputIndex == x) {
				return stringLengths[x].length;
			}
		}
		return std::numeric_limits<size_t>::max();
	}

	template<size_t inputIndex, const auto& stringLengths> constexpr auto getCurrentStartingIndex() {
		size_t currentStartingIndex{};
		for (size_t x = 0; x < inputIndex; ++x) {
			currentStartingIndex += stringLengths[x].count;
		}
		return currentStartingIndex;
	}

	template<typename value_type> constexpr auto sortedTupleReferences{ sortTupleReferences<value_type>() };
	template<typename value_type> constexpr auto uniqueStringLengthCount{ countUniqueStringLengths(sortedTupleReferences<value_type>) };
	template<typename value_type> constexpr auto uniqueStringLengths{ collectUniqueStringLengths<uniqueStringLengthCount<value_type>>(sortedTupleReferences<value_type>) };

	template<typename value_type, size_t... indices> constexpr auto createGroupedSubTuple(std::index_sequence<indices...>) {
		constexpr auto& oldTuple  = jsonifier::concepts::coreV<value_type>;
		constexpr auto& tupleRefs = sortedTupleReferences<value_type>;
		return std::make_tuple(std::get<tupleRefs[indices].oldIndex>(oldTuple)...);
	}

	template<typename value_type, size_t startIndex, size_t count> constexpr auto createGroupedTuple() {
		return createGroupedSubTuple<value_type>(custom_index_sequence<startIndex, (startIndex + count) - 1>{});
	}

	template<typename value_type, size_t... indices> constexpr auto createNewTupleInternal(std::index_sequence<indices...>) {
		constexpr auto& stringLengths = uniqueStringLengths<value_type>;
		return std::make_tuple(createGroupedTuple<value_type, getCurrentStartingIndex<indices, stringLengths>(), stringLengths[indices].count>()...);
	}

	template<typename value_type> constexpr auto createNewTuple() {
		constexpr auto& tupleRefs = sortedTupleReferences<value_type>;
		return createNewTupleInternal<value_type>(std::make_index_sequence<countUniqueStringLengths(tupleRefs)>{});
	}

	template<typename value_type> constexpr auto finalTuple = createNewTuple<unwrap_t<value_type>>();

	template<typename value_type> using final_tuple_t = decltype(finalTuple<value_type>);

	template<typename value_type01, typename value_type02> constexpr bool contains(const value_type01* hashData, value_type02 byteToCheckFor, size_t size) {
		for (size_t x = 0; x < size; ++x) {
			if (hashData[x] == byteToCheckFor) {
				return true;
			}
		}
		return false;
	}

	template<size_t size, size_t length> constexpr size_t getMaxSizeIndex(const std::array<size_t, length>& maxSizes) {
		for (size_t x = 0; x < std::size(maxSizes); ++x) {
			if (size <= maxSizes[x]) {
				return x;
			}
		}
		return std::size(maxSizes) - 1;
	}

	template<typename value_type, size_t subTupleIndex> constexpr size_t getActualCount() {
		constexpr auto& subTuple = std::get<subTupleIndex>(finalTuple<unwrap_t<value_type>>);
		using sub_tuple_type	 = unwrap_t<decltype(subTuple)>;
		return std::tuple_size_v<sub_tuple_type>;
	}

	template<size_t length> constexpr size_t setSimdWidth() {
		return length >= 64 && bytesPerStep >= 64 ? 64 : length >= 32 && bytesPerStep >= 32 ? 32 : 16;
	}

	constexpr std::array<size_t, 5> hashTupleMaxSizes{ 16, 32, 64, 128, 256 };

	template<size_t length> struct tuple_simd {
		using type = std::conditional_t<length >= 64 && bytesPerStep >= 64, jsonifier_simd_int_512,
			std::conditional_t<length >= 32 && bytesPerStep >= 32, jsonifier_simd_int_256, jsonifier_simd_int_128>>;
	};

	template<size_t length> using tuple_simd_t = tuple_simd<length>::type;

	template<const auto& tuple, size_t index> JSONIFIER_INLINE bool compareStringFunctionNonConst(const char* string01) {
		if constexpr (index < std::tuple_size_v<jsonifier_internal::unwrap_t<decltype(tuple)>>) {
			static constexpr auto currentKey = jsonifier_internal::getKey<tuple, index>();
			if (!string01) {
				std::cout << "SOUGHT KEY: " << currentKey << std::endl;
				return false;
			}
			return jsonifier_internal::compare<currentKey.size()>(currentKey.data(), string01);
		} else {
			return false;
		}
	}

	template<const auto& tuple, size_t index> JSONIFIER_INLINE constexpr bool compareStringFunctionConst(const char* string01) {
		if constexpr (index < std::tuple_size_v<jsonifier_internal::unwrap_t<decltype(tuple)>>) {
			constexpr auto currentKey = jsonifier_internal::getKey<tuple, index>();
			return currentKey == jsonifier::string_view{ string01, currentKey.size() };
		} else {
			return false;
		}
	}

	template<const auto& tuple, size_t index> using compare_string_function_non_const_tuple_ptr = decltype(&compareStringFunctionNonConst<tuple, index>);

	template<const auto& tuple, size_t index> using compare_string_function_const_tuple_ptr = decltype(&compareStringFunctionConst<tuple, index>);

	template<const auto& tuple, size_t currentIndex> constexpr auto getCurrentConstFunction() {
		return &compareStringFunctionConst<tuple, currentIndex>;
	}

	template<const auto& tuple, size_t... indices> JSONIFIER_INLINE constexpr auto generateConstCompareStringFunctionPtrArrayInternal(std::index_sequence<indices...>) {
		return std::array<compare_string_function_non_const_tuple_ptr<tuple, 0>, sizeof...(indices)>{ { getCurrentConstFunction<tuple, indices>()... } };
	}

	template<uint64_t size, const auto& tuple> JSONIFIER_INLINE constexpr auto generateConstCompareStringFunctionPtrArray() {
		return generateConstCompareStringFunctionPtrArrayInternal<tuple>(std::make_index_sequence<size>{});
	}

	template<const auto& tuple, size_t currentIndex> constexpr auto getCurrentNonConstFunction() {
		return &compareStringFunctionNonConst<tuple, currentIndex>;
	}

	template<const auto& tuple, size_t... indices> JSONIFIER_INLINE constexpr auto generateNonConstCompareStringFunctionPtrArrayInternal(std::index_sequence<indices...>) {
		return std::array<compare_string_function_non_const_tuple_ptr<tuple, 0>, sizeof...(indices)>{ { getCurrentNonConstFunction<tuple, indices>()... } };
	}

	template<uint64_t size, const auto& tuple> JSONIFIER_INLINE constexpr auto generateNonConstCompareStringFunctionPtrArray() {
		return generateNonConstCompareStringFunctionPtrArrayInternal<tuple>(std::make_index_sequence<size>{});
	}

	enum class sub_tuple_type {
		simd		 = 0,
		minimal_char = 1,
	};

	template<typename key_type_new, typename value_type, size_t subTupleIndexNew, size_t maxSizeIndexNew> struct sub_tuple_construction_data : public key_hasher {
		std::array<jsonifier::string_view, getActualCount<value_type, subTupleIndexNew>()> pairs{};
		std::array<hash_record, getActualCount<value_type, subTupleIndexNew>()> hashRecords{};
		JSONIFIER_ALIGN std::array<uint8_t, hashTupleMaxSizes[maxSizeIndexNew]> controlBytes{};
		JSONIFIER_ALIGN std::array<size_t, hashTupleMaxSizes[maxSizeIndexNew] + 1> indices{};
		size_t stringLength{ std::numeric_limits<size_t>::max() };
		size_t maxSizeIndex{ std::numeric_limits<size_t>::max() };
		size_t subTupleIndex{ subTupleIndexNew };
		sub_tuple_type type{};
		size_t actualCount{};
		size_t storageSize{};
		size_t bucketSize{};
		size_t numGroups{};

		constexpr sub_tuple_construction_data() noexcept = default;
	};

	template<const auto& tuple, size_t maxIndex, size_t currentIndex = 0> constexpr auto getKeysInternal(std::array<jsonifier::string_view, maxIndex>& pairsNew) {
		if constexpr (currentIndex < maxIndex) {
			pairsNew[currentIndex] = getKey<tuple, currentIndex>();
			return getKeysInternal<tuple, maxIndex, currentIndex + 1>(pairsNew);
		} else {
			return pairsNew;
		}
	}

	template<const auto& tuple, size_t maxIndex> constexpr auto getKeys() {
		std::array<jsonifier::string_view, maxIndex> pairsNew{};
		return getKeysInternal<tuple, maxIndex, 0>(pairsNew);
	}

	template<typename key_type, typename value_type, size_t subTupleIndex, template<typename, typename, size_t, size_t> typename sub_tuple_construction_t>
	using sub_tuple_construction_data_variant = std::variant<sub_tuple_construction_data<key_type, value_type, subTupleIndex, 0>,
		sub_tuple_construction_data<key_type, value_type, subTupleIndex, 1>, sub_tuple_construction_data<key_type, value_type, subTupleIndex, 2>,
		sub_tuple_construction_data<key_type, value_type, subTupleIndex, 3>, sub_tuple_construction_data<key_type, value_type, subTupleIndex, 4>>;

	template<typename key_type, typename value_type, size_t subTupleIndex, size_t... indices> constexpr auto collectSimdSubTupleData(std::index_sequence<indices...>) {
		constexpr auto& finalSubTuple = std::get<subTupleIndex>(finalTuple<value_type>);
		constexpr auto tupleSize	  = std::tuple_size_v<unwrap_t<decltype(finalSubTuple)>>;
		constexpr auto keyStatsVal	  = keyStats<finalSubTuple>();
		constexpr auto pairsNew		  = std::array<jsonifier::string_view, sizeof...(indices)>{ getKey<finalSubTuple, indices>()... };
		xoshiro256 prng{};
		auto constructForGivenStringLength =
			[&](const auto maxSizeIndex, auto&& constructForGivenStringLength,
				size_t stringLength) mutable -> sub_tuple_construction_data_variant<key_type, value_type, subTupleIndex, sub_tuple_construction_data> {
			constexpr size_t bucketSize	 = setSimdWidth<hashTupleMaxSizes[maxSizeIndex]>();
			constexpr size_t storageSize = hashTupleMaxSizes[maxSizeIndex];
			constexpr size_t numGroups	 = storageSize > bucketSize ? storageSize / bucketSize : 1;
			sub_tuple_construction_data<key_type, value_type, subTupleIndex, maxSizeIndex> returnValues{};
			returnValues.actualCount   = tupleSize;
			returnValues.bucketSize	   = bucketSize;
			returnValues.maxSizeIndex  = maxSizeIndex;
			returnValues.numGroups	   = numGroups;
			returnValues.storageSize   = storageSize;
			returnValues.subTupleIndex = subTupleIndex;
			returnValues.type		   = sub_tuple_type::simd;
			returnValues.stringLength  = stringLength;
			returnValues.pairs		   = pairsNew;
			size_t bucketSizes[numGroups]{};
			bool collided{};
			for (size_t x = 0; x < 2; ++x) {
				std::fill(returnValues.hashRecords.begin(), returnValues.hashRecords.end(), hash_record{});
				std::fill(returnValues.indices.begin(), returnValues.indices.end(), returnValues.indices.size() - 1);
				std::fill(returnValues.controlBytes.begin(), returnValues.controlBytes.end(), std::numeric_limits<uint8_t>::max());
				returnValues.setSeedCt(prng());
				collided = false;
				for (size_t y = 0; y < tupleSize; ++y) {
					const auto keySize							   = returnValues.pairs[y].size() > stringLength ? stringLength : returnValues.pairs[y].size();
					const auto hash								   = returnValues.hashKeyCt(returnValues.pairs[y].data(), keySize);
					const auto groupPos							   = (hash >> 8) % numGroups;
					const auto ctrlByte							   = static_cast<uint8_t>(hash);
					const auto bucketSizeNew					   = bucketSizes[groupPos]++;
					const auto slot								   = ((groupPos * bucketSize) + bucketSizeNew);
					returnValues.hashRecords[y].hashValue		   = hash;
					returnValues.hashRecords[y].hashedString	   = returnValues.pairs[y];
					returnValues.hashRecords[y].seed			   = returnValues.operator unsigned long long();
					returnValues.hashRecords[y].hashedStringLength = keySize;

					if (bucketSizeNew >= bucketSize || contains(returnValues.controlBytes.data() + groupPos * bucketSize, ctrlByte, bucketSize)) {
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
			if (collided) {
				if constexpr (maxSizeIndex < std::size(hashTupleMaxSizes) - 1) {
					return sub_tuple_construction_data_variant<key_type, value_type, subTupleIndex, sub_tuple_construction_data>{ constructForGivenStringLength(
						std::integral_constant<size_t, maxSizeIndex + 1>{}, constructForGivenStringLength, stringLength) };
				} else if (stringLength <= keyStatsVal.maxLength) {
					return sub_tuple_construction_data_variant<key_type, value_type, subTupleIndex, sub_tuple_construction_data>{ constructForGivenStringLength(
						std::integral_constant<size_t, 0>{}, constructForGivenStringLength, stringLength + 1) };
				}
				returnValues.stringLength = std::numeric_limits<size_t>::max();
				returnValues.maxSizeIndex = std::numeric_limits<size_t>::max();
				return returnValues;
			}
			returnValues.stringLength = stringLength;
			returnValues.maxSizeIndex = maxSizeIndex;
			return returnValues;
		};

		return constructForGivenStringLength(std::integral_constant<size_t, 0>{}, constructForGivenStringLength, 1);
	}

	template<typename value_type, size_t subTupleIndex> JSONIFIER_INLINE constexpr auto generateSubTupleConstructionDataImpl() {
		constexpr auto tupleSize = std::tuple_size_v<jsonifier_internal::unwrap_t<decltype(std::get<subTupleIndex>(finalTuple<value_type>))>>;

		if constexpr (tupleSize == 0) {
			return nullptr;
		} else {
			constexpr auto newData = collectSimdSubTupleData<jsonifier::string_view, value_type, subTupleIndex>(std::make_index_sequence<tupleSize>{});
			constexpr auto index   = newData.index();
			return std::get<index>(newData);
		}
	}

	template<typename value_type, size_t subTupleIndex> JSONIFIER_INLINE constexpr auto generateSubTupleConstructionData() {
		return generateSubTupleConstructionDataImpl<value_type, subTupleIndex>();
	}

	template<typename value_type, size_t subTupleIndex> struct sub_tuple_construction_data_static {
		static constexpr auto staticData{ generateSubTupleConstructionData<value_type, subTupleIndex>() };
	};

	template<typename value_type, size_t subTupleIndexNew> struct simd_hash_tuple : public jsonifier_internal::key_hasher {
		static constexpr auto& constructionData{ sub_tuple_construction_data_static<value_type, subTupleIndexNew>::staticData };
		using simd_type	   = tuple_simd_t<constructionData.storageSize>;
		using size_type	   = size_t;
		using control_type = uint8_t;
		static constexpr auto& tuple{ std::get<subTupleIndexNew>(finalTuple<value_type>) };
		static constexpr auto subTupleIndex{ subTupleIndexNew };
		static constexpr auto storageSize{ constructionData.storageSize };
		static constexpr auto numGroups{ constructionData.numGroups };
		static constexpr auto bucketSize{ constructionData.bucketSize };
		static constexpr auto actualCount{ constructionData.actualCount };
		static constexpr auto controlBytes{ constructionData.controlBytes };
		static constexpr auto indices{ constructionData.indices };
		static constexpr auto stringLength{ constructionData.stringLength };

		static constexpr auto nonConstCompareStringFunctions{ generateNonConstCompareStringFunctionPtrArray<constructionData.storageSize + 1, tuple>() };

		static constexpr auto constCompareStringFunctions{ generateConstCompareStringFunctionPtrArray<constructionData.storageSize + 1, tuple>() };

		constexpr simd_hash_tuple() noexcept {
			setSeedCt(constructionData.operator size_t());
		}

		template<const auto& functionPtrs> JSONIFIER_INLINE constexpr auto find(const char* iter) const noexcept {
			if (!std::is_constant_evaluated()) {
				JSONIFIER_ALIGN const auto hash		   = hashKeyRt(iter, constructionData.stringLength);
				JSONIFIER_ALIGN const auto resultIndex = ((hash >> 8) % constructionData.numGroups) * constructionData.bucketSize;
				JSONIFIER_ALIGN const auto finalIndex  = (simd_internal::tzcnt(simd_internal::opCmpEq(simd_internal::gatherValue<simd_type>(static_cast<control_type>(hash)),
															  simd_internal::gatherValues<simd_type>(controlBytes.data() + resultIndex))) +
					 resultIndex);
				std::cout << "CHECKING NEW KEYS:" << std::endl;
				for (uint64_t x = 0; x < nonConstCompareStringFunctions.size(); ++x) {
					nonConstCompareStringFunctions[x](nullptr);
				}
				if (nonConstCompareStringFunctions[indices[finalIndex]](iter)) {
					return functionPtrs.data() + indices[finalIndex];
				} else {
					return functionPtrs.data() + functionPtrs.size();
				}
			} else {
				JSONIFIER_ALIGN const auto hash		   = hashKeyCt(iter, constructionData.stringLength);
				JSONIFIER_ALIGN const auto resultIndex = ((hash >> 8) % constructionData.numGroups) * constructionData.bucketSize;
				JSONIFIER_ALIGN const auto finalIndex  = (constMatch(controlBytes.data() + resultIndex, static_cast<control_type>(hash)) + resultIndex);
				if (constCompareStringFunctions[indices[finalIndex]](iter)) {
					return functionPtrs.data() + indices[finalIndex];
				} else {
					return functionPtrs.data() + functionPtrs.size();
				}
			}
		}

		JSONIFIER_INLINE constexpr auto find(const char* iter, size_type keyLength) const noexcept {
			if (!std::is_constant_evaluated()) {
				JSONIFIER_ALIGN const auto keySize	   = constructionData.stringLength;
				JSONIFIER_ALIGN const auto hash		   = hashKeyCt(iter, keySize);
				JSONIFIER_ALIGN const auto resultIndex = ((hash >> 8) % constructionData.numGroups) * constructionData.bucketSize;
				JSONIFIER_ALIGN const auto finalIndex  = (simd_internal::tzcnt(simd_internal::opCmpEq(simd_internal::gatherValue<simd_type>(static_cast<control_type>(hash)),
															  simd_internal::gatherValues<simd_type>(controlBytes.data() + resultIndex))) +
					 resultIndex);
				if (constCompareStringFunctions[indices[finalIndex]](iter)) {
					return true;
				} else {
					throw false;
				}
			} else {
				JSONIFIER_ALIGN const auto keySize	   = keyLength > constructionData.stringLength ? constructionData.stringLength : keyLength;
				JSONIFIER_ALIGN const auto hash		   = hashKeyCt(iter, keySize);
				JSONIFIER_ALIGN const auto resultIndex = ((hash >> 8) % constructionData.numGroups) * constructionData.bucketSize;
				JSONIFIER_ALIGN const auto finalIndex  = (constMatch(controlBytes.data() + resultIndex, static_cast<control_type>(hash)) + resultIndex);
				if (constCompareStringFunctions[indices[finalIndex]](iter)) {
					return true;
				} else {
					return false;
				}
			}
		}

	  protected:
		constexpr size_type tzcnt(size_type value) const {
			size_type count{};
			while ((value & 1) == 0 && value != 0) {
				value >>= 1;
				++count;
			}
			return count;
		}

		constexpr size_type constMatch(const control_type* hashData, control_type hash) const {
			size_type mask = 0;
			for (size_type x = 0; x < constructionData.bucketSize; ++x) {
				if (hashData[x] == hash) {
					mask |= (1ull << x);
				}
			}
			return tzcnt(mask);
		}
	};

	template<size_t inputIndex, const auto& stringLengths> constexpr auto getCurrentSubTupleIndex() {
		for (size_t x = 0; x < stringLengths.size(); ++x) {
			if (inputIndex == stringLengths[x].length) {
				return x;
			}
		}
		return std::numeric_limits<size_t>::max();
	}

	template<const auto& function, typename value_type, size_t stringLength, typename... arg_types> JSONIFIER_INLINE auto collectSubTuple(arg_types&&... args) {
		using tuple_type	   = unwrap_t<decltype(finalTuple<value_type>)>;
		constexpr size_t index = getCurrentSubTupleIndex<stringLength, uniqueStringLengths<value_type>>();
		if constexpr (index < std::tuple_size_v<tuple_type>) {
			return function(std::integral_constant<size_t, index>{}, std::forward<arg_types>(args)...);
		} else {
			return false;
		}
	};

	template<const auto& function, typename value_type, size_t index, typename... arg_types> using collect_sub_tuple_ptr =
		decltype(&collectSubTuple<function, value_type, index, arg_types...>);

	template<const auto& function, typename value_type, typename... arg_types, size_t... indices>
	JSONIFIER_INLINE constexpr auto generateCollectSubTuplePtrArrayInternal(std::index_sequence<indices...>) {
		return std::array<collect_sub_tuple_ptr<function, value_type, 0, arg_types...>, sizeof...(indices)>{ { &collectSubTuple<function, value_type, indices, arg_types...>... } };
	}

	template<const auto& function, typename value_type, typename... arg_types> JSONIFIER_INLINE constexpr auto generateCollectSubTuplePtrArray() {
		return generateCollectSubTuplePtrArrayInternal<function, value_type, arg_types...>(std::make_index_sequence<512>{});
	}

	template<typename value_type, size_t subTupleIndex> constexpr auto collectSubTuple() {
		if (sub_tuple_construction_data_static<value_type, subTupleIndex>::staticData.type == sub_tuple_type::simd) {
			return simd_hash_tuple<value_type, subTupleIndex>{};
		}
	}

	template<typename value_type, size_t... indices> JSONIFIER_INLINE constexpr auto collectSubTupleImpl(std::index_sequence<indices...>) {
		return std::make_tuple(collectSubTuple<value_type, indices>()...);
	}

	template<typename value_type> JSONIFIER_INLINE constexpr auto makeHashTupleInternal() {
		constexpr auto tupleSize{ std::tuple_size_v<decltype(finalTuple<value_type>)> };
		return collectSubTupleImpl<value_type>(std::make_index_sequence<tupleSize>{});
	}

	template<typename value_type> struct hash_tuple {
		using size_type = size_t;
		static constexpr auto tuple{ makeHashTupleInternal<value_type>() };

		constexpr hash_tuple() noexcept {};

		template<const auto& function, typename... arg_types> JSONIFIER_INLINE constexpr auto find(size_type stringLength, arg_types&&... args) const {
			constexpr auto getSubTuplePtrArray = generateCollectSubTuplePtrArray<function, unwrap_t<value_type>, arg_types...>();
			return getSubTuplePtrArray[stringLength](std::forward<arg_types>(args)...);
		}
	};

	template<typename value_type> JSONIFIER_INLINE constexpr auto makeHashTuple() {
		return hash_tuple<unwrap_t<value_type>>{};
	}

}