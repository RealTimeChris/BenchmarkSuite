#pragma once

#include <jsonifier/Config.hpp>
#include "Hash.hpp"

namespace jsonifier {

	struct rt_key_hasher;

	template<typename key_type_new, typename mapped_type_new, typename hasher = rt_key_hasher, typename key_equals = std::equal_to<key_type_new>,
		typename allocator = jsonifier_internal::alloc_wrapper<std::pair<key_type_new, mapped_type_new>>>
	class unordered_map;

	template<typename map_iterator, typename key_type, typename value_type>
	concept map_container_iterator_t = std::same_as<typename unordered_map<key_type, value_type>::iterator, std::decay_t<map_iterator>>;

	constexpr std::array<uint64_t, 135> prns{ { 1033321092324544984ull, 2666561049963377653ull, 3901177690447069239ull, 4218182233242110882ull, 5911765535454950103ull,
		6788651254494793497ull, 7100864855074445223ull, 8121427956336305945ull, 9038010914689427860ull, 14840306302415334885ull, 2861875790078914964ull, 3162274379479658823ull,
		4716213344225307449ull, 540950270129450019ull, 6138393194460717092ull, 7344427311844191385ull, 8475133706542525636ull, 9707373313909664576ull, 13125261184447140558ull,
		2935828130652229499ull, 3352961464321085856ull, 4654333323360932970ull, 5071886467123008198ull, 6337413869067417456ull, 7068363609472928302ull, 8706829452892616150ull,
		9383326841165471636ull, 16102866716245881820ull, 2811014628691939071ull, 3268225168635854144ull, 4143407405368768949ull, 5597712091605167573ull, 6100647393685909969ull,
		7810560643861675820ull, 8193567265249468576ull, 9274898615585908930ull, 1186958974127274710ull, 246203706832441443ull, 3668316000003001120ull, 4918933721064431133ull,
		5627034507966762943ull, 6439181573813589114ull, 7007274452014357082ull, 8727797399712164036ull, 9543719554692367837ull, 17847026727775507706ull, 2455135551339952688ull,
		3249111793759010315ull, 4777639692643446085ull, 5509261895474102266ull, 6044529605818700585ull, 7171416927005376731ull, 8039758273674712696ull, 9590025961231307183ull,
		11664492738409977550ull, 2284380607188886295ull, 3813446608469001272ull, 4331825983983119949ull, 5837226587704917004ull, 6635783511790253542ull, 705765947185415012ull,
		8161069307156878177ull, 9569010482089894937ull, 10396430003774290631ull, 2470420036324932164ull, 3824517089721070566ull, 4514057289782578484ull, 5632633334704453746ull,
		6925174598726443993ull, 7516137935779736019ull, 824571755531910559ull, 9361703638678697870ull, 13796235445108347584ull, 2481146946454909733ull, 3823008181066037442ull,
		4754601782272553608ull, 588747081408207180ull, 6322894155156329605ull, 7737051281621502357ull, 8728044800884920985ull, 9923282424466541678ull, 18161647536849824815ull,
		2607456623799892498ull, 3651449820230355939ull, 4624058760756378704ull, 564341449426358799ull, 6732980169420780216ull, 7082954320121844082ull, 8326246156222992233ull,
		9417642353078551282ull, 17539099562248686315ull, 226802774388233589ull, 3258991441457498839ull, 4386515027804287469ull, 5492870100834679754ull, 6249105792560430415ull,
		7289628920991893817ull, 8241072433031030544ull, 9727644451441173921ull, 13586305903807621608ull, 200516020872926547ull, 3730616597292952024ull, 4256645544584917949ull,
		5613544969337462956ull, 6264647669092059269ull, 7960331042009250727ull, 8582958636583556090ull, 9171663339272942914ull, 1619903645133495717ull, 2840616349619109328ull,
		3166096472286566799ull, 4494229804275778550ull, 5497884137148100871ull, 6572487097017879223ull, 738706937289335047ull, 8122825727823277447ull, 9131968469543030694ull,
		14054393997887833558ull, 2874030832643593377ull, 3673904271267944876ull, 4542812785880908260ull, 5621946313585701459ull, 6176632143181793702ull, 7512972502278041818ull,
		8724494295438506783ull, 9277533619161797917ull, 13495127262014153477ull, 2883303557104387784ull, 3039599040070277986ull, 4196273005435491662ull, 5417879022829474871ull,
		6476778602757520149ull, 7959620869796075525ull, 8518936512742009562ull, 9635246566869230345ull } };

	struct rt_key_hasher {
		constexpr rt_key_hasher() noexcept : seed{ prns[0] } {};

		uint64_t seed{};

		constexpr void updateSeed(uint64_t seedNew) noexcept {
			seed = seedNew;
		}

		JSONIFIER_ALWAYS_INLINE constexpr uint64_t bitmix(uint64_t h) const noexcept {
			h *= seed;
			return h ^ std::rotr(h, 49);
		};

		template<size_t size> JSONIFIER_ALWAYS_INLINE uint64_t operator()(const char (&stringVal)[size]) const {
			uint64_t seed64{};
			auto length{ size };
			auto value = stringVal;
			while (length >= 8) {
				std::memcpy(&returnValue64, value, 8);
				seed64 ^= returnValue64;
				value += 8;
				length -= 8;
			}

			if (length >= 4) {
				std::memcpy(&returnValue32, value, 4);
				seed64 ^= returnValue32;
				value += 4;
				length -= 4;
			}

			if (length >= 2) {
				std::memcpy(&returnValue16, value, 2);
				seed64 ^= returnValue16;
				value += 2;
				length -= 2;
			}

			if (length == 1) {
				seed64 ^= value[0];
			}
			return bitmix(seed64);
		}

		template<jsonifier::concepts::string_t value_type_new> JSONIFIER_ALWAYS_INLINE uint64_t operator()(value_type_new&& valueNew) const {
			uint64_t seed64{};
			auto length{ valueNew.size() };
			auto value = valueNew.data();
			while (length >= 8) {
				std::memcpy(&returnValue64, value, 8);
				seed64 ^= returnValue64;
				value += 8;
				length -= 8;
			}

			if (length >= 4) {
				std::memcpy(&returnValue32, value, 4);
				seed64 ^= returnValue32;
				value += 4;
				length -= 4;
			}

			if (length >= 2) {
				std::memcpy(&returnValue16, value, 2);
				seed64 ^= returnValue16;
				value += 2;
				length -= 2;
			}

			if (length == 1) {
				seed64 ^= value[0];
			}
			return bitmix(seed64);
		}

		/**
		 * @brief Hashes a key at runtime.
		 *
		 * @param value The value to be hashed.
		 * @param length The length of the value.
		 * @return The hashed value.
		 */
		JSONIFIER_ALWAYS_INLINE uint64_t hashKeyRt(const char* value, uint64_t length) const noexcept {
			uint64_t seed64{};
			while (length >= 8) {
				std::memcpy(&returnValue64, value, 8);
				seed64 ^= returnValue64;
				value += 8;
				length -= 8;
			}

			if (length >= 4) {
				std::memcpy(&returnValue32, value, 4);
				seed64 ^= returnValue32;
				value += 4;
				length -= 4;
			}

			if (length >= 2) {
				std::memcpy(&returnValue16, value, 2);
				seed64 ^= returnValue16;
				value += 2;
				length -= 2;
			}

			if (length == 1) {
				seed64 ^= *value;
			}
			return bitmix(seed64);
		}

	  protected:
		mutable uint64_t returnValue64{};
		mutable uint32_t returnValue32{};
		mutable uint16_t returnValue16{};
	};

	template<typename value_type> struct key_accessor {};

	template<jsonifier::concepts::pair_t value_type> struct key_accessor<value_type> {
		using key_type = typename value_type::first_type;
		template<jsonifier::concepts::pair_t value_type_new> JSONIFIER_ALWAYS_INLINE const key_type& operator()(value_type_new&& value) const {
			return std ::forward<value_type_new>(value).first;
		}
	};

	enum class control_byte : int8_t {
		empty	 = -128,
		deleted	 = -2,
		sentinel = -1,
	};

	template<size_t shift = 0, typename value_type = uint16_t> struct bit_mask {
		value_type bits{};

		JSONIFIER_ALWAYS_INLINE bit_mask() noexcept = default;

		JSONIFIER_ALWAYS_INLINE bit_mask(value_type valNew) : bits{ valNew } {};

		JSONIFIER_ALWAYS_INLINE size_t lowestBitSet() const {
			return simd_internal::tzcnt(bits) >> shift;
		}

		JSONIFIER_ALWAYS_INLINE uint16_t operator*() noexcept {
			return simd_internal::tzcnt(bits) >> shift;
		}

		JSONIFIER_ALWAYS_INLINE bit_mask& operator++() noexcept {
			this->bits = blsr(this->bits);
			return *this;
		}

		JSONIFIER_ALWAYS_INLINE bit_mask operator++(int32_t) noexcept {
			bit_mask temp = *this;
			++*this;
			return temp;
		}

		JSONIFIER_ALWAYS_INLINE operator bool() const noexcept {
			return bits != 0;
		}

		JSONIFIER_ALWAYS_INLINE bool operator==(const bit_mask& other) const noexcept {
			return bits == 0;
		}

		JSONIFIER_ALWAYS_INLINE auto begin() const {
			return bit_mask{ bits };
		}

		JSONIFIER_ALWAYS_INLINE auto end() const {
			return bit_mask{};
		}

		JSONIFIER_ALWAYS_INLINE bool isEmpty() const noexcept {
			return bits == 0;
		}
	};

	struct byte_group {
#if JSONIFIER_CHECK_FOR_INSTRUCTION(JSONIFIER_NEON)
		uint8x8_t ctrl;

		JSONIFIER_ALWAYS_INLINE byte_group(int8_t* ctrlNew) : ctrl{ vld1_s8(ctrlNew) } {};
#else
		jsonifier_simd_int_128 ctrl;

		JSONIFIER_ALWAYS_INLINE byte_group(int8_t* ctrlNew) : ctrl{ simd_internal::gatherValues<jsonifier_simd_int_128>(ctrlNew) } {};
#endif

		JSONIFIER_ALWAYS_INLINE auto maskEmpty() const {
#if JSONIFIER_CHECK_FOR_INSTRUCTION(JSONIFIER_NEON)
			uint64_t mask = vget_lane_u64(vreinterpret_u64_u8(vceq_s8(vdup_n_s8(static_cast<int8_t>(control_byte::empty)), vreinterpret_s8_u8(ctrl))), 0);
			return bit_mask<3, uint64_t>{ mask };
#else
			return bit_mask{ static_cast<uint16_t>(_mm_movemask_epi8(_mm_sign_epi8(ctrl, ctrl))) };
#endif
		}
		
		JSONIFIER_ALWAYS_INLINE auto match(int8_t hash) {
#if JSONIFIER_CHECK_FOR_INSTRUCTION(JSONIFIER_NEON)
			uint8x8_t dup			= vdup_n_u8(hash);
			auto mask				= vceq_u8(ctrl, dup);
			constexpr uint64_t msbs = 0x8080808080808080ULL;
			return bit_mask<3, uint64_t>{ vget_lane_u64(vreinterpret_u64_u8(mask), 0) & msbs };
#else
			auto bytesToCheckFor{ _mm_set1_epi8(hash) };
			return bit_mask{ static_cast<uint16_t>(_mm_movemask_epi8(_mm_cmpeq_epi8(bytesToCheckFor, ctrl))) };
#endif
		}
	};

	JSONIFIER_ALWAYS_INLINE bool isEmpty(int8_t c) {
		return static_cast<control_byte>(c) == control_byte::empty;
	}

	JSONIFIER_ALWAYS_INLINE bool isFull(int8_t c) {
		return static_cast<std::underlying_type_t<control_byte>>(c) >= 0;
	}

	JSONIFIER_ALWAYS_INLINE bool isDeleted(int8_t c) {
		return static_cast<control_byte>(c) == control_byte::deleted;
	}

	JSONIFIER_ALWAYS_INLINE bool isEmptyOrDeleted(int8_t c) {
		return static_cast<control_byte>(c) < control_byte::sentinel;
	}

	template<typename value_type_internal_new> class hash_map_iterator {
	  public:
		using iterator_category	  = std::forward_iterator_tag;
		using value_type_internal = value_type_internal_new;
		using value_type		  = typename value_type_internal::value_type;
		using reference			  = value_type&;
		using pointer			  = value_type*;
		using pointer_internal	  = value_type_internal*;
		using size_type			  = uint64_t;
		using ctrl_byte_type	  = int8_t;

		JSONIFIER_INLINE hash_map_iterator() = default;

		JSONIFIER_INLINE hash_map_iterator(pointer valueNew, ctrl_byte_type* ctrlBytesNew, ctrl_byte_type* ctrlBytesEndNew)
			: values{ valueNew }, ctrlBytes{ ctrlBytesNew }, ctrlBytesEnd{ ctrlBytesEndNew } {};

		JSONIFIER_INLINE hash_map_iterator& operator++() {
			skipEmptySlots();
			return *this;
		}

		JSONIFIER_INLINE hash_map_iterator& operator+(size_type amountToAdd) {
			for (size_type x = 0; x < amountToAdd; ++x) {
				skipEmptySlots();
			}
			return *this;
		}

		JSONIFIER_INLINE pointer getRawPtr() {
			return values;
		}

		JSONIFIER_INLINE bool operator==(const hash_map_iterator&) const {
			return !values || ctrlBytes == ctrlBytesEnd;
		}

		JSONIFIER_INLINE pointer operator->() {
			return values;
		}

		JSONIFIER_INLINE reference operator*() {
			return *values;
		}

	  protected:
		ctrl_byte_type* ctrlBytesEnd{};
		ctrl_byte_type* ctrlBytes{};
		pointer values{};

		void skipEmptySlots() {
			if (ctrlBytes < ctrlBytesEnd) {
				++ctrlBytes;
				++values;
				while (isEmpty(*ctrlBytes) && ctrlBytes < ctrlBytesEnd) {
					++ctrlBytes;
					++values;
				};
			}
		}
	};

	template<typename key_type_new, typename mapped_type_new, typename hasher = rt_key_hasher, typename key_equals = std::equal_to<key_type_new>,
		typename allocator = jsonifier_internal::alloc_wrapper<std::pair<key_type_new, mapped_type_new>>>
	struct hash_table : protected jsonifier_internal::alloc_wrapper<int8_t>, protected allocator, protected key_equals, protected hasher {
#if JSONIFIER_CHECK_FOR_INSTRUCTION(JSONIFIER_NEON)
		static constexpr size_t probeDistance{ 8 };
#else 
		static constexpr size_t probeDistance{ 16 };
#endif
		using mapped_type					= mapped_type_new;
		using key_type						= key_type_new;
		using value_type					= std::pair<key_type, mapped_type>;
		using allocator_type				= jsonifier_internal::alloc_wrapper<value_type>;
		using object_allocator_traits		= std::allocator_traits<allocator_type>;
		using control_byte_allocator_traits = std::allocator_traits<jsonifier_internal::alloc_wrapper<int8_t>>;
		using size_type						= uint64_t;
		using difference_type				= int64_t;
		using pointer						= typename object_allocator_traits::pointer;
		using const_pointer					= typename object_allocator_traits::const_pointer;
		using reference						= value_type&;
		using const_reference				= const value_type&;
		using iterator						= hash_map_iterator<hash_table>;
		using const_iterator				= hash_map_iterator<const hash_table>;
		using key_compare					= key_equals;
		using key_hasher_new				= hasher;

		friend iterator;
		friend const_iterator;

		JSONIFIER_ALWAYS_INLINE hash_table() {
			resize(128);
		};

		JSONIFIER_ALWAYS_INLINE const_iterator begin() const {
			for (size_type x{ 0 }; x < capacityVal; ++x) {
				if (!isEmpty(sentinelVector[x])) {
					return { data + x, sentinelVector.data() + x, sentinelVector.data() + capacityVal };
				}
			}
			return end();
		}

		JSONIFIER_ALWAYS_INLINE size_t H1(size_t hash) {
			return hash >> 7;
		}

		JSONIFIER_ALWAYS_INLINE int8_t H2(size_t hash) {
			return hash & 0x7F;
		}

		JSONIFIER_ALWAYS_INLINE const_iterator end() const {
			return { data + capacityVal, sentinelVector.data() + capacityVal, sentinelVector.data() + capacityVal };
		}

		JSONIFIER_ALWAYS_INLINE iterator begin() {
			for (size_type x{ 0 }; x < capacityVal; ++x) {
				if (!isEmpty(sentinelVector[x])) {
					return { data + x, sentinelVector.data() + x, sentinelVector.data() + capacityVal };
				}
			}
			return end();
		}

		JSONIFIER_ALWAYS_INLINE iterator end() {
			return { data + capacityVal, sentinelVector.data() + capacityVal, sentinelVector.data() + capacityVal };
		}

		template<typename key_type_newer> JSONIFIER_ALWAYS_INLINE bool contains(key_type_newer&& key) const {
			if (sizeVal > 0) {
				auto hash = indexForHash(key);
				for (size_type x{}; x < probeDistance; ++x, ++hash) {
					if (!isEmpty(sentinelVector[hash]) && key_compare()(data[hash].first, key)) {
						return true;
					}
				}
			}
			return false;
		}

		template<map_container_iterator_t<key_type, mapped_type> map_iterator> JSONIFIER_ALWAYS_INLINE iterator erase(map_iterator&& iter) {
			if (sizeVal > 0) {
				auto hash = static_cast<size_type>(iter.getRawPtr() - data);
				for (size_type x{}; x < probeDistance; ++x, ++hash) {
					if (sentinelVector[hash] > 0 && key_compare()(data[hash].first, iter.operator*().first)) {
						object_allocator_traits::destroy(*this, data + hash);
						sentinelVector[hash] = static_cast<int8_t>(control_byte::deleted);
						--sizeVal;
						return { this, ++hash };
					}
				}
			}
			return end();
		}

		template<typename key_type_newer> JSONIFIER_ALWAYS_INLINE iterator erase(key_type_newer&& key) {
			if (sizeVal > 0) {
				auto hash = indexForHash(key);
				for (size_type x{}; x < probeDistance; ++x, ++hash) {
					if (sentinelVector[hash] > 0 && key_compare()(data[hash].first, key)) {
						object_allocator_traits::destroy(*this, data + hash);
						sentinelVector[hash] = static_cast<int8_t>(control_byte::deleted);
						--sizeVal;
						return { this, ++hash };
					}
				}
			}
			return end();
		}

		template<typename... Args> JSONIFIER_ALWAYS_INLINE iterator emplace(Args&&... value) {
			return emplaceInternal(std::forward<Args>(value)...);
		}

		template<typename key_type_newer> JSONIFIER_ALWAYS_INLINE const mapped_type& at(key_type_newer&& key) const {
			auto iter = find(std::forward<key_type_newer>(key));
			if (iter == end()) {
				throw std::runtime_error{ "Sorry, but an object by that key doesn't exist in this map." };
			}
			return iter->second;
		}

		template<typename key_type_newer> JSONIFIER_ALWAYS_INLINE mapped_type& at(key_type_newer&& key) {
			auto iter = find(std::forward<key_type_newer>(key));
			if (iter == end()) {
				throw std::runtime_error{ "Sorry, but an object by that key doesn't exist in this map." };
			}
			return iter->second;
		}

		template<typename key_type_newer> JSONIFIER_ALWAYS_INLINE const_iterator find(key_type_newer&& key) const {
			if (sizeVal > 0) {
				size_t hash;
				if constexpr (std::is_same_v<key_type, std::remove_cvref_t<key_type_newer>>) {
					hash = key_hasher_new()(key);
				} else if (std::is_convertible_v<std::remove_cvref_t<key_type_newer>, key_type>) {
					hash = key_hasher_new()(static_cast<key_type>(key));
				}
				size_type group = H1(hash) % groupCount;
				while (true) {
					byte_group g{ sentinelVector.data() + (group * probeDistance) };
					auto bitMask = g.match(H2(hash));
					for (int32_t i: bitMask) {
						if (key == data[group * probeDistance + i].first) {
							return iterator{ data + ((group * probeDistance) + i), sentinelVector.data() + ((group * probeDistance) + i), sentinelVector.data() + capacityVal };
						}
					}
					if (g.maskEmpty()) {
						return end();
					} 
					group = (group + 1) % groupCount;
				}
			}
			return end();
		}

		template<typename key_type_newer> JSONIFIER_ALWAYS_INLINE iterator find(key_type_newer&& key) {
			if (sizeVal > 0) {
				size_t hash;
				if constexpr (std::is_same_v<key_type, std::remove_cvref_t<key_type_newer>>) {
					hash = key_hasher_new()(key);
				} else if (std::is_convertible_v<std::remove_cvref_t<key_type_newer>, key_type>) {
					hash = key_hasher_new()(static_cast<key_type>(key));
				}
				size_type group = H1(hash) % groupCount;
				while (true) {
					byte_group g{ sentinelVector.data() + (group * probeDistance) };
					auto bitMask = g.match(H2(hash));
					for (int32_t i: bitMask) {
						if (key == data[group * probeDistance + i].first) {
							return iterator{ data + ((group * probeDistance) + i), sentinelVector.data() + ((group * probeDistance) + i), sentinelVector.data() + capacityVal };
						}
					}
					if (g.maskEmpty()) {
						return end();
					} 
					group = (group + 1) % groupCount;
				}
			}
			return end();
		}

		JSONIFIER_ALWAYS_INLINE void clear() {
			for (size_type x = 0; x < sentinelVector.size(); ++x) {
				if (!isEmpty(sentinelVector[x])) {
					object_allocator_traits::destroy(*this, data + x);
					sentinelVector[x] = static_cast<int8_t>(control_byte::empty);
				}
			}
			sizeVal = 0;
		}

		JSONIFIER_ALWAYS_INLINE bool full() const {
			return static_cast<float>(sizeVal) >= static_cast<float>(capacityVal) * 0.90f;
		}

		template<typename mask_type> JSONIFIER_ALWAYS_INLINE auto getInsertionOffset(mask_type mask) {
			return mask.lowestBitSet();
		}

		template<typename key_type_newer, typename... mapped_type_newer> JSONIFIER_INLINE iterator emplaceInternal(key_type_newer&& key, mapped_type_newer&&... value) {
			if (full() || capacityVal == 0) {
				resize((capacityVal + 1) * 8);
			}
			size_t hash;
			if constexpr (std::is_same_v<key_type, std::remove_cvref_t<key_type_newer>>) {
				hash = key_hasher_new()(key);
			} else if constexpr (std::is_convertible_v<std::remove_cvref_t<key_type_newer>, key_type>) {
				hash = key_hasher_new()(static_cast<key_type>(key));
			}
			size_type group = (H1(hash) % groupCount) * probeDistance;
			//std::cout << "HASH VALUE: " << H1(hash) << std::endl;
			//byte_group g{ sentinelVector.data() + (group) };
			for (size_type x{}; x < probeDistance; ++x, ++group) {
				if (isEmpty(sentinelVector[group])) {
					//std::cout << "WE ARE EMPTY: " << key << std::endl;
					if constexpr ((( !std::is_void_v<mapped_type_newer> ) || ...)) {
						new (std::addressof(data[group])) value_type{ std::make_pair(std::forward<key_type_newer>(key), std::forward<mapped_type_newer>(value)...) };
					} else {
						new (std::addressof(data[group])) value_type{ std::make_pair(std::forward<key_type_newer>(key), mapped_type{}) };
					}
					sentinelVector[group] = hash & 0x7F;
					sizeVal++;
					return { data + group, sentinelVector.data() + group, sentinelVector.data() + capacityVal };
				} else if (!isEmpty(sentinelVector[group]) && key_compare()(data[group].first, key)) {
					//std::cout << "WE ARE NOT EMPTY-AND EQUAL: " << key << std::endl;
					if constexpr ((( !std::is_void_v<mapped_type_newer> ) || ...)) {
						if constexpr (std::is_same_v<mapped_type, mapped_type_newer...>) {
							data[group].second = mapped_type{ std::forward<mapped_type_newer>(value)... };
						} else if constexpr (std::is_convertible_v<mapped_type_newer..., mapped_type>) {
							data[group].second = mapped_type{ static_cast<mapped_type>(std::forward<mapped_type_newer...>(value...)) };
						}
					}
					return { data + group, sentinelVector.data() + group, sentinelVector.data() + capacityVal };
				}
			}
			resize((capacityVal + 1) * 8);
			return emplaceInternal(std::forward<key_type_newer>(key), std::forward<mapped_type_newer>(value)...);
		}

		JSONIFIER_ALWAYS_INLINE void reserve(size_type capacityNew) {
			resize(capacityNew);
		}

		JSONIFIER_ALWAYS_INLINE void resize(size_type capacityNew) {
			auto newSize = jsonifier_internal::roundUpToMultiple<probeDistance>(capacityNew);
			if (newSize > capacityVal) {
				jsonifier::vector<int8_t> oldSentinelVector = std::move(sentinelVector);
				auto oldCapacity							= capacityVal;
				auto oldSize								= sizeVal;
				auto oldPtr									= data;
				sizeVal										= 0;
				groupCount									= newSize / probeDistance;
				data										= object_allocator_traits::allocate(*this, newSize);
				sentinelVector.resize(newSize);
				std::memset(sentinelVector.data(), static_cast<int32_t>(control_byte::empty), newSize);
				capacityVal = newSize;
				for (size_type x = 0, y = 0; y < oldSize && x < oldCapacity; ++x) {
					if (!isEmpty(oldSentinelVector[x])) {
						++y;
						emplaceInternal(std::move(oldPtr[x].first), std::move(oldPtr[x].second));
					}
				}
				if (oldPtr && oldCapacity) {
					object_allocator_traits::deallocate(*this, oldPtr, oldCapacity);
				}
			}
		}

		JSONIFIER_ALWAYS_INLINE void reset() {
			if (data && sizeVal > 0) {
				for (uint64_t x = 0; x < sentinelVector.size(); ++x) {
					if (sentinelVector.at(x) > 0) {
						object_allocator_traits::destroy(*this, data + x);
						sentinelVector.at(x) = 0;
					}
				}
				object_allocator_traits::deallocate(*this, data, capacityVal);
				sentinelVector.clear();
				sizeVal		= 0;
				capacityVal = 0;
				data		= nullptr;
			}
		}

	  protected:
		jsonifier::vector<int8_t> sentinelVector{};
		size_type capacityVal{};
		size_type groupCount{};
		size_type sizeVal{};
		value_type* data{};
	};
}
