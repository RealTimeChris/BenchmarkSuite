#include <iostream>

template<typename... types> struct tuple {
	constexpr tuple() noexcept = default;
};

template<typename first_type, typename... types> struct tuple<first_type, types...> {
	constexpr tuple() noexcept = default;
	template<typename first_type_new, typename... types_new> constexpr tuple(first_type_new&& argOne, types_new&&... args)
		: val{ std::forward<first_type_new>(argOne) }, restVals{ std::forward<types_new>(args)... } {};
	tuple<types...> restVals{};
	first_type val{};
};

template<typename first_type> struct tuple<first_type> {
	constexpr tuple() noexcept = default;
	template<typename first_type_new> constexpr tuple(first_type_new&& argOne) : val{ std::forward<first_type_new>(argOne) } {}
	first_type val{};
};

template<typename... types> tuple(const types&...) -> tuple<types...>;
template<typename... types> tuple(types&&...) -> tuple<types...>;
template<typename... types, typename first_type> tuple(const types&..., const first_type&) -> tuple<types..., first_type>;
template<typename... types, typename first_type> tuple(types&&..., first_type&&) -> tuple<types..., first_type>;
template<typename first_type> tuple(const first_type&) -> tuple<first_type>;
template<typename first_type> tuple(first_type&&) -> tuple<first_type>;

template<std::size_t index, typename tuple_type>
	requires(index == 0)
static decltype(auto) get(tuple_type&& t) {
	return std::move(t.val);
};

template<std::size_t index, typename tuple_type> static decltype(auto) get(tuple_type&& t) {
	return get<index - 1>(std::move(t.restVals));
};

template<std::size_t index, typename tuple_type>
	requires(index == 0)
static auto& get(const tuple_type& t) {
	return t.val;
};

template<std::size_t index, typename tuple_type> static auto& get(const tuple_type& t) {
	return get<index - 1>(t.restVals);
};

template<typename... types> struct tuple;

template<size_t index, typename tuple_type> struct tuple_element;

template<typename first_type, typename... types> struct tuple_element<0, const tuple<first_type, types...>> {
	using type = first_type;
};

template<size_t index, typename first_type, typename... types> struct tuple_element<index, const tuple<first_type, types...>> {
	static_assert(index < sizeof...(types) + 1, "Index out of bounds");
	using type = typename tuple_element<index - 1, const tuple<types...>>::type;
};

template<typename first_type, typename... types> struct tuple_element<0, tuple<first_type, types...>> {
	using type = first_type;
};

template<size_t index, typename first_type, typename... types> struct tuple_element<index, tuple<first_type, types...>> {
	static_assert(index < sizeof...(types) + 1, "Index out of bounds");
	using type = typename tuple_element<index - 1, tuple<types...>>::type;
};

template<size_t index, typename...types>using tuple_element_t = tuple_element<index,types...>::type;

template<typename value_type> struct tuple_size;

template<typename... types> struct tuple_size<tuple<types...>> {
	static constexpr std::size_t value = sizeof...(types);
};

template<typename... types> constexpr decltype(auto) makeTuple(types&&... args) {
	return tuple{ std::forward<types>(args)... };
}

template<typename value_type> inline constexpr std::size_t tuple_size_v = tuple_size<value_type>::value;

template<typename T> struct type_printer {
	static void print() {
		std::cout << "Type: " << typeid(T).name() << std::endl;
	}
};

template<typename tuple1, typename tuple2> struct concat_tuples;

template<typename... types1, typename... types2> struct concat_tuples<tuple<types1...>, tuple<types2...>> {
	using type = tuple<types1..., types2...>;
};

template<typename type01, typename type02> using concat_tuples_t = typename concat_tuples<type01, type02>::type;

template<typename tuple> struct print_tuple_types;

template<typename... types> struct print_tuple_types<tuple<types...>> {
	static void print() {
		(type_printer<types>::print(), ...);
	}
};

int main() {
	// Create a source tuple with some values
	tuple<int, double, char> sourceTuple(42, 3.14, 'A');

	// Extract and insert values into a new tuple
	//auto newTuple = extract_and_insert(sourceTuple);

	std::cout << "New Tuple Types:" << std::endl;
	//print_tuple_types<decltype(newTuple)>::print();// Print types of the new tuple
	using tuple1 = tuple<int, double>;
	using tuple2 = tuple<char, float>;

	using concatenated_tuple = concat_tuples_t<tuple1, tuple2>;
	std::cout << "Concatenated Tuple Type: " << typeid(concatenated_tuple).name() << std::endl;

	std::cout << "Concatenated Tuple types:" << std::endl;
	print_tuple_types<concatenated_tuple>::print();

	// Create tuples to concatenate
	tuple myTuple01(42, 3.14, "Hello");
	tuple myTuple02(43, 2.71, "World");

	// Concatenate tuples
	//auto newTuple = tupleCat(myTuple01, myTuple02);
	//std::cout << "Concatenated Tuple Type: " << typeid(decltype(newTuple)).name() << std::endl;
	std::tuple myTuple03(42, 3.14, "Hello");

	auto mytuple04 = makeTuple(std::string_view{ "test" }, int32_t{});

	using typestd = std::tuple_element_t<0, decltype(myTuple03)>;
	using type	  = tuple_element_t<0, decltype(myTuple02)>;
	using type01  = tuple_element_t<1, decltype(myTuple02)>;
	//using type02  = tuple_element_t<2, decltype(myTuple02)>;

	std::cout << "Tuple Size: " << tuple_size_v<decltype(myTuple01)> << std::endl;
	std::cout << "Type at index 0: " << typeid(typestd).name() << std::endl;
	std::cout << "Type at index 0: " << typeid(type).name() << std::endl;
	std::cout << "Type at index 1: " << typeid(type01).name() << std::endl;
	//std::cout << "Type at index 2: " << typeid(type02).name() << std::endl;
	// Accessing the first element (index 0)
	std::cout << "Value at index 0: " << get<0>(myTuple01) << std::endl;

	// Accessing the second element (index 1)
	std::cout << "Value at index 1: " << get<1>(myTuple01) << std::endl;

	// Accessing the third element (index 2)
	std::cout << "Value at index 2: " << get<2>(myTuple01) << std::endl;
	return 0;
}
