#include <jsonifier/Index.hpp>
#include <iostream>
#include <tuple>
#include <type_traits>
#include "Jsonifier.hpp"



int main() {
	static constexpr jsonifier_internal ::string_literal string{ "testing" };
	static constexpr auto newString = string.operator std::string();
	jsonifier::jsonifier_core parser{};
	//parser.parseJson(twitter_message{}, std::string{});
	parser.serializeJson(twitter_message{}, std::string{});
	return 0;
}
