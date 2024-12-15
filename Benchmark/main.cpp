#include <jsonifier/Index.hpp>

struct custom_struct_raw {
	std::string testString{ "test_string" };
	int64_t testInt{ 234234 };
	bool testBool{true};
	std::vector<std::string> testVector{ "test01", "test02", "test03", "test04" };
};

template<> struct jsonifier::core<custom_struct_raw>{
	static constexpr auto parseValue = createValue<&custom_struct_raw::testBool, &custom_struct_raw::testInt, &custom_struct_raw::testString, &custom_struct_raw::testVector>();
};

int main() {
	custom_struct_raw testData{};
	jsonifier::raw_json_data testRawData{};
	jsonifier::jsonifier_core parser{};
	std::string testString{};
	parser.serializeJson(testData, testString);
	std::cout << "CURRENT DATA: " << testString << std::endl;
	parser.parseJson(testRawData, testString);
	std::cout << "CURRENT VALUES: " << testRawData.rawJson() << std::endl;
	for (auto& [key, value]: testRawData.getObject()) {
		if (value.getType() == jsonifier::json_token_type::Array) {
			for (auto& valueNew: value.getArray()) {
				std::cout << "ValueNew: " << valueNew.rawJson() << std::endl;
			}
		} else if (value.getType() != jsonifier::json_token_type::Unset) {
			std::cout << "The value was parsed here, for Key: " << key << std::endl;
		}
		std::cout << "Key: " << key << ", Value: " << value.rawJson() << std::endl;
	}
	return 0;
}
