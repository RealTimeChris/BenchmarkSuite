#include <BnchSwt/BenchmarkSuite.hpp>
#include<source_location>
#include <iostream>
#include <vector>
#include <array>
#include <bit>

enum class log_type {
	warn  = 0,
	error = 1,
	abort = 2,
	log	  = 3,
};

template<log_type type, typename log_stream = std::ostream> BNCH_SWT_INLINE void oiml_log(const char* error_string,
	log_stream* log_out = nullptr, std::source_location location = std::source_location::current()) {
#if defined(DEBUG)
	if constexpr (type == log_type::warn) {
		if (log_out) {
			*log_out << "Error: " << error_string << ", in File: " << location.file_name() << ", on Line: " << location.line() << std::endl;
		}
	} else if constexpr (type == log_type::error) {
		if (log_out) {
			*log_out << "Error: " << error_string << ", in File: " << location.file_name() << ", on Line: " << location.line() << std::endl;
		}
	} else if constexpr (type == log_type::abort) {
		if (log_out) {
			*log_out << "Error: " << error_string << ", in File: " << location.file_name() << ", on Line: " << location.line() << std::endl;
		}
		std::abort();
	} else if constexpr (type == log_type::log) {
		if (log_out) {
			*log_out << "Error: " << error_string << ", in File: " << location.file_name() << ", on Line: " << location.line() << std::endl;
		}
	}
#endif
}

int main() {
	oiml_log<log_type::error>("Testing", &std::cout);
	return 0;
}
