#pragma once

#include <oiml/legacy/oiml-legacy-common/oiml-final.hpp>
#include <mutex>

inline static std::mutex oiml_critical_section_mutex;

inline static void oiml_critical_section_start() {
	oiml_critical_section_mutex.lock();
}

inline static void oiml_critical_section_end() {
	oiml_critical_section_mutex.unlock();
}