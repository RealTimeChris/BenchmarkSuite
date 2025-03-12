#pragma once
#include <oiml/legacy/oiml-legacy-common/oiml-backend-impl.hpp>
#include <oiml/legacy/oiml-legacy-common/oiml-backend.hpp>
#include <oiml/legacy/oiml-legacy-common/oiml-impl.hpp>
#include <algorithm>
#include <codecvt>
#include <cstring>
#include <filesystem>
#include <locale>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#ifdef _WIN32
	#define WIN32_LEAN_AND_MEAN
	#ifndef NOMINMAX
		#define NOMINMAX
	#endif
	#include <windows.h>
#elif defined(__APPLE__)
	#include <mach-o/dyld.h>
	#include <dlfcn.h>
#else
	#include <dlfcn.h>
	#include <unistd.h>
#endif

// Backend registry
#ifdef OIML_USE_CPU
	#include <oiml/legacy/oiml-legacy-common/oiml-cpu.hpp>
#endif

#ifdef OIML_USE_CUDA
	#include <oiml/legacy/oiml-legacy-common/oiml-cuda.hpp>
#endif

// disable C++17 deprecation warning for std::codecvt_utf8
#if defined(__clang__)
	#pragma clang diagnostic push
	#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif

OIML_INLINE std::wstring utf8_to_utf16(const std::string& str) {
	std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
	return converter.from_bytes(str);
}

OIML_INLINE std::string utf16_to_utf8(const std::wstring& str) {
	std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
	return converter.to_bytes(str);
}

#if defined(__clang__)
	#pragma clang diagnostic pop
#endif

#ifdef _WIN32

using dl_handle = std::remove_pointer_t<HMODULE>;

struct dl_handle_deleter {
	void operator()(HMODULE handle) {
		FreeLibrary(handle);
	}
};

OIML_INLINE dl_handle* dl_load_library(const std::wstring& path) {
	// suppress error dialogs for missing DLLs
	DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
	SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

	HMODULE handle = LoadLibraryW(path.c_str());

	SetErrorMode(old_mode);

	return handle;
}

OIML_INLINE void* dl_get_sym(dl_handle* handle, const char* name) {
	DWORD old_mode = SetErrorMode(SEM_FAILCRITICALERRORS);
	SetErrorMode(old_mode | SEM_FAILCRITICALERRORS);

	void* p = ( void* )GetProcAddress(handle, name);

	SetErrorMode(old_mode);

	return p;
}

#else

using dl_handle = void;

struct dl_handle_deleter {
	void operator()(void* handle) {
		dlclose(handle);
	}
};

static void* dl_load_library(const std::wstring& path) {
	dl_handle* handle = dlopen(utf16_to_utf8(path).c_str(), RTLD_NOW | RTLD_LOCAL);

	return handle;
}

static void* dl_get_sym(dl_handle* handle, const char* name) {
	return dlsym(handle, name);
}

#endif

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

struct oiml_backend_reg_entry {
	oiml_backend_reg_t reg;
	dl_handle_ptr handle;
};

struct oiml_backend_registry {
	std::vector<oiml_backend_reg_entry> backends;
	std::vector<oiml_backend_dev_t> devices;

	oiml_backend_registry() {
#ifdef OIML_USE_CUDA
		register_backend(oiml_backend_cuda_reg());
#endif
		register_backend(oiml_backend_cpu_reg());
	}

	~oiml_backend_registry() {
		// FIXME: backends cannot be safely unloaded without a function to destroy all the backend resources,
		// since backend threads may still be running and accessing resources from the dynamic library
		for (auto& entry: backends) {
			if (entry.handle) {
				entry.handle.release();// NOLINT
			}
		}
	}

	void register_backend(oiml_backend_reg_t reg, dl_handle_ptr handle = nullptr) {
		if (!reg) {
			return;
		}

#ifndef NDEBUG
		OIML_LOG_DEBUG("%s: registered backend %s (%zu devices)\n", __func__, oiml_backend_reg_name(reg), oiml_backend_reg_dev_count(reg));
#endif
		backends.push_back({ reg, std::move(handle) });
		for (size_t i = 0; i < oiml_backend_reg_dev_count(reg); i++) {
			register_device(oiml_backend_reg_dev_get(reg, i));
		}
	}

	void register_device(oiml_backend_dev_t device) {
#ifndef NDEBUG
		OIML_LOG_DEBUG("%s: registered device %s (%s)\n", __func__, oiml_backend_dev_name(device), oiml_backend_dev_description(device));
#endif
		devices.push_back(device);
	}

	oiml_backend_reg_t load_backend(const std::wstring& path, bool silent) {
		dl_handle_ptr handle{ dl_load_library(path) };
		if (!handle) {
			if (!silent) {
				OIML_LOG_ERROR("%s: failed to load %s\n", __func__, utf16_to_utf8(path).c_str());
			}
			return nullptr;
		}

		auto score_fn = ( oiml_backend_score_t )dl_get_sym(handle.get(), "oiml_backend_score");
		if (score_fn && score_fn() == 0) {
			if (!silent) {
				OIML_LOG_INFO("%s: backend %s is not supported on this system\n", __func__, utf16_to_utf8(path).c_str());
			}
			return nullptr;
		}

		auto backend_init_fn = ( oiml_backend_init_t )dl_get_sym(handle.get(), "oiml_backend_init");
		if (!backend_init_fn) {
			if (!silent) {
				OIML_LOG_ERROR("%s: failed to find oiml_backend_init in %s\n", __func__, utf16_to_utf8(path).c_str());
			}
			return nullptr;
		}

		oiml_backend_reg_t reg = backend_init_fn();
		if (!reg || reg->api_version != OIML_API_VERSION) {
			if (!silent) {
				if (!reg) {
					OIML_LOG_ERROR("%s: failed to initialize backend from %s: oiml_backend_init returned NULL\n", __func__, utf16_to_utf8(path).c_str());
				} else {
					OIML_LOG_ERROR("%s: failed to initialize backend from %s: incompatible API version (backend: %d, current: %d)\n", __func__, utf16_to_utf8(path).c_str(),
						reg->api_version, OIML_API_VERSION);
				}
			}
			return nullptr;
		}

		OIML_LOG_INFO("%s: loaded %s backend from %s\n", __func__, oiml_backend_reg_name(reg), utf16_to_utf8(path).c_str());

		register_backend(reg, std::move(handle));

		return reg;
	}

	void unload_backend(oiml_backend_reg_t reg, bool silent) {
		auto it = std::find_if(backends.begin(), backends.end(), [reg](const oiml_backend_reg_entry& entry) {
			return entry.reg == reg;
		});

		if (it == backends.end()) {
			if (!silent) {
				OIML_LOG_ERROR("%s: backend not found\n", __func__);
			}
			return;
		}

		if (!silent) {
			OIML_LOG_DEBUG("%s: unloading %s backend\n", __func__, oiml_backend_reg_name(reg));
		}

		// remove devices
		devices.erase(std::remove_if(devices.begin(), devices.end(),
						  [reg](oiml_backend_dev_t dev) {
							  return oiml_backend_dev_backend_reg(dev) == reg;
						  }),
			devices.end());

		// remove backend
		backends.erase(it);
	}
};

OIML_INLINE oiml_backend_registry& get_reg() {
	static oiml_backend_registry reg;
	return reg;
}

// Internal API
OIML_INLINE void oiml_backend_register(oiml_backend_reg_t reg) {
	get_reg().register_backend(reg);
}

OIML_INLINE void oiml_backend_device_register(oiml_backend_dev_t device) {
	get_reg().register_device(device);
}

// Backend (reg) enumeration
OIML_INLINE bool striequals(const char* a, const char* b) {
	for (; *a && *b; a++, b++) {
		if (std::tolower(*a) != std::tolower(*b)) {
			return false;
		}
	}
	return *a == *b;
}

OIML_INLINE size_t oiml_backend_reg_count() {
	return get_reg().backends.size();
}

OIML_INLINE oiml_backend_reg_t oiml_backend_reg_get(size_t index) {
	OIML_ASSERT(index < oiml_backend_reg_count());
	return get_reg().backends[index].reg;
}

OIML_INLINE oiml_backend_reg_t oiml_backend_reg_by_name(const char* name) {
	for (size_t i = 0; i < oiml_backend_reg_count(); i++) {
		oiml_backend_reg_t reg = oiml_backend_reg_get(i);
		if (striequals(oiml_backend_reg_name(reg), name)) {
			return reg;
		}
	}
	return nullptr;
}

// Device enumeration
OIML_INLINE size_t oiml_backend_dev_count() {
	return get_reg().devices.size();
}

OIML_INLINE oiml_backend_dev_t oiml_backend_dev_get(size_t index) {
	OIML_ASSERT(index < oiml_backend_dev_count());
	return get_reg().devices[index];
}

OIML_INLINE oiml_backend_dev_t oiml_backend_dev_by_name(const char* name) {
	for (size_t i = 0; i < oiml_backend_dev_count(); i++) {
		oiml_backend_dev_t dev = oiml_backend_dev_get(i);
		if (striequals(oiml_backend_dev_name(dev), name)) {
			return dev;
		}
	}
	return nullptr;
}

OIML_INLINE oiml_backend_dev_t oiml_backend_dev_by_type(enum oiml_backend_device_types type) {
	for (size_t i = 0; i < oiml_backend_dev_count(); i++) {
		oiml_backend_dev_t dev = oiml_backend_dev_get(i);
		if (oiml_backend_device_type(dev) == type) {
			return dev;
		}
	}
	return nullptr;
}

// Convenience functions
OIML_INLINE oiml_backend_t oiml_backend_init_by_name(const char* name, const char* params) {
	oiml_backend_dev_t dev = oiml_backend_dev_by_name(name);
	if (!dev) {
		return nullptr;
	}
	return oiml_backend_dev_init(dev, params);
}

OIML_INLINE oiml_backend_t oiml_backend_init_by_type(enum oiml_backend_device_types type, const char* params) {
	oiml_backend_dev_t dev = oiml_backend_dev_by_type(type);
	if (!dev) {
		return nullptr;
	}
	return oiml_backend_dev_init(dev, params);
}

OIML_INLINE oiml_backend_t oiml_backend_init_best() {
	oiml_backend_dev_t dev = oiml_backend_dev_by_type(gpu);
	if (!dev) {
		dev = oiml_backend_dev_by_type(cpu);
	}
	if (!dev) {
		return nullptr;
	}
	return oiml_backend_dev_init(dev, nullptr);
}

// Dynamic loading
OIML_INLINE oiml_backend_reg_t oiml_backend_load(const char* path) {
	return get_reg().load_backend(utf8_to_utf16(path), false);
}

OIML_INLINE void oiml_backend_unload(oiml_backend_reg_t reg) {
	get_reg().unload_backend(reg, true);
}

OIML_INLINE std::wstring get_executable_path() {
#if defined(__APPLE__)
	// get executable path
	std::vector<char> path;
	uint32_t size;
	while (true) {
		size = path.size();
		if (_NSGetExecutablePath(path.data(), &size) == 0) {
			break;
		}
		path.resize(size);
	}
	std::string base_path(path.data(), size);
	// remove executable name
	auto last_slash = base_path.find_last_of('/');
	if (last_slash != std::string::npos) {
		base_path = base_path.substr(0, last_slash);
	}
	return utf8_to_utf16(base_path + "/");
#elif defined(__linux__) || defined(__FreeBSD__)
	std::string base_path = ".";
	std::vector<char> path(1024);
	while (true) {
		// get executable path
	#if defined(__linux__)
		ssize_t len = readlink("/proc/self/exe", path.data(), path.size());
	#elif defined(__FreeBSD__)
		ssize_t len = readlink("/proc/curproc/file", path.data(), path.size());
	#endif
		if (len == -1) {
			break;
		}
		if (len < ( ssize_t )path.size()) {
			base_path = std::string(path.data(), len);
			// remove executable name
			auto last_slash = base_path.find_last_of('/');
			if (last_slash != std::string::npos) {
				base_path = base_path.substr(0, last_slash);
			}
			break;
		}
		path.resize(path.size() * 2);
	}

	return utf8_to_utf16(base_path + "/");
#elif defined(_WIN32)
	std::vector<wchar_t> path(MAX_PATH);
	DWORD len = GetModuleFileNameW(NULL, path.data(), path.size());
	if (len == 0) {
		return {};
	}
	std::wstring base_path(path.data(), len);
	// remove executable name
	auto last_slash = base_path.find_last_of('\\');
	if (last_slash != std::string::npos) {
		base_path = base_path.substr(0, last_slash);
	}
	return base_path + L"\\";
#else
	return {};
#endif
}

OIML_INLINE std::wstring backend_filename_prefix() {
#ifdef _WIN32
	return L"oiml-";
#else
	return L"liboiml-";
#endif
}

OIML_INLINE std::wstring backend_filename_suffix() {
#ifdef _WIN32
	return L".dll";
#else
	return L".so";
#endif
}

OIML_INLINE std::wstring path_separator() {
#ifdef _WIN32
	return L"\\";
#else
	return L"/";
#endif
}

OIML_INLINE oiml_backend_reg_t oiml_backend_load_best(const char* name, bool silent, const char* user_search_path) {
	// enumerate all the files that match [lib]oiml-name-*.[so|dll] in the search paths
	// TODO: search system paths
	std::wstring file_prefix = backend_filename_prefix() + utf8_to_utf16(name) + L"-";
	std::vector<std::wstring> search_paths;
	if (user_search_path == nullptr) {
		search_paths.push_back(L"." + path_separator());
		search_paths.push_back(get_executable_path());
	} else {
		search_paths.push_back(utf8_to_utf16(user_search_path) + path_separator());
	}

	int best_score = 0;
	std::wstring best_path;

	namespace fs = std::filesystem;
	for (const auto& search_path: search_paths) {
		if (!fs::exists(search_path)) {
			continue;
		}
		fs::directory_iterator dir_it(search_path, fs::directory_options::skip_permission_denied);
		for (const auto& entry: dir_it) {
			if (entry.is_regular_file()) {
				std::wstring filename = entry.path().filename().wstring();
				std::wstring ext	  = entry.path().extension().wstring();
				if (filename.find(file_prefix) == 0 && ext == backend_filename_suffix()) {
					dl_handle_ptr handle{ dl_load_library(entry.path().wstring()) };
					if (!handle && !silent) {
						OIML_LOG_ERROR("%s: failed to load %s\n", __func__, utf16_to_utf8(entry.path().wstring()).c_str());
					}
					if (handle) {
						auto score_fn = ( oiml_backend_score_t )dl_get_sym(handle.get(), "oiml_backend_score");
						if (score_fn) {
							int s = score_fn();
#ifndef NDEBUG
							OIML_LOG_DEBUG("%s: %s score: %d\n", __func__, utf16_to_utf8(entry.path().wstring()).c_str(), s);
#endif
							if (s > best_score) {
								best_score = s;
								best_path  = entry.path().wstring();
							}
						} else {
							if (!silent) {
								OIML_LOG_INFO("%s: failed to find oiml_backend_score in %s\n", __func__, utf16_to_utf8(entry.path().wstring()).c_str());
							}
						}
					}
				}
			}
		}
	}

	if (best_score == 0) {
		// try to load the base backend
		for (const auto& search_path: search_paths) {
			std::wstring path = search_path + backend_filename_prefix() + utf8_to_utf16(name) + backend_filename_suffix();
			if (fs::exists(path)) {
				return get_reg().load_backend(path, silent);
			}
		}
		return nullptr;
	}

	return get_reg().load_backend(best_path, silent);
}

OIML_INLINE void oiml_backend_load_all() {
	oiml_backend_load_all_from_path(nullptr);
}

OIML_INLINE void oiml_backend_load_all_from_path(const char* dir_path) {
#ifdef NDEBUG
	bool silent = true;
#else
	bool silent = false;
#endif

	oiml_backend_load_best("blas", silent, dir_path);
	oiml_backend_load_best("cann", silent, dir_path);
	oiml_backend_load_best("cuda", silent, dir_path);
	oiml_backend_load_best("hip", silent, dir_path);
	oiml_backend_load_best("kompute", silent, dir_path);
	oiml_backend_load_best("metal", silent, dir_path);
	oiml_backend_load_best("rpc", silent, dir_path);
	oiml_backend_load_best("sycl", silent, dir_path);
	oiml_backend_load_best("vulkan", silent, dir_path);
	oiml_backend_load_best("opencl", silent, dir_path);
	oiml_backend_load_best("musa", silent, dir_path);
	oiml_backend_load_best("cpu", silent, dir_path);
	// check the environment variable OIML_BACKEND_PATH to load an out-of-tree backend
	const char* backend_path = std::getenv("OIML_BACKEND_PATH");
	if (backend_path) {
		oiml_backend_load(backend_path);
	}
}
