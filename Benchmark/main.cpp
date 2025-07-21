#include <BnchSwt/BenchmarkSuite.hpp>
#include <stdexcept>
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <random>
#include <unordered_map>
#include <memory>
#include <cstring>

const char* const BASE_FILE_PATH = "C:\\users\\chris\\Desktop\\test_file_";

static constexpr uint64_t total_iterations{ 100 };
static constexpr uint64_t measured_iterations{ 10 };

// 🔥 GLOBAL VERIFICATION DATA STORAGE! 🔥
struct verify_data_storage {
	std::unordered_map<uint64_t, std::unique_ptr<char[]>> data_map;

	void store_data(uint64_t size, char* data) {
		auto buffer = std::make_unique<char[]>(size);
		std::memcpy(buffer.get(), data, size);
		data_map[size] = std::move(buffer);
		std::cout << "💾 STORED " << size << " BYTES OF VERIFICATION DATA!" << std::endl;
	}

	char* get_data(uint64_t size) {
		auto it = data_map.find(size);
		if (it != data_map.end()) {
			return it->second.get();
		}
		return nullptr;
	}

	bool verify_data(uint64_t size, const char* read_data) {
		char* original = get_data(size);
		if (!original) {
			std::cerr << "❌ NO ORIGINAL DATA FOUND FOR SIZE " << size << "!" << std::endl;
			return false;
		}

		if (std::memcmp(original, read_data, size) != 0) {
			std::cerr << "❌ DATA MISMATCH FOR SIZE " << size << "!" << std::endl;
			return false;
		}

		std::cout << "✅ DATA VERIFIED FOR SIZE " << size << " BYTES!" << std::endl;
		return true;
	}
};

inline verify_data_storage global_verify_data;

template<uint64_t FILE_SIZE> std::string get_file_path() {
	std::string path = BASE_FILE_PATH;

	if constexpr (FILE_SIZE < 1024) {
		path += std::to_string(FILE_SIZE) + "B.void";
	} else if constexpr (FILE_SIZE < 1024 * 1024) {
		path += std::to_string(FILE_SIZE / 1024) + "KB.void";
	} else if constexpr (FILE_SIZE < 1024 * 1024 * 1024) {
		path += std::to_string(FILE_SIZE / (1024 * 1024)) + "MB.void";
	} else {
		path += std::to_string(FILE_SIZE / (1024 * 1024 * 1024)) + "GB.void";
	}

	return path;
}

#ifdef _WIN32
	#define PLATFORM_WINDOWS 1
	#define PLATFORM_LINUX 0
	#define PLATFORM_MACOS 0
#elif defined(__linux__)
	#define PLATFORM_WINDOWS 0
	#define PLATFORM_LINUX 1
	#define PLATFORM_MACOS 0
#elif defined(__APPLE__)
	#define PLATFORM_WINDOWS 0
	#define PLATFORM_LINUX 0
	#define PLATFORM_MACOS 1
#endif

constexpr uint64_t alignment_size = 4096;

BNCH_SWT_INLINE static char* allocate(uint64_t count_new) noexcept {
	if (count_new == 0) {
		return nullptr;
	}
#if defined(PLATFORM_WINDOWS) || defined(PLATFORM_LINUX)
	return static_cast<char*>(_mm_malloc(count_new * sizeof(char), alignment_size));
#else
	return static_cast<char*>(aligned_alloc(alignment_size, count_new));
#endif
}

BNCH_SWT_INLINE static void free_impl(void* ptr) noexcept {
	if (!ptr) {
		return;
	}
#if defined(PLATFORM_WINDOWS) || defined(PLATFORM_LINUX)
	_mm_free(ptr);
#else
	aligned_free(ptr);
#endif
}

#include <string_view>
#include <cstring>

#ifdef _WIN32
	#include <windows.h>
#elif defined(__linux__)
	#include <fcntl.h>
	#include <unistd.h>
	#include <sys/stat.h>
	#include <liburing.h>
	#include <errno.h>
#elif defined(__APPLE__)
	#include <fcntl.h>
	#include <unistd.h>
	#include <sys/types.h>
	#include <sys/stat.h>
	#include <aio.h>
	#include <errno.h>
#endif


enum class file_size_cutoffs : uint64_t {
#if PLATFORM_WINDOWS
	tiny   = 4ull * 1024ull,
	small  = 64ull * 1024ull,
	medium = 2ull * 1024ull * 1024ull,
	large,
#elif PLATFORM_LINUX
	tiny   = 4ull * 1024ull,
	small  = 16ull * 1024ull * 1024ull,
	medium = 128ull * 1024ull * 1024ull,
	large,
#elif PLATFORM_MAC
	tiny   = 4ull * 1024ull,
	small  = 64ull * 1024ull,
	medium = 2ull * 1024ull * 1024ull,
	large,
#endif
};

enum class file_access_statuses {
	success,
	file_open_fail,
	file_create_fail,
	file_not_found,
	file_size_query_fail,
	file_extend_fail,
	set_file_pointer_fail,
	completion_port_create_fail,
	event_create_fail,
	buffer_allocation_fail,
	read_fail,
	write_fail,
	read_pending_timeout,
	write_pending_timeout,
	memory_allocation_fail,
	overlapped_result_fail,
	wait_operation_fail,
	completion_status_fail,
	io_pending_error,
	invalid_handle,
	file_not_open,
	buffer_not_allocated,
	invalid_file_size,
	alignment_error,
	chunk_size_error,
	unknown_error,
	not_initialized,
	uring_init_fail,
	aio_fail,
	count
};

enum class file_access_types {
	read,
	write,
	read_write,
};

template<file_access_types mode> struct io_file {
	BNCH_SWT_INLINE io_file() noexcept {
	}

	BNCH_SWT_INLINE io_file(io_file&& other) noexcept
		: file_access_status(other.file_access_status), file_path(std::move(other.file_path)), file_offset(other.file_offset), file_size(other.file_size),
		  file_active(other.file_active)
#if PLATFORM_WINDOWS
		  ,
		  h_completion_port(other.h_completion_port), overlapped(other.overlapped), h_event(other.h_event), h_file(other.h_file)
#else
		  ,
		  fd(other.fd)
#endif
#if PLATFORM_LINUX
		  ,
		  ring(other.ring)
#endif
	{
#if PLATFORM_WINDOWS
		other.h_file			= INVALID_HANDLE_VALUE;
		other.h_event			= NULL;
		other.h_completion_port = NULL;
#else
		other.fd = -1;
#endif
#if PLATFORM_LINUX
		other.ring = {};
#endif
		other.file_active = false;
	}

	BNCH_SWT_INLINE io_file& operator=(io_file&& other) noexcept {
		if (this != &other) {
			cleanup();
			file_access_status = other.file_access_status;
			file_path		   = std::move(other.file_path);
			file_offset		   = other.file_offset;
			file_size		   = other.file_size;
			file_active		   = other.file_active;
#if PLATFORM_WINDOWS
			h_completion_port		= other.h_completion_port;
			overlapped				= other.overlapped;
			h_event					= other.h_event;
			h_file					= other.h_file;
			other.h_file			= INVALID_HANDLE_VALUE;
			other.h_event			= NULL;
			other.h_completion_port = NULL;
#else
			fd		 = other.fd;
			other.fd = -1;
#endif
#if PLATFORM_LINUX
			ring	   = other.ring;
			other.ring = {};
#endif
			other.file_active = false;
		}
		return *this;
	}

	io_file(const io_file&)			   = delete;
	io_file& operator=(const io_file&) = delete;

	BNCH_SWT_INLINE static io_file<mode> open_file(std::string_view file_path_new, uint64_t size_or_offset_new = 0) {
		io_file<mode> return_value{};
		return_value.file_path = file_path_new;

		if constexpr (mode == file_access_types::write || mode == file_access_types::read_write) {
			return_value.file_size							  = size_or_offset_new;
			const uint64_t aligned_size						  = ((return_value.file_size + alignment_size - 1) / alignment_size) * alignment_size;
			[[maybe_unused]] const bool needs_completion_port = return_value.file_size > static_cast<uint64_t>(file_size_cutoffs::medium);
			[[maybe_unused]] const bool needs_async			  = return_value.file_size > static_cast<uint64_t>(file_size_cutoffs::small);

#if PLATFORM_WINDOWS
			DWORD access_flags	= (mode == file_access_types::read_write) ? (GENERIC_READ | GENERIC_WRITE) : GENERIC_WRITE;
			DWORD flags			= needs_async ? (FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED) : FILE_ATTRIBUTE_NORMAL;
			return_value.h_file = CreateFile(return_value.file_path.c_str(), access_flags, 0, NULL, CREATE_ALWAYS, flags, NULL);
			if (return_value.h_file == INVALID_HANDLE_VALUE) {
				std::cerr << "Failed to create file. Error: " << GetLastError() << std::endl;
				return_value.file_access_status = file_access_statuses::file_create_fail;
				return return_value;
			}
			if (needs_async) {
				LARGE_INTEGER file_size_struct;
				file_size_struct.QuadPart = aligned_size;
				if (!SetFilePointerEx(return_value.h_file, file_size_struct, NULL, FILE_BEGIN)) {
					std::cerr << "Failed to set file pointer. Error: " << GetLastError() << std::endl;
					CloseHandle(return_value.h_file);
					return_value.h_file				= INVALID_HANDLE_VALUE;
					return_value.file_access_status = file_access_statuses::set_file_pointer_fail;
					return return_value;
				}
				if (!SetEndOfFile(return_value.h_file)) {
					std::cerr << "Failed to extend file. Error: " << GetLastError() << std::endl;
					CloseHandle(return_value.h_file);
					return_value.h_file				= INVALID_HANDLE_VALUE;
					return_value.file_access_status = file_access_statuses::file_extend_fail;
					return return_value;
				}
			}
			if (needs_completion_port) {
				return_value.h_completion_port = CreateIoCompletionPort(return_value.h_file, NULL, 0, 0);
				if (!return_value.h_completion_port) {
					std::cerr << "Failed to create completion port. Error: " << GetLastError() << std::endl;
					CloseHandle(return_value.h_file);
					return_value.h_file				= INVALID_HANDLE_VALUE;
					return_value.file_access_status = file_access_statuses::completion_port_create_fail;
					return return_value;
				}
			} else if (needs_async) {
				return_value.h_event = CreateEvent(NULL, TRUE, FALSE, NULL);
				if (!return_value.h_event) {
					CloseHandle(return_value.h_file);
					return_value.h_file				= INVALID_HANDLE_VALUE;
					return_value.file_access_status = file_access_statuses::event_create_fail;
					return return_value;
				}
			}
#else
			int flags = (mode == file_access_types::read_write) ? (O_RDWR | O_CREAT | O_TRUNC) : (O_WRONLY | O_CREAT | O_TRUNC);
	#if PLATFORM_LINUX
			if (needs_async)
				flags |= O_DIRECT;
	#endif
			return_value.fd = open(return_value.file_path.c_str(), flags, 0644);
			if (return_value.fd < 0) {
				std::cerr << "Failed to create file. Error: " << strerror(errno) << std::endl;
				return_value.file_access_status = file_access_statuses::file_create_fail;
				return return_value;
			}
			if (ftruncate(return_value.fd, static_cast<off_t>(aligned_size)) != 0) {
				std::cerr << "Failed to extend file. Error: " << strerror(errno) << std::endl;
				close(return_value.fd);
				return_value.fd					= -1;
				return_value.file_access_status = file_access_statuses::file_extend_fail;
				return return_value;
			}
	#if PLATFORM_LINUX
			if (needs_async && fallocate(return_value.fd, 0, 0, static_cast<off_t>(aligned_size)) != 0) {
				std::cerr << "Failed to allocate file space. Error: " << strerror(errno) << std::endl;
				close(return_value.fd);
				return_value.fd					= -1;
				return_value.file_access_status = file_access_statuses::file_extend_fail;
				return return_value;
			}
			if (needs_completion_port) {
				if (io_uring_queue_init(64, &return_value.ring, 0) < 0) {
					std::cerr << "Failed to initialize io_uring. Error: " << strerror(errno) << std::endl;
					close(return_value.fd);
					return_value.fd					= -1;
					return_value.file_access_status = file_access_statuses::uring_init_fail;
					return return_value;
				}
			}
	#endif
#endif
			return_value.file_active		= true;
			return_value.file_access_status = file_access_statuses::success;
			return return_value;
		} else {
			return_value.file_offset = size_or_offset_new;
#if PLATFORM_WINDOWS
			HANDLE temp_handle = CreateFile(return_value.file_path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
			if (temp_handle == INVALID_HANDLE_VALUE) {
				std::cerr << "Failed to open file. Error: " << GetLastError() << std::endl;
				return_value.file_access_status = file_access_statuses::file_not_found;
				return return_value;
			}
			LARGE_INTEGER file_size_li;
			if (!GetFileSizeEx(temp_handle, &file_size_li)) {
				std::cerr << "Failed to get file size. Error: " << GetLastError() << std::endl;
				CloseHandle(temp_handle);
				return_value.file_access_status = file_access_statuses::file_size_query_fail;
				return return_value;
			}
			uint64_t total_file_size = file_size_li.QuadPart;
			if (return_value.file_offset >= total_file_size) {
				std::cerr << "File offset exceeds file size." << std::endl;
				CloseHandle(temp_handle);
				return_value.file_access_status = file_access_statuses::invalid_file_size;
				return return_value;
			}
			return_value.file_size = total_file_size - return_value.file_offset;
			CloseHandle(temp_handle);
#else
			struct stat st;
			if (stat(return_value.file_path.c_str(), &st) != 0) {
				std::cerr << "Failed to stat file. Error: " << strerror(errno) << std::endl;
				return_value.file_access_status = file_access_statuses::file_not_found;
				return return_value;
			}
			uint64_t total_file_size = static_cast<uint64_t>(st.st_size);
			if (return_value.file_offset >= total_file_size) {
				std::cerr << "File offset exceeds file size." << std::endl;
				return_value.file_access_status = file_access_statuses::invalid_file_size;
				return return_value;
			}
			return_value.file_size = total_file_size - return_value.file_offset;
#endif
			[[maybe_unused]] const bool needs_completion_port = return_value.file_size > static_cast<uint64_t>(file_size_cutoffs::medium);
			[[maybe_unused]] const bool needs_async			  = return_value.file_size > static_cast<uint64_t>(file_size_cutoffs::small);
#if PLATFORM_WINDOWS
			DWORD flags			= needs_async ? (FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED) : FILE_ATTRIBUTE_NORMAL;
			return_value.h_file = CreateFile(return_value.file_path.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, flags, NULL);
			if (return_value.h_file == INVALID_HANDLE_VALUE) {
				std::cerr << "Failed to open file. Error: " << GetLastError() << std::endl;
				return_value.file_access_status = file_access_statuses::file_not_found;
				return return_value;
			}
			if (needs_completion_port) {
				return_value.h_completion_port = CreateIoCompletionPort(return_value.h_file, NULL, 0, 0);
				if (!return_value.h_completion_port) {
					std::cerr << "Failed to create completion port. Error: " << GetLastError() << std::endl;
					CloseHandle(return_value.h_file);
					return_value.h_file				= INVALID_HANDLE_VALUE;
					return_value.file_access_status = file_access_statuses::completion_port_create_fail;
					return return_value;
				}
			} else if (needs_async) {
				return_value.h_event = CreateEvent(NULL, TRUE, FALSE, NULL);
				if (!return_value.h_event) {
					CloseHandle(return_value.h_file);
					return_value.h_file				= INVALID_HANDLE_VALUE;
					return_value.file_access_status = file_access_statuses::event_create_fail;
					return return_value;
				}
			}
#else
			int flags = O_RDONLY;
	#if PLATFORM_LINUX
			if (needs_async)
				flags |= O_DIRECT;
	#endif
			return_value.fd = open(return_value.file_path.c_str(), flags);
			if (return_value.fd < 0) {
				std::cerr << "Failed to open file. Error: " << strerror(errno) << std::endl;
				return_value.file_access_status = file_access_statuses::file_not_found;
				return return_value;
			}
	#if PLATFORM_LINUX
			if (needs_completion_port) {
				if (io_uring_queue_init(64, &return_value.ring, 0) < 0) {
					std::cerr << "Failed to initialize io_uring. Error: " << strerror(errno) << std::endl;
					close(return_value.fd);
					return_value.fd					= -1;
					return_value.file_access_status = file_access_statuses::uring_init_fail;
					return return_value;
				}
			}
	#endif
#endif
			return_value.file_active		= true;
			return_value.file_access_status = file_access_statuses::success;
			return return_value;
		}
	}

	BNCH_SWT_INLINE void write_data(uint64_t offset, uint64_t length_to_write, void* data)
		requires(mode == file_access_types::write || mode == file_access_types::read_write)
	{
		if (length_to_write <= static_cast<uint64_t>(file_size_cutoffs::tiny)) {
			return impl_tiny_write(offset, length_to_write, data);
		} else if (length_to_write <= static_cast<uint64_t>(file_size_cutoffs::small)) {
			return impl_small_write(offset, length_to_write, data);
		} else if (length_to_write <= static_cast<uint64_t>(file_size_cutoffs::medium)) {
			return impl_medium_write(offset, length_to_write, data);
		} else {
			return impl_large_write(offset, length_to_write, data);
		}
	}

	BNCH_SWT_INLINE void read_data(uint64_t offset, uint64_t length_to_read, void* data)
		requires(mode == file_access_types::read || mode == file_access_types::read_write)
	{
		if (length_to_read <= static_cast<uint64_t>(file_size_cutoffs::tiny)) {
			return impl_tiny_read(offset, length_to_read, data);
		} else if (length_to_read <= static_cast<uint64_t>(file_size_cutoffs::small)) {
			return impl_small_read(offset, length_to_read, data);
		} else if (length_to_read <= static_cast<uint64_t>(file_size_cutoffs::medium)) {
			return impl_medium_read(offset, length_to_read, data);
		} else {
			return impl_large_read(offset, length_to_read, data);
		}
	}

	BNCH_SWT_INLINE ~io_file() {
		cleanup();
	}

	BNCH_SWT_INLINE void cleanup() {
		if (!file_active)
			return;
#if PLATFORM_WINDOWS
		if (h_completion_port) {
			CloseHandle(h_completion_port);
			h_completion_port = NULL;
		}
		if (h_event) {
			CloseHandle(h_event);
			h_event = NULL;
		}
		if (h_file != INVALID_HANDLE_VALUE) {
			CloseHandle(h_file);
			h_file = INVALID_HANDLE_VALUE;
		}
#else
	#if PLATFORM_LINUX
		if (ring.ring_fd >= 0) {
			io_uring_queue_exit(&ring);
			ring = {};
		}
	#endif
		if (fd >= 0) {
			close(fd);
			fd = -1;
		}
#endif
		file_active = false;
	}
	file_access_statuses file_access_status{};

	BNCH_SWT_INLINE operator bool() {
		return file_active;
	}

  protected:
	std::string file_path{};
	uint64_t file_offset{};
	uint64_t file_size{};
	bool file_active{};
#if PLATFORM_WINDOWS
	HANDLE h_completion_port{};
	OVERLAPPED overlapped{};
	HANDLE h_event{};
	HANDLE h_file{};
#else
	int fd{ -1 };
#endif
#if PLATFORM_LINUX
	io_uring ring{};
#endif

	BNCH_SWT_INLINE void impl_tiny_write(uint64_t offset, uint64_t length, void* buffer) {
#if PLATFORM_WINDOWS
		DWORD bytes_written = 0;
		LARGE_INTEGER file_pos;
		file_pos.QuadPart = offset;
		if (!SetFilePointerEx(h_file, file_pos, NULL, FILE_BEGIN)) {
			std::cerr << "SetFilePointerEx failed. Error: " << GetLastError() << std::endl;
			file_access_status = file_access_statuses::set_file_pointer_fail;
			return;
		}
		if (!WriteFile(h_file, buffer, static_cast<DWORD>(length), &bytes_written, NULL)) {
			std::cerr << "WriteFile failed. Error: " << GetLastError() << std::endl;
			file_access_status = file_access_statuses::write_fail;
			return;
		}
#else
		lseek(fd, static_cast<off_t>(offset), SEEK_SET);
		ssize_t bytes_written = write(fd, buffer, length);
		if (bytes_written < 0) {
			std::cerr << "write() failed. Error: " << strerror(errno) << std::endl;
			file_access_status = file_access_statuses::write_fail;
			return;
		}
#endif
		file_access_status = file_access_statuses::success;
	}

	BNCH_SWT_INLINE void impl_small_write(uint64_t offset, uint64_t length, void* buffer) {
#if PLATFORM_WINDOWS
		LARGE_INTEGER file_pos;
		file_pos.QuadPart = offset;
		if (!SetFilePointerEx(h_file, file_pos, NULL, FILE_BEGIN)) {
			std::cerr << "SetFilePointerEx failed. Error: " << GetLastError() << std::endl;
			file_access_status = file_access_statuses::set_file_pointer_fail;
			return;
		}
		DWORD bytes_written = 0;
		if (!WriteFile(h_file, buffer, static_cast<DWORD>(length), &bytes_written, NULL)) {
			std::cerr << "WriteFile failed. Error: " << GetLastError() << std::endl;
			file_access_status = file_access_statuses::write_fail;
			return;
		}
#else
		lseek(fd, static_cast<off_t>(offset), SEEK_SET);
		ssize_t bytes_written = write(fd, buffer, length);
		if (bytes_written < 0) {
			std::cerr << "write() failed. Error: " << strerror(errno) << std::endl;
			file_access_status = file_access_statuses::write_fail;
			return;
		}
#endif
		file_access_status = file_access_statuses::success;
	}

	BNCH_SWT_INLINE void impl_medium_write(uint64_t offset, uint64_t length, void* buffer) {
		const uint64_t aligned_size = ((length + alignment_size - 1) / alignment_size) * alignment_size;
#if PLATFORM_WINDOWS
		ResetEvent(h_event);
		overlapped.Offset		= static_cast<DWORD>(offset & 0xFFFFFFFF);
		overlapped.OffsetHigh	= static_cast<DWORD>(offset >> 32);
		overlapped.hEvent		= h_event;
		overlapped.Internal		= 0;
		overlapped.InternalHigh = 0;
		DWORD bytes_written		= 0;
		BOOL result				= WriteFile(h_file, buffer, static_cast<DWORD>(aligned_size), &bytes_written, &overlapped);
		if (!result) {
			DWORD error = GetLastError();
			if (error != ERROR_IO_PENDING) {
				std::cerr << "WriteFile failed. Error: " << error << std::endl;
				file_access_status = file_access_statuses::write_fail;
				return;
			}
			DWORD wait_result = WaitForSingleObject(h_event, INFINITE);
			if (wait_result != WAIT_OBJECT_0) {
				std::cerr << "Wait failed. Error: " << GetLastError() << std::endl;
				file_access_status = file_access_statuses::wait_operation_fail;
				return;
			}
			if (!GetOverlappedResult(h_file, &overlapped, &bytes_written, FALSE)) {
				std::cerr << "GetOverlappedResult failed. Error: " << GetLastError() << std::endl;
				file_access_status = file_access_statuses::overlapped_result_fail;
				return;
			}
		}
#else
		ssize_t bytes_written = pwrite(fd, buffer, aligned_size, static_cast<off_t>(offset));
		if (bytes_written < 0) {
			std::cerr << "pwrite() failed. Error: " << strerror(errno) << std::endl;
			file_access_status = file_access_statuses::write_fail;
			return;
		}
#endif
		file_access_status = file_access_statuses::success;
	}

	BNCH_SWT_INLINE void impl_large_write(uint64_t offset, uint64_t length, void* buffer) {
		const uint64_t aligned_size		 = ((length + alignment_size - 1) / alignment_size) * alignment_size;
		const uint64_t base_chunk_size	 = (length >= 256 * 1024 * 1024) ? (8 * 1024 * 1024) : (length >= 64 * 1024 * 1024) ? (4 * 1024 * 1024) : (2 * 1024 * 1024);
		constexpr uint64_t max_chunks	 = 64;
		const uint64_t calculated_chunks = (aligned_size + base_chunk_size - 1) / base_chunk_size;
		const uint64_t chunks			 = (calculated_chunks > max_chunks) ? max_chunks : calculated_chunks;
		const uint64_t actual_chunk_size =
			(calculated_chunks > max_chunks) ? (((aligned_size + max_chunks - 1) / max_chunks + alignment_size - 1) / alignment_size) * alignment_size : base_chunk_size;
#if PLATFORM_WINDOWS
		struct IOOperation {
			OVERLAPPED overlapped;
			uint64_t offset;
			uint64_t size;
		};
		std::vector<IOOperation> io_operations(chunks);
		for (uint64_t i = 0; i < chunks; ++i) {
			uint64_t local_offset  = i * actual_chunk_size;
			uint64_t file_position = offset + local_offset;
			uint64_t chunk_size	   = (i == chunks - 1) ? (aligned_size - local_offset) : actual_chunk_size;
			memset(&io_operations[i].overlapped, 0, sizeof(OVERLAPPED));
			io_operations[i].overlapped.Offset	   = static_cast<DWORD>(file_position & 0xFFFFFFFF);
			io_operations[i].overlapped.OffsetHigh = static_cast<DWORD>(file_position >> 32);
			io_operations[i].offset				   = file_position;
			io_operations[i].size				   = chunk_size;
			DWORD bytes_written					   = 0;
			BOOL result = WriteFile(h_file, static_cast<char*>(buffer) + local_offset, static_cast<DWORD>(chunk_size), &bytes_written, &io_operations[i].overlapped);
			if (!result && GetLastError() != ERROR_IO_PENDING) {
				std::cerr << "WriteFile failed for chunk " << i << ". Error: " << GetLastError() << std::endl;
				file_access_status = file_access_statuses::write_fail;
				return;
			}
		}
		for (uint64_t i = 0; i < chunks; ++i) {
			DWORD bytes_transferred	  = 0;
			ULONG_PTR completion_key  = 0;
			LPOVERLAPPED p_overlapped = nullptr;
			if (!GetQueuedCompletionStatus(h_completion_port, &bytes_transferred, &completion_key, &p_overlapped, INFINITE)) {
				std::cerr << "GetQueuedCompletionStatus failed. Error: " << GetLastError() << std::endl;
				file_access_status = file_access_statuses::completion_status_fail;
				return;
			}
		}
#elif PLATFORM_LINUX
		static constexpr uint64_t batch_size = 4ull;
		for (uint64_t batch_start = 0; batch_start < chunks; batch_start += batch_size) {
			uint64_t batch_end	 = (batch_start + batch_size < chunks) ? (batch_start + batch_size) : chunks;
			uint64_t batch_count = batch_end - batch_start;
			for (uint64_t i = batch_start; i < batch_end; ++i) {
				uint64_t local_offset  = i * actual_chunk_size;
				uint64_t file_position = offset + local_offset;
				uint64_t chunk_size	   = (i == chunks - 1) ? (aligned_size - local_offset) : actual_chunk_size;
				io_uring_sqe* sqe	   = io_uring_get_sqe(&ring);
				if (!sqe) {
					std::cerr << "Failed to get SQE" << std::endl;
					file_access_status = file_access_statuses::write_fail;
					return;
				}
				io_uring_prep_write(sqe, fd, static_cast<char*>(buffer) + local_offset, static_cast<uint32_t>(chunk_size), file_position);
				io_uring_sqe_set_data(sqe, reinterpret_cast<void*>(i));
			}
			int submitted = io_uring_submit(&ring);
			if (submitted < 0) {
				std::cerr << "io_uring_submit failed. Error: " << strerror(-submitted) << std::endl;
				file_access_status = file_access_statuses::write_fail;
				return;
			}
			bool had_error = false;
			for (uint64_t i = 0; i < batch_count; ++i) {
				io_uring_cqe* cqe;
				int ret = io_uring_wait_cqe(&ring, &cqe);
				if (ret < 0) {
					std::cerr << "io_uring_wait_cqe failed. Error: " << strerror(-ret) << std::endl;
					had_error = true;
				} else if (cqe->res < 0) {
					std::cerr << "Write operation failed for chunk " << (batch_start + i) << ". Error: " << strerror(-cqe->res) << std::endl;
					had_error = true;
				}
				if (cqe) {
					io_uring_cqe_seen(&ring, cqe);
				}
			}
			if (had_error) {
				file_access_status = file_access_statuses::write_fail;
				return;
			}
		}
#elif PLATFORM_MAC
		struct aiocb* cbs = new struct aiocb[chunks];
		memset(cbs, 0, sizeof(struct aiocb) * chunks);
		for (uint64_t i = 0; i < chunks; ++i) {
			uint64_t local_offset  = i * actual_chunk_size;
			uint64_t file_position = offset + local_offset;
			uint64_t chunk_size	   = (i == chunks - 1) ? (aligned_size - local_offset) : actual_chunk_size;
			cbs[i].aio_fildes	   = fd;
			cbs[i].aio_buf		   = static_cast<char*>(buffer) + local_offset;
			cbs[i].aio_nbytes	   = chunk_size;
			cbs[i].aio_offset	   = static_cast<off_t>(file_position);
			if (aio_write(&cbs[i]) < 0) {
				std::cerr << "aio_write failed for chunk " << i << ". Error: " << strerror(errno) << std::endl;
				delete[] cbs;
				file_access_status = file_access_statuses::write_fail;
				return;
			}
		}
		std::vector<const struct aiocb*> cbs_list(chunks);
		for (uint64_t i = 0; i < chunks; ++i) {
			cbs_list[i] = &cbs[i];
		}
		if (aio_suspend(cbs_list.data(), static_cast<int>(chunks), nullptr) < 0) {
			std::cerr << "aio_suspend failed. Error: " << strerror(errno) << std::endl;
			delete[] cbs;
			file_access_status = file_access_statuses::aio_fail;
			return;
		}
		for (uint64_t i = 0; i < chunks; ++i) {
			int err = aio_error(&cbs[i]);
			if (err != 0) {
				std::cerr << "AIO operation " << i << " failed. Error: " << strerror(err) << std::endl;
				delete[] cbs;
				file_access_status = file_access_statuses::write_fail;
				return;
			}
			ssize_t ret = aio_return(&cbs[i]);
			if (ret < 0) {
				std::cerr << "AIO operation " << i << " return failed." << std::endl;
				delete[] cbs;
				file_access_status = file_access_statuses::write_fail;
				return;
			}
		}
		delete[] cbs;
#endif
		file_access_status = file_access_statuses::success;
	}

	BNCH_SWT_INLINE void impl_tiny_read(uint64_t offset, uint64_t length, void* buffer) {
#if PLATFORM_WINDOWS
		DWORD bytes_read = 0;
		LARGE_INTEGER file_pos;
		file_pos.QuadPart = offset;
		if (!SetFilePointerEx(h_file, file_pos, NULL, FILE_BEGIN)) {
			std::cerr << "SetFilePointerEx failed. Error: " << GetLastError() << std::endl;
			file_access_status = file_access_statuses::set_file_pointer_fail;
			return;
		}
		if (!ReadFile(h_file, buffer, static_cast<DWORD>(length), &bytes_read, NULL)) {
			std::cerr << "ReadFile failed. Error: " << GetLastError() << std::endl;
			file_access_status = file_access_statuses::read_fail;
			return;
		}
#else
		lseek(fd, static_cast<off_t>(offset), SEEK_SET);
		ssize_t bytes_read = read(fd, buffer, length);
		if (bytes_read < 0) {
			std::cerr << "read() failed. Error: " << strerror(errno) << std::endl;
			file_access_status = file_access_statuses::read_fail;
			return;
		}
#endif
		file_access_status = file_access_statuses::success;
	}

	BNCH_SWT_INLINE void impl_small_read(uint64_t offset, uint64_t length, void* buffer) {
#if PLATFORM_WINDOWS
		LARGE_INTEGER file_pos;
		file_pos.QuadPart = offset;
		if (!SetFilePointerEx(h_file, file_pos, NULL, FILE_BEGIN)) {
			std::cerr << "SetFilePointerEx failed. Error: " << GetLastError() << std::endl;
			file_access_status = file_access_statuses::set_file_pointer_fail;
			return;
		}
		DWORD bytes_read = 0;
		if (!ReadFile(h_file, buffer, static_cast<DWORD>(length), &bytes_read, NULL)) {
			std::cerr << "ReadFile failed. Error: " << GetLastError() << std::endl;
			file_access_status = file_access_statuses::read_fail;
			return;
		}
#else
		lseek(fd, static_cast<off_t>(offset), SEEK_SET);
		ssize_t bytes_read = read(fd, buffer, length);
		if (bytes_read < 0) {
			std::cerr << "read() failed. Error: " << strerror(errno) << std::endl;
			file_access_status = file_access_statuses::read_fail;
			return;
		}
#endif
		file_access_status = file_access_statuses::success;
	}

	BNCH_SWT_INLINE void impl_medium_read(uint64_t offset, uint64_t length, void* buffer) {
		const uint64_t aligned_size = ((length + alignment_size - 1) / alignment_size) * alignment_size;
#if PLATFORM_WINDOWS
		ResetEvent(h_event);
		overlapped.Offset		= static_cast<DWORD>(offset & 0xFFFFFFFF);
		overlapped.OffsetHigh	= static_cast<DWORD>(offset >> 32);
		overlapped.hEvent		= h_event;
		overlapped.Internal		= 0;
		overlapped.InternalHigh = 0;
		DWORD bytes_read		= 0;
		BOOL result				= ReadFile(h_file, buffer, static_cast<DWORD>(aligned_size), &bytes_read, &overlapped);
		if (!result) {
			DWORD error = GetLastError();
			if (error != ERROR_IO_PENDING) {
				std::cerr << "ReadFile failed. Error: " << error << std::endl;
				file_access_status = file_access_statuses::read_fail;
				return;
			}
			DWORD wait_result = WaitForSingleObject(h_event, INFINITE);
			if (wait_result != WAIT_OBJECT_0) {
				std::cerr << "Wait failed. Error: " << GetLastError() << std::endl;
				file_access_status = file_access_statuses::wait_operation_fail;
				return;
			}
			if (!GetOverlappedResult(h_file, &overlapped, &bytes_read, FALSE)) {
				std::cerr << "GetOverlappedResult failed. Error: " << GetLastError() << std::endl;
				file_access_status = file_access_statuses::overlapped_result_fail;
				return;
			}
		}
#else
		ssize_t bytes_read = pread(fd, buffer, aligned_size, static_cast<off_t>(offset));
		if (bytes_read < 0) {
			std::cerr << "pread() failed. Error: " << strerror(errno) << std::endl;
			file_access_status = file_access_statuses::read_fail;
			return;
		}
#endif
		file_access_status = file_access_statuses::success;
	}

	BNCH_SWT_INLINE void impl_large_read(uint64_t offset, uint64_t length, void* buffer) {
		const uint64_t aligned_size		 = ((length + alignment_size - 1) / alignment_size) * alignment_size;
		const uint64_t base_chunk_size	 = (length >= 256 * 1024 * 1024) ? (8 * 1024 * 1024) : (length >= 64 * 1024 * 1024) ? (4 * 1024 * 1024) : (2 * 1024 * 1024);
		constexpr uint64_t max_chunks	 = 64;
		const uint64_t calculated_chunks = (aligned_size + base_chunk_size - 1) / base_chunk_size;
		const uint64_t chunks			 = (calculated_chunks > max_chunks) ? max_chunks : calculated_chunks;
		const uint64_t actual_chunk_size =
			(calculated_chunks > max_chunks) ? (((aligned_size + max_chunks - 1) / max_chunks + alignment_size - 1) / alignment_size) * alignment_size : base_chunk_size;
#if PLATFORM_WINDOWS
		struct IOOperation {
			OVERLAPPED overlapped;
			uint64_t offset;
			uint64_t size;
		};
		std::vector<IOOperation> io_operations(chunks);
		for (uint64_t i = 0; i < chunks; ++i) {
			uint64_t local_offset  = i * actual_chunk_size;
			uint64_t file_position = offset + local_offset;
			uint64_t chunk_size	   = (i == chunks - 1) ? (aligned_size - local_offset) : actual_chunk_size;
			memset(&io_operations[i].overlapped, 0, sizeof(OVERLAPPED));
			io_operations[i].overlapped.Offset	   = static_cast<DWORD>(file_position & 0xFFFFFFFF);
			io_operations[i].overlapped.OffsetHigh = static_cast<DWORD>(file_position >> 32);
			io_operations[i].offset				   = file_position;
			io_operations[i].size				   = chunk_size;
			DWORD bytes_read					   = 0;
			BOOL result =
				ReadFile(h_file, static_cast<LPVOID>(static_cast<char*>(buffer) + local_offset), static_cast<DWORD>(chunk_size), &bytes_read, &io_operations[i].overlapped);
			if (!result && GetLastError() != ERROR_IO_PENDING) {
				std::cerr << "ReadFile failed for chunk " << i << ". Error: " << GetLastError() << std::endl;
				file_access_status = file_access_statuses::read_fail;
				return;
			}
		}
		for (uint64_t i = 0; i < chunks; ++i) {
			DWORD bytes_transferred	  = 0;
			ULONG_PTR completion_key  = 0;
			LPOVERLAPPED p_overlapped = nullptr;
			if (!GetQueuedCompletionStatus(h_completion_port, &bytes_transferred, &completion_key, &p_overlapped, INFINITE)) {
				std::cerr << "GetQueuedCompletionStatus failed. Error: " << GetLastError() << std::endl;
				file_access_status = file_access_statuses::completion_status_fail;
				return;
			}
		}
#elif PLATFORM_LINUX
		static constexpr uint64_t batch_size = 16ull;
		for (uint64_t batch_start = 0; batch_start < chunks; batch_start += batch_size) {
			uint64_t batch_end	 = (batch_start + batch_size < chunks) ? (batch_start + batch_size) : chunks;
			uint64_t batch_count = batch_end - batch_start;
			for (uint64_t i = batch_start; i < batch_end; ++i) {
				uint64_t local_offset  = i * actual_chunk_size;
				uint64_t file_position = offset + local_offset;
				uint64_t chunk_size	   = (i == chunks - 1) ? (aligned_size - local_offset) : actual_chunk_size;
				io_uring_sqe* sqe	   = io_uring_get_sqe(&ring);
				if (!sqe) {
					std::cerr << "Failed to get SQE" << std::endl;
					file_access_status = file_access_statuses::read_fail;
					return;
				}
				io_uring_prep_read(sqe, fd, buffer + local_offset, static_cast<uint32_t>(chunk_size), file_position);
				io_uring_sqe_set_data(sqe, reinterpret_cast<void*>(i));
			}
			int submitted = io_uring_submit(&ring);
			if (submitted < 0) {
				std::cerr << "io_uring_submit failed. Error: " << strerror(-submitted) << std::endl;
				file_access_status = file_access_statuses::read_fail;
				return;
			}
			bool had_error = false;
			for (uint64_t i = 0; i < batch_count; ++i) {
				io_uring_cqe* cqe;
				int ret = io_uring_wait_cqe(&ring, &cqe);
				if (ret < 0) {
					std::cerr << "io_uring_wait_cqe failed. Error: " << strerror(-ret) << std::endl;
					had_error = true;
				} else if (cqe->res < 0) {
					std::cerr << "Read operation failed for chunk " << (batch_start + i) << ". Error: " << strerror(-cqe->res) << std::endl;
					had_error = true;
				}
				if (cqe) {
					io_uring_cqe_seen(&ring, cqe);
				}
			}
			if (had_error) {
				file_access_status = file_access_statuses::read_fail;
				return;
			}
		}
#elif PLATFORM_MAC
		struct aiocb* cbs = new struct aiocb[chunks];
		memset(cbs, 0, sizeof(struct aiocb) * chunks);
		for (uint64_t i = 0; i < chunks; ++i) {
			uint64_t local_offset  = i * actual_chunk_size;
			uint64_t file_position = offset + local_offset;
			uint64_t chunk_size	   = (i == chunks - 1) ? (aligned_size - local_offset) : actual_chunk_size;
			cbs[i].aio_fildes	   = fd;
			cbs[i].aio_buf		   = buffer + local_offset;
			cbs[i].aio_nbytes	   = chunk_size;
			cbs[i].aio_offset	   = static_cast<off_t>(file_position);
			if (aio_read(&cbs[i]) < 0) {
				std::cerr << "aio_read failed for chunk " << i << ". Error: " << strerror(errno) << std::endl;
				delete[] cbs;
				file_access_status = file_access_statuses::read_fail;
				return;
			}
		}
		std::vector<const struct aiocb*> cbs_list(chunks);
		for (uint64_t i = 0; i < chunks; ++i) {
			cbs_list[i] = &cbs[i];
		}
		if (aio_suspend(cbs_list.data(), static_cast<int>(chunks), nullptr) < 0) {
			std::cerr << "aio_suspend failed. Error: " << strerror(errno) << std::endl;
			delete[] cbs;
			file_access_status = file_access_statuses::aio_fail;
			return;
		}
		for (uint64_t i = 0; i < chunks; ++i) {
			int err = aio_error(&cbs[i]);
			if (err != 0) {
				std::cerr << "AIO operation " << i << " failed. Error: " << strerror(err) << std::endl;
				delete[] cbs;
				file_access_status = file_access_statuses::read_fail;
				return;
			}
			ssize_t ret = aio_return(&cbs[i]);
			if (ret < 0) {
				std::cerr << "AIO operation " << i << " return failed." << std::endl;
				delete[] cbs;
				file_access_status = file_access_statuses::read_fail;
				return;
			}
		}
		delete[] cbs;
#endif
		file_access_status = file_access_statuses::success;
	}
};

template<uint64_t FILE_SIZE> struct io_read_benchmark {
	inline static io_file<file_access_types::read> file{};
	inline static std::string file_path;
	inline static char* data{};
	static bool setup() {
		file_path = get_file_path<FILE_SIZE>();
		data	  = allocate(FILE_SIZE);
		file	  = io_file<file_access_types::read>::open_file(file_path);

		if (!file) {
			std::cerr << "File not opened successfully!" << std::endl;
			return false;
		}

		// 🔥 VERIFY DATA BEFORE BENCHMARKING! 🔥
		std::cout << "🔍 VERIFYING DATA FOR " << FILE_SIZE << " BYTES..." << std::endl;
		file.read_data(0, FILE_SIZE, data);
		if (!file) {
			std::cerr << "❌ FAILED TO READ DATA FOR VERIFICATION!" << std::endl;
			return false;
		}

		if (!global_verify_data.verify_data(FILE_SIZE, data)) {
			std::cerr << "❌ VERIFICATION FAILED!" << std::endl;
			return false;
		}

		return true;
	}

	static void cleanup() {
		file.~io_file();
	}

	BNCH_SWT_INLINE static uint64_t impl(uint64_t& current_index) {
		file.read_data(0, FILE_SIZE, data);
		current_index++;
		return FILE_SIZE;
	}
};

template<uint64_t FILE_SIZE> struct io_write_benchmark {
	inline static io_file<file_access_types::write> file{};
	inline static std::string filePath;
	inline static char* data{};

	static bool setup() {
		filePath = get_file_path<FILE_SIZE>();
		data	 = static_cast<char*>(allocate(FILE_SIZE));

		// 🔥 GENERATE RANDOM DATA! 🔥
		std::cout << "🎲 GENERATING RANDOM DATA FOR " << FILE_SIZE << " BYTES..." << std::endl;
		std::random_device rd;
		std::mt19937_64 gen(rd());
		std::uniform_int_distribution<uint64_t> dis;

		uint64_t* data_as_uint64 = reinterpret_cast<uint64_t*>(data);
		uint64_t num_uint64s	 = FILE_SIZE / sizeof(uint64_t);

		for (uint64_t i = 0; i < num_uint64s; ++i) {
			data_as_uint64[i] = dis(gen);
		}

		// Fill any remaining bytes
		uint64_t remaining_bytes = FILE_SIZE % sizeof(uint64_t);
		if (remaining_bytes > 0) {
			char* remaining_data = data + (num_uint64s * sizeof(uint64_t));
			uint64_t random_val	 = dis(gen);
			std::memcpy(remaining_data, &random_val, remaining_bytes);
		}

		// 💾 STORE THE DATA FOR VERIFICATION! 💾
		global_verify_data.store_data(FILE_SIZE, data);

		file = io_file<file_access_types::write>::open_file(filePath, FILE_SIZE);

		return true;
	}

	static void cleanup() {
		file.~io_file();
		free_impl(data);
	}

	BNCH_SWT_INLINE static uint64_t impl(uint64_t& current_index) {
		file.write_data(0, FILE_SIZE, data);
		current_index++;
		return FILE_SIZE;
	}
};

template<bnch_swt::string_literal BenchmarkName, uint64_t FILE_SIZE, typename BenchmarkType> BNCH_SWT_INLINE void run_io_benchmark() {
	if (!BenchmarkType::setup()) {
		throw std::runtime_error("Setup failed.");
	}

	std::vector<std::vector<std::string>> values_to_test_01(total_iterations);
	std::vector<std::vector<uint64_t>> values_tested01(total_iterations);
	uint64_t current_index = 0;

	static constexpr bnch_swt::string_literal test_name{ "Adaptive-IO-Benchmark-" + BenchmarkName };

	bnch_swt::benchmark_stage<test_name, total_iterations, measured_iterations>::template runBenchmark<BenchmarkName, BenchmarkType>(current_index);

	bnch_swt::benchmark_stage<test_name, total_iterations, measured_iterations>::printResults();

	BenchmarkType::cleanup();
}

int main() {
	try {
		std::cout << "\n";
		std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
		std::cout << "║  ADAPTIVE I/O BENCHMARK SUITE - SAMSUNG 9100 PRO (WRITE)      ║\n";
		std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
		std::cout << "\n";

		std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
		std::cout << "TIER 1: TINY FILES WRITE (Trivial Sync)\n";
		std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
		run_io_benchmark<"2B", 2, io_write_benchmark<2>>();
		run_io_benchmark<"4B", 4, io_write_benchmark<4>>();
		run_io_benchmark<"8B", 8, io_write_benchmark<8>>();
		run_io_benchmark<"16B", 16, io_write_benchmark<16>>();
		run_io_benchmark<"32B", 32, io_write_benchmark<32>>();
		run_io_benchmark<"64B", 64, io_write_benchmark<64>>();
		run_io_benchmark<"128B", 128, io_write_benchmark<128>>();
		run_io_benchmark<"256B", 256, io_write_benchmark<256>>();
		run_io_benchmark<"512B", 512, io_write_benchmark<512>>();
		run_io_benchmark<"1KB", 1024, io_write_benchmark<1024>>();
		run_io_benchmark<"2KB", 2048, io_write_benchmark<2048>>();
		run_io_benchmark<"4KB", 4096, io_write_benchmark<4096>>();

		std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
		std::cout << "TIER 2: SMALL FILES WRITE (Buffered Sync)\n";
		std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
		run_io_benchmark<"16KB", 16 * 1024, io_write_benchmark<16 * 1024>>();
		run_io_benchmark<"32KB", 32 * 1024, io_write_benchmark<32 * 1024>>();
		run_io_benchmark<"64KB", 64 * 1024, io_write_benchmark<64 * 1024>>();

		std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
		std::cout << "TIER 3: MEDIUM FILES WRITE (Single Async Direct I/O)\n";
		std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
		run_io_benchmark<"256KB", 256 * 1024, io_write_benchmark<256 * 1024>>();
		run_io_benchmark<"512KB", 512 * 1024, io_write_benchmark<512 * 1024>>();
		run_io_benchmark<"1MB", 1 * 1024 * 1024, io_write_benchmark<1 * 1024 * 1024>>();
		run_io_benchmark<"2MB", 2 * 1024 * 1024, io_write_benchmark<2 * 1024 * 1024>>();

		std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
		std::cout << "TIER 4: LARGE FILES WRITE (Chunked Async Direct I/O)\n";
		std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
		run_io_benchmark<"8MB", 8 * 1024 * 1024, io_write_benchmark<8 * 1024 * 1024>>();
		run_io_benchmark<"16MB", 16 * 1024 * 1024, io_write_benchmark<16 * 1024 * 1024>>();
		run_io_benchmark<"32MB", 32 * 1024 * 1024, io_write_benchmark<32 * 1024 * 1024>>();
		run_io_benchmark<"64MB", 64 * 1024 * 1024, io_write_benchmark<64 * 1024 * 1024>>();

		std::cout << "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
		std::cout << "TIER 5: HUGE FILES WRITE (Adaptive Chunked Async Direct I/O)\n";
		std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
		run_io_benchmark<"128MB", 128 * 1024 * 1024, io_write_benchmark<128 * 1024 * 1024>>();
		run_io_benchmark<"256MB", 256 * 1024 * 1024, io_write_benchmark<256 * 1024 * 1024>>();
		run_io_benchmark<"512MB", 512 * 1024 * 1024, io_write_benchmark<512 * 1024 * 1024>>();
		run_io_benchmark<"1GB", 1024 * 1024 * 1024, io_write_benchmark<1024 * 1024 * 1024>>();


		std::cout << "\n\n";
		std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
		std::cout << "║  ADAPTIVE I/O BENCHMARK SUITE - SAMSUNG 9100 PRO (READ)       ║\n";
		std::cout << "╚═══════════════════════════════════════════════════════════════╝\n";
		std::cout << "\n";

		std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
		std::cout << "TIER 1: TINY FILES READ (Trivial Sync)\n";
		std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";

		run_io_benchmark<"2B", 2, io_read_benchmark<2>>();
		run_io_benchmark<"4B", 4, io_read_benchmark<4>>();
		run_io_benchmark<"8B", 8, io_read_benchmark<8>>();
		run_io_benchmark<"16B", 16, io_read_benchmark<16>>();
		run_io_benchmark<"32B", 32, io_read_benchmark<32>>();
		run_io_benchmark<"64B", 64, io_read_benchmark<64>>();
		run_io_benchmark<"128B", 128, io_read_benchmark<128>>();
		run_io_benchmark<"256B", 256, io_read_benchmark<256>>();
		run_io_benchmark<"512B", 512, io_read_benchmark<512>>();
		run_io_benchmark<"1KB", 1024, io_read_benchmark<1024>>();
		run_io_benchmark<"2KB", 2048, io_read_benchmark<2048>>();
		run_io_benchmark<"4KB", 4096, io_read_benchmark<4096>>();

		run_io_benchmark<"16KB", 16 * 1024, io_read_benchmark<16 * 1024>>();
		run_io_benchmark<"32KB", 32 * 1024, io_read_benchmark<32 * 1024>>();
		run_io_benchmark<"64KB", 64 * 1024, io_read_benchmark<64 * 1024>>();

		run_io_benchmark<"256KB", 256 * 1024, io_read_benchmark<256 * 1024>>();
		run_io_benchmark<"512KB", 512 * 1024, io_read_benchmark<512 * 1024>>();
		run_io_benchmark<"1MB", 1 * 1024 * 1024, io_read_benchmark<1 * 1024 * 1024>>();
		run_io_benchmark<"2MB", 2 * 1024 * 1024, io_read_benchmark<2 * 1024 * 1024>>();

		run_io_benchmark<"8MB", 8 * 1024 * 1024, io_read_benchmark<8 * 1024 * 1024>>();
		run_io_benchmark<"16MB", 16 * 1024 * 1024, io_read_benchmark<16 * 1024 * 1024>>();
		run_io_benchmark<"32MB", 32 * 1024 * 1024, io_read_benchmark<32 * 1024 * 1024>>();
		run_io_benchmark<"64MB", 64 * 1024 * 1024, io_read_benchmark<64 * 1024 * 1024>>();

		run_io_benchmark<"128MB", 128 * 1024 * 1024, io_read_benchmark<128 * 1024 * 1024>>();
		run_io_benchmark<"256MB", 256 * 1024 * 1024, io_read_benchmark<256 * 1024 * 1024>>();
		run_io_benchmark<"512MB", 512 * 1024 * 1024, io_read_benchmark<512 * 1024 * 1024>>();
		run_io_benchmark<"1GB", 1024 * 1024 * 1024, io_read_benchmark<1024 * 1024 * 1024>>();

	} catch (const std::exception& e) {
		std::cerr << "💀 Fatal Error: " << e.what() << std::endl;
		return 1;
	}
	return 0;
}