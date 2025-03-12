#pragma once

#include <oiml/legacy/oiml-legacy-common/oiml-final.hpp>
#include <oiml/legacy/oiml-legacy-common/oiml-backend.hpp>

#ifdef OIML_USE_HIP
	#define OIML_CUDA_NAME "ROCm"
	#define OIML_CUBLAS_NAME "hipBLAS"
#elif defined(OIML_USE_MUSA)
	#define OIML_CUDA_NAME "MUSA"
	#define OIML_CUBLAS_NAME "muBLAS"
#else
	#define OIML_CUDA_NAME "CUDA"
	#define OIML_CUBLAS_NAME "cuBLAS"
#endif
#define OIML_CUDA_MAX_DEVICES 16


// backend API
oiml_backend_t oiml_backend_cuda_init(int device);

bool oiml_backend_is_cuda(oiml_backend_t backend);

// device buffer
oiml_backend_buffer_type_t oiml_backend_cuda_buffer_type(int device);

// split tensor buffer that splits matrices by rows across multiple devices
oiml_backend_buffer_type_t oiml_backend_cuda_split_buffer_type(int main_device, const float* tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
oiml_backend_buffer_type_t oiml_backend_cuda_host_buffer_type();

int oiml_backend_cuda_get_device_count();
void oiml_backend_cuda_get_device_description(int device, char* description, size_t description_size);
void oiml_backend_cuda_get_device_memory(int device, size_t* free, size_t* total);

bool oiml_backend_cuda_register_host_buffer(void* buffer, size_t size);
void oiml_backend_cuda_unregister_host_buffer(void* buffer);

oiml_backend_reg_t oiml_backend_cuda_reg();
