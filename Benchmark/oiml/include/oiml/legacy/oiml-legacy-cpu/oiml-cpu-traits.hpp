#pragma once
#include <oiml/legacy/oiml-legacy-common/oiml-backend-impl.hpp>
#include <oiml/legacy/oiml-legacy-cpu/oiml-cpu-impl.hpp>
#include <oiml/legacy/oiml-legacy-common/oiml-final.hpp>

#include <vector>

namespace oiml_legacy::cpu {
	// register in tensor->extra
	class tensor_traits {
	  public:
		virtual ~tensor_traits() {};
		virtual bool work_size(int n_threads, const oiml_tensor* op, size_t& size) = 0;
		virtual bool compute_forward(oiml_compute_params* params, oiml_tensor* op) = 0;
	};

	class extra_buffer_type {
	  public:
		virtual ~extra_buffer_type() {};
		virtual bool supports_op(const oiml_tensor* op)					= 0;
		virtual tensor_traits* get_tensor_traits(const oiml_tensor* op) = 0;
	};
}// namespace oiml_legacy::cpu

// implemented in oiml-cpu.cpp.
std::vector<oiml_backend_buffer_type_t>& oiml_backend_cpu_get_extra_buffers_type();

OIML_INLINE static bool oiml_cpu_extra_compute_forward(oiml_compute_params* params, oiml_tensor* op) {
	for (auto extra: oiml_backend_cpu_get_extra_buffers_type()) {
		if (extra && extra->context) {
			auto buf_extra	   = ( oiml_legacy::cpu::extra_buffer_type* )extra->context;
			auto tensor_traits = buf_extra->get_tensor_traits(op);
			if (tensor_traits && tensor_traits->compute_forward(params, op)) {
				return true;
			}
		}
	}
	return false;
}

OIML_INLINE static bool oiml_cpu_extra_work_size(int n_threads, const oiml_tensor* op, size_t* size) {
	for (auto extra: oiml_backend_cpu_get_extra_buffers_type()) {
		if (extra && extra->context) {
			auto buf_extra	   = ( oiml_legacy::cpu::extra_buffer_type* )extra->context;
			auto tensor_traits = buf_extra->get_tensor_traits(op);
			if (tensor_traits && tensor_traits->work_size(n_threads, op, *size)) {
				return true;
			}
		}
	}
	return false;
}
