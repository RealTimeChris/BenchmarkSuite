#pragma once
#include <oiml/legacy/oiml-legacy-common/oiml-alloc-final.hpp>
#include <oiml/legacy/oiml-legacy-common/oiml-backend-impl.hpp>
#include <oiml/legacy/oiml-legacy-common/oiml-final.hpp>
#include <oiml/legacy/oiml-legacy-common/oiml-impl.hpp>
#include <assert.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef MAX
	#undef MAX
#endif
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MAX_FREE_BLOCKS 256

//#define OIML_ALLOCATOR_DEBUG

//#define AT_PRINTF(...) OIML_LOG_DEBUG(__VA_ARGS__)
#define AT_PRINTF(...)


OIML_INLINE static bool oiml_is_view(const oiml_tensor* t) {
	return t->view_src != NULL;
}

// ops that return true for this function must not use __restrict pointers for their backend implementations
OIML_INLINE static bool oiml_op_can_inplace(enum oiml_op op) {
	switch (op) {
		case OIML_OP_SCALE:
		case OIML_OP_DIAG_MASK_ZERO:
		case OIML_OP_DIAG_MASK_INF:
		case OIML_OP_ADD:
		case OIML_OP_ADD1:
		case OIML_OP_SUB:
		case OIML_OP_MUL:
		case OIML_OP_DIV:
		case OIML_OP_SQR:
		case OIML_OP_SQRT:
		case OIML_OP_LOG:
		case OIML_OP_UNARY:
		case OIML_OP_ROPE:
		case OIML_OP_ROPE_BACK:
		case OIML_OP_SILU_BACK:
		case OIML_OP_RMS_NORM:
		case OIML_OP_RMS_NORM_BACK:
		case OIML_OP_SOFT_MAX:
		case OIML_OP_SOFT_MAX_BACK:
			return true;

		default:
			return false;
	}
}

OIML_INLINE static size_t aligned_offset(const void* buffer, size_t offset, size_t alignment) {
	assert(alignment && !(alignment & (alignment - 1)));// power of 2
	size_t align = (alignment - ((( uintptr_t )buffer + offset) % alignment)) % alignment;
	return offset + align;
}

OIML_INLINE static bool oiml_are_same_layout(const struct oiml_tensor* a, const struct oiml_tensor* b) {
	if (a->type != b->type) {
		return false;
	}
	for (int i = 0; i < OIML_MAX_DIMS; i++) {
		if (a->ne[i] != b->ne[i]) {
			return false;
		}
		if (a->nb[i] != b->nb[i]) {
			return false;
		}
	}
	return true;
}

// tallocr

OIML_INLINE struct oiml_tallocr oiml_tallocr_new(oiml_backend_buffer_t buffer) {
	void* base	 = oiml_backend_buffer_get_base(buffer);
	size_t align = oiml_backend_buffer_get_alignment(buffer);

	assert(align && !(align & (align - 1)));// power of 2

	struct oiml_tallocr talloc = oiml_tallocr{
		/*.buffer    = */ buffer,
		/*.base      = */ base,
		/*.alignment = */ align,
		/*.offset    = */ aligned_offset(base, 0, align),
	};
	return talloc;
}

OIML_INLINE void oiml_tallocr_alloc(oiml_tallocr* talloc, oiml_tensor* tensor) {
	size_t size = oiml_backend_buffer_get_alloc_size(talloc->buffer, tensor);
	size		= OIML_PAD(size, talloc->alignment);

	if (talloc->offset + size > oiml_backend_buffer_get_size(talloc->buffer)) {
		OIML_LOG_ERROR("%s: not enough space in the buffer to allocate %s (needed %zu, available %zu)\n", __func__, tensor->name, size,
			oiml_backend_buffer_get_size(talloc->buffer) - talloc->offset);
		OIML_ABORT("not enough space in the buffer");
	}

	void* addr = ( char* )oiml_backend_buffer_get_base(talloc->buffer) + talloc->offset;
	talloc->offset += size;

	assert((( uintptr_t )addr % talloc->alignment) == 0);

	oiml_backend_tensor_alloc(talloc->buffer, tensor, addr);
}

// dynamic tensor allocator

struct free_block {
	size_t offset;
	size_t size;
};

struct oiml_dyn_tallocr {
	size_t alignment;
	int n_free_blocks;
	struct free_block free_blocks[MAX_FREE_BLOCKS];
	size_t max_size;

#ifdef OIML_ALLOCATOR_DEBUG
	struct {
		const oiml_tensor* tensor;
		size_t offset;
	} allocated_tensors[1024];
#endif
};

#ifdef OIML_ALLOCATOR_DEBUG
static void add_allocated_tensor(oiml_dyn_tallocr* alloc, size_t offset, const oiml_tensor* tensor) {
	for (int i = 0; i < 1024; i++) {
		if (alloc->allocated_tensors[i].tensor == NULL) {
			alloc->allocated_tensors[i].tensor = tensor;
			alloc->allocated_tensors[i].offset = offset;
			return;
		}
	}
	OIML_ABORT("out of allocated_tensors");
}
static void remove_allocated_tensor(oiml_dyn_tallocr* alloc, size_t offset, const oiml_tensor* tensor) {
	for (int i = 0; i < 1024; i++) {
		if (alloc->allocated_tensors[i].offset == offset) {
			alloc->allocated_tensors[i].tensor = NULL;
			return;
		}
	}
	OIML_ABORT("tried to free tensor %s not found\n", tensor->name);
}
#endif

OIML_INLINE static size_t oiml_dyn_tallocr_alloc(oiml_dyn_tallocr* alloc, size_t size, const oiml_tensor* tensor) {
	size = aligned_offset(NULL, size, alloc->alignment);

	AT_PRINTF("%s: allocating %s (%zu bytes) - ", __func__, tensor->name, size);

	size_t max_avail = 0;

	// find the best fitting free block besides the last block
	int best_fit_block	 = -1;
	size_t best_fit_size = SIZE_MAX;
	for (int i = 0; i < alloc->n_free_blocks - 1; i++) {
		free_block* block = &alloc->free_blocks[i];
		max_avail		  = MAX(max_avail, block->size);
		if (block->size >= size && block->size <= best_fit_size) {
			best_fit_block = i;
			best_fit_size  = block->size;
		}
	}

	if (best_fit_block == -1) {
		// the last block is our last resort
		free_block* block = &alloc->free_blocks[alloc->n_free_blocks - 1];
		max_avail		  = MAX(max_avail, block->size);
		if (block->size >= size) {
			best_fit_block = alloc->n_free_blocks - 1;
		} else {
			// this should never happen
			OIML_LOG_ERROR("%s: not enough space in the buffer to allocate %zu bytes, largest block available %zu bytes\n", __func__, size, max_avail);
			OIML_ABORT("not enough space in the buffer");
		}
	}

	free_block* block = &alloc->free_blocks[best_fit_block];
	size_t offset	  = block->offset;
	block->offset	  = offset + size;
	block->size -= size;
	if (block->size == 0) {
		// remove block if empty
		alloc->n_free_blocks--;
		for (int j = best_fit_block; j < alloc->n_free_blocks; j++) {
			alloc->free_blocks[j] = alloc->free_blocks[j + 1];
		}
	}

	AT_PRINTF("block %d, offset %zu\n", best_fit_block, offset);

#ifdef OIML_ALLOCATOR_DEBUG
	add_allocated_tensor(alloc, offset, tensor);
	size_t cur_max = offset + size;
	if (cur_max > alloc->max_size) {
		// sort allocated_tensors by offset
		for (int i = 0; i < 1024; i++) {
			for (int j = i + 1; j < 1024; j++) {
				if (alloc->allocated_tensors[i].offset > alloc->allocated_tensors[j].offset) {
					const oiml_tensor* tmp_tensor	   = alloc->allocated_tensors[i].tensor;
					size_t tmp_offset				   = alloc->allocated_tensors[i].offset;
					alloc->allocated_tensors[i].tensor = alloc->allocated_tensors[j].tensor;
					alloc->allocated_tensors[i].offset = alloc->allocated_tensors[j].offset;
					alloc->allocated_tensors[j].tensor = tmp_tensor;
					alloc->allocated_tensors[j].offset = tmp_offset;
				}
			}
		}
		OIML_LOG_DEBUG("max_size = %.2f MB: tensors: ", cur_max / 1024.0 / 1024.0);
		for (int i = 0; i < 1024; i++) {
			if (alloc->allocated_tensors[i].tensor) {
				OIML_LOG_DEBUG("%s [%zx-%zx] (%.2f MB) ", alloc->allocated_tensors[i].tensor->name, alloc->allocated_tensors[i].offset,
					alloc->allocated_tensors[i].offset + oiml_nbytes(alloc->allocated_tensors[i].tensor), oiml_nbytes(alloc->allocated_tensors[i].tensor) / 1024.0 / 1024.0);
			}
		}
		OIML_LOG_DEBUG("\n");
	}
#endif

	alloc->max_size = MAX(alloc->max_size, offset + size);

	return offset;

	OIML_UNUSED(tensor);
}

// this is a very naive implementation, but for our case the number of free blocks should be very small
OIML_INLINE static void oiml_dyn_tallocr_free_tensor(oiml_dyn_tallocr* alloc, size_t offset, size_t size, const oiml_tensor* tensor) {
	size = aligned_offset(NULL, size, alloc->alignment);

	AT_PRINTF("%s: freeing %s at %zu (%zu bytes) - n_free_blocks = %d\n", __func__, tensor->name, offset, size, alloc->n_free_blocks);

#ifdef OIML_ALLOCATOR_DEBUG
	remove_allocated_tensor(alloc, offset, tensor);
#endif

	// see if we can merge with an existing block
	for (int i = 0; i < alloc->n_free_blocks; i++) {
		free_block* block = &alloc->free_blocks[i];
		// check if ptr is at the end of the block
		if (block->offset + block->size == offset) {
			block->size += size;
			// check if we can merge with the next block
			if (i < alloc->n_free_blocks - 1 && block->offset + block->size == alloc->free_blocks[i + 1].offset) {
				block->size += alloc->free_blocks[i + 1].size;
				alloc->n_free_blocks--;
				for (int j = i + 1; j < alloc->n_free_blocks; j++) {
					alloc->free_blocks[j] = alloc->free_blocks[j + 1];
				}
			}
			return;
		}
		// check if ptr is at the beginning of the block
		if (offset + size == block->offset) {
			block->offset = offset;
			block->size += size;
			// check if we can merge with the previous block
			if (i > 0 && alloc->free_blocks[i - 1].offset + alloc->free_blocks[i - 1].size == block->offset) {
				alloc->free_blocks[i - 1].size += block->size;
				alloc->n_free_blocks--;
				for (int j = i; j < alloc->n_free_blocks; j++) {
					alloc->free_blocks[j] = alloc->free_blocks[j + 1];
				}
			}
			return;
		}
	}
	// otherwise, add a new block
	OIML_ASSERT(alloc->n_free_blocks < MAX_FREE_BLOCKS && "out of free blocks");
	// insert the new block in the correct position to keep the array sorted by address (to make merging blocks faster)
	int insert_pos = 0;
	while (insert_pos < alloc->n_free_blocks && alloc->free_blocks[insert_pos].offset < offset) {
		insert_pos++;
	}
	// shift all blocks from insert_pos onward to make room for the new block
	for (int i = alloc->n_free_blocks; i > insert_pos; i--) {
		alloc->free_blocks[i] = alloc->free_blocks[i - 1];
	}
	// insert the new block
	alloc->free_blocks[insert_pos].offset = offset;
	alloc->free_blocks[insert_pos].size	  = size;
	alloc->n_free_blocks++;

	OIML_UNUSED(tensor);
}

OIML_INLINE static void oiml_dyn_tallocr_reset(oiml_dyn_tallocr* alloc) {
	alloc->n_free_blocks		 = 1;
	alloc->free_blocks[0].offset = 0;
	alloc->free_blocks[0].size	 = SIZE_MAX / 2;// __restrict maximum size of a measure allocator to half size_t max to avoid overflows
	alloc->max_size				 = 0;

#ifdef OIML_ALLOCATOR_DEBUG
	for (int i = 0; i < 1024; i++) {
		alloc->allocated_tensors[i].tensor = NULL;
	}
#endif
}

OIML_INLINE static oiml_dyn_tallocr* oiml_dyn_tallocr_new(size_t alignment) {
	oiml_dyn_tallocr* alloc = ( oiml_dyn_tallocr* )malloc(sizeof(struct oiml_dyn_tallocr));

	*alloc = oiml_dyn_tallocr{
		/*.alignment     = */ alignment,
		/*.n_free_blocks = */ 0,
		/*.free_blocks   = */ { { 0 } },
		/*.max_size      = */ 0,
#ifdef OIML_ALLOCATOR_DEBUG
		/*.allocated_tensors = */ { { 0 } },
#endif
	};

	oiml_dyn_tallocr_reset(alloc);

	return alloc;
}

OIML_INLINE static void oiml_dyn_tallocr_free(oiml_dyn_tallocr* alloc) {
	free(alloc);
}

OIML_INLINE static size_t oiml_dyn_tallocr_max_size(oiml_dyn_tallocr* alloc) {
	return alloc->max_size;
}


/////////////////////////////////////

// graph allocator

struct hash_node {
	int n_children;
	int n_views;
	int buffer_id;
	size_t offset;// offset within the buffer
	bool allocated;
};

struct tensor_alloc {
	int buffer_id;
	size_t offset;
	size_t size_max;// 0 = pre-allocated, unused, or view
};

struct leaf_alloc {
	struct tensor_alloc leaf;
};

struct node_alloc {
	struct tensor_alloc dst;
	struct tensor_alloc src[OIML_MAX_SRC];
};

struct oiml_gallocr {
	oiml_backend_buffer_type_t* bufts;// [n_buffers]
	oiml_backend_buffer_t* buffers;// [n_buffers]
	oiml_dyn_tallocr** buf_tallocs;// [n_buffers]
	int n_buffers;

	struct oiml_hash_set hash_set;
	hash_node* hash_values;// [hash_set.size]

	node_alloc* node_allocs;// [n_nodes]
	int n_nodes;

	leaf_alloc* leaf_allocs;// [n_leafs]
	int n_leafs;
};

OIML_INLINE oiml_gallocr_t oiml_gallocr_new_n(oiml_backend_buffer_type_t* bufts, int n_bufs) {
	oiml_gallocr_t galloc = ( oiml_gallocr_t )calloc(1, sizeof(struct oiml_gallocr));
	OIML_ASSERT(galloc != NULL);

	galloc->bufts = static_cast<oiml_backend_buffer_type_t*>(calloc(n_bufs, sizeof(oiml_backend_buffer_type_t)));
	OIML_ASSERT(galloc->bufts != NULL);

	galloc->buffers = static_cast<oiml_backend_buffer_t*>(calloc(n_bufs, sizeof(oiml_backend_buffer_t)));
	OIML_ASSERT(galloc->buffers != NULL);

	galloc->buf_tallocs = static_cast<oiml_dyn_tallocr**>(calloc(n_bufs, sizeof(oiml_dyn_tallocr*)));
	OIML_ASSERT(galloc->buf_tallocs != NULL);

	for (int i = 0; i < n_bufs; i++) {
		galloc->bufts[i]   = bufts[i];
		galloc->buffers[i] = NULL;

		// check if the same buffer type is used multiple times and reuse the same allocator
		for (int j = 0; j < i; j++) {
			if (bufts[i] == bufts[j]) {
				galloc->buf_tallocs[i] = galloc->buf_tallocs[j];
				break;
			}
		}

		if (galloc->buf_tallocs[i] == NULL) {
			size_t alignment	   = oiml_backend_buft_get_alignment(bufts[i]);
			galloc->buf_tallocs[i] = oiml_dyn_tallocr_new(alignment);
		}
	}
	galloc->n_buffers = n_bufs;

	return galloc;
}

OIML_INLINE oiml_gallocr_t oiml_gallocr_new(oiml_backend_buffer_type_t buft) {
	return oiml_gallocr_new_n(&buft, 1);
}

OIML_INLINE void oiml_gallocr_free(oiml_gallocr_t galloc) {
	if (galloc == NULL) {
		return;
	}

	for (int i = 0; i < galloc->n_buffers; i++) {
		if (galloc->buffers != NULL) {
			// skip if already freed
			bool freed = false;
			for (int j = 0; j < i; j++) {
				if (galloc->buffers[j] == galloc->buffers[i]) {
					freed = true;
					break;
				}
			}
			if (!freed) {
				oiml_backend_buffer_free(galloc->buffers[i]);
			}
		}
		if (galloc->buf_tallocs != NULL) {
			// skip if already freed
			bool freed = false;
			for (int j = 0; j < i; j++) {
				if (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
					freed = true;
					break;
				}
			}
			if (!freed) {
				oiml_dyn_tallocr_free(galloc->buf_tallocs[i]);
			}
		}
	}

	oiml_hash_set_free(&galloc->hash_set);
	free(galloc->hash_values);
	free(galloc->bufts);
	free(galloc->buffers);
	free(galloc->buf_tallocs);
	free(galloc->node_allocs);
	free(galloc->leaf_allocs);
	free(galloc);
}

typedef oiml_gallocr* oiml_gallocr_t;

OIML_INLINE static hash_node* oiml_gallocr_hash_get(oiml_gallocr_t galloc, oiml_tensor* t) {
	size_t i = oiml_hash_find_or_insert(&galloc->hash_set, t);
	return &galloc->hash_values[i];
}

OIML_INLINE static bool oiml_gallocr_is_own(oiml_gallocr_t galloc, oiml_tensor* t) {
	return oiml_gallocr_hash_get(galloc, t)->allocated;
}

OIML_INLINE static bool oiml_gallocr_is_allocated(oiml_gallocr_t galloc, oiml_tensor* t) {
	return t->data != NULL || oiml_gallocr_hash_get(galloc, t)->allocated;
}

OIML_INLINE static void oiml_gallocr_allocate_node(oiml_gallocr_t galloc, oiml_tensor* node, int buffer_id) {
	OIML_ASSERT(buffer_id >= 0);
	hash_node* hn = oiml_gallocr_hash_get(galloc, node);

	if (!oiml_gallocr_is_allocated(galloc, node) && !oiml_is_view(node)) {
		hn->allocated = true;
		assert(hn->offset == 0);

		// try to reuse a parent's buffer (inplace)
		if (oiml_op_can_inplace(node->op)) {
			for (int i = 0; i < OIML_MAX_SRC; i++) {
				oiml_tensor* parent = node->src[i];
				if (parent == NULL) {
					continue;
				}

				// if the node's data is external, then we cannot re-use it
				if (!oiml_gallocr_is_own(galloc, parent)) {
					AT_PRINTF("not reusing parent %s for %s as %p is external\n", parent->name, node->name, parent->data);
					continue;
				}

				// outputs cannot be reused
				if (parent->flags & OIML_TENSOR_FLAG_OUTPUT || (parent->view_src != NULL && parent->view_src->flags & OIML_TENSOR_FLAG_OUTPUT)) {
					AT_PRINTF("not reusing parent %s for %s as it is an output\n", parent->name, node->name);
					continue;
				}

				if (!oiml_are_same_layout(node, parent)) {
					AT_PRINTF("not reusing parent %s for %s as layouts are different\n", parent->name, node->name);
					continue;
				}

				hash_node* p_hn = oiml_gallocr_hash_get(galloc, parent);
				if (p_hn->n_children == 1 && p_hn->n_views == 0) {
					if (oiml_is_view(parent)) {
						oiml_tensor* view_src  = parent->view_src;
						hash_node* view_src_hn = oiml_gallocr_hash_get(galloc, view_src);
						if (view_src_hn->n_views == 1 && view_src_hn->n_children == 0 && view_src->data == parent->data) {
							AT_PRINTF("reusing view parent %s (%s) for %s\n", parent->name, view_src->name, node->name);
							assert(view_src_hn->offset == p_hn->offset);
							hn->buffer_id		   = p_hn->buffer_id;
							hn->offset			   = p_hn->offset;
							p_hn->allocated		   = false;// avoid freeing the parent
							view_src_hn->allocated = false;
							return;
						}
					} else {
						AT_PRINTF("reusing parent %s for %s\n", parent->name, node->name);
						hn->buffer_id	= p_hn->buffer_id;
						hn->offset		= p_hn->offset;
						p_hn->allocated = false;// avoid freeing the parent
						return;
					}
				}
			}
		}
		// allocate tensor from the buffer
		oiml_dyn_tallocr* alloc			= galloc->buf_tallocs[buffer_id];
		oiml_backend_buffer_type_t buft = galloc->bufts[buffer_id];
		size_t size						= oiml_backend_buft_get_alloc_size(buft, node);
		size_t offset					= oiml_dyn_tallocr_alloc(alloc, size, node);
		hn->buffer_id					= buffer_id;
		hn->offset						= offset;
	}
}

OIML_INLINE static void oiml_gallocr_free_node(oiml_gallocr_t galloc, oiml_tensor* node) {
	// graph outputs are never freed
	if (node->flags & OIML_TENSOR_FLAG_OUTPUT) {
		AT_PRINTF("not freeing output %s\n", node->name);
		return;
	}

	hash_node* hn					= oiml_gallocr_hash_get(galloc, node);
	size_t offset					= hn->offset;
	int buffer_id					= hn->buffer_id;
	oiml_dyn_tallocr* alloc			= galloc->buf_tallocs[buffer_id];
	oiml_backend_buffer_type_t buft = galloc->bufts[buffer_id];
	size_t size						= oiml_backend_buft_get_alloc_size(buft, node);
	oiml_dyn_tallocr_free_tensor(alloc, offset, size, node);
	hn->allocated = false;
}

OIML_INLINE static int get_node_buffer_id(const int* node_buffer_ids, int i) {
	return node_buffer_ids ? node_buffer_ids[i] : 0;
}

OIML_INLINE static void oiml_gallocr_alloc_graph_impl(oiml_gallocr_t galloc, oiml_cgraph* graph, const int* node_buffer_ids, const int* leaf_buffer_ids) {
	// clear hash tables
	oiml_hash_set_reset(&galloc->hash_set);
	memset(galloc->hash_values, 0, sizeof(struct hash_node) * galloc->hash_set.size);

	// allocate leafs
	// these may be tensors that the application is not using in the graph, but may still want to allocate for other purposes
	for (int i = 0; i < graph->n_leafs; i++) {
		oiml_tensor* leaf = graph->leafs[i];
		oiml_gallocr_allocate_node(galloc, leaf, get_node_buffer_id(leaf_buffer_ids, i));
	}

	// count number of children and views
	// allocate other graph inputs and leafs first to avoid overwriting them
	for (int i = 0; i < graph->n_nodes; i++) {
		oiml_tensor* node = graph->nodes[i];

		// TODO: better way to add external dependencies
		// OIML_OP_NONE does not appear normally in the graph nodes, but is used by oiml-backend to add dependencies to
		// control when some tensors are allocated and freed. in this case, the dependencies are in `src`, but the node
		// itself is never used and should not be considered a dependency
		if (oiml_is_view(node) && node->op != OIML_OP_NONE) {
			oiml_tensor* view_src = node->view_src;
			oiml_gallocr_hash_get(galloc, view_src)->n_views += 1;
		}

		if (node->flags & OIML_TENSOR_FLAG_INPUT) {
			oiml_gallocr_allocate_node(galloc, graph->nodes[i], get_node_buffer_id(node_buffer_ids, i));
		}

		for (int j = 0; j < OIML_MAX_SRC; j++) {
			oiml_tensor* src = node->src[j];
			if (src == NULL) {
				continue;
			}

			oiml_gallocr_hash_get(galloc, src)->n_children += 1;

			// allocate explicit inputs
			if (src->flags & OIML_TENSOR_FLAG_INPUT) {
				oiml_gallocr_allocate_node(galloc, src, get_node_buffer_id(node_buffer_ids, i));
			}
		}
	}

	// allocate tensors
	for (int i = 0; i < graph->n_nodes; i++) {
		oiml_tensor* node = graph->nodes[i];
		int buffer_id	  = get_node_buffer_id(node_buffer_ids, i);

		// allocate parents (only leafs need to be allocated at this point)
		for (int j = 0; j < OIML_MAX_SRC; j++) {
			oiml_tensor* parent = node->src[j];
			if (parent == NULL) {
				continue;
			}
			oiml_gallocr_allocate_node(galloc, parent, buffer_id);
		}

		// allocate node
		oiml_gallocr_allocate_node(galloc, node, buffer_id);

		AT_PRINTF("exec: %s (%s) <= ", oiml_op_desc(node), node->name);
		for (int j = 0; j < OIML_MAX_SRC; j++) {
			oiml_tensor* parent = node->src[j];
			if (parent == NULL) {
				continue;
			}
			AT_PRINTF("%s", parent->name);
			if (j < OIML_MAX_SRC - 1 && node->src[j + 1] != NULL) {
				AT_PRINTF(", ");
			}
		}
		AT_PRINTF("\n");

		// update parents
		for (int j = 0; j < OIML_MAX_SRC; j++) {
			oiml_tensor* parent = node->src[j];
			if (parent == NULL) {
				continue;
			}
			hash_node* p_hn = oiml_gallocr_hash_get(galloc, parent);
			p_hn->n_children -= 1;

			AT_PRINTF("parent %s: %d children, %d views, allocated: %d\n", parent->name, p_hn->n_children, p_hn->n_views, p_hn->allocated);

			if (p_hn->n_children == 0 && p_hn->n_views == 0) {
				if (oiml_is_view(parent)) {
					oiml_tensor* view_src  = parent->view_src;
					hash_node* view_src_hn = oiml_gallocr_hash_get(galloc, view_src);
					view_src_hn->n_views -= 1;
					AT_PRINTF("view_src %s: %d children, %d views\n", view_src->name, view_src_hn->n_children, view_src_hn->n_views);
					if (view_src_hn->n_views == 0 && view_src_hn->n_children == 0 && view_src_hn->allocated) {
						oiml_gallocr_free_node(galloc, view_src);
					}
				} else if (p_hn->allocated) {
					oiml_gallocr_free_node(galloc, parent);
				}
			}
			AT_PRINTF("\n");
		}
	}
}

OIML_INLINE bool oiml_gallocr_reserve_n(oiml_gallocr_t galloc, oiml_cgraph* graph, const int* node_buffer_ids, const int* leaf_buffer_ids) {
	size_t min_hash_size = graph->n_nodes + graph->n_leafs;
	// add 25% margin to avoid hash collisions
	min_hash_size += min_hash_size / 4;

	// initialize hash table
	if (galloc->hash_set.size < min_hash_size) {
		oiml_hash_set_free(&galloc->hash_set);
		galloc->hash_set = oiml_hash_set_new(min_hash_size);
		OIML_ASSERT(galloc->hash_set.keys != NULL);

		free(galloc->hash_values);
		galloc->hash_values = static_cast<hash_node*>(malloc(sizeof(hash_node) * galloc->hash_set.size));
		OIML_ASSERT(galloc->hash_values != NULL);
	}

	// reset allocators
	for (int i = 0; i < galloc->n_buffers; i++) {
		oiml_dyn_tallocr_reset(galloc->buf_tallocs[i]);
	}

	// allocate in hash table
	oiml_gallocr_alloc_graph_impl(galloc, graph, node_buffer_ids, leaf_buffer_ids);

	// set the node_allocs from the hash table
	if (galloc->n_nodes < graph->n_nodes) {
		free(galloc->node_allocs);
		galloc->node_allocs = static_cast<node_alloc*>(calloc(graph->n_nodes, sizeof(node_alloc)));
		OIML_ASSERT(galloc->node_allocs != NULL);
	}
	galloc->n_nodes = graph->n_nodes;
	for (int i = 0; i < graph->n_nodes; i++) {
		oiml_tensor* node	   = graph->nodes[i];
		node_alloc* node_alloc = &galloc->node_allocs[i];
		if (node->view_src || node->data) {
			node_alloc->dst.buffer_id = -1;
			node_alloc->dst.offset	  = SIZE_MAX;
			node_alloc->dst.size_max  = 0;
		} else {
			hash_node* hn			  = oiml_gallocr_hash_get(galloc, node);
			node_alloc->dst.buffer_id = hn->buffer_id;
			node_alloc->dst.offset	  = hn->offset;
			node_alloc->dst.size_max  = oiml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], node);
		}
		for (int j = 0; j < OIML_MAX_SRC; j++) {
			oiml_tensor* src = node->src[j];
			if (!src || src->view_src || src->data) {
				node_alloc->src[j].buffer_id = -1;
				node_alloc->src[j].offset	 = SIZE_MAX;
				node_alloc->src[j].size_max	 = 0;
			} else {
				hash_node* hn				 = oiml_gallocr_hash_get(galloc, src);
				node_alloc->src[j].buffer_id = hn->buffer_id;
				node_alloc->src[j].offset	 = hn->offset;
				node_alloc->src[j].size_max	 = oiml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], src);
			}
		}
	}
	if (galloc->n_leafs < graph->n_leafs) {
		free(galloc->leaf_allocs);
		galloc->leaf_allocs = static_cast<leaf_alloc*>(calloc(graph->n_leafs, sizeof(galloc->leaf_allocs[0])));
		OIML_ASSERT(galloc->leaf_allocs != NULL);
	}
	galloc->n_leafs = graph->n_leafs;
	for (int i = 0; i < graph->n_leafs; i++) {
		oiml_tensor* leaf = graph->leafs[i];
		hash_node* hn	  = oiml_gallocr_hash_get(galloc, leaf);
		if (leaf->view_src || leaf->data) {
			galloc->leaf_allocs[i].leaf.buffer_id = -1;
			galloc->leaf_allocs[i].leaf.offset	  = SIZE_MAX;
			galloc->leaf_allocs[i].leaf.size_max  = 0;
		} else {
			galloc->leaf_allocs[i].leaf.buffer_id = hn->buffer_id;
			galloc->leaf_allocs[i].leaf.offset	  = hn->offset;
			galloc->leaf_allocs[i].leaf.size_max  = oiml_backend_buft_get_alloc_size(galloc->bufts[hn->buffer_id], leaf);
		}
	}

	// reallocate buffers if needed
	for (int i = 0; i < galloc->n_buffers; i++) {
		// if the buffer type is used multiple times, we reuse the same buffer
		for (int j = 0; j < i; j++) {
			if (galloc->buf_tallocs[j] == galloc->buf_tallocs[i]) {
				galloc->buffers[i] = galloc->buffers[j];
				break;
			}
		}

		size_t cur_size = galloc->buffers[i] ? oiml_backend_buffer_get_size(galloc->buffers[i]) : 0;
		size_t new_size = oiml_dyn_tallocr_max_size(galloc->buf_tallocs[i]);

		// even if there are no tensors allocated in this buffer, we still need to allocate it to initialize views
		if (new_size > cur_size || galloc->buffers[i] == NULL) {
#ifndef NDEBUG
			OIML_LOG_DEBUG("%s: reallocating %s buffer from size %.02f MiB to %.02f MiB\n", __func__, oiml_backend_buft_name(galloc->bufts[i]), cur_size / 1024.0 / 1024.0,
				new_size / 1024.0 / 1024.0);
#endif

			oiml_backend_buffer_free(galloc->buffers[i]);
			galloc->buffers[i] = oiml_backend_buft_alloc_buffer(galloc->bufts[i], new_size);
			if (galloc->buffers[i] == NULL) {
				OIML_LOG_ERROR("%s: failed to allocate %s buffer of size %zu\n", __func__, oiml_backend_buft_name(galloc->bufts[i]), new_size);
				return false;
			}
			oiml_backend_buffer_set_usage(galloc->buffers[i], OIML_BACKEND_BUFFER_USAGE_COMPUTE);
		}
	}

	return true;
}

OIML_INLINE bool oiml_gallocr_reserve(oiml_gallocr_t galloc, oiml_cgraph* graph) {
	return oiml_gallocr_reserve_n(galloc, graph, NULL, NULL);
}

OIML_INLINE static void oiml_gallocr_init_tensor(oiml_gallocr_t galloc, oiml_tensor* tensor, tensor_alloc* tensor_alloc) {
	int buffer_id = tensor_alloc->buffer_id;
	assert(tensor->data || tensor->view_src || oiml_backend_buffer_get_alloc_size(galloc->buffers[buffer_id], tensor) <= tensor_alloc->size_max);

	if (tensor->view_src != NULL) {
		if (tensor->buffer == NULL) {
			assert(tensor_alloc->offset == SIZE_MAX);
			if (tensor->view_src->buffer == NULL) {
				// this tensor was allocated without oiml-backend
				return;
			}
			oiml_backend_view_init(tensor);
		}
	} else {
		if (tensor->data == NULL) {
			assert(tensor_alloc->offset != SIZE_MAX);
			assert(oiml_backend_buffer_get_alloc_size(galloc->buffers[buffer_id], tensor) <= tensor_alloc->size_max);
			void* base = oiml_backend_buffer_get_base(galloc->buffers[buffer_id]);
			void* addr = ( char* )base + tensor_alloc->offset;
			oiml_backend_tensor_alloc(galloc->buffers[buffer_id], tensor, addr);
		} else {
			if (tensor->buffer == NULL) {
				// this tensor was allocated without oiml-backend
				return;
			}
		}
	}
}

OIML_INLINE static bool oiml_gallocr_node_needs_realloc(oiml_gallocr_t galloc, oiml_tensor* node, tensor_alloc* talloc) {
	size_t node_size = 0;
	if (!node->data && !node->view_src) {
		OIML_ASSERT(talloc->buffer_id >= 0);// prevent segfault when misusing the API
		node_size = oiml_backend_buft_get_alloc_size(galloc->bufts[talloc->buffer_id], node);
	}
	return talloc->size_max >= node_size;
}

OIML_INLINE static bool oiml_gallocr_needs_realloc(oiml_gallocr_t galloc, oiml_cgraph* graph) {
	if (galloc->n_nodes != graph->n_nodes) {
#ifndef NDEBUG
		OIML_LOG_DEBUG("%s: graph has different number of nodes\n", __func__);
#endif
		return true;
	}

	if (galloc->n_leafs != graph->n_leafs) {
#ifndef NDEBUG
		OIML_LOG_DEBUG("%s: graph has different number of leafs\n", __func__);
#endif
		return true;
	}

	for (int i = 0; i < graph->n_nodes; i++) {
		oiml_tensor* node	   = graph->nodes[i];
		node_alloc* node_alloc = &galloc->node_allocs[i];

		if (!oiml_gallocr_node_needs_realloc(galloc, node, &node_alloc->dst)) {
#ifndef NDEBUG
			OIML_LOG_DEBUG("%s: node %s is not valid\n", __func__, node->name);
#endif
			return true;
		}

		for (int j = 0; j < OIML_MAX_SRC; j++) {
			oiml_tensor* src = node->src[j];
			if (src == NULL) {
				continue;
			}
			if (!oiml_gallocr_node_needs_realloc(galloc, src, &node_alloc->src[j])) {
#ifndef NDEBUG
				OIML_LOG_DEBUG("%s: src %d (%s) of node %s is not valid\n", __func__, j, src->name, node->name);
#endif
				return true;
			}
		}
	}

	return false;
}

OIML_INLINE bool oiml_gallocr_alloc_graph(oiml_gallocr_t galloc, oiml_cgraph* graph) {
	if (oiml_gallocr_needs_realloc(galloc, graph)) {
		if (galloc->n_buffers == 1) {
#ifndef NDEBUG
			OIML_LOG_DEBUG("%s: reallocating buffers automatically\n", __func__);
#endif
			if (!oiml_gallocr_reserve(galloc, graph)) {
				return false;
			}
		} else {
#ifndef NDEBUG
			OIML_LOG_DEBUG("%s: cannot reallocate multi buffer graph automatically, call reserve\n", __func__);
#endif
			return false;
		}
	}

	// reset buffers
	for (int i = 0; i < galloc->n_buffers; i++) {
		if (galloc->buffers[i] != NULL) {
			oiml_backend_buffer_reset(galloc->buffers[i]);
		}
	}

	// allocate the graph tensors from the previous assignments
	// leafs
	for (int i = 0; i < graph->n_leafs; i++) {
		oiml_tensor* leaf	   = graph->leafs[i];
		leaf_alloc* leaf_alloc = &galloc->leaf_allocs[i];
		oiml_gallocr_init_tensor(galloc, leaf, &leaf_alloc->leaf);
	}
	// nodes
	for (int i = 0; i < graph->n_nodes; i++) {
		oiml_tensor* node	   = graph->nodes[i];
		node_alloc* node_alloc = &galloc->node_allocs[i];
		for (int j = 0; j < OIML_MAX_SRC; j++) {
			oiml_tensor* src = node->src[j];
			if (src == NULL) {
				continue;
			}
			oiml_gallocr_init_tensor(galloc, src, &node_alloc->src[j]);
		}
		oiml_gallocr_init_tensor(galloc, node, &node_alloc->dst);
	}

	return true;
}

OIML_INLINE size_t oiml_gallocr_get_buffer_size(oiml_gallocr_t galloc, int buffer_id) {
	OIML_ASSERT(buffer_id >= 0 && buffer_id < galloc->n_buffers);

	if (galloc->buffers[buffer_id] == NULL) {
		return 0;
	}

	for (int i = 0; i < buffer_id; i++) {
		if (galloc->buffers[i] == galloc->buffers[buffer_id]) {
			// this buffer is the same as a previous one due to the same buffer type being used multiple times
			// only return the buffer size the first time it appears to avoid double counting
			return 0;
		}
	}

	return oiml_backend_buffer_get_size(galloc->buffers[buffer_id]);
}

// utils

OIML_INLINE static bool alloc_tensor_range(oiml_context* ctx, oiml_tensor* first, oiml_tensor* last, oiml_backend_buffer_type_t buft, size_t size, oiml_backend_buffer_t** buffers,
	size_t* n_buffers) {
	oiml_backend_buffer_t buffer = oiml_backend_buft_alloc_buffer(buft, size);
	if (buffer == NULL) {
#ifndef NDEBUG
		OIML_LOG_DEBUG("%s: failed to allocate %s buffer of size %zu\n", __func__, oiml_backend_buft_name(buft), size);
#endif
		for (size_t i = 0; i < *n_buffers; i++) {
			oiml_backend_buffer_free((*buffers)[i]);
		}
		free(*buffers);
		return false;
	}

	struct oiml_tallocr tallocr = oiml_tallocr_new(buffer);

	for (oiml_tensor* t = first; t != last; t = oiml_get_next_tensor(ctx, t)) {
		if (t->data == NULL) {
			if (t->view_src == NULL) {
				oiml_tallocr_alloc(&tallocr, t);
			} else if (t->buffer == NULL) {
				oiml_backend_view_init(t);
			}
		} else {
			if (t->view_src != NULL && t->buffer == NULL) {
				// view of a pre-allocated tensor
				oiml_backend_view_init(t);
			}
		}
	}

	*buffers				   = static_cast<oiml_backend_buffer_t*>(realloc(*buffers, sizeof(oiml_backend_buffer_t) * (*n_buffers + 1)));
	(*buffers)[(*n_buffers)++] = buffer;

	return true;
}

OIML_INLINE oiml_backend_buffer_t oiml_backend_alloc_ctx_tensors_from_buft(oiml_context* ctx, oiml_backend_buffer_type_t buft) {
	OIML_ASSERT(oiml_get_no_alloc(ctx) == true);

	size_t alignment = oiml_backend_buft_get_alignment(buft);
	size_t max_size	 = oiml_backend_buft_get_max_size(buft);

	oiml_backend_buffer_t* buffers = NULL;
	size_t n_buffers			   = 0;

	size_t cur_buf_size = 0;
	oiml_tensor* first	= oiml_get_first_tensor(ctx);
	for (oiml_tensor* t = first; t != NULL; t = oiml_get_next_tensor(ctx, t)) {
		size_t this_size = 0;
		if (t->data == NULL && t->view_src == NULL) {
			this_size = OIML_PAD(oiml_backend_buft_get_alloc_size(buft, t), alignment);
		}

		if (this_size > max_size) {
			OIML_LOG_ERROR("%s: tensor %s is too large to fit in a %s buffer (tensor size: %zu, max buffer size: %zu)\n", __func__, t->name, oiml_backend_buft_name(buft),
				this_size, max_size);
			for (size_t i = 0; i < n_buffers; i++) {
				oiml_backend_buffer_free(buffers[i]);
			}
			free(buffers);
			return NULL;
		}

		if ((cur_buf_size + this_size) > max_size) {
			// allocate tensors in the current buffer
			if (!alloc_tensor_range(ctx, first, t, buft, cur_buf_size, &buffers, &n_buffers)) {
				return NULL;
			}
			first		 = t;
			cur_buf_size = this_size;
		} else {
			cur_buf_size += this_size;
		}
	}

	// allocate remaining tensors
	if (cur_buf_size > 0) {
		if (!alloc_tensor_range(ctx, first, NULL, buft, cur_buf_size, &buffers, &n_buffers)) {
			return NULL;
		}
	}

	if (n_buffers == 0) {
#ifndef NDEBUG
		OIML_LOG_DEBUG("%s: all tensors in the context are already allocated\n", __func__);
#endif
		return NULL;
	}

	oiml_backend_buffer_t buffer;
	if (n_buffers == 1) {
		buffer = buffers[0];
	} else {
		buffer = oiml_backend_multi_buffer_alloc_buffer(buffers, n_buffers);
	}
	free(buffers);
	return buffer;
}

OIML_INLINE oiml_backend_buffer_t oiml_backend_alloc_ctx_tensors(oiml_context* ctx, oiml_backend_t backend) {
	return oiml_backend_alloc_ctx_tensors_from_buft(ctx, oiml_backend_get_default_buffer_type(backend));
}
