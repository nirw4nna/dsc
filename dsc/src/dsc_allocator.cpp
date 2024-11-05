// Copyright (c) 2024, Christian Gilli <christian.gilli@dspcraft.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#include "dsc_allocator.h"
#include "dsc_tracing.h"

// ============================================================
// Utilities

DSC_MALLOC void *dsc_obj_alloc(dsc_allocator *allocator,
                               const usize nb,
                               const usize alignment) noexcept {
    DSC_TRACE_OBJ_ALLOC(nb, allocator->type);
    return allocator->alloc(allocator->buf, nb, alignment);
}

void dsc_obj_free(dsc_allocator *allocator, void *ptr) noexcept {
    DSC_TRACE_OBJ_FREE(allocator->type, (uintptr_t) ptr);
    allocator->free(allocator->buf, ptr);
}

void dsc_clear_buffer(dsc_allocator *allocator) noexcept {
    allocator->clear_buffer(allocator->buf);
}

usize dsc_buffer_used_mem(dsc_allocator *allocator) noexcept {
    return allocator->used_memory(allocator->buf);
}

// ============================================================
// General Purpose Allocator API

#define dsc_generic_buffer(PTR) (dsc_generic_buf *) (PTR + 1)

struct dsc_generic_node {
    usize size;
    void *padding;
};

struct dsc_generic_free_node {
    dsc_generic_free_node *next;
    usize size;
};

struct dsc_generic_buf {
    usize used_mem;
    dsc_generic_free_node *head;
};

static DSC_INLINE dsc_generic_free_node *generic_find_best(dsc_generic_buf *gb,
                                                           const usize required_size,
                                                           dsc_generic_free_node **prev) noexcept {
    dsc_generic_free_node *node = gb->head;
    dsc_generic_free_node *best = node->size >= required_size ? node : nullptr;
    dsc_generic_free_node *prev_node = nullptr;
    while (node->next != nullptr) {
        if ((node->next->size >= required_size) &&
            (best == nullptr || best->size >= node->next->size)) {
            prev_node = node;
            best = node->next;
        }
        node = node->next;
    }
    
    *prev = prev_node;
    
    return best;
}

static DSC_INLINE void generic_list_insert(dsc_generic_free_node **head,
                                           dsc_generic_free_node *prev,
                                           dsc_generic_free_node *to_insert) noexcept {
    if (prev == nullptr) {
        if (*head != nullptr) {
            to_insert->next = *head;
        }
        *head = to_insert;
    } else {
        if (prev->next == nullptr) {
            prev->next = to_insert;
            to_insert->next = nullptr;
        } else {
            to_insert->next = prev->next;
            prev->next = to_insert;
        }
    }
}

static DSC_INLINE void generic_list_remove(dsc_generic_free_node **head,
                                           dsc_generic_free_node *prev,
                                           dsc_generic_free_node *to_remove) noexcept {
    if (prev == nullptr) {
        *head = to_remove->next;
    } else {
        prev->next = to_remove->next;
    }
}

static DSC_MALLOC void *generic_alloc(dsc_buffer *buf,
                                      const usize nb,
                                      const usize alignment) noexcept {
    DSC_ASSERT(buf != nullptr);
    DSC_ASSERT(nb > 0);

    dsc_generic_buf *gb = dsc_generic_buffer(buf);

    const usize required_size = DSC_ALIGN(nb + sizeof(dsc_generic_node), alignment);

    dsc_generic_free_node *prev = nullptr;
    dsc_generic_free_node *node = generic_find_best(gb, required_size, &prev);
    if (node == nullptr) {
        DSC_LOG_FATAL("error allocating %.2fKB", DSC_B_TO_KB(required_size));
    }

    // node->size is always at least equal to required_size otherwise generic_find_best won't return
    usize left = node->size - required_size;
    // It doesn't make sense to add a free node with a size less than the header.
    // Not only that, allowing such nodes could lead to serious bugs like double-frees and memory leaks.
    if (left > sizeof(dsc_generic_free_node)) {
        dsc_generic_free_node *new_node = (dsc_generic_free_node *) ((byte *) node + required_size);
        new_node->size = left;
        generic_list_insert(&gb->head, node, new_node);
    }

    generic_list_remove(&gb->head, prev, node);

    dsc_generic_node *obj = (dsc_generic_node *) node;
    obj->size = required_size;
    gb->used_mem += required_size;
    return (void *) (obj + 1);
}

static void generic_clear(dsc_buffer *buf) noexcept {
    // For now assume that this is a NOP when using a general purpose allocator
    DSC_UNUSED(buf);
}

static void generic_free(dsc_buffer *buf, void *ptr) noexcept {
    DSC_ASSERT(buf != nullptr);
    DSC_ASSERT(ptr != nullptr);

    dsc_generic_buf *gb = dsc_generic_buffer(buf);

    const uintptr_t ptr_addr = (uintptr_t) ((byte *) ptr - sizeof(dsc_generic_node));

    dsc_generic_node *obj = (dsc_generic_node *) ((byte *) ptr - sizeof(dsc_generic_node));
    dsc_generic_free_node *new_node = (dsc_generic_free_node *) obj;

    dsc_generic_free_node *node = gb->head, *prev = nullptr;
    bool already_freed = true;
    while (node != nullptr) {
        const uintptr_t free_range_start = (uintptr_t) ((byte *) node);
        const uintptr_t free_range_stop = (uintptr_t) ((byte *) node + node->size);

        if (ptr_addr >= free_range_start && ptr_addr < free_range_stop) {
            // The idea here is that if we are trying to free an object that has already
            // been freed we will get a pointer to a range that is part of the free list.
            // It that's the case simple return without updating anything.
            // (This WILL happen if we rely on __del__ in Python to take care of freeing
            // DSC objects).
            already_freed = true;
            break;
        }

        if (ptr < node) {
            already_freed = false;
            new_node->size = obj->size;
            new_node->next = nullptr;
            generic_list_insert(&gb->head, prev, new_node);
            break;
        }

        prev = node;
        node = node->next;
    }

    if (already_freed) {
        DSC_LOG_DEBUG("careful, you are trying to free %p multiple times!", ptr);
        return;
    }

    gb->used_mem -= new_node->size;

    // Coalescence
    if ((new_node->next != nullptr) &&
        (void *) ((byte *) new_node + new_node->size) == new_node->next) {
        new_node->size += new_node->next->size;
        generic_list_remove(&gb->head, new_node, new_node->next);
    }

    if ((prev != nullptr && prev->next != nullptr) &&
        (void *) ((byte *) prev + prev->size) == new_node) {
        prev->size += new_node->size;
        generic_list_remove(&gb->head, prev, new_node);
    }
}

static usize generic_used_memory(dsc_buffer *buf) noexcept {
    dsc_generic_buf *gb = dsc_generic_buffer(buf);
    return gb->used_mem;
}

dsc_allocator *dsc_generic_allocator(dsc_buffer *buf) noexcept {
    // Initialize the general purpose allocator
    dsc_generic_buf *gb = dsc_generic_buffer(buf);
    gb->used_mem = 0;
    dsc_generic_free_node *first = (dsc_generic_free_node *) (gb + 1);
    first->next = nullptr;
    first->size = buf->size - sizeof(dsc_generic_buf);
    gb->head = first;
    static dsc_allocator generic = {
        /* .buf             = */ buf,
        /* .type            = */ dsc_allocator_type::GENERAL_PURPOSE,
        /* .alloc           = */ generic_alloc,
        /* .clear_buffer    = */ generic_clear,
        /* .free            = */ generic_free,
        /* .used_memory     = */ generic_used_memory,
    };
    return &generic;
}

// ============================================================
// Linear (Arena) Allocator API

#define dsc_linear_buffer(PTR) (dsc_linear_buf *) (PTR + 1)

struct dsc_obj {
    usize offset;
    usize size;
};

struct dsc_linear_buf {
    dsc_obj *last;
    int n_objs;
};

static DSC_MALLOC void *linear_alloc(dsc_buffer *buf,
                                     usize nb,
                                     const usize alignment) noexcept {
    nb = DSC_ALIGN(nb, alignment);

    dsc_linear_buf *lb = dsc_linear_buffer(buf);

    const usize last_offset = lb->last == nullptr ? 0 : lb->last->offset;
    const usize last_size = lb->last == nullptr ? 0 : lb->last->size;
    const usize last_end = last_offset + last_size;
    
    if (nb + sizeof(dsc_obj) + last_end > buf->size) {
        DSC_LOG_FATAL("can't allocate %.2fKB", DSC_B_TO_KB(nb));
    }

    // The actual buffer starts after the 'header' of the arena struct.
    dsc_obj *new_obj = (dsc_obj *) ((byte *) lb + last_end + sizeof(dsc_linear_buf));
    // The offset refers to the actual offset of the "contained" object which comes after
    // the dsc_object "header".
    new_obj->offset = last_end + sizeof(dsc_obj);
    new_obj->size = nb;

    lb->n_objs++;
    lb->last = new_obj;

    return (void *) ((byte *) lb + sizeof(dsc_obj) + lb->last->offset);
}

static void linear_clear(dsc_buffer *buf) noexcept {
    dsc_linear_buf *lb = dsc_linear_buffer(buf);

    DSC_LOG_DEBUG("clearing buffer %p of %ldMB n_objs=%d",
                  (void *) buf,
                  (usize) DSC_B_TO_MB(buf->size),
                  lb->n_objs
    );
    lb->last = nullptr;
    lb->n_objs = 0;
}

static void linear_free(dsc_buffer *buf,
                        void *ptr) noexcept {
    // NOP
    DSC_UNUSED(buf);
    DSC_UNUSED(ptr);
}

static usize linear_used_memory(dsc_buffer *buf) noexcept {
    dsc_linear_buf *lb = dsc_linear_buffer(buf);
    return lb->last == nullptr ? 0 : lb->last->offset + lb->last->size;
}

dsc_allocator *dsc_linear_allocator(dsc_buffer *buf) noexcept {
    // Initialize the linear buffer
    dsc_linear_buf *lb = dsc_linear_buffer(buf);
    lb->n_objs = 0;
    lb->last = nullptr;
    static dsc_allocator linear = {
        /* .buf             = */ buf,
        /* .type            = */ dsc_allocator_type::LINEAR,
        /* .alloc           = */ linear_alloc,
        /* .clear_buffer    = */ linear_clear,
        /* .free            = */ linear_free,
        /* .used_memory     = */ linear_used_memory,
    };
    return &linear;
}
