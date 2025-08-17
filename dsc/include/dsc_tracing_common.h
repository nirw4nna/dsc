// Copyright (c) 2024-2025, Christian Gilli <christian.gilli11@gmail.com>
// All rights reserved.
//
// This code is licensed under the terms of the 3-clause BSD license
// (https://opensource.org/license/bsd-3-clause).

#pragma once

#include "dsc.h"

#if defined(DSC_TRACING)

#include "dsc_device.h"
#include <cstdlib>
#include <ctime>        // timespec
#include <cinttypes>    // PRIxPTR
#include <cstring>
#include <cmath>        // log


#define DSC_TRACE_SET_TENSOR(X, field)                                                         \
    args__.field.n_dim = (X)->n_dim;                                                           \
    args__.field.ne = (X)->ne;                                                                 \
    memcpy(args__.field.shape, &dsc_tensor_get_dim((X), 0), (X)->n_dim * sizeof(*(X)->shape)); \
    args__.field.dtype = (X)->dtype;                                                           \
    args__.field.device = (X)->device;                                                         \
    args__.field.addr = (uintptr_t) (X)

#define DSC_TRACE_TENSOR_NEW(DEV, shape_, n_dim_, dtype_, device_, lazy_, data_, data_device_, ...) \
    dsc_tensor_alloc_args args__{};                                                                 \
    memcpy(&args__.x.shape, (shape_), (n_dim_) * sizeof(*(shape_)));                                \
    args__.x.n_dim = (n_dim_);                                                                      \
    args__.x.dtype = (dtype_);                                                                      \
    args__.x.device = (device_);                                                                    \
    args__.x.addr = 0;                                                                              \
    args__.data = (data_);                                                                          \
    args__.data_device = (data_device_);                                                            \
    args__.lazy = (lazy_);                                                                          \
    DSC_INSERT_TYPED_TRACE((DEV), dsc_tensor_alloc_args, DSC_TENSOR_ALLOC, ##__VA_ARGS__)

#define DSC_TRACE_TENSOR_FREE(DEV, X, ...) \
    dsc_tensor_free_args args__{};         \
    DSC_TRACE_SET_TENSOR(X, x);            \
    DSC_INSERT_TYPED_TRACE((DEV), dsc_tensor_free_args, DSC_TENSOR_FREE, ##__VA_ARGS__)

#define DSC_TRACE_CAST_OP(DEV, X, OUT, ...) \
    dsc_cast_args args__{};                 \
    DSC_TRACE_SET_TENSOR(X, x);             \
    args__.new_dtype = (OUT)->dtype;        \
    DSC_INSERT_TYPED_TRACE((DEV), dsc_cast_args, DSC_CAST_OP, ##__VA_ARGS__)

#define DSC_TRACE_BINARY_OP(DEV, XA, XB, OUT, ...) \
    dsc_binary_args args__{};                      \
    DSC_TRACE_SET_TENSOR(XA, xa);                  \
    DSC_TRACE_SET_TENSOR(XB, xb);                  \
    DSC_TRACE_SET_TENSOR(OUT, out);                \
    DSC_INSERT_TYPED_TRACE((DEV), dsc_binary_args, DSC_BINARY_OP, ##__VA_ARGS__)

#define DSC_TRACE_UNARY_OP(DEV, X, OUT, ...) \
    dsc_unary_args args__{};                 \
    DSC_TRACE_SET_TENSOR(X, x);              \
    DSC_TRACE_SET_TENSOR(OUT, out);          \
    DSC_INSERT_TYPED_TRACE((DEV), dsc_unary_args, DSC_UNARY_OP, ##__VA_ARGS__)

#define DSC_TRACE_UNARY_AXIS_OP(DEV, X, OUT, axis_, ...) \
    dsc_unary_axis_args args__{};                        \
    DSC_TRACE_SET_TENSOR(X, x);                          \
    DSC_TRACE_SET_TENSOR(OUT, out);                      \
    args__.axis = (axis_);                               \
    DSC_INSERT_TYPED_TRACE((DEV), dsc_unary_axis_args, DSC_UNARY_AXIS_OP, ##__VA_ARGS__)

#define DSC_TRACE_MATMUL_OP(DEV, XA, XB, trans_b_, OUT, is_gevm_, ...) \
    dsc_matmul_args args__{};                                          \
    DSC_TRACE_SET_TENSOR(XA, xa);                                      \
    DSC_TRACE_SET_TENSOR(XB, xb);                                      \
    DSC_TRACE_SET_TENSOR(OUT, out);                                    \
    args__.trans_b = (trans_b_);                                       \
    DSC_INSERT_NAMED_TRACE((DEV), dsc_matmul_args, DSC_MATMUL_OP, ((trans_b_) && (is_gevm_)) ? "dsc_gevm" : "dsc_gemm", ##__VA_ARGS__)

#define DSC_TRACE_MASK_OP(DEV, X, MASK, value_, ...) \
    dsc_mask_args args__{};                          \
    DSC_TRACE_SET_TENSOR(X, x);                      \
    DSC_TRACE_SET_TENSOR(MASK, mask);                \
    args__.value = (value_);                         \
    DSC_INSERT_TYPED_TRACE((DEV), dsc_mask_args, DSC_MASK_OP, ##__VA_ARGS__)

#define DSC_TRACE_OUTER_OP(DEV, XA, XB, OUT, ...) \
    dsc_outer_args args__{};                      \
    DSC_TRACE_SET_TENSOR(XA, xa);                 \
    DSC_TRACE_SET_TENSOR(XB, xb);                 \
    DSC_TRACE_SET_TENSOR(OUT, out);               \
    DSC_INSERT_TYPED_TRACE((DEV), dsc_outer_args, DSC_OUTER_OP, ##__VA_ARGS__)

#define DSC_TRACE_WHERE_OP(DEV, CONDITION, INPUT, OTHER, OUT, ...) \
    dsc_where_args args__{};                                       \
    DSC_TRACE_SET_TENSOR(CONDITION, condition);                    \
    DSC_TRACE_SET_TENSOR(INPUT, input);                            \
    DSC_TRACE_SET_TENSOR(OTHER, other);                            \
    DSC_TRACE_SET_TENSOR(OUT, out);                                \
    DSC_INSERT_TYPED_TRACE((DEV), dsc_where_args, DSC_WHERE_OP, ##__VA_ARGS__)

#define DSC_TRACE_GET_SLICE(DEV, X, OUT, slices_, n_slices_, ...)       \
    dsc_get_slice_args args__{};                                        \
    DSC_TRACE_SET_TENSOR(X, x);                                         \
    DSC_TRACE_SET_TENSOR(OUT, out);                                     \
    memcpy(args__.slices, (slices_), (n_slices_) * sizeof(*(slices_))); \
    args__.n_slices = (n_slices_);                                      \
    DSC_INSERT_TYPED_TRACE((DEV), dsc_get_slice_args, DSC_GET_SLICE, ##__VA_ARGS__)

#define DSC_TRACE_GET_TENSOR(DEV, X, INDEXES, ...) \
    dsc_get_tensor_args args__{};                  \
    DSC_TRACE_SET_TENSOR(X, x);                    \
    DSC_TRACE_SET_TENSOR(INDEXES, indexes);        \
    DSC_INSERT_TYPED_TRACE((DEV), dsc_get_tensor_args, DSC_GET_TENSOR, ##__VA_ARGS__)

#define DSC_TRACE_GET_IDX(DEV, X, indexes_, n_indexes_, out_shape_, out_n_dim_, ...) \
    dsc_get_idx_args args__{};                                                       \
    DSC_TRACE_SET_TENSOR(X, x);                                                      \
    memcpy(args__.indexes, (indexes_), (n_indexes_) * sizeof(*(indexes_)));          \
    memcpy(args__.out_shape, (out_shape_), (out_n_dim_) * sizeof(*(out_shape_)));    \
    args__.n_indexes = (n_indexes_);                                                 \
    args__.out_n_dim = (out_n_dim_);                                                 \
    DSC_INSERT_TYPED_TRACE((DEV), dsc_get_idx_args, DSC_GET_IDX, ##__VA_ARGS__)

#define DSC_TRACE_SET_SLICE(DEV, XA, XB, slices_, n_slices_, ...)       \
    dsc_set_slice_args args__{};                                        \
    DSC_TRACE_SET_TENSOR(XA, xa);                                       \
    DSC_TRACE_SET_TENSOR(XB, xb);                                       \
    memcpy(args__.slices, (slices_), (n_slices_) * sizeof(*(slices_))); \
    args__.n_slices = (n_slices_);                                      \
    DSC_INSERT_TYPED_TRACE((DEV), dsc_set_slice_args, DSC_SET_SLICE, ##__VA_ARGS__)

#define DSC_TRACE_RANDN_OP(DEV, X, ...) \
    dsc_randn_args args__{};            \
    DSC_TRACE_SET_TENSOR(X, x);         \
    DSC_INSERT_TYPED_TRACE((DEV), dsc_randn_args, DSC_RANDN_OP, ##__VA_ARGS__)

#define DSC_TRACE_TOPK_OP(DEV, X, k_, axis_, largest_, ...) \
    dsc_topk_args args__{};                                 \
    DSC_TRACE_SET_TENSOR(X, x);                             \
    args__.k = (k_);                                        \
    args__.axis = (axis_);                                  \
    args__.largest = (largest_);                            \
    DSC_INSERT_TYPED_TRACE((DEV), dsc_topk_args, DSC_TOPK_OP, ##__VA_ARGS__)

#define DSC_TRACE_MULTINOMIAL_OP(DEV, X, OUT, num_samples_, ...) \
    dsc_multinomial_args args__{};                               \
    DSC_TRACE_SET_TENSOR(X, x);                                  \
    DSC_TRACE_SET_TENSOR(OUT, out);                              \
    args__.num_samples = (num_samples_);                         \
    DSC_INSERT_TYPED_TRACE((DEV), dsc_multinomial_args, DSC_MULTINOMIAL_OP, ##__VA_ARGS__)

#define DSC_TRACE_ARANGE_OP(DEV, X, start_, step_, ...) \
    dsc_arange_args args__{};                           \
    DSC_TRACE_SET_TENSOR(X, x);                         \
    args__.start = (start_);                            \
    args__.step = (step_);                              \
    DSC_INSERT_TYPED_TRACE((DEV), dsc_arange_args, DSC_ARANGE_OP, ##__VA_ARGS__)

#define DSC_TRACE_REPEAT_OP(DEV, X, OUT, repeats_, axis_, ...) \
    dsc_repeat_args args__{};                                  \
    DSC_TRACE_SET_TENSOR(X, x);                                \
    DSC_TRACE_SET_TENSOR(OUT, out);                            \
    args__.repeats = (repeats_);                               \
    args__.axis = (axis_);                                     \
    DSC_INSERT_TYPED_TRACE((DEV), dsc_repeat_args, DSC_REPEAT_OP, ##__VA_ARGS__)

#define DSC_TRACE_COPY_OP(DEV, X, data_, nb_, data_device_, ...) \
    dsc_copy_args args__{};                                      \
    DSC_TRACE_SET_TENSOR(X, x);                                  \
    args__.data = (uintptr_t) (data_);                           \
    args__.nb = (nb_);                                           \
    args__.data_device = (data_device_);                         \
    DSC_INSERT_TYPED_TRACE((DEV), dsc_copy_args, DSC_COPY_OP, ##__VA_ARGS__)

#define DSC_TRACE_TO_OP(DEV, X, new_device_, ...) \
    dsc_to_args args__{};                         \
    DSC_TRACE_SET_TENSOR(X, x);                   \
    args__.new_device = (new_device_);            \
    DSC_INSERT_TYPED_TRACE((DEV), dsc_to_args, DSC_TO_OP, ##__VA_ARGS__)

#define DSC_TRACE_CONCAT_OP(DEV, OUT, tensors_, axis_, ...) \
    dsc_concat_args args__{};                               \
    DSC_TRACE_SET_TENSOR(OUT, out);                         \
    args__.tensors = (tensors_);                            \
    args__.axis = (axis_);                                  \
    DSC_INSERT_TYPED_TRACE((DEV), dsc_concat_args, DSC_CONCAT_OP, ##__VA_ARGS__)

#define DSC_TRACE_TRANSPOSE_OP(DEV, X, OUT, ...) \
    dsc_transpose_args args__{};                 \
    DSC_TRACE_SET_TENSOR(X, x);                  \
    DSC_TRACE_SET_TENSOR(OUT, out);              \
    DSC_INSERT_TYPED_TRACE((DEV), dsc_transpose_args, DSC_TRANSPOSE_OP, ##__VA_ARGS__)

#define DSC_TRACE_TRIL_OP(DEV, X, OUT, diagonal_, ...) \
    dsc_tril_args args__{};                            \
    DSC_TRACE_SET_TENSOR(X, x);                        \
    DSC_TRACE_SET_TENSOR(OUT, out);                    \
    args__.diagonal = (diagonal_);                     \
    DSC_INSERT_TYPED_TRACE((DEV), dsc_tril_args, DSC_TRIL_OP, ##__VA_ARGS__)

#define TYPED_FILL(NAME, ARGS)                       \
    if constexpr (dsc_is_type<T, ARGS>()) {          \
        const ARGS *args_ = (const ARGS *) args;     \
        memcpy(&trace->NAME, args_, sizeof(*args_)); \
    }

#define TYPED_DUMP(TYPE, ARGS)    \
    case TYPE:                    \
        trace->ARGS.json_dump(f); \
        break


namespace internal::tracing {
DSC_INLINE void dump_indexes(FILE *f, const int *indexes,
                             const int n_indexes) {
    if (n_indexes > 1) {
        fprintf(f, "\"[");
        for (int i = 0; i < n_indexes; ++i) {
            fprintf(f, "%d", indexes[i]);
            if (i < n_indexes - 1) fprintf(f, ", ");
        }
        fprintf(f, "]\"");
    } else {
        fprintf(f, "%d", indexes[0]);
    }
}

DSC_INLINE void dump_slices(FILE *f, const dsc_slice *slices,
                            const int n_slices) {
    if (n_slices > 1) {
        fprintf(f, "\"[");
        for (int i = 0; i < n_slices; ++i) {
            fprintf(f, "%d:%d:%d",
                    slices[i].start,
                    slices[i].stop,
                    slices[i].step);
            if (i < n_slices - 1) fprintf(f, ", ");
        }
        fprintf(f, "]\"");
    } else {
        fprintf(f, "\"%d:%d:%d\"",
                slices[0].start,
                slices[0].stop,
                slices[0].step);
    }
}
}

struct dsc_empty_args {
    static DSC_INLINE u64 rw_bytes() { return 0; }
    static DSC_INLINE void json_dump(FILE *) {}
};

struct dsc_tensor_args {
    int shape[DSC_MAX_DIMS];
    uintptr_t addr;
    int n_dim, ne;
    dsc_device_type device;
    dsc_dtype dtype;

    DSC_INLINE u64 rw_bytes() const { return ne * DSC_DTYPE_SIZE[dtype]; }
    DSC_INLINE void json_dump(FILE *f) const {
        fprintf(f, R"({"shape":)");
        internal::tracing::dump_indexes(f, shape, n_dim);
        fprintf(f, R"(,"dtype":"%s","device":"%s")",
                DSC_DTYPE_NAMES[dtype],
                DSC_DEVICE_NAMES[device]);
        if (addr != 0) fprintf(f, ",\"addr\":\"0x%" PRIxPTR "\"", addr);
        fprintf(f, "}");
    }
};

struct dsc_tensor_alloc_args {
    dsc_tensor_args x;
    const void *data;
    dsc_device_type data_device;
    bool lazy;

    DSC_INLINE u64 rw_bytes() const { return data ? x.rw_bytes() : 0; }
    DSC_INLINE void json_dump(FILE *f) const {
        fprintf(f, R"(,"x":)");
        x.json_dump(f);
        fprintf(f,R"(,"lazy":"%s")", lazy ? "True" : "False");
        if (data) fprintf(f, ",\"data\":\"0x%" PRIxPTR "\",\"data_device\":\"%s\"",
                          (uintptr_t) data, DSC_DEVICE_NAMES[data_device]);
    }
};

struct dsc_tensor_free_args {
    dsc_tensor_args x;

    static DSC_INLINE u64 rw_bytes() { return 0; }
    DSC_INLINE void json_dump(FILE *f) const {
        fprintf(f, R"(,"x":)");
        x.json_dump(f);
    }
};

struct dsc_cast_args {
    dsc_tensor_args x;
    dsc_dtype new_dtype;

    DSC_INLINE u64 rw_bytes() const { return x.rw_bytes() + x.ne * DSC_DTYPE_SIZE[new_dtype]; }
    DSC_INLINE void json_dump(FILE *f) const {
        fprintf(f, R"(,"x":)");
        x.json_dump(f);
        fprintf(f, R"(,"new_dtype":"%s")", DSC_DTYPE_NAMES[new_dtype]);
    }
};

struct dsc_binary_args {
    dsc_tensor_args xa, xb, out;

    DSC_INLINE u64 rw_bytes() const { return xa.rw_bytes() + xb.rw_bytes() + out.rw_bytes(); }
    DSC_INLINE void json_dump(FILE * f) const {
        fprintf(f, R"(,"xa": )");
        xa.json_dump(f);
        fprintf(f, R"(,"xb":)");
        xb.json_dump(f);
        fprintf(f, R"(,"out":)");
        out.json_dump(f);
    }
};

struct dsc_unary_args {
    dsc_tensor_args x, out;

    DSC_INLINE u64 rw_bytes() const { return x.rw_bytes() + out.rw_bytes(); }
    DSC_INLINE void json_dump(FILE *f) const {
        fprintf(f, R"(,"x":)");
        x.json_dump(f);
        fprintf(f, R"(,"out":)");
        out.json_dump(f);
    }
};

struct dsc_unary_axis_args {
    dsc_tensor_args x, out;
    int axis;

    DSC_INLINE u64 rw_bytes() const { return x.rw_bytes() + out.rw_bytes(); }
    DSC_INLINE void json_dump(FILE *f) const {
        fprintf(f, R"(,"axis": %d,"x":)", axis);
        x.json_dump(f);
        fprintf(f, R"(,"out":)");
        out.json_dump(f);
    }
};

struct dsc_matmul_args {
    dsc_tensor_args xa, xb, out;
    bool trans_b;

    DSC_INLINE u64 rw_bytes() const { return xa.rw_bytes() + xb.rw_bytes() + out.rw_bytes(); }
    DSC_INLINE void json_dump(FILE * f) const {
        fprintf(f, R"(,"xa":)");
        xa.json_dump(f);
        fprintf(f, R"(,"xb":)");
        xb.json_dump(f);
        fprintf(f, R"(,"transposed":"%s")", trans_b ? "True" : "False");
        fprintf(f, R"(,"out":)");
        out.json_dump(f);
    }
};

struct dsc_mask_args {
    dsc_tensor_args x, mask;
    f64 value;

    DSC_INLINE u64 rw_bytes() const { return 2 * x.rw_bytes() + mask.rw_bytes(); }
    DSC_INLINE void json_dump(FILE * f) const {
        char value_str[16];
        fprintf(f, R"(,"x":)");
        x.json_dump(f);
        fprintf(f, R"(,"mask":)");
        mask.json_dump(f);
        snprintf(value_str, 16, "%f", value);
        fprintf(f, R"(,"value":"%s")", value_str);
    }
};

struct dsc_outer_args {
    dsc_tensor_args xa, xb, out;

    DSC_INLINE u64 rw_bytes() const { return xa.rw_bytes() + xb.rw_bytes() + out.rw_bytes(); }
    DSC_INLINE void json_dump(FILE * f) const {
        fprintf(f, R"(,"xa":)");
        xa.json_dump(f);
        fprintf(f, R"(,"xb":)");
        xb.json_dump(f);
        fprintf(f, R"(,"out":)");
        out.json_dump(f);
    }
};

struct dsc_where_args {
    dsc_tensor_args condition, input, other, out;

    DSC_INLINE u64 rw_bytes() const { return condition.rw_bytes() + input.rw_bytes() + other.rw_bytes() + out.rw_bytes(); }
    DSC_INLINE void json_dump(FILE * f) const {
        fprintf(f, R"(,"cond":)");
        condition.json_dump(f);
        fprintf(f, R"(,"input":)");
        input.json_dump(f);
        fprintf(f, R"(,"other":)");
        other.json_dump(f);
        fprintf(f, R"(,"out":)");
        out.json_dump(f);
    }
};

struct dsc_get_idx_args {
    dsc_tensor_args x;
    int indexes[DSC_MAX_DIMS], out_shape[DSC_MAX_DIMS];
    int n_indexes, out_n_dim;

    DSC_INLINE u64 rw_bytes() const { int ne = 1; for (int i = 0; i < out_n_dim; ++i) ne *= out_shape[i]; return 2 * ne * DSC_DTYPE_SIZE[x.dtype]; }
    DSC_INLINE void json_dump(FILE * f) const {
        fprintf(f, R"(,"x":)");
        x.json_dump(f);
        fprintf(f, R"(,"idx":)");
        internal::tracing::dump_indexes(f, indexes, n_indexes);
    }
};

struct dsc_get_slice_args {
    dsc_tensor_args x, out;
    dsc_slice slices[DSC_MAX_DIMS];
    int n_slices;

    DSC_INLINE u64 rw_bytes() const { return x.rw_bytes() + out.rw_bytes(); }
    DSC_INLINE void json_dump(FILE * f) const {
        fprintf(f, R"(,"x":)");
        x.json_dump(f);
        fprintf(f, R"(,"slice":)");
        internal::tracing::dump_slices(f, slices, n_slices);
    }
};

struct dsc_get_tensor_args {
    dsc_tensor_args x, indexes;

    DSC_INLINE u64 rw_bytes() const { return 2 * x.shape[DSC_MAX_DIMS - 1] * indexes.ne * DSC_DTYPE_SIZE[x.dtype]; }
    DSC_INLINE void json_dump(FILE * f) const {
        fprintf(f, R"(,"x":)");
        x.json_dump(f);
        fprintf(f, R"(,"idx":)");
        indexes.json_dump(f);
    }
};

struct dsc_set_slice_args {
    dsc_tensor_args xa, xb;
    dsc_slice slices[DSC_MAX_DIMS];
    int n_slices;

    DSC_INLINE u64 rw_bytes() const { return xa.rw_bytes() + xb.rw_bytes(); }
    DSC_INLINE void json_dump(FILE * f) const {
        fprintf(f, R"(,"xa":)");
        xa.json_dump(f);
        fprintf(f, R"(,"xb":)");
        xb.json_dump(f);
        fprintf(f, R"(,"slice":)");
        internal::tracing::dump_slices(f, slices, n_slices);
    }
};

struct dsc_randn_args {
    dsc_tensor_args x;

    DSC_INLINE u64 rw_bytes() const { return x.rw_bytes(); }
    DSC_INLINE void json_dump(FILE * f) const {
        fprintf(f, R"(,"x":)");
        x.json_dump(f);
    }
};

struct dsc_topk_args {
    dsc_tensor_args x;
    int k, axis;
    bool largest;

    // This is a rough bw estimate for topk sampling: N is for the read, NlogN for the sorting
    DSC_INLINE u64 rw_bytes() const { const f64 n = x.ne; return (u64) (n * (1. + n * log(n)) * (f64) DSC_DTYPE_SIZE[x.dtype]); }
    DSC_INLINE void json_dump(FILE * f) const {
        fprintf(f, R"(,"x":)");
        x.json_dump(f);
        fprintf(f, R"(,"k": %d,"axis": %d,"largest":"%s")", k, axis, largest ? "True" : "False");
    }
};

struct dsc_multinomial_args {
    dsc_tensor_args x, out;
    int num_samples;

    DSC_INLINE u64 rw_bytes() const { return x.rw_bytes() + out.rw_bytes(); }
    DSC_INLINE void json_dump(FILE * f) const {
        fprintf(f, R"(,"x":)");
        x.json_dump(f);
        fprintf(f, R"(,"out":)");
        out.json_dump(f);
        fprintf(f, R"(,"num_samples":%d)", num_samples);
    }
};

struct dsc_arange_args {
    dsc_tensor_args x;
    f64 start, step;

    DSC_INLINE u64 rw_bytes() const { return x.rw_bytes(); }
    DSC_INLINE void json_dump(FILE * f) const {
        fprintf(f, R"(,"x":)");
        x.json_dump(f);
        fprintf(f, R"(,"start":%.1f,"step":%.1f)", start, step);
    }
};

struct dsc_repeat_args {
    dsc_tensor_args x, out;
    int repeats, axis;

    DSC_INLINE u64 rw_bytes() const { return x.rw_bytes() + out.rw_bytes(); }
    DSC_INLINE void json_dump(FILE * f) const {
        fprintf(f, R"(,"x":)");
        x.json_dump(f);
        fprintf(f, R"(,"out":)");
        out.json_dump(f);
        fprintf(f, R"(,"repeats":%d,"axis":%d)", repeats, axis);
    }
};

struct dsc_copy_args {
    dsc_tensor_args x;
    uintptr_t data;
    usize nb;
    dsc_device_type data_device;

    DSC_INLINE u64 rw_bytes() const { return 2 * nb; }
    DSC_INLINE void json_dump(FILE * f) const {
        fprintf(f, R"(,"x":)");
        x.json_dump(f);
        fprintf(f, ",\"addr\":\"0x%" PRIxPTR "\"", data);
        fprintf(f, R"(,"nb":%ld,"data_device":"%s")", nb, DSC_DEVICE_NAMES[data_device]);
    }
};

struct dsc_to_args {
    dsc_tensor_args x;
    dsc_device_type new_device;

    DSC_INLINE u64 rw_bytes() const { return 2 * x.rw_bytes(); }
    DSC_INLINE void json_dump(FILE * f) const {
        fprintf(f, R"(,"x":)");
        x.json_dump(f);
        fprintf(f, R"(,"new_device":%s)", DSC_DEVICE_NAMES[new_device]);
    }
};

struct dsc_concat_args {
    dsc_tensor_args out;
    int tensors;
    int axis;

    DSC_INLINE u64 rw_bytes() const { return 2 * out.rw_bytes(); }
    DSC_INLINE void json_dump(FILE * f) const {
        char axis_str[16];
        fprintf(f, R"(,"out":)");
        out.json_dump(f);
        if (axis == DSC_VALUE_NONE) snprintf(axis_str, 16, "Flatten");
        else snprintf(axis_str, 16, "%d", axis);
        fprintf(f, R"(,"tensors":%d,"axis":"%s")", tensors, axis_str);
    }
};

struct dsc_transpose_args {
    dsc_tensor_args x, out;

    DSC_INLINE u64 rw_bytes() const { return x.rw_bytes() + out.rw_bytes(); }
    DSC_INLINE void json_dump(FILE * f) const {
        fprintf(f, R"(,"x":)");
        x.json_dump(f);
        fprintf(f, R"(,"out":)");
        out.json_dump(f);
    }
};

struct dsc_tril_args {
    dsc_tensor_args x, out;
    int diagonal;

    DSC_INLINE u64 rw_bytes() const { return x.rw_bytes() + out.rw_bytes(); }
    DSC_INLINE void json_dump(FILE * f) const {
        fprintf(f, R"(,"x":)");
        x.json_dump(f);
        fprintf(f, R"(,"out":)");
        out.json_dump(f);
        fprintf(f, R"(,"diagonal":%d)", diagonal);
    }
};

enum dsc_trace_type : u8 {
    DSC_TRACE_EMPY, // Trace without any args
    DSC_TRACE_CUSTOM,
    DSC_TENSOR_ALLOC,
    DSC_TENSOR_FREE,
    DSC_UNARY_OP,
    DSC_UNARY_AXIS_OP,
    DSC_BINARY_OP,
    DSC_MATMUL_OP,
    DSC_MASK_OP,
    DSC_OUTER_OP,
    DSC_WHERE_OP,
    DSC_GET_IDX,
    DSC_GET_SLICE,
    DSC_GET_TENSOR,
    DSC_SET_SLICE,
    DSC_CAST_OP,
    DSC_RANDN_OP,
    DSC_TOPK_OP,
    DSC_MULTINOMIAL_OP,
    DSC_ARANGE_OP,
    DSC_REPEAT_OP,
    DSC_COPY_OP,
    DSC_TO_OP,
    DSC_CONCAT_OP,
    DSC_TRANSPOSE_OP,
    DSC_TRIL_OP
};

static constexpr const char *DSC_TRACE_CATEGORY[] = {
    "",
    "custom",
    "alloc",
    "free",
    "op;unary",
    "op;reduce",
    "op;binary",
    "op;matmul",
    "op;mask",
    "op;outer",
    "op;where",
    "op;slice;get",
    "op;tensor;get",
    "op;slice;set",
    "op;cast",
    "op;randn",
    "op;topk",
    "op;multinomial",
    "op;arange",
    "op;repeat",
    "op;copy",
    "op;to",
    "op;concat",
    "op;transpose",
    "op;tril",
};

struct dsc_trace_common {
    char name[DSC_TRACE_NAME_MAX];
    u64 rw_bytes;
    u64 ingestion_time_us;

    dsc_trace_type type;
    union {
        dsc_empty_args empty;
        dsc_tensor_alloc_args tensor_alloc;
        dsc_tensor_alloc_args tensor_free;
        dsc_unary_args unary;
        dsc_unary_axis_args unary_axis;
        dsc_binary_args binary;
        dsc_matmul_args matmul;
        dsc_mask_args mask;
        dsc_outer_args outer;
        dsc_where_args where;
        dsc_get_idx_args get_idx;
        dsc_get_slice_args get_slice;
        dsc_get_tensor_args get_tensor;
        dsc_set_slice_args set_slice;
        dsc_cast_args cast;
        dsc_randn_args randn;
        dsc_topk_args topk;
        dsc_multinomial_args multinomial;
        dsc_arange_args arange;
        dsc_repeat_args repeat;
        dsc_copy_args copy;
        dsc_to_args to;
        dsc_concat_args concat;
        dsc_transpose_args transpose;
        dsc_tril_args tril;
    };
};

struct dsc_trace_ctx {
    void *traces[DSC_MAX_CHUNKS];
    void *current_trace;
    int current_trace_idx;
    int n_chunks, n_traces;
};

namespace internal::tracing {
DSC_INLINE u64 time_us() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (u64) (ts.tv_sec * 1'000'000ULL) + (u64) (ts.tv_nsec / 1'000ULL);
}

template<typename T>
DSC_INLINE void advance_current_trace(dsc_trace_ctx *ctx) {
    ctx->current_trace_idx++;
    const int chunk = ctx->current_trace_idx / DSC_MAX_TRACES_PER_CHUNK;
    const int idx_in_chunk = ctx->current_trace_idx % DSC_MAX_TRACES_PER_CHUNK;

    if (chunk >= ctx->n_chunks || idx_in_chunk >= ctx->n_traces) {
        // We are done
        ctx->current_trace = nullptr;
        ctx->current_trace_idx = 0;
    }

    T **traces = (T **) ctx->traces;
    ctx->current_trace = &traces[chunk][idx_in_chunk];
}

template<typename T>
DSC_INLINE void traces_allocate_chunk(dsc_trace_ctx *ctx) {
    ctx->traces[ctx->n_chunks] = (T *) malloc(DSC_MAX_TRACES_PER_CHUNK * sizeof(T));
    ctx->n_chunks++;
    // Reset n_traces
    ctx->n_traces = 0;
}

template<typename T>
DSC_INLINE dsc_trace_ctx *init() {
    static dsc_trace_ctx ctx{
        .traces = {},
        .current_trace = nullptr,
        .current_trace_idx = 0,
        .n_chunks = 0,
        .n_traces = 0,
    };

    traces_allocate_chunk<T>(&ctx);
    T **traces = (T **) ctx.traces;
    ctx.current_trace = &traces[0][0];
    return &ctx;
}

DSC_INLINE void dispose(const dsc_trace_ctx *ctx) {
    for (int i = 0; i < ctx->n_chunks; i++) {
        free(ctx->traces[i]);
    }
}

template<typename T>
DSC_INLINE void check_if_full(dsc_trace_ctx *ctx) {
    if (ctx->n_traces >= DSC_MAX_TRACES_PER_CHUNK) {

        if (ctx->n_chunks >= DSC_MAX_CHUNKS) {
            DSC_LOG_FATAL("can't allocate any more traces!");
        }

        // Allocate a brand-new chunk
        traces_allocate_chunk<T>(ctx);
    }
}

template<typename T>
DSC_INLINE T *next_empty_trace(dsc_trace_ctx *ctx) {
    T **traces = (T **) ctx->traces;
    return &traces[ctx->n_chunks - 1][ctx->n_traces++];
}

template<typename T = dsc_empty_args>
DSC_INLINE void fill_trace(dsc_trace_common *trace,
                           const char *name,
                           const dsc_trace_type type,
                           const T *args = nullptr) {
    trace->ingestion_time_us = time_us();
    strncpy(trace->name, name, DSC_TRACE_NAME_MAX);
    trace->type = type;
    trace->rw_bytes = args->rw_bytes();

    TYPED_FILL(tensor_alloc, dsc_tensor_alloc_args)
    TYPED_FILL(tensor_free, dsc_tensor_free_args)
    TYPED_FILL(unary, dsc_unary_args)
    TYPED_FILL(unary_axis, dsc_unary_axis_args)
    TYPED_FILL(binary, dsc_binary_args)
    TYPED_FILL(matmul, dsc_matmul_args)
    TYPED_FILL(mask, dsc_mask_args)
    TYPED_FILL(outer, dsc_outer_args)
    TYPED_FILL(where, dsc_where_args)
    TYPED_FILL(get_slice, dsc_get_slice_args)
    TYPED_FILL(get_tensor, dsc_get_tensor_args)
    TYPED_FILL(get_idx, dsc_get_idx_args)
    TYPED_FILL(set_slice, dsc_set_slice_args)
    TYPED_FILL(cast, dsc_cast_args)
    TYPED_FILL(randn, dsc_randn_args)
    TYPED_FILL(topk, dsc_topk_args)
    TYPED_FILL(multinomial, dsc_multinomial_args)
    TYPED_FILL(arange, dsc_arange_args)
    TYPED_FILL(repeat, dsc_repeat_args)
    TYPED_FILL(copy, dsc_copy_args)
    TYPED_FILL(to, dsc_to_args)
    TYPED_FILL(concat, dsc_concat_args)
    TYPED_FILL(transpose, dsc_transpose_args)
    TYPED_FILL(tril, dsc_tril_args)
}

DSC_INLINE void dump_trace_base(FILE *f, const dsc_trace_common *trace) {
    switch (trace->type) {
        TYPED_DUMP(DSC_TENSOR_ALLOC, tensor_alloc);
        TYPED_DUMP(DSC_TENSOR_FREE, tensor_free);
        TYPED_DUMP(DSC_UNARY_OP, unary);
        TYPED_DUMP(DSC_UNARY_AXIS_OP, unary_axis);
        TYPED_DUMP(DSC_BINARY_OP, binary);
        TYPED_DUMP(DSC_MATMUL_OP, matmul);
        TYPED_DUMP(DSC_MASK_OP, mask);
        TYPED_DUMP(DSC_OUTER_OP, outer);
        TYPED_DUMP(DSC_WHERE_OP, where);
        TYPED_DUMP(DSC_GET_SLICE, get_slice);
        TYPED_DUMP(DSC_GET_IDX, get_idx);
        TYPED_DUMP(DSC_GET_TENSOR, get_tensor);
        TYPED_DUMP(DSC_SET_SLICE, set_slice);
        TYPED_DUMP(DSC_CAST_OP, cast);
        TYPED_DUMP(DSC_RANDN_OP, randn);
        TYPED_DUMP(DSC_TOPK_OP, topk);
        TYPED_DUMP(DSC_MULTINOMIAL_OP, multinomial);
        TYPED_DUMP(DSC_ARANGE_OP, arange);
        TYPED_DUMP(DSC_REPEAT_OP, repeat);
        TYPED_DUMP(DSC_COPY_OP, copy);
        TYPED_DUMP(DSC_TO_OP, to);
        TYPED_DUMP(DSC_CONCAT_OP, concat);
        TYPED_DUMP(DSC_TRANSPOSE_OP, transpose);
        TYPED_DUMP(DSC_TRIL_OP, tril);
        default:
            break;
    }
}

DSC_INLINE bool is_valid_trace(const void *trace) {
    if (!trace) return false;

    const dsc_trace_common *base = (const dsc_trace_common *) trace;
    return base->type != DSC_TRACE_EMPY;
}

DSC_INLINE bool trace_var_is_set() {
    const char *trace_str = std::getenv("TRACE");
    bool enabled = false;

    if (trace_str) {
        const int tracing_flag = std::atoi(trace_str);
        enabled = tracing_flag != 0;
    }

    return enabled;
}
}

static DSC_INLINE bool dsc_tracing_is_enabled() {
    static const bool tracing_enabled = internal::tracing::trace_var_is_set();

    return tracing_enabled;
}

static DSC_INLINE void dsc_tracing_dump(dsc_ctx *ctx) {
    if (!dsc_tracing_is_enabled()) return;

    dsc_trace_ctx *tracing_ctxs[DSC_MAX_DEVICES];
    for (int i = 0; i < DSC_MAX_DEVICES; ++i) {
        const dsc_device *device = ctx->devices[i];
        tracing_ctxs[i] = device->trace_ctx;
    }

    FILE *json_file = fopen("traces.json", "wt");
    DSC_ASSERT(json_file);

    fprintf(json_file, "[\n");

    // Dump json metadata before dumping actual traces
    for (int i = 0; i < DSC_MAX_DEVICES; ++i) {
        const dsc_device *device = ctx->devices[i];
        device->dump_json_metadata(json_file, device->extra_info);
    }

    printf("\n");

    // NOTE: this doesn't make sense here!
    while (true) {
        // Find the first potential trace for dumping
        int dump_device_idx = 0;
        void *trace_to_dump = nullptr;
        for (int i = 0; i < DSC_MAX_DEVICES && !internal::tracing::is_valid_trace(trace_to_dump); ++i) {
            trace_to_dump = tracing_ctxs[i]->current_trace;
            dump_device_idx = i;
        }

        if (!internal::tracing::is_valid_trace(trace_to_dump)) break;

        const dsc_trace_common *base = (dsc_trace_common *)trace_to_dump;

        // At each step iterate over all the devices and get the current trace (if any)
        for (int i = 0; i < DSC_MAX_DEVICES; ++i) {
            // Skip if the device is the same of the current trace to dump
            if (i == dump_device_idx) continue;

            void *this_trace = tracing_ctxs[i]->current_trace;
            if (!internal::tracing::is_valid_trace(this_trace)) continue;

            // If we found a valid trace compare the ingestion timestamp
            if (const dsc_trace_common *this_base = (dsc_trace_common *) this_trace;
                this_base->ingestion_time_us < base->ingestion_time_us) {
                trace_to_dump = this_trace;
                dump_device_idx = i;
            }
        }

        // Dump only the trace that came in first and advance the current pointer only for that device
        ctx->devices[dump_device_idx]->dump_trace(trace_to_dump, json_file);
        ctx->devices[dump_device_idx]->next_trace(tracing_ctxs[dump_device_idx]);
    }
    printf("\n");
    fflush(stdout);

    fprintf(json_file, "]");
    fclose(json_file);
}

#undef TYPED_FILL
#undef TYPED_DUMP

#else

#define DSC_TRACE_TENSOR_NEW(DEV, shape_, n_dim_, dtype_, device_, lazy_, data_, data_device_, ...)  (DSC_UNUSED(DEV))
#define DSC_TRACE_TENSOR_FREE(DEV, X, ...)                                                           (DSC_UNUSED(DEV))
#define DSC_TRACE_CAST_OP(DEV, X, OUT, ...)                                                          (DSC_UNUSED(DEV))
#define DSC_TRACE_BINARY_OP(DEV, XA, XB, OUT, ...)                                                   (DSC_UNUSED(DEV))
#define DSC_TRACE_UNARY_OP(DEV, X, OUT, ...)                                                         (DSC_UNUSED(DEV))
#define DSC_TRACE_UNARY_AXIS_OP(DEV, X, OUT, axis_, ...)                                             (DSC_UNUSED(DEV))
#define DSC_TRACE_MATMUL_OP(DEV, XA, XB, trans_b_, OUT, is_gevm_, ...)                               (DSC_UNUSED(DEV))
#define DSC_TRACE_MASK_OP(DEV, X, MASK, value_, ...)                                                 (DSC_UNUSED(DEV))
#define DSC_TRACE_OUTER_OP(DEV, XA, XB, OUT, ...)                                                    (DSC_UNUSED(DEV))
#define DSC_TRACE_WHERE_OP(DEV, CONDITION, INPUT, OTHER, OUT, ...)                                   (DSC_UNUSED(DEV))
#define DSC_TRACE_GET_SLICE(DEV, X, OUT, slices_, n_slices_, ...)                                    (DSC_UNUSED(DEV))
#define DSC_TRACE_GET_TENSOR(DEV, X, INDEXES, ...)                                                   (DSC_UNUSED(DEV))
#define DSC_TRACE_GET_IDX(DEV, X, indexes_, n_indexes_, out_shape_, out_n_dim_, ...)                 (DSC_UNUSED(DEV))
#define DSC_TRACE_SET_SLICE(DEV, XA, XB, slices_, n_slices_, ...)                                    (DSC_UNUSED(DEV))
#define DSC_TRACE_RANDN_OP(DEV, X, ...)                                                              (DSC_UNUSED(DEV))
#define DSC_TRACE_TOPK_OP(DEV, X, k_, axis_, largest_, ...)                                          (DSC_UNUSED(DEV))
#define DSC_TRACE_MULTINOMIAL_OP(DEV, X, OUT, num_samples_, ...)                                     (DSC_UNUSED(DEV))
#define DSC_TRACE_ARANGE_OP(DEV, X, start_, step_, ...)                                              (DSC_UNUSED(DEV))
#define DSC_TRACE_REPEAT_OP(DEV, X, OUT, repeats_, axis_, ...)                                       (DSC_UNUSED(DEV))
#define DSC_TRACE_COPY_OP(DEV, X, data_, nb_, data_device_, ...)                                     (DSC_UNUSED(DEV))
#define DSC_TRACE_TO_OP(DEV, X, new_device_, ...)                                                    (DSC_UNUSED(DEV))
#define DSC_TRACE_CONCAT_OP(DEV, OUT, tensors_, axis_, ...)                                          (DSC_UNUSED(DEV))
#define DSC_TRACE_TRANSPOSE_OP(DEV, X, OUT, ...)                                                     (DSC_UNUSED(DEV))
#define DSC_TRACE_TRIL_OP(DEV, X, OUT, diagonal_, ...)                                               (DSC_UNUSED(DEV))

static consteval bool dsc_tracing_is_enabled() { return false; }
static DSC_INLINE void dsc_tracing_dump(dsc_ctx *) {}

#endif // DSC_TRACING