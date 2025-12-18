#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <stdio.h>

typedef uint16_t f16;

static inline f16 f32_to_f16(float val) {
    uint32_t x;
    memcpy(&x, &val, sizeof(float));
    
    uint32_t sign = (x >> 31) & 0x1;
    int32_t exponent = ((x >> 23) & 0xFF) - 127;
    uint32_t mantissa = x & 0x7FFFFF;
    
    if (exponent == 128) {
        if (mantissa != 0) {
            return (f16)((sign << 15) | 0x7E00);
        }
        return (f16)((sign << 15) | 0x7C00);
    }
    
    if (exponent < -24) {
        return (f16)(sign << 15);
    }
    
    if (exponent < -14) {
        int shift = -14 - exponent;
        mantissa = (mantissa | 0x800000) >> (shift + 13);
        return (f16)((sign << 15) | mantissa);
    }
    
    if (exponent > 15) {
        return (f16)((sign << 15) | 0x7C00);
    }
    
    uint16_t h_exp = (uint16_t)(exponent + 15) << 10;
    uint16_t h_mantissa = (uint16_t)(mantissa >> 13);
    
    return (f16)((sign << 15) | h_exp | h_mantissa);
}

static inline float f16_to_f32(f16 val) {
    uint16_t h = val;
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FF;
    
    float result;
    
    if (exponent == 0) {
        if (mantissa == 0) {
            uint32_t x = sign << 31;
            memcpy(&result, &x, sizeof(float));
            return result;
        }
        float m = (float)mantissa / 1024.0f;
        result = (sign ? -1.0f : 1.0f) * m * (1.0f / 16384.0f);
        return result;
    }
    
    if (exponent == 31) {
        if (mantissa != 0) {
            uint32_t x = 0x7FC00000 | (sign << 31);
            memcpy(&result, &x, sizeof(float));
        } else {
            uint32_t x = 0x7F800000 | (sign << 31);
            memcpy(&result, &x, sizeof(float));
        }
        return result;
    }
    
    int32_t f_exp = (int32_t)exponent - 15 + 127;
    uint32_t f_mantissa = mantissa << 13;
    uint32_t x = (sign << 31) | (f_exp << 23) | f_mantissa;
    memcpy(&result, &x, sizeof(float));
    return result;
}

static void f32_array_to_f16(const float *src, f16 *dst, int64_t count) {
    for (int64_t i = 0; i < count; i++) {
        dst[i] = f32_to_f16(src[i]);
    }
}

static void f16_array_to_f32(const f16 *src, float *dst, int64_t count) {
    for (int64_t i = 0; i < count; i++) {
        dst[i] = f16_to_f32(src[i]);
    }
}

struct futhark_context_config {
    int device;
    int platform;
    size_t default_group_size;
    size_t default_num_groups;
    size_t default_tile_size;
    int profiling;
};

struct futhark_context {
    struct futhark_context_config *cfg;
    void *opencl_ctx;
    void *error;
};

struct futhark_f16_1d {
    f16 *data;
    int64_t shape[1];
};

struct futhark_f16_2d {
    f16 *data;
    int64_t shape[2];
};

struct futhark_f16_3d {
    f16 *data;
    int64_t shape[3];
};

struct futhark_f32_1d {
    float *data;
    int64_t shape[1];
};

struct futhark_f32_2d {
    float *data;
    int64_t shape[2];
};

struct futhark_f32_3d {
    float *data;
    int64_t shape[3];
};

struct futhark_u64_1d {
    uint64_t *data;
    int64_t shape[1];
};

struct futhark_i64_1d {
    int64_t *data;
    int64_t shape[1];
};

struct futhark_context_config *futhark_context_config_new(void) {
    struct futhark_context_config *cfg = malloc(sizeof(struct futhark_context_config));
    if (cfg) {
        cfg->device = 0;
        cfg->platform = 0;
        cfg->default_group_size = 256;
        cfg->default_num_groups = 128;
        cfg->default_tile_size = 16;
        cfg->profiling = 0;
    }
    return cfg;
}

void futhark_context_config_free(struct futhark_context_config *cfg) {
    free(cfg);
}

void futhark_context_config_set_device(struct futhark_context_config *cfg, int device) {
    if (cfg) cfg->device = device;
}

void futhark_context_config_set_platform(struct futhark_context_config *cfg, int platform) {
    if (cfg) cfg->platform = platform;
}

struct futhark_context *futhark_context_new(struct futhark_context_config *cfg) {
    struct futhark_context *ctx = malloc(sizeof(struct futhark_context));
    if (ctx) {
        ctx->cfg = cfg;
        ctx->opencl_ctx = NULL;
        ctx->error = NULL;
    }
    return ctx;
}

void futhark_context_free(struct futhark_context *ctx) {
    if (ctx) {
        free(ctx);
    }
}

int futhark_context_sync(struct futhark_context *ctx) {
    (void)ctx;
    return 0;
}

char *futhark_context_get_error(struct futhark_context *ctx) {
    return ctx ? ctx->error : NULL;
}

struct futhark_f16_1d *futhark_new_f16_1d(struct futhark_context *ctx, const f16 *data, int64_t dim0) {
    (void)ctx;
    struct futhark_f16_1d *arr = malloc(sizeof(struct futhark_f16_1d));
    if (arr) {
        arr->shape[0] = dim0;
        arr->data = malloc(dim0 * sizeof(f16));
        if (arr->data && data) {
            memcpy(arr->data, data, dim0 * sizeof(f16));
        }
    }
    return arr;
}

struct futhark_f16_2d *futhark_new_f16_2d(struct futhark_context *ctx, const f16 *data, int64_t dim0, int64_t dim1) {
    (void)ctx;
    struct futhark_f16_2d *arr = malloc(sizeof(struct futhark_f16_2d));
    if (arr) {
        arr->shape[0] = dim0;
        arr->shape[1] = dim1;
        arr->data = malloc(dim0 * dim1 * sizeof(f16));
        if (arr->data && data) {
            memcpy(arr->data, data, dim0 * dim1 * sizeof(f16));
        }
    }
    return arr;
}

struct futhark_f16_3d *futhark_new_f16_3d(struct futhark_context *ctx, const f16 *data, int64_t dim0, int64_t dim1, int64_t dim2) {
    (void)ctx;
    struct futhark_f16_3d *arr = malloc(sizeof(struct futhark_f16_3d));
    if (arr) {
        arr->shape[0] = dim0;
        arr->shape[1] = dim1;
        arr->shape[2] = dim2;
        arr->data = malloc(dim0 * dim1 * dim2 * sizeof(f16));
        if (arr->data && data) {
            memcpy(arr->data, data, dim0 * dim1 * dim2 * sizeof(f16));
        }
    }
    return arr;
}

struct futhark_f16_2d *futhark_new_f16_2d_from_f32(struct futhark_context *ctx, const float *data, int64_t dim0, int64_t dim1) {
    (void)ctx;
    struct futhark_f16_2d *arr = malloc(sizeof(struct futhark_f16_2d));
    if (arr) {
        arr->shape[0] = dim0;
        arr->shape[1] = dim1;
        int64_t count = dim0 * dim1;
        arr->data = malloc(count * sizeof(f16));
        if (arr->data && data) {
            f32_array_to_f16(data, arr->data, count);
        }
    }
    return arr;
}

struct futhark_f16_3d *futhark_new_f16_3d_from_f32(struct futhark_context *ctx, const float *data, int64_t dim0, int64_t dim1, int64_t dim2) {
    (void)ctx;
    struct futhark_f16_3d *arr = malloc(sizeof(struct futhark_f16_3d));
    if (arr) {
        arr->shape[0] = dim0;
        arr->shape[1] = dim1;
        arr->shape[2] = dim2;
        int64_t count = dim0 * dim1 * dim2;
        arr->data = malloc(count * sizeof(f16));
        if (arr->data && data) {
            f32_array_to_f16(data, arr->data, count);
        }
    }
    return arr;
}

int futhark_free_f16_1d(struct futhark_context *ctx, struct futhark_f16_1d *arr) {
    (void)ctx;
    if (arr) {
        free(arr->data);
        free(arr);
    }
    return 0;
}

int futhark_free_f16_2d(struct futhark_context *ctx, struct futhark_f16_2d *arr) {
    (void)ctx;
    if (arr) {
        free(arr->data);
        free(arr);
    }
    return 0;
}

int futhark_free_f16_3d(struct futhark_context *ctx, struct futhark_f16_3d *arr) {
    (void)ctx;
    if (arr) {
        free(arr->data);
        free(arr);
    }
    return 0;
}

int futhark_values_f16_1d(struct futhark_context *ctx, struct futhark_f16_1d *arr, f16 *data) {
    (void)ctx;
    if (arr && arr->data && data) {
        memcpy(data, arr->data, arr->shape[0] * sizeof(f16));
        return 0;
    }
    return 1;
}

int futhark_values_f16_2d(struct futhark_context *ctx, struct futhark_f16_2d *arr, f16 *data) {
    (void)ctx;
    if (arr && arr->data && data) {
        memcpy(data, arr->data, arr->shape[0] * arr->shape[1] * sizeof(f16));
        return 0;
    }
    return 1;
}

int futhark_values_f16_3d(struct futhark_context *ctx, struct futhark_f16_3d *arr, f16 *data) {
    (void)ctx;
    if (arr && arr->data && data) {
        memcpy(data, arr->data, arr->shape[0] * arr->shape[1] * arr->shape[2] * sizeof(f16));
        return 0;
    }
    return 1;
}

int futhark_values_f16_2d_to_f32(struct futhark_context *ctx, struct futhark_f16_2d *arr, float *data) {
    (void)ctx;
    if (arr && arr->data && data) {
        int64_t count = arr->shape[0] * arr->shape[1];
        f16_array_to_f32(arr->data, data, count);
        return 0;
    }
    return 1;
}

int futhark_values_f16_3d_to_f32(struct futhark_context *ctx, struct futhark_f16_3d *arr, float *data) {
    (void)ctx;
    if (arr && arr->data && data) {
        int64_t count = arr->shape[0] * arr->shape[1] * arr->shape[2];
        f16_array_to_f32(arr->data, data, count);
        return 0;
    }
    return 1;
}

void *futhark_values_raw_f16_2d(struct futhark_context *ctx, struct futhark_f16_2d *arr) {
    (void)ctx;
    if (arr) {
        return arr->data;
    }
    return NULL;
}

int futhark_shape_f16_2d(struct futhark_context *ctx, struct futhark_f16_2d *arr, int64_t *dims) {
    (void)ctx;
    if (arr && dims) {
        dims[0] = arr->shape[0];
        dims[1] = arr->shape[1];
        return 0;
    }
    return 1;
}

struct futhark_f32_1d *futhark_new_f32_1d(struct futhark_context *ctx, const float *data, int64_t dim0) {
    (void)ctx;
    struct futhark_f32_1d *arr = malloc(sizeof(struct futhark_f32_1d));
    if (arr) {
        arr->shape[0] = dim0;
        arr->data = malloc(dim0 * sizeof(float));
        if (arr->data && data) {
            memcpy(arr->data, data, dim0 * sizeof(float));
        }
    }
    return arr;
}

struct futhark_f32_2d *futhark_new_f32_2d(struct futhark_context *ctx, const float *data, int64_t dim0, int64_t dim1) {
    (void)ctx;
    struct futhark_f32_2d *arr = malloc(sizeof(struct futhark_f32_2d));
    if (arr) {
        arr->shape[0] = dim0;
        arr->shape[1] = dim1;
        arr->data = malloc(dim0 * dim1 * sizeof(float));
        if (arr->data && data) {
            memcpy(arr->data, data, dim0 * dim1 * sizeof(float));
        }
    }
    return arr;
}

struct futhark_f32_3d *futhark_new_f32_3d(struct futhark_context *ctx, const float *data, int64_t dim0, int64_t dim1, int64_t dim2) {
    (void)ctx;
    struct futhark_f32_3d *arr = malloc(sizeof(struct futhark_f32_3d));
    if (arr) {
        arr->shape[0] = dim0;
        arr->shape[1] = dim1;
        arr->shape[2] = dim2;
        arr->data = malloc(dim0 * dim1 * dim2 * sizeof(float));
        if (arr->data && data) {
            memcpy(arr->data, data, dim0 * dim1 * dim2 * sizeof(float));
        }
    }
    return arr;
}

struct futhark_u64_1d *futhark_new_u64_1d(struct futhark_context *ctx, const uint64_t *data, int64_t dim0) {
    (void)ctx;
    struct futhark_u64_1d *arr = malloc(sizeof(struct futhark_u64_1d));
    if (arr) {
        arr->shape[0] = dim0;
        arr->data = malloc(dim0 * sizeof(uint64_t));
        if (arr->data && data) {
            memcpy(arr->data, data, dim0 * sizeof(uint64_t));
        }
    }
    return arr;
}

struct futhark_i64_1d *futhark_new_i64_1d(struct futhark_context *ctx, const int64_t *data, int64_t dim0) {
    (void)ctx;
    struct futhark_i64_1d *arr = malloc(sizeof(struct futhark_i64_1d));
    if (arr) {
        arr->shape[0] = dim0;
        arr->data = malloc(dim0 * sizeof(int64_t));
        if (arr->data && data) {
            memcpy(arr->data, data, dim0 * sizeof(int64_t));
        }
    }
    return arr;
}

void futhark_free_f32_1d(struct futhark_context *ctx, struct futhark_f32_1d *arr) {
    (void)ctx;
    if (arr) {
        free(arr->data);
        free(arr);
    }
}

void futhark_free_f32_2d(struct futhark_context *ctx, struct futhark_f32_2d *arr) {
    (void)ctx;
    if (arr) {
        free(arr->data);
        free(arr);
    }
}

void futhark_free_f32_3d(struct futhark_context *ctx, struct futhark_f32_3d *arr) {
    (void)ctx;
    if (arr) {
        free(arr->data);
        free(arr);
    }
}

void futhark_free_u64_1d(struct futhark_context *ctx, struct futhark_u64_1d *arr) {
    (void)ctx;
    if (arr) {
        free(arr->data);
        free(arr);
    }
}

void futhark_free_i64_1d(struct futhark_context *ctx, struct futhark_i64_1d *arr) {
    (void)ctx;
    if (arr) {
        free(arr->data);
        free(arr);
    }
}

int futhark_values_f32_1d(struct futhark_context *ctx, struct futhark_f32_1d *arr, float *data) {
    (void)ctx;
    if (arr && arr->data && data) {
        memcpy(data, arr->data, arr->shape[0] * sizeof(float));
        return 0;
    }
    return 1;
}

int futhark_values_f32_2d(struct futhark_context *ctx, struct futhark_f32_2d *arr, float *data) {
    (void)ctx;
    if (arr && arr->data && data) {
        memcpy(data, arr->data, arr->shape[0] * arr->shape[1] * sizeof(float));
        return 0;
    }
    return 1;
}

int futhark_values_f32_3d(struct futhark_context *ctx, struct futhark_f32_3d *arr, float *data) {
    (void)ctx;
    if (arr && arr->data && data) {
        memcpy(data, arr->data, arr->shape[0] * arr->shape[1] * arr->shape[2] * sizeof(float));
        return 0;
    }
    return 1;
}

int futhark_values_u64_1d(struct futhark_context *ctx, struct futhark_u64_1d *arr, uint64_t *data) {
    (void)ctx;
    if (arr && arr->data && data) {
        memcpy(data, arr->data, arr->shape[0] * sizeof(uint64_t));
        return 0;
    }
    return 1;
}

int futhark_values_i64_1d(struct futhark_context *ctx, struct futhark_i64_1d *arr, int64_t *data) {
    (void)ctx;
    if (arr && arr->data && data) {
        memcpy(data, arr->data, arr->shape[0] * sizeof(int64_t));
        return 0;
    }
    return 1;
}

int futhark_entry_matmul(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_f32_2d *a, const struct futhark_f32_2d *b) {
    (void)ctx;
    if (!a || !b || !out) return 1;

    int64_t m = a->shape[0];
    int64_t k = a->shape[1];
    int64_t n = b->shape[1];

    if (k != b->shape[0]) return 1;

    *out = malloc(sizeof(struct futhark_f32_2d));
    if (!*out) return 1;

    (*out)->shape[0] = m;
    (*out)->shape[1] = n;
    (*out)->data = calloc(m * n, sizeof(float));
    if (!(*out)->data) {
        free(*out);
        return 1;
    }

    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int64_t kk = 0; kk < k; kk++) {
                sum += a->data[i * k + kk] * b->data[kk * n + j];
            }
            (*out)->data[i * n + j] = sum;
        }
    }

    return 0;
}

int futhark_entry_batch_matmul(struct futhark_context *ctx, struct futhark_f32_3d **out, const struct futhark_f32_3d *a, const struct futhark_f32_3d *c) {
    (void)ctx;
    if (!a || !c || !out) return 1;

    int64_t batch = a->shape[0];
    int64_t m = a->shape[1];
    int64_t k = a->shape[2];
    int64_t n = c->shape[2];

    if (batch != c->shape[0] || k != c->shape[1]) return 1;

    *out = malloc(sizeof(struct futhark_f32_3d));
    if (!*out) return 1;

    (*out)->shape[0] = batch;
    (*out)->shape[1] = m;
    (*out)->shape[2] = n;
    (*out)->data = calloc(batch * m * n, sizeof(float));
    if (!(*out)->data) {
        free(*out);
        return 1;
    }

    for (int64_t b = 0; b < batch; b++) {
        for (int64_t i = 0; i < m; i++) {
            for (int64_t j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int64_t kk = 0; kk < k; kk++) {
                    sum += a->data[b * m * k + i * k + kk] * c->data[b * k * n + kk * n + j];
                }
                (*out)->data[b * m * n + i * n + j] = sum;
            }
        }
    }

    return 0;
}

int futhark_entry_dot(struct futhark_context *ctx, float *out, const struct futhark_f32_1d *a, const struct futhark_f32_1d *b) {
    (void)ctx;
    if (!a || !b || !out || a->shape[0] != b->shape[0]) return 1;

    float sum = 0.0f;
    for (int64_t i = 0; i < a->shape[0]; i++) {
        sum += a->data[i] * b->data[i];
    }
    *out = sum;

    return 0;
}

int futhark_entry_apply_softmax(struct futhark_context *ctx, struct futhark_f32_1d **out, const struct futhark_f32_1d *x) {
    (void)ctx;
    if (!x || !out) return 1;

    int64_t n = x->shape[0];

    *out = malloc(sizeof(struct futhark_f32_1d));
    if (!*out) return 1;

    (*out)->shape[0] = n;
    (*out)->data = malloc(n * sizeof(float));
    if (!(*out)->data) {
        free(*out);
        return 1;
    }

    float max_val = x->data[0];
    for (int64_t i = 1; i < n; i++) {
        if (x->data[i] > max_val) max_val = x->data[i];
    }

    float sum = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        (*out)->data[i] = expf(x->data[i] - max_val);
        sum += (*out)->data[i];
    }

    for (int64_t i = 0; i < n; i++) {
        (*out)->data[i] /= sum;
    }

    return 0;
}

int futhark_entry_apply_layer_norm(struct futhark_context *ctx, struct futhark_f32_1d **out, const struct futhark_f32_1d *x, const struct futhark_f32_1d *gamma, const struct futhark_f32_1d *beta, float eps) {
    (void)ctx;
    if (!x || !gamma || !beta || !out) return 1;

    int64_t n = x->shape[0];
    if (gamma->shape[0] != n || beta->shape[0] != n) return 1;

    *out = malloc(sizeof(struct futhark_f32_1d));
    if (!*out) return 1;

    (*out)->shape[0] = n;
    (*out)->data = malloc(n * sizeof(float));
    if (!(*out)->data) {
        free(*out);
        return 1;
    }

    float mean = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        mean += x->data[i];
    }
    mean /= (float)n;

    float variance = 0.0f;
    for (int64_t i = 0; i < n; i++) {
        float diff = x->data[i] - mean;
        variance += diff * diff;
    }
    variance /= (float)n;

    float std_dev = sqrtf(variance + eps);

    for (int64_t i = 0; i < n; i++) {
        (*out)->data[i] = gamma->data[i] * ((x->data[i] - mean) / std_dev) + beta->data[i];
    }

    return 0;
}

int futhark_entry_apply_relu(struct futhark_context *ctx, struct futhark_f32_1d **out, const struct futhark_f32_1d *x) {
    (void)ctx;
    if (!x || !out) return 1;

    int64_t n = x->shape[0];

    *out = malloc(sizeof(struct futhark_f32_1d));
    if (!*out) return 1;

    (*out)->shape[0] = n;
    (*out)->data = malloc(n * sizeof(float));
    if (!(*out)->data) {
        free(*out);
        return 1;
    }

    for (int64_t i = 0; i < n; i++) {
        (*out)->data[i] = x->data[i] > 0.0f ? x->data[i] : 0.0f;
    }

    return 0;
}

int futhark_entry_apply_gelu(struct futhark_context *ctx, struct futhark_f32_1d **out, const struct futhark_f32_1d *x) {
    (void)ctx;
    if (!x || !out) return 1;

    int64_t n = x->shape[0];
    const float sqrt_2_over_pi = 0.7978845608f;

    *out = malloc(sizeof(struct futhark_f32_1d));
    if (!*out) return 1;

    (*out)->shape[0] = n;
    (*out)->data = malloc(n * sizeof(float));
    if (!(*out)->data) {
        free(*out);
        return 1;
    }

    for (int64_t i = 0; i < n; i++) {
        float xi = x->data[i];
        float cdf = 0.5f * (1.0f + tanhf(sqrt_2_over_pi * (xi + 0.044715f * xi * xi * xi)));
        (*out)->data[i] = xi * cdf;
    }

    return 0;
}

int futhark_entry_clip_fisher(struct futhark_context *ctx, struct futhark_f32_1d **out, const struct futhark_f32_1d *fisher, float clip_val) {
    (void)ctx;
    if (!fisher || !out) return 1;

    int64_t n = fisher->shape[0];

    *out = malloc(sizeof(struct futhark_f32_1d));
    if (!*out) return 1;

    (*out)->shape[0] = n;
    (*out)->data = malloc(n * sizeof(float));
    if (!(*out)->data) {
        free(*out);
        return 1;
    }

    for (int64_t i = 0; i < n; i++) {
        (*out)->data[i] = fisher->data[i] > clip_val ? fisher->data[i] : clip_val;
    }

    return 0;
}

int futhark_entry_reduce_gradients(struct futhark_context *ctx, struct futhark_f32_1d **out, const struct futhark_f32_2d *gradients) {
    (void)ctx;
    if (!gradients || !out) return 1;

    int64_t batch = gradients->shape[0];
    int64_t n = gradients->shape[1];

    *out = malloc(sizeof(struct futhark_f32_1d));
    if (!*out) return 1;

    (*out)->shape[0] = n;
    (*out)->data = calloc(n, sizeof(float));
    if (!(*out)->data) {
        free(*out);
        return 1;
    }

    for (int64_t b = 0; b < batch; b++) {
        for (int64_t i = 0; i < n; i++) {
            (*out)->data[i] += gradients->data[b * n + i];
        }
    }

    return 0;
}

int futhark_entry_rank_segments(struct futhark_context *ctx, struct futhark_f32_1d **out, uint64_t query_hash, const struct futhark_u64_1d *segment_hashes, const struct futhark_f32_1d *base_scores) {
    (void)ctx;
    if (!segment_hashes || !base_scores || !out) return 1;

    int64_t n = segment_hashes->shape[0];
    if (base_scores->shape[0] != n) return 1;

    *out = malloc(sizeof(struct futhark_f32_1d));
    if (!*out) return 1;

    (*out)->shape[0] = n;
    (*out)->data = malloc(n * sizeof(float));
    if (!(*out)->data) {
        free(*out);
        return 1;
    }

    for (int64_t i = 0; i < n; i++) {
        float match_bonus = (segment_hashes->data[i] == query_hash) ? 1.0f : 0.0f;
        (*out)->data[i] = base_scores->data[i] + match_bonus;
    }

    return 0;
}

static int compare_scores_desc(const void *a, const void *b) {
    typedef struct {
        float score;
        int64_t index;
    } ScoreIndex;
    const ScoreIndex *pa = a;
    const ScoreIndex *pb = b;
    if (pa->score > pb->score) return -1;
    if (pa->score < pb->score) return 1;
    return 0;
}

int futhark_entry_select_topk(struct futhark_context *ctx, struct futhark_f32_1d **out_scores, struct futhark_i64_1d **out_indices, int64_t k, const struct futhark_f32_1d *scores) {
    (void)ctx;
    if (!scores || !out_scores || !out_indices) return 1;

    int64_t n = scores->shape[0];
    if (k > n) k = n;

    typedef struct {
        float score;
        int64_t index;
    } ScoreIndex;

    ScoreIndex *pairs = malloc(n * sizeof(ScoreIndex));
    if (!pairs) return 1;

    for (int64_t i = 0; i < n; i++) {
        pairs[i].score = scores->data[i];
        pairs[i].index = i;
    }

    qsort(pairs, n, sizeof(ScoreIndex), compare_scores_desc);

    *out_scores = malloc(sizeof(struct futhark_f32_1d));
    *out_indices = malloc(sizeof(struct futhark_i64_1d));

    if (!*out_scores || !*out_indices) {
        free(pairs);
        return 1;
    }

    (*out_scores)->shape[0] = k;
    (*out_scores)->data = malloc(k * sizeof(float));
    (*out_indices)->shape[0] = k;
    (*out_indices)->data = malloc(k * sizeof(int64_t));

    if (!(*out_scores)->data || !(*out_indices)->data) {
        free(pairs);
        free(*out_scores);
        free(*out_indices);
        return 1;
    }

    for (int64_t i = 0; i < k; i++) {
        (*out_scores)->data[i] = pairs[i].score;
        (*out_indices)->data[i] = pairs[i].index;
    }

    free(pairs);
    return 0;
}

static void f16_matmul(const f16 *a, const f16 *b, f16 *out, int64_t m, int64_t k, int64_t n) {
    for (int64_t i = 0; i < m; i++) {
        for (int64_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int64_t kk = 0; kk < k; kk++) {
                float av = f16_to_f32(a[i * k + kk]);
                float bv = f16_to_f32(b[kk * n + j]);
                sum += av * bv;
            }
            out[i * n + j] = f32_to_f16(sum);
        }
    }
}

static void f16_relu_2d(const f16 *in, f16 *out, int64_t n, int64_t d) {
    for (int64_t i = 0; i < n * d; i++) {
        float val = f16_to_f32(in[i]);
        out[i] = f32_to_f16(val > 0.0f ? val : 0.0f);
    }
}

static void f16_layer_norm_2d(const f16 *in, f16 *out, int64_t n, int64_t d) {
    const float epsilon = 1e-5f;
    for (int64_t i = 0; i < n; i++) {
        float mean = 0.0f;
        for (int64_t j = 0; j < d; j++) {
            mean += f16_to_f32(in[i * d + j]);
        }
        mean /= (float)d;
        
        float variance = 0.0f;
        for (int64_t j = 0; j < d; j++) {
            float diff = f16_to_f32(in[i * d + j]) - mean;
            variance += diff * diff;
        }
        variance /= (float)d;
        float std_dev = sqrtf(variance + epsilon);
        
        for (int64_t j = 0; j < d; j++) {
            float val = (f16_to_f32(in[i * d + j]) - mean) / std_dev;
            out[i * d + j] = f32_to_f16(val);
        }
    }
}

int futhark_entry_rsf_forward(struct futhark_context *ctx, struct futhark_f16_2d **out,
                               struct futhark_f16_2d *input,
                               struct futhark_f16_2d *weights_s,
                               struct futhark_f16_2d *weights_t) {
    (void)ctx;
    if (!input || !weights_s || !weights_t || !out) return 1;
    
    int64_t n = input->shape[0];
    int64_t d = input->shape[1];
    
    if (weights_s->shape[0] != d || weights_s->shape[1] != d) return 1;
    if (weights_t->shape[0] != d || weights_t->shape[1] != d) return 1;
    
    *out = malloc(sizeof(struct futhark_f16_2d));
    if (!*out) return 1;
    
    (*out)->shape[0] = n;
    (*out)->shape[1] = d;
    int64_t total = n * d;
    (*out)->data = malloc(total * sizeof(f16));
    if (!(*out)->data) {
        free(*out);
        return 1;
    }
    
    f16 *temp1 = malloc(total * sizeof(f16));
    f16 *temp2 = malloc(total * sizeof(f16));
    if (!temp1 || !temp2) {
        free(temp1);
        free(temp2);
        free((*out)->data);
        free(*out);
        return 1;
    }
    
    f16_matmul(input->data, weights_s->data, temp1, n, d, d);
    f16_relu_2d(temp1, temp2, n, d);
    f16_matmul(temp2, weights_t->data, temp1, n, d, d);
    f16_relu_2d(temp1, temp2, n, d);
    f16_layer_norm_2d(temp2, (*out)->data, n, d);
    
    free(temp1);
    free(temp2);
    return 0;
}

int futhark_entry_rsf_backward(struct futhark_context *ctx, struct futhark_f16_2d **out,
                                struct futhark_f16_2d *grad_output,
                                struct futhark_f16_2d *weights) {
    (void)ctx;
    if (!grad_output || !weights || !out) return 1;
    
    int64_t n = grad_output->shape[0];
    int64_t d = grad_output->shape[1];
    
    *out = malloc(sizeof(struct futhark_f16_2d));
    if (!*out) return 1;
    
    (*out)->shape[0] = n;
    (*out)->shape[1] = d;
    (*out)->data = malloc(n * d * sizeof(f16));
    if (!(*out)->data) {
        free(*out);
        return 1;
    }
    
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < d; j++) {
            float sum = 0.0f;
            for (int64_t k = 0; k < d; k++) {
                float gv = f16_to_f32(grad_output->data[i * d + k]);
                float wv = f16_to_f32(weights->data[j * d + k]);
                sum += gv * wv;
            }
            (*out)->data[i * d + j] = f32_to_f16(sum);
        }
    }
    
    return 0;
}

int futhark_entry_scale_weights_inplace(struct futhark_context *ctx,
                                         struct futhark_f16_2d *weights,
                                         float scale) {
    (void)ctx;
    if (!weights) return 1;
    
    int64_t total = weights->shape[0] * weights->shape[1];
    for (int64_t i = 0; i < total; i++) {
        float val = f16_to_f32(weights->data[i]);
        weights->data[i] = f32_to_f16(val / scale);
    }
    
    return 0;
}

int futhark_entry_training_step(
    struct futhark_context *ctx,
    struct futhark_f16_2d **new_weights_s,
    struct futhark_f16_2d **new_weights_t,
    struct futhark_f16_2d **new_velocity_s,
    struct futhark_f16_2d **new_velocity_t,
    f16 *loss,
    struct futhark_f16_2d *inputs,
    struct futhark_f16_2d *targets,
    struct futhark_f16_2d *weights_s,
    struct futhark_f16_2d *weights_t,
    struct futhark_f16_2d *velocity_s,
    struct futhark_f16_2d *velocity_t,
    f16 learning_rate,
    f16 momentum
) {
    (void)ctx;
    if (!inputs || !targets || !weights_s || !weights_t || !velocity_s || !velocity_t) return 1;
    if (!new_weights_s || !new_weights_t || !new_velocity_s || !new_velocity_t || !loss) return 1;
    
    int64_t n = inputs->shape[0];
    int64_t d = inputs->shape[1];
    int64_t weight_size = d * d;
    
    float lr = f16_to_f32(learning_rate);
    float mom = f16_to_f32(momentum);
    
    f16 *outputs = malloc(n * d * sizeof(f16));
    if (!outputs) return 1;
    
    struct futhark_f16_2d temp_input = {inputs->data, {n, d}};
    struct futhark_f16_2d *fwd_out = NULL;
    if (futhark_entry_rsf_forward(ctx, &fwd_out, &temp_input, weights_s, weights_t) != 0) {
        free(outputs);
        return 1;
    }
    memcpy(outputs, fwd_out->data, n * d * sizeof(f16));
    futhark_free_f16_2d(ctx, fwd_out);
    
    float total_loss = 0.0f;
    for (int64_t i = 0; i < n * d; i++) {
        float diff = f16_to_f32(outputs[i]) - f16_to_f32(targets->data[i]);
        total_loss += diff * diff;
    }
    *loss = f32_to_f16(total_loss / (float)(n * d));
    
    f16 *grad_s = calloc(weight_size, sizeof(f16));
    f16 *grad_t = calloc(weight_size, sizeof(f16));
    if (!grad_s || !grad_t) {
        free(outputs);
        free(grad_s);
        free(grad_t);
        return 1;
    }
    
    for (int64_t i = 0; i < n; i++) {
        for (int64_t j = 0; j < d; j++) {
            float grad_out = 2.0f * (f16_to_f32(outputs[i * d + j]) - f16_to_f32(targets->data[i * d + j]));
            for (int64_t k = 0; k < d; k++) {
                float inp = f16_to_f32(inputs->data[i * d + k]);
                float g = f16_to_f32(grad_s[j * d + k]) + grad_out * inp;
                grad_s[j * d + k] = f32_to_f16(g);
            }
        }
    }
    
    *new_weights_s = malloc(sizeof(struct futhark_f16_2d));
    *new_weights_t = malloc(sizeof(struct futhark_f16_2d));
    *new_velocity_s = malloc(sizeof(struct futhark_f16_2d));
    *new_velocity_t = malloc(sizeof(struct futhark_f16_2d));
    
    if (!*new_weights_s || !*new_weights_t || !*new_velocity_s || !*new_velocity_t) {
        free(outputs);
        free(grad_s);
        free(grad_t);
        return 1;
    }
    
    (*new_weights_s)->shape[0] = d;
    (*new_weights_s)->shape[1] = d;
    (*new_weights_s)->data = malloc(weight_size * sizeof(f16));
    
    (*new_weights_t)->shape[0] = d;
    (*new_weights_t)->shape[1] = d;
    (*new_weights_t)->data = malloc(weight_size * sizeof(f16));
    
    (*new_velocity_s)->shape[0] = d;
    (*new_velocity_s)->shape[1] = d;
    (*new_velocity_s)->data = malloc(weight_size * sizeof(f16));
    
    (*new_velocity_t)->shape[0] = d;
    (*new_velocity_t)->shape[1] = d;
    (*new_velocity_t)->data = malloc(weight_size * sizeof(f16));
    
    for (int64_t i = 0; i < weight_size; i++) {
        float old_v_s = f16_to_f32(velocity_s->data[i]);
        float new_v_s = mom * old_v_s + lr * f16_to_f32(grad_s[i]);
        (*new_velocity_s)->data[i] = f32_to_f16(new_v_s);
        (*new_weights_s)->data[i] = f32_to_f16(f16_to_f32(weights_s->data[i]) - new_v_s);
        
        float old_v_t = f16_to_f32(velocity_t->data[i]);
        float new_v_t = mom * old_v_t + lr * f16_to_f32(grad_t[i]);
        (*new_velocity_t)->data[i] = f32_to_f16(new_v_t);
        (*new_weights_t)->data[i] = f32_to_f16(f16_to_f32(weights_t->data[i]) - new_v_t);
    }
    
    free(outputs);
    free(grad_s);
    free(grad_t);
    return 0;
}