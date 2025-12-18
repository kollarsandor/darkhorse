const std = @import("std");
const mem = std.mem;
const math = std.math;
const Allocator = mem.Allocator;
const types = @import("types.zig");
const Error = types.Error;
const Fixed32_32 = types.Fixed32_32;

pub const Tensor = struct {
    data: []f32,
    shape: []usize,
    strides: []usize,
    ndim: usize,
    allocator: Allocator,
    refcount: *usize,

    pub fn init(allocator: Allocator, shape: []const usize) !Tensor {
        if (shape.len == 0) return Error.EmptyInput;
        var total_size: usize = 1;
        for (shape) |dim| {
            if (dim == 0) return Error.InvalidShape;
            total_size *= dim;
        }
        const data = try allocator.alloc(f32, total_size);
        @memset(data, 0);
        var strides = try allocator.alloc(usize, shape.len);
        strides[shape.len - 1] = 1;
        if (shape.len > 1) {
            var i: usize = shape.len - 1;
            while (i > 0) : (i -= 1) {
                strides[i - 1] = strides[i] * shape[i];
            }
        }
        const refcount = try allocator.create(usize);
        refcount.* = 1;
        return .{ .data = data, .shape = try allocator.dupe(usize, shape), .strides = strides, .ndim = shape.len, .allocator = allocator, .refcount = refcount };
    }

    pub fn retain(self: *Tensor) void {
        _ = @atomicRmw(usize, self.refcount, .Add, 1, .AcqRel);
    }

    pub fn release(self: *Tensor) void {
        const prev = @atomicRmw(usize, self.refcount, .Sub, 1, .AcqRel);
        if (prev == 1) {
            self.deallocate();
        }
    }

    fn deallocate(self: *Tensor) void {
        self.allocator.free(self.data);
        self.allocator.free(self.shape);
        self.allocator.free(self.strides);
        self.allocator.destroy(self.refcount);
    }

    pub fn deinit(self: *Tensor) void {
        const prev = @atomicRmw(usize, self.refcount, .Sub, 1, .AcqRel);
        if (prev == 1) {
            self.deallocate();
        }
    }

    pub fn copy(self: *const Tensor, allocator: Allocator) !Tensor {
        const new_t = try init(allocator, self.shape);
        @memcpy(new_t.data, self.data);
        return new_t;
    }

    pub fn reshape(self: *Tensor, new_shape: []const usize) !void {
        if (new_shape.len == 0) return Error.InvalidShape;
        var total: usize = 1;
        for (new_shape) |dim| total *= dim;
        if (total != self.data.len) return Error.InvalidShape;
        self.allocator.free(self.shape);
        self.shape = try self.allocator.dupe(usize, new_shape);
        self.ndim = new_shape.len;
        self.allocator.free(self.strides);
        self.strides = try self.allocator.alloc(usize, new_shape.len);
        self.strides[new_shape.len - 1] = 1;
        if (new_shape.len > 1) {
            var i: usize = new_shape.len - 1;
            while (i > 0) : (i -= 1) {
                self.strides[i - 1] = self.strides[i] * new_shape[i];
            }
        }
    }

    pub fn unsqueeze(self: *const Tensor, allocator: Allocator, axis: usize) !Tensor {
        if (axis > self.ndim) return Error.InvalidAxis;
        var new_shape = try allocator.alloc(usize, self.ndim + 1);
        var j: usize = 0;
        var i: usize = 0;
        while (i < self.ndim + 1) : (i += 1) {
            if (i == axis) {
                new_shape[i] = 1;
            } else {
                new_shape[i] = self.shape[j];
                j += 1;
            }
        }
        const result = try init(allocator, new_shape);
        @memcpy(result.data, self.data);
        allocator.free(new_shape);
        return result;
    }

    pub fn transpose(self: *const Tensor, allocator: Allocator, axes: []const usize) !Tensor {
        if (axes.len != self.ndim) return Error.InvalidAxis;
        var new_shape = try allocator.alloc(usize, self.ndim);
        var i: usize = 0;
        while (i < self.ndim) : (i += 1) {
            new_shape[i] = self.shape[axes[i]];
        }
        var new_strides = try allocator.alloc(usize, self.ndim);
        i = 0;
        while (i < self.ndim) : (i += 1) {
            new_strides[i] = self.strides[axes[i]];
        }
        var new_t = try self.copy(allocator);
        allocator.free(new_t.shape);
        allocator.free(new_t.strides);
        new_t.shape = new_shape;
        new_t.strides = new_strides;
        return new_t;
    }

    pub fn get(self: *const Tensor, indices: []const usize) !f32 {
        if (indices.len != self.ndim) return Error.InvalidAxis;
        var idx: usize = 0;
        var i: usize = 0;
        while (i < self.ndim) : (i += 1) {
            if (indices[i] >= self.shape[i]) return Error.OutOfBounds;
            idx += indices[i] * self.strides[i];
        }
        return self.data[idx];
    }

    pub fn set(self: *Tensor, indices: []const usize, value: f32) !void {
        if (indices.len != self.ndim) return Error.InvalidAxis;
        var idx: usize = 0;
        var i: usize = 0;
        while (i < self.ndim) : (i += 1) {
            if (indices[i] >= self.shape[i]) return Error.OutOfBounds;
            idx += indices[i] * self.strides[i];
        }
        self.data[idx] = value;
    }

    pub fn fill(self: *Tensor, value: f32) void {
        @memset(self.data, value);
    }

    pub fn add(self: *Tensor, other: *const Tensor) !void {
        if (!mem.eql(usize, self.shape, other.shape)) return Error.ShapeMismatch;
        var i: usize = 0;
        while (i < self.data.len) : (i += 1) {
            self.data[i] += other.data[i];
        }
    }

    pub fn sub(self: *Tensor, other: *const Tensor) !void {
        if (!mem.eql(usize, self.shape, other.shape)) return Error.ShapeMismatch;
        var i: usize = 0;
        while (i < self.data.len) : (i += 1) {
            self.data[i] -= other.data[i];
        }
    }

    pub fn mul(self: *Tensor, other: *const Tensor) !void {
        if (!mem.eql(usize, self.shape, other.shape)) return Error.ShapeMismatch;
        var i: usize = 0;
        while (i < self.data.len) : (i += 1) {
            self.data[i] *= other.data[i];
        }
    }

    pub fn div(self: *Tensor, other: *const Tensor) !void {
        if (!mem.eql(usize, self.shape, other.shape)) return Error.ShapeMismatch;
        var i: usize = 0;
        while (i < self.data.len) : (i += 1) {
            if (other.data[i] == 0) return Error.DivideByZero;
            self.data[i] /= other.data[i];
        }
    }

    pub fn addScalar(self: *Tensor, scalar: f32) void {
        for (self.data) |*val| {
            val.* += scalar;
        }
    }

    pub fn subScalar(self: *Tensor, scalar: f32) void {
        for (self.data) |*val| {
            val.* -= scalar;
        }
    }

    pub fn mulScalar(self: *Tensor, scalar: f32) void {
        for (self.data) |*val| {
            val.* *= scalar;
        }
    }

    pub fn divScalar(self: *Tensor, scalar: f32) void {
        if (scalar == 0) return;
        for (self.data) |*val| {
            val.* /= scalar;
        }
    }

    pub fn exp(self: *Tensor) void {
        for (self.data) |*val| {
            val.* = @exp(val.*);
        }
    }

    pub fn log(self: *Tensor) void {
        for (self.data) |*val| {
            if (val.* <= 0.0) {
                val.* = -math.inf(f32);
            } else {
                val.* = @log(val.*);
            }
        }
    }

    pub fn sin(self: *Tensor) void {
        for (self.data) |*val| {
            val.* = @sin(val.*);
        }
    }

    pub fn cos(self: *Tensor) void {
        for (self.data) |*val| {
            val.* = @cos(val.*);
        }
    }

    pub fn tan(self: *Tensor) void {
        for (self.data) |*val| {
            val.* = @tan(val.*);
        }
    }

    pub fn sqrt(self: *Tensor) void {
        for (self.data) |*val| {
            if (val.* < 0.0) {
                val.* = math.nan(f32);
            } else {
                val.* = @sqrt(val.*);
            }
        }
    }

    pub fn pow(self: *Tensor, exponent: f32) void {
        for (self.data) |*val| {
            if (val.* < 0.0 and @floor(exponent) != exponent) {
                val.* = math.nan(f32);
            } else if (val.* == 0.0 and exponent < 0.0) {
                val.* = math.inf(f32);
            } else {
                val.* = math.pow(f32, val.*, exponent);
            }
        }
    }

    pub fn abs(self: *Tensor) void {
        for (self.data) |*val| {
            val.* = math.fabs(val.*);
        }
    }

    pub fn max(self: *const Tensor, allocator: Allocator, axis: usize) !Tensor {
        if (axis >= self.ndim) return Error.InvalidAxis;
        var reduced_shape = try allocator.alloc(usize, self.ndim - 1);
        defer allocator.free(reduced_shape);
        var j: usize = 0;
        var i: usize = 0;
        while (i < self.ndim) : (i += 1) {
            if (i != axis) {
                reduced_shape[j] = self.shape[i];
                j += 1;
            }
        }
        const result = try init(allocator, reduced_shape);
        var out_idx: usize = 0;
        var in_indices = try allocator.alloc(usize, self.ndim);
        defer allocator.free(in_indices);
        @memset(in_indices, 0);
        while (true) {
            var max_val: f32 = -math.inf(f32);
            var k: usize = 0;
            while (k < self.shape[axis]) : (k += 1) {
                in_indices[axis] = k;
                const val = try self.get(in_indices);
                if (val > max_val) max_val = val;
            }
            result.data[out_idx] = max_val;
            out_idx += 1;
            var carry = true;
            var dim = self.ndim;
            while (carry and dim > 0) : (dim -= 1) {
                if (dim - 1 != axis) {
                    in_indices[dim - 1] += 1;
                    if (in_indices[dim - 1] < self.shape[dim - 1]) {
                        carry = false;
                    } else {
                        in_indices[dim - 1] = 0;
                    }
                }
            }
            if (carry) break;
        }
        return result;
    }

    pub fn min(self: *const Tensor, allocator: Allocator, axis: usize) !Tensor {
        if (axis >= self.ndim) return Error.InvalidAxis;
        var reduced_shape = try allocator.alloc(usize, self.ndim - 1);
        defer allocator.free(reduced_shape);
        var j: usize = 0;
        var i: usize = 0;
        while (i < self.ndim) : (i += 1) {
            if (i != axis) {
                reduced_shape[j] = self.shape[i];
                j += 1;
            }
        }
        const result = try init(allocator, reduced_shape);
        var out_idx: usize = 0;
        var in_indices = try allocator.alloc(usize, self.ndim);
        defer allocator.free(in_indices);
        @memset(in_indices, 0);
        while (true) {
            var min_val: f32 = math.inf(f32);
            var k: usize = 0;
            while (k < self.shape[axis]) : (k += 1) {
                in_indices[axis] = k;
                const val = try self.get(in_indices);
                if (val < min_val) min_val = val;
            }
            result.data[out_idx] = min_val;
            out_idx += 1;
            var carry = true;
            var dim = self.ndim;
            while (carry and dim > 0) : (dim -= 1) {
                if (dim - 1 != axis) {
                    in_indices[dim - 1] += 1;
                    if (in_indices[dim - 1] < self.shape[dim - 1]) {
                        carry = false;
                    } else {
                        in_indices[dim - 1] = 0;
                    }
                }
            }
            if (carry) break;
        }
        return result;
    }

    pub fn sum(self: *const Tensor, allocator: Allocator, axis: usize) !Tensor {
        if (axis >= self.ndim) return Error.InvalidAxis;
        var reduced_shape = try allocator.alloc(usize, self.ndim - 1);
        defer allocator.free(reduced_shape);
        var j: usize = 0;
        var i: usize = 0;
        while (i < self.ndim) : (i += 1) {
            if (i != axis) {
                reduced_shape[j] = self.shape[i];
                j += 1;
            }
        }
        const result = try init(allocator, reduced_shape);
        var out_idx: usize = 0;
        var in_indices = try allocator.alloc(usize, self.ndim);
        defer allocator.free(in_indices);
        @memset(in_indices, 0);
        while (true) {
            var total: f32 = 0.0;
            var k: usize = 0;
            while (k < self.shape[axis]) : (k += 1) {
                in_indices[axis] = k;
                total += try self.get(in_indices);
            }
            result.data[out_idx] = total;
            out_idx += 1;
            var carry = true;
            var dim = self.ndim;
            while (carry and dim > 0) : (dim -= 1) {
                if (dim - 1 != axis) {
                    in_indices[dim - 1] += 1;
                    if (in_indices[dim - 1] < self.shape[dim - 1]) {
                        carry = false;
                    } else {
                        in_indices[dim - 1] = 0;
                    }
                }
            }
            if (carry) break;
        }
        return result;
    }

    pub fn mean(self: *const Tensor, allocator: Allocator, axis: usize) !Tensor {
        var summed = try self.sum(allocator, axis);
        const axis_size = self.shape[axis];
        summed.divScalar(@as(f32, @floatFromInt(axis_size)));
        return summed;
    }

    pub fn variance(self: *const Tensor, allocator: Allocator, axis: usize) !Tensor {
        var mean_t = try self.mean(allocator, axis);
        defer mean_t.deinit();
        var mean_unsqueezed = try mean_t.unsqueeze(allocator, axis);
        defer mean_unsqueezed.deinit();
        var mean_broadcasted = try mean_unsqueezed.broadcast(allocator, self.shape);
        defer mean_broadcasted.deinit();
        var diff = try self.copy(allocator);
        defer diff.deinit();
        try diff.sub(&mean_broadcasted);
        var sq = try diff.copy(allocator);
        defer sq.deinit();
        try sq.mul(&diff);
        return try sq.mean(allocator, axis);
    }

    pub fn stddev(self: *const Tensor, allocator: Allocator, axis: usize) !Tensor {
        var var_t = try self.variance(allocator, axis);
        var_t.sqrt();
        return var_t;
    }

    pub fn argmax(self: *const Tensor, allocator: Allocator, axis: usize) !Tensor {
        if (axis >= self.ndim) return Error.InvalidAxis;
        var reduced_shape = try allocator.alloc(usize, self.ndim - 1);
        defer allocator.free(reduced_shape);
        var j: usize = 0;
        var i: usize = 0;
        while (i < self.ndim) : (i += 1) {
            if (i != axis) {
                reduced_shape[j] = self.shape[i];
                j += 1;
            }
        }
        const result = try init(allocator, reduced_shape);
        var out_idx: usize = 0;
        var in_indices = try allocator.alloc(usize, self.ndim);
        defer allocator.free(in_indices);
        @memset(in_indices, 0);
        while (true) {
            var max_idx: usize = 0;
            var max_val: f32 = try self.get(in_indices);
            var k: usize = 1;
            while (k < self.shape[axis]) : (k += 1) {
                in_indices[axis] = k;
                const val = try self.get(in_indices);
                if (val > max_val) {
                    max_val = val;
                    max_idx = k;
                }
            }
            in_indices[axis] = 0;
            result.data[out_idx] = @as(f32, @floatFromInt(max_idx));
            out_idx += 1;
            var carry = true;
            var dim = self.ndim;
            while (carry and dim > 0) : (dim -= 1) {
                if (dim - 1 != axis) {
                    in_indices[dim - 1] += 1;
                    if (in_indices[dim - 1] < self.shape[dim - 1]) {
                        carry = false;
                    } else {
                        in_indices[dim - 1] = 0;
                    }
                }
            }
            if (carry) break;
        }
        return result;
    }

    pub fn argmin(self: *const Tensor, allocator: Allocator, axis: usize) !Tensor {
        if (axis >= self.ndim) return Error.InvalidAxis;
        var reduced_shape = try allocator.alloc(usize, self.ndim - 1);
        defer allocator.free(reduced_shape);
        var j: usize = 0;
        var i: usize = 0;
        while (i < self.ndim) : (i += 1) {
            if (i != axis) {
                reduced_shape[j] = self.shape[i];
                j += 1;
            }
        }
        const result = try init(allocator, reduced_shape);
        var out_idx: usize = 0;
        var in_indices = try allocator.alloc(usize, self.ndim);
        defer allocator.free(in_indices);
        @memset(in_indices, 0);
        while (true) {
            var min_idx: usize = 0;
            var min_val: f32 = try self.get(in_indices);
            var k: usize = 1;
            while (k < self.shape[axis]) : (k += 1) {
                in_indices[axis] = k;
                const val = try self.get(in_indices);
                if (val < min_val) {
                    min_val = val;
                    min_idx = k;
                }
            }
            in_indices[axis] = 0;
            result.data[out_idx] = @as(f32, @floatFromInt(min_idx));
            out_idx += 1;
            var carry = true;
            var dim = self.ndim;
            while (carry and dim > 0) : (dim -= 1) {
                if (dim - 1 != axis) {
                    in_indices[dim - 1] += 1;
                    if (in_indices[dim - 1] < self.shape[dim - 1]) {
                        carry = false;
                    } else {
                        in_indices[dim - 1] = 0;
                    }
                }
            }
            if (carry) break;
        }
        return result;
    }

    fn isBroadcastCompatible(self_shape: []const usize, target_shape: []const usize) bool {
        if (target_shape.len < self_shape.len) return false;
        const offset = target_shape.len - self_shape.len;
        var i: usize = 0;
        while (i < self_shape.len) : (i += 1) {
            const self_dim = self_shape[i];
            const target_dim = target_shape[offset + i];
            if (self_dim != target_dim and self_dim != 1) {
                return false;
            }
        }
        return true;
    }

    pub fn broadcast(self: *const Tensor, allocator: Allocator, target_shape: []const usize) !Tensor {
        if (!isBroadcastCompatible(self.shape, target_shape)) {
            return Error.ShapeMismatch;
        }
        var new_t = try init(allocator, target_shape);
        var target_indices = try allocator.alloc(usize, target_shape.len);
        defer allocator.free(target_indices);
        @memset(target_indices, 0);
        const offset = target_shape.len - self.ndim;
        var flat_idx: usize = 0;
        while (flat_idx < new_t.data.len) : (flat_idx += 1) {
            var src_idx: usize = 0;
            var i: usize = 0;
            while (i < self.ndim) : (i += 1) {
                const target_i = target_indices[offset + i];
                const src_i = if (self.shape[i] == 1) 0 else target_i;
                src_idx += src_i * self.strides[i];
            }
            new_t.data[flat_idx] = self.data[src_idx];
            var carry = true;
            var dim = target_shape.len;
            while (carry and dim > 0) : (dim -= 1) {
                target_indices[dim - 1] += 1;
                if (target_indices[dim - 1] < target_shape[dim - 1]) {
                    carry = false;
                } else {
                    target_indices[dim - 1] = 0;
                }
            }
            if (carry) break;
        }
        return new_t;
    }

    pub fn matmul(a: *const Tensor, b: *const Tensor, allocator: Allocator) !Tensor {
        if (a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]) return Error.ShapeMismatch;
        const m = a.shape[0];
        const k = a.shape[1];
        const n = b.shape[1];
        const c = try init(allocator, &.{ m, n });
        var i: usize = 0;
        while (i < m) : (i += 1) {
            var j: usize = 0;
            while (j < n) : (j += 1) {
                var sum_result: f32 = 0.0;
                var l: usize = 0;
                while (l < k) : (l += 1) {
                    sum_result += a.data[i * k + l] * b.data[l * n + j];
                }
                c.data[i * n + j] = sum_result;
            }
        }
        return c;
    }

    pub fn conv2d(self: *const Tensor, kernel: *const Tensor, allocator: Allocator, stride: [2]usize, padding: [2]usize) !Tensor {
        if (self.ndim != 4 or kernel.ndim != 4 or self.shape[3] != kernel.shape[2]) return Error.InvalidConv2D;
        const batch = self.shape[0];
        const in_h = self.shape[1];
        const in_w = self.shape[2];
        const in_c = self.shape[3];
        const k_h = kernel.shape[0];
        const k_w = kernel.shape[1];
        const out_c = kernel.shape[3];
        const out_h = ((in_h + 2 * padding[0] - k_h) / stride[0]) + 1;
        const out_w = ((in_w + 2 * padding[1] - k_w) / stride[1]) + 1;
        const output = try init(allocator, &.{ batch, out_h, out_w, out_c });
        var padded_input = if (padding[0] > 0 or padding[1] > 0) try self.pad(allocator, &.{ .{ padding[0], padding[0] }, .{ padding[1], padding[1] }, .{ 0, 0 }, .{ 0, 0 } }) else self.*;
        defer if (padding[0] > 0 or padding[1] > 0) padded_input.deinit();
        var b: usize = 0;
        while (b < batch) : (b += 1) {
            var oh: usize = 0;
            while (oh < out_h) : (oh += 1) {
                var ow: usize = 0;
                while (ow < out_w) : (ow += 1) {
                    var oc: usize = 0;
                    while (oc < out_c) : (oc += 1) {
                        var sum_result: f32 = 0.0;
                        var kh: usize = 0;
                        while (kh < k_h) : (kh += 1) {
                            var kw: usize = 0;
                            while (kw < k_w) : (kw += 1) {
                                var ic: usize = 0;
                                while (ic < in_c) : (ic += 1) {
                                    const ih = oh * stride[0] + kh;
                                    const iw = ow * stride[1] + kw;
                                    sum_result += try padded_input.get(&.{ b, ih, iw, ic }) * try kernel.get(&.{ kh, kw, ic, oc });
                                }
                            }
                        }
                        try output.set(&.{ b, oh, ow, oc }, sum_result);
                    }
                }
            }
        }
        return output;
    }

    pub fn pad(self: *const Tensor, allocator: Allocator, pads: [][2]usize) !Tensor {
        if (pads.len != self.ndim) return Error.InvalidPads;
        var new_shape = try allocator.alloc(usize, self.ndim);
        defer allocator.free(new_shape);
        var i: usize = 0;
        while (i < self.ndim) : (i += 1) {
            new_shape[i] = self.shape[i] + pads[i][0] + pads[i][1];
        }
        const new_t = try init(allocator, new_shape);
        new_t.fill(0.0);
        var indices = try allocator.alloc(usize, self.ndim);
        defer allocator.free(indices);
        @memset(indices, 0);
        while (true) {
            var src_indices = try allocator.alloc(usize, self.ndim);
            defer allocator.free(src_indices);
            var is_pad = false;
            i = 0;
            while (i < self.ndim) : (i += 1) {
                if (indices[i] < pads[i][0] or indices[i] >= pads[i][0] + self.shape[i]) {
                    is_pad = true;
                } else {
                    src_indices[i] = indices[i] - pads[i][0];
                }
            }
            if (!is_pad) {
                const val = try self.get(src_indices);
                try new_t.set(indices, val);
            }
            var carry = true;
            var dim = self.ndim;
            while (carry and dim > 0) : (dim -= 1) {
                indices[dim - 1] += 1;
                if (indices[dim - 1] < new_shape[dim - 1]) {
                    carry = false;
                } else {
                    indices[dim - 1] = 0;
                }
            }
            if (carry) break;
        }
        return new_t;
    }

    pub fn relu(self: *Tensor) void {
        for (self.data) |*val| {
            val.* = @max(0.0, val.*);
        }
    }

    pub fn softmax(self: *Tensor, axis: usize) !void {
        if (axis >= self.ndim) return Error.InvalidAxis;
        
        if (self.ndim == 1) {
            var max_val: f32 = -math.inf(f32);
            for (self.data) |val| {
                if (val > max_val) max_val = val;
            }
            var sum_val: f32 = 0.0;
            for (self.data) |*val| {
                val.* = @exp(val.* - max_val);
                sum_val += val.*;
            }
            for (self.data) |*val| {
                val.* /= sum_val;
            }
            return;
        }
        
        var max_t = try self.max(self.allocator, axis);
        defer max_t.deinit();
        var b_shape = try self.allocator.alloc(usize, self.ndim);
        defer self.allocator.free(b_shape);
        @memcpy(b_shape, self.shape);
        b_shape[axis] = 1;
        var b_max = try max_t.broadcast(self.allocator, b_shape);
        defer b_max.deinit();
        try self.sub(&b_max);
        self.exp();
        var sum_t = try self.sum(self.allocator, axis);
        defer sum_t.deinit();
        var b_sum = try sum_t.broadcast(self.allocator, b_shape);
        defer b_sum.deinit();
        try self.div(&b_sum);
    }

    pub fn toFixed(self: *const Tensor, allocator: Allocator) !Tensor {
        const fixed_t = try init(allocator, self.shape);
        var i: usize = 0;
        while (i < self.data.len) : (i += 1) {
            fixed_t.data[i] = Fixed32_32.init(self.data[i]).toFloat();
        }
        return fixed_t;
    }

    pub fn tile(self: *const Tensor, allocator: Allocator, reps: []const usize) !Tensor {
        if (reps.len != self.ndim) return Error.InvalidReps;
        var new_shape = try allocator.alloc(usize, self.ndim);
        defer allocator.free(new_shape);
        var i: usize = 0;
        while (i < self.ndim) : (i += 1) {
            new_shape[i] = self.shape[i] * reps[i];
        }
        const new_t = try init(allocator, new_shape);
        var indices = try allocator.alloc(usize, self.ndim);
        defer allocator.free(indices);
        @memset(indices, 0);
        while (true) {
            var src_indices = try allocator.alloc(usize, self.ndim);
            defer allocator.free(src_indices);
            i = 0;
            while (i < self.ndim) : (i += 1) {
                src_indices[i] = indices[i] % self.shape[i];
            }
            const val = try self.get(src_indices);
            try new_t.set(indices, val);
            var carry = true;
            var dim = self.ndim;
            while (carry and dim > 0) : (dim -= 1) {
                indices[dim - 1] += 1;
                if (indices[dim - 1] < new_shape[dim - 1]) {
                    carry = false;
                } else {
                    indices[dim - 1] = 0;
                }
            }
            if (carry) break;
        }
        return new_t;
    }

    pub fn clip(self: *Tensor, min_val: f32, max_val: f32) void {
        for (self.data) |*val| {
            val.* = math.clamp(val.*, min_val, max_val);
        }
    }

    pub fn norm(self: *const Tensor, order: f32) f32 {
        var total: f32 = 0.0;
        for (self.data) |val| {
            total += math.pow(f32, math.fabs(val), order);
        }
        return math.pow(f32, total, 1.0 / order);
    }

    pub fn trace(self: *const Tensor) !f32 {
        if (self.ndim != 2 or self.shape[0] != self.shape[1]) return Error.MustBeSquare;
        var sum_result: f32 = 0.0;
        const n = self.shape[0];
        var i: usize = 0;
        while (i < n) : (i += 1) {
            sum_result += self.data[i * n + i];
        }
        return sum_result;
    }

    pub fn det(self: *const Tensor, allocator: Allocator) !f32 {
        if (self.ndim != 2 or self.shape[0] != self.shape[1]) return Error.MustBeSquare;
        const n = self.shape[0];
        var mat = try self.copy(allocator);
        defer mat.deinit();
        var det_val: f32 = 1.0;
        var i: usize = 0;
        while (i < n) : (i += 1) {
            var pivot = i;
            var j: usize = i + 1;
            while (j < n) : (j += 1) {
                if (math.fabs(mat.data[j * n + i]) > math.fabs(mat.data[pivot * n + i])) {
                    pivot = j;
                }
            }
            if (math.fabs(mat.data[pivot * n + i]) < 1e-10) return 0.0;
            if (pivot != i) {
                var k: usize = 0;
                while (k < n) : (k += 1) {
                    const temp = mat.data[i * n + k];
                    mat.data[i * n + k] = mat.data[pivot * n + k];
                    mat.data[pivot * n + k] = temp;
                }
                det_val = -det_val;
            }
            det_val *= mat.data[i * n + i];
            j = i + 1;
            while (j < n) : (j += 1) {
                const factor = mat.data[j * n + i] / mat.data[i * n + i];
                var k: usize = i;
                while (k < n) : (k += 1) {
                    mat.data[j * n + k] -= factor * mat.data[i * n + k];
                }
            }
        }
        return det_val;
    }

    pub fn eye(allocator: Allocator, n: usize) !Tensor {
        const t = try init(allocator, &.{ n, n });
        var i: usize = 0;
        while (i < n) : (i += 1) {
            t.data[i * n + i] = 1.0;
        }
        return t;
    }

    pub fn zeros(allocator: Allocator, shape: []const usize) !Tensor {
        return init(allocator, shape);
    }

    pub fn ones(allocator: Allocator, shape: []const usize) !Tensor {
        var t = try init(allocator, shape);
        t.fill(1.0);
        return t;
    }

    pub fn full(allocator: Allocator, shape: []const usize, value: f32) !Tensor {
        var t = try init(allocator, shape);
        t.fill(value);
        return t;
    }

    pub fn arange(allocator: Allocator, start: f32, end: f32, step: f32) !Tensor {
        const size = @as(usize, @intFromFloat(@ceil((end - start) / step)));
        const t = try init(allocator, &.{size});
        var val = start;
        for (t.data) |*d| {
            d.* = val;
            val += step;
        }
        return t;
    }

    pub fn linspace(allocator: Allocator, start: f32, end: f32, num: usize) !Tensor {
        const t = try init(allocator, &.{num});
        if (num == 0) return t;
        const step = (end - start) / @as(f32, @floatFromInt(num - 1));
        var val = start;
        var i: usize = 0;
        while (i < num - 1) : (i += 1) {
            t.data[i] = val;
            val += step;
        }
        t.data[num - 1] = end;
        return t;
    }

    pub fn randomNormal(allocator: Allocator, shape: []const usize, mean_val: f32, stddev_val: f32, seed: u64) !Tensor {
        var prng = types.PRNG.init(seed);
        const t = try init(allocator, shape);
        for (t.data) |*val| {
            var u = prng.float();
            var v = prng.float();
            while (u <= 0.0) u = prng.float();
            while (v == 0.0) v = prng.float();
            const z = @sqrt(-2.0 * @log(u)) * @cos(2.0 * math.pi * v);
            val.* = mean_val + stddev_val * z;
        }
        return t;
    }

    pub fn randomUniform(allocator: Allocator, shape: []const usize, min_val: f32, max_val: f32, seed: u64) !Tensor {
        var prng = types.PRNG.init(seed);
        const t = try init(allocator, shape);
        for (t.data) |*val| {
            val.* = prng.float() * (max_val - min_val) + min_val;
        }
        return t;
    }

    pub fn slice(self: *const Tensor, starts: []const usize, ends: []const usize, allocator: Allocator) !Tensor {
        if (starts.len != self.ndim or ends.len != self.ndim) return Error.InvalidAxis;
        var new_shape = try allocator.alloc(usize, self.ndim);
        defer allocator.free(new_shape);
        var i: usize = 0;
        while (i < self.ndim) : (i += 1) {
            new_shape[i] = ends[i] - starts[i];
        }
        var new_t = try init(allocator, new_shape);
        var indices = try allocator.alloc(usize, self.ndim);
        defer allocator.free(indices);
        @memset(indices, 0);
        while (true) {
            var src_indices = try allocator.alloc(usize, self.ndim);
            defer allocator.free(src_indices);
            i = 0;
            while (i < self.ndim) : (i += 1) {
                src_indices[i] = starts[i] + indices[i];
            }
            const val = try self.get(src_indices);
            try new_t.set(indices, val);
            var carry = true;
            var dim = self.ndim;
            while (carry and dim > 0) : (dim -= 1) {
                indices[dim - 1] += 1;
                if (indices[dim - 1] < new_shape[dim - 1]) {
                    carry = false;
                } else {
                    indices[dim - 1] = 0;
                }
            }
            if (carry) break;
        }
        return new_t;
    }

    pub fn concat(allocator: Allocator, tensors: []const Tensor, axis: usize) !Tensor {
        if (tensors.len == 0) return Error.EmptyInput;
        const ndim = tensors[0].ndim;
        if (axis >= ndim) return Error.InvalidAxis;
        for (tensors) |ten| {
            if (ten.ndim != ndim) return Error.ShapeMismatch;
            var i: usize = 0;
            while (i < ndim) : (i += 1) {
                if (i != axis and ten.shape[i] != tensors[0].shape[i]) return Error.ShapeMismatch;
            }
        }
        var new_shape = try allocator.alloc(usize, ndim);
        defer allocator.free(new_shape);
        @memcpy(new_shape, tensors[0].shape);
        var total_axis: usize = 0;
        for (tensors) |ten| {
            total_axis += ten.shape[axis];
        }
        new_shape[axis] = total_axis;
        const new_t = try init(allocator, new_shape);
        var reduced_shape = try allocator.alloc(usize, ndim - 1);
        defer allocator.free(reduced_shape);
        var j: usize = 0;
        var i: usize = 0;
        while (i < ndim) : (i += 1) {
            if (i != axis) {
                reduced_shape[j] = tensors[0].shape[i];
                j += 1;
            }
        }
        var common_indices = try allocator.alloc(usize, ndim - 1);
        defer allocator.free(common_indices);
        @memset(common_indices, 0);
        while (true) {
            var current_offset: usize = 0;
            i = 0;
            while (i < tensors.len) : (i += 1) {
                const ten = tensors[i];
                var ax: usize = 0;
                while (ax < ten.shape[axis]) : (ax += 1) {
                    var new_indices = try allocator.alloc(usize, ndim);
                    defer allocator.free(new_indices);
                    var ten_indices = try allocator.alloc(usize, ndim);
                    defer allocator.free(ten_indices);
                    var k: usize = 0;
                    var d: usize = 0;
                    while (d < ndim) : (d += 1) {
                        if (d == axis) {
                            new_indices[d] = current_offset + ax;
                            ten_indices[d] = ax;
                        } else {
                            new_indices[d] = common_indices[k];
                            ten_indices[d] = common_indices[k];
                            k += 1;
                        }
                    }
                    const val = try ten.get(ten_indices);
                    try new_t.set(new_indices, val);
                }
                current_offset += ten.shape[axis];
            }
            var carry = true;
            var dim: isize = @as(isize, @intCast(ndim - 1)) - 1;
            while (carry and dim >= 0) : (dim -= 1) {
                common_indices[@intCast(dim)] += 1;
                if (common_indices[@intCast(dim)] < reduced_shape[@intCast(dim)]) {
                    carry = false;
                } else {
                    common_indices[@intCast(dim)] = 0;
                }
            }
            if (carry) break;
        }
        return new_t;
    }

    pub fn stack(allocator: Allocator, tensors: []const Tensor, axis: usize) !Tensor {
        if (tensors.len == 0) return Error.EmptyInput;
        const ndim = tensors[0].ndim;
        if (axis > ndim) return Error.InvalidAxis;
        for (tensors) |ten| {
            if (ten.ndim != ndim or !mem.eql(usize, ten.shape, tensors[0].shape)) return Error.ShapeMismatch;
        }
        var new_shape = try allocator.alloc(usize, ndim + 1);
        defer allocator.free(new_shape);
        new_shape[axis] = tensors.len;
        var k: usize = 0;
        var i: usize = 0;
        while (i < ndim + 1) : (i += 1) {
            if (i == axis) continue;
            new_shape[i] = tensors[0].shape[k];
            k += 1;
        }
        const new_t = try init(allocator, new_shape);
        var reduced_shape = try allocator.alloc(usize, ndim);
        defer allocator.free(reduced_shape);
        @memcpy(reduced_shape, tensors[0].shape);
        var common_indices = try allocator.alloc(usize, ndim);
        defer allocator.free(common_indices);
        @memset(common_indices, 0);
        while (true) {
            var t_idx: usize = 0;
            while (t_idx < tensors.len) : (t_idx += 1) {
                const ten = tensors[t_idx];
                var new_indices = try allocator.alloc(usize, ndim + 1);
                defer allocator.free(new_indices);
                var ten_indices = try allocator.alloc(usize, ndim);
                defer allocator.free(ten_indices);
                var kk: usize = 0;
                var d: usize = 0;
                while (d < ndim + 1) : (d += 1) {
                    if (d == axis) {
                        new_indices[d] = t_idx;
                    } else {
                        new_indices[d] = common_indices[kk];
                        ten_indices[kk] = common_indices[kk];
                        kk += 1;
                    }
                }
                const val = try ten.get(ten_indices);
                try new_t.set(new_indices, val);
            }
            var carry = true;
            var dim: isize = @as(isize, @intCast(ndim)) - 1;
            while (carry and dim >= 0) : (dim -= 1) {
                common_indices[@intCast(dim)] += 1;
                if (common_indices[@intCast(dim)] < reduced_shape[@intCast(dim)]) {
                    carry = false;
                } else {
                    common_indices[@intCast(dim)] = 0;
                }
            }
            if (carry) break;
        }
        return new_t;
    }

    pub fn unique(self: *const Tensor, allocator: Allocator) !Tensor {
        var unique_set = std.AutoHashMap(f32, void).init(allocator);
        defer unique_set.deinit();
        for (self.data) |val| {
            try unique_set.put(val, {});
        }
        const unique_len = unique_set.count();
        const unique_t = try init(allocator, &.{unique_len});
        var iter = unique_set.iterator();
        var i: usize = 0;
        while (iter.next()) |entry| {
            unique_t.data[i] = entry.key_ptr.*;
            i += 1;
        }
        return unique_t;
    }

    pub fn sort(self: *const Tensor, allocator: Allocator, axis: usize, descending: bool) !Tensor {
        if (axis >= self.ndim) return Error.InvalidAxis;
        const new_t = try self.copy(allocator);
        const Context = struct {
            pub fn lessThan(_: void, a: f32, b: f32) bool {
                return a < b;
            }
            pub fn greaterThan(_: void, a: f32, b: f32) bool {
                return a > b;
            }
        };
        var reduced_shape = try allocator.alloc(usize, self.ndim - 1);
        defer allocator.free(reduced_shape);
        var j: usize = 0;
        var i: usize = 0;
        while (i < self.ndim) : (i += 1) {
            if (i != axis) {
                reduced_shape[j] = self.shape[i];
                j += 1;
            }
        }
        var common_indices = try allocator.alloc(usize, self.ndim - 1);
        defer allocator.free(common_indices);
        @memset(common_indices, 0);
        var temp = try allocator.alloc(f32, self.shape[axis]);
        defer allocator.free(temp);
        while (true) {
            var base_idx: usize = 0;
            var k: usize = 0;
            i = 0;
            while (i < self.ndim) : (i += 1) {
                if (i == axis) {
                    // skip
                } else {
                    base_idx += common_indices[k] * new_t.strides[i];
                    k += 1;
                }
            }
            i = 0;
            while (i < self.shape[axis]) : (i += 1) {
                const idx = base_idx + i * new_t.strides[axis];
                temp[i] = new_t.data[idx];
            }
            if (descending) {
                std.mem.sort(f32, temp, {}, Context.greaterThan);
            } else {
                std.mem.sort(f32, temp, {}, Context.lessThan);
            }
            i = 0;
            while (i < self.shape[axis]) : (i += 1) {
                const idx = base_idx + i * new_t.strides[axis];
                new_t.data[idx] = temp[i];
            }
            var carry = true;
            var dim: isize = @as(isize, @intCast(self.ndim - 1)) - 1;
            while (carry and dim >= 0) : (dim -= 1) {
                common_indices[@intCast(dim)] += 1;
                if (common_indices[@intCast(dim)] < reduced_shape[@intCast(dim)]) {
                    carry = false;
                } else {
                    common_indices[@intCast(dim)] = 0;
                }
            }
            if (carry) break;
        }
        return new_t;
    }

    pub fn cumsum(self: *const Tensor, allocator: Allocator, axis: usize) !Tensor {
        var new_t = try self.copy(allocator);
        if (axis >= self.ndim) return Error.InvalidAxis;
        var indices = try allocator.alloc(usize, self.ndim);
        defer allocator.free(indices);
        @memset(indices, 0);
        while (true) {
            if (indices[axis] > 0) {
                var prev_indices = try allocator.alloc(usize, self.ndim);
                defer allocator.free(prev_indices);
                @memcpy(prev_indices, indices);
                prev_indices[axis] -= 1;
                const prev = try new_t.get(prev_indices);
                const curr = try new_t.get(indices);
                try new_t.set(indices, prev + curr);
            }
            var carry = true;
            var dim = self.ndim;
            while (carry and dim > 0) : (dim -= 1) {
                indices[dim - 1] += 1;
                if (indices[dim - 1] < self.shape[dim - 1]) {
                    carry = false;
                } else {
                    indices[dim - 1] = 0;
                }
            }
            if (carry) break;
        }
        return new_t;
    }

    pub fn oneHot(self: *const Tensor, allocator: Allocator, num_classes: usize) !Tensor {
        if (self.ndim != 1) return Error.InvalidForOneHot;
        const new_shape = &.{ self.shape[0], num_classes };
        const new_t = try init(allocator, new_shape);
        new_t.fill(0.0);
        var i: usize = 0;
        while (i < self.shape[0]) : (i += 1) {
            const idx = @as(usize, @intFromFloat(try self.get(&.{i})));
            if (idx < num_classes) {
                try new_t.set(&.{ i, idx }, 1.0);
            }
        }
        return new_t;
    }

    pub fn isClose(self: *const Tensor, other: *const Tensor, rtol: f32, atol: f32) !bool {
        if (!mem.eql(usize, self.shape, other.shape)) return false;
        var i: usize = 0;
        while (i < self.data.len) : (i += 1) {
            const diff = math.fabs(self.data[i] - other.data[i]);
            if (diff > atol + rtol * math.fabs(other.data[i])) return false;
        }
        return true;
    }

    pub fn toInt(self: *const Tensor, allocator: Allocator) !Tensor {
        const new_t = try init(allocator, self.shape);
        var i: usize = 0;
        while (i < self.data.len) : (i += 1) {
            new_t.data[i] = @trunc(self.data[i]);
        }
        return new_t;
    }

    pub fn spectralNorm(self: *const Tensor, allocator: Allocator, max_iter: u32, tol: f32) !f32 {
        if (self.ndim != 2 or self.shape[0] != self.shape[1]) return Error.MustBeSquare;
        const n = self.shape[0];
        var v = try randomUniform(allocator, &.{n}, -1.0, 1.0, 42);
        defer v.deinit();
        const norm_v = v.norm(2.0);
        v.divScalar(norm_v);
        var last_radius: f32 = 0.0;
        var iter: usize = 0;
        while (iter < max_iter) : (iter += 1) {
            var av = try matmul(self, &v, allocator);
            defer av.deinit();
            const norm_av = av.norm(2.0);
            if (norm_av == 0.0) return 0.0;
            av.divScalar(norm_av);
            v.deinit();
            v = av;
            var radius: f32 = 0.0;
            var i: usize = 0;
            while (i < n) : (i += 1) {
                radius += v.data[i] * self.data[i * n + i];
            }
            if (math.fabs(radius - last_radius) < tol) return math.fabs(radius);
            last_radius = radius;
        }
        return math.fabs(last_radius);
    }

    pub fn normL2(self: *const Tensor) f32 {
        var sum_sq: f32 = 0.0;
        for (self.data) |val| {
            sum_sq += val * val;
        }
        return @sqrt(sum_sq);
    }

    pub fn dot(self: *const Tensor, other: *const Tensor) !f32 {
        if (self.data.len != other.data.len) return Error.ShapeMismatch;
        var sum_result: f32 = 0.0;
        var i: usize = 0;
        while (i < self.data.len) : (i += 1) {
            sum_result += self.data[i] * other.data[i];
        }
        return sum_result;
    }

    pub fn outer(allocator: Allocator, a: *const Tensor, b: *const Tensor) !Tensor {
        if (a.ndim == 1 and b.ndim == 1) {
            const m = a.shape[0];
            const n = b.shape[0];
            const result = try init(allocator, &.{ m, n });
            var i: usize = 0;
            while (i < m) : (i += 1) {
                var j: usize = 0;
                while (j < n) : (j += 1) {
                    result.data[i * n + j] = a.data[i] * b.data[j];
                }
            }
            return result;
        } else if (a.ndim == 2 and b.ndim == 2) {
            if (a.shape[0] != b.shape[0]) return Error.ShapeMismatch;
            const batch = a.shape[0];
            const m = a.shape[1];
            const n = b.shape[1];
            const result = try init(allocator, &.{ m, n });
            var batch_idx: usize = 0;
            while (batch_idx < batch) : (batch_idx += 1) {
                var i: usize = 0;
                while (i < m) : (i += 1) {
                    var j: usize = 0;
                    while (j < n) : (j += 1) {
                        result.data[i * n + j] += a.data[batch_idx * m + i] * b.data[batch_idx * n + j];
                    }
                }
            }
            return result;
        } else {
            return Error.ShapeMismatch;
        }
    }

    pub fn inverse(self: *const Tensor, allocator: Allocator) !Tensor {
        if (self.ndim != 2 or self.shape[0] != self.shape[1]) return Error.MustBeSquare;
        const n = self.shape[0];
        var mat = try self.copy(allocator);
        defer mat.deinit();
        var inv = try eye(allocator, n);
        var i: usize = 0;
        while (i < n) : (i += 1) {
            var pivot = i;
            var j: usize = i + 1;
            while (j < n) : (j += 1) {
                if (math.fabs(mat.data[j * n + i]) > math.fabs(mat.data[pivot * n + i])) {
                    pivot = j;
                }
            }
            if (math.fabs(mat.data[pivot * n + i]) < 1e-10) return Error.SingularMatrix;
            if (pivot != i) {
                var k: usize = 0;
                while (k < n) : (k += 1) {
                    const temp_mat = mat.data[i * n + k];
                    mat.data[i * n + k] = mat.data[pivot * n + k];
                    mat.data[pivot * n + k] = temp_mat;
                    const temp_inv = inv.data[i * n + k];
                    inv.data[i * n + k] = inv.data[pivot * n + k];
                    inv.data[pivot * n + k] = temp_inv;
                }
            }
            const diag = mat.data[i * n + i];
            var k: usize = 0;
            while (k < n) : (k += 1) {
                mat.data[i * n + k] /= diag;
                inv.data[i * n + k] /= diag;
            }
            j = 0;
            while (j < n) : (j += 1) {
                if (j != i) {
                    const factor = mat.data[j * n + i];
                    k = 0;
                    while (k < n) : (k += 1) {
                        mat.data[j * n + k] -= factor * mat.data[i * n + k];
                        inv.data[j * n + k] -= factor * inv.data[i * n + k];
                    }
                }
            }
        }
        return inv;
    }

    pub fn eig(self: *const Tensor, allocator: Allocator) !struct { vals: Tensor, vecs: Tensor } {
        if (self.ndim != 2 or self.shape[0] != self.shape[1]) return Error.MustBeSquare;
        const n = self.shape[0];
        var mat = try self.copy(allocator);
        defer mat.deinit();
        var vecs = try eye(allocator, n);
        var iter: usize = 0;
        while (iter < 100) : (iter += 1) {
            var qr_result = try mat.qr(allocator);
            var new_mat = try matmul(&qr_result.r, &qr_result.q, allocator);
            mat.deinit();
            mat = new_mat;
            var new_vecs = try matmul(&vecs, &qr_result.q, allocator);
            vecs.deinit();
            vecs = new_vecs;
            qr_result.q.deinit();
            qr_result.r.deinit();
            var converged = true;
            var ii: usize = 1;
            while (ii < n) : (ii += 1) {
                if (math.fabs(mat.data[ii * n + (ii - 1)]) > 1e-10) {
                    converged = false;
                    break;
                }
            }
            if (converged) break;
        }
        var vals = try init(allocator, &.{n});
        var i: usize = 0;
        while (i < n) : (i += 1) {
            vals.data[i] = mat.data[i * n + i];
        }
        return .{ .vals = vals, .vecs = vecs };
    }

    pub fn qr(self: *const Tensor, allocator: Allocator) !struct { q: Tensor, r: Tensor } {
        const m = self.shape[0];
        const n = self.shape[1];
        var q = try eye(allocator, m);
        var r = try self.copy(allocator);
        var j: usize = 0;
        while (j < @min(m, n)) : (j += 1) {
            var x = try allocator.alloc(f32, m - j);
            defer allocator.free(x);
            var i: usize = j;
            while (i < m) : (i += 1) {
                x[i - j] = r.data[i * n + j];
            }
            var norm_x: f32 = 0.0;
            for (x) |val| norm_x += val * val;
            norm_x = @sqrt(norm_x);
            if (norm_x == 0.0) continue;
            const sign: f32 = if (x[0] >= 0.0) 1.0 else -1.0;
            var u = try allocator.alloc(f32, m - j);
            defer allocator.free(u);
            u[0] = x[0] + sign * norm_x;
            i = 1;
            while (i < m - j) : (i += 1) u[i] = x[i];
            var norm_u: f32 = 0.0;
            for (u) |val| norm_u += val * val;
            norm_u = @sqrt(norm_u);
            for (u) |*val| val.* /= norm_u;
            var k: usize = j;
            while (k < n) : (k += 1) {
                var dot_prod: f32 = 0.0;
                i = j;
                while (i < m) : (i += 1) {
                    dot_prod += r.data[i * n + k] * u[i - j];
                }
                dot_prod *= 2.0;
                i = j;
                while (i < m) : (i += 1) {
                    r.data[i * n + k] -= dot_prod * u[i - j];
                }
            }
            k = 0;
            while (k < m) : (k += 1) {
                var dot_prod: f32 = 0.0;
                i = j;
                while (i < m) : (i += 1) {
                    dot_prod += q.data[i * m + k] * u[i - j];
                }
                dot_prod *= 2.0;
                i = j;
                while (i < m) : (i += 1) {
                    q.data[i * m + k] -= dot_prod * u[i - j];
                }
            }
        }
        return .{ .q = q, .r = r };
    }

    pub fn svd(self: *const Tensor, allocator: Allocator) !struct { u: Tensor, s: Tensor, v: Tensor } {
        var ata = try self.transpose(allocator, &.{1, 0});
        defer ata.deinit();
        var ata_self = try matmul(&ata, self, allocator);
        defer ata_self.deinit();
        var eigen = try eig(&ata_self, allocator);
        var s = try init(allocator, &.{eigen.vals.shape[0]});
        var i: usize = 0;
        while (i < s.data.len) : (i += 1) {
            s.data[i] = @sqrt(@max(0.0, eigen.vals.data[i]));
        }
        defer eigen.vals.deinit();
        var v = eigen.vecs;
        var u = try matmul(self, &v, allocator);
        i = 0;
        while (i < s.data.len) : (i += 1) {
            if (s.data[i] > 1e-10) {
                var jj: usize = 0;
                while (jj < u.shape[0]) : (jj += 1) {
                    u.data[jj * u.shape[1] + i] /= s.data[i];
                }
            }
        }
        return .{ .u = u, .s = s, .v = v };
    }

    pub fn cholesky(self: *const Tensor, allocator: Allocator) !Tensor {
        if (self.ndim != 2 or self.shape[0] != self.shape[1]) return Error.MustBeSquare;
        const n = self.shape[0];
        var l = try init(allocator, self.shape);
        l.fill(0.0);
        var i: usize = 0;
        while (i < n) : (i += 1) {
            var j: usize = 0;
            while (j < i + 1) : (j += 1) {
                var sum_result: f32 = 0.0;
                var k: usize = 0;
                while (k < j) : (k += 1) {
                    sum_result += l.data[i * n + k] * l.data[j * n + k];
                }
                if (i == j) {
                    l.data[i * n + j] = @sqrt(self.data[i * n + j] - sum_result);
                } else {
                    l.data[i * n + j] = (self.data[i * n + j] - sum_result) / l.data[j * n + j];
                }
            }
        }
        return l;
    }

    pub fn solve(self: *const Tensor, b: *const Tensor, allocator: Allocator) !Tensor {
        const lu_result = try self.lu(allocator);
        defer lu_result.l.deinit();
        defer lu_result.u.deinit();
        var y = try matmul(&lu_result.l, b, allocator);
        defer y.deinit();
        return matmul(&lu_result.u, &y, allocator);
    }

    pub fn lu(self: *const Tensor, allocator: Allocator) !struct { l: Tensor, u: Tensor } {
        const n = self.shape[0];
        var l = try eye(allocator, n);
        var u = try self.copy(allocator);
        var i: usize = 0;
        while (i < n) : (i += 1) {
            var j: usize = i;
            while (j < n) : (j += 1) {
                var sum_result: f32 = 0.0;
                var k: usize = 0;
                while (k < i) : (k += 1) {
                    sum_result += l.data[j * n + k] * u.data[k * n + i];
                }
                u.data[j * n + i] = self.data[j * n + i] - sum_result;
            }
            j = i + 1;
            while (j < n) : (j += 1) {
                var sum_result2: f32 = 0.0;
                var k: usize = 0;
                while (k < i) : (k += 1) {
                    sum_result2 += l.data[j * n + k] * u.data[k * n + i];
                }
                l.data[j * n + i] = (self.data[j * n + i] - sum_result2) / u.data[i * n + i];
            }
        }
        return .{ .l = l, .u = u };
    }

    pub fn toString(self: *const Tensor, allocator: Allocator) ![]u8 {
        var buf = std.ArrayList(u8).init(allocator);
        const writer = buf.writer();
        try writer.print("Tensor(shape=[", .{});
        var i: usize = 0;
        while (i < self.shape.len) : (i += 1) {
            const dim = self.shape[i];
            try writer.print("{d}", .{dim});
            if (i < self.shape.len - 1) try writer.print(", ", .{});
        }
        try writer.print("], data=[", .{});
        i = 0;
        while (i < self.data.len) : (i += 1) {
            const val = self.data[i];
            try writer.print("{d:.4}", .{val});
            if (i < self.data.len - 1) try writer.print(", ", .{});
        }
        try writer.print("])", .{});
        return buf.toOwnedSlice();
    }

    pub fn save(self: *const Tensor, writer: anytype) !void {
        try writer.writeInt(usize, self.ndim, .Little);
        for (self.shape) |dim| {
            try writer.writeInt(usize, dim, .Little);
        }
        for (self.data) |val| {
            try writer.writeAll(mem.asBytes(&val));
        }
    }

    pub fn load(allocator: Allocator, reader: anytype) !Tensor {
        const ndim = try reader.readInt(usize, .Little);
        var shape = try allocator.alloc(usize, ndim);
        errdefer allocator.free(shape);
        var i: usize = 0;
        while (i < ndim) : (i += 1) {
            shape[i] = try reader.readInt(usize, .Little);
        }
        const tensor = try init(allocator, shape);
        allocator.free(shape);
        for (tensor.data) |*val| {
            var buf: [@sizeOf(f32)]u8 = undefined;
            _ = try reader.readAll(&buf);
            val.* = @bitCast(buf);
        }
        return tensor;
    }
};

test "Tensor basic ops" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var t1 = try Tensor.init(gpa, &.{ 2, 2 });
    defer t1.deinit();
    t1.fill(1.0);
    var t2 = try Tensor.init(gpa, &.{ 2, 2 });
    defer t2.deinit();
    t2.fill(2.0);
    try t1.add(&t2);
    try testing.expectEqual(@as(f32, 3.0), t1.data[0]);
}

test "Matmul" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var a = try Tensor.init(gpa, &.{ 2, 3 });
    defer a.deinit();
    @memcpy(a.data, &[_]f32{ 1, 2, 3, 4, 5, 6 });
    var b = try Tensor.init(gpa, &.{ 3, 2 });
    defer b.deinit();
    @memcpy(b.data, &[_]f32{ 7, 8, 9, 10, 11, 12 });
    var c = try Tensor.matmul(&a, &b, gpa);
    defer c.deinit();
    try testing.expectApproxEqAbs(@as(f32, 58.0), c.data[0], 1e-5);
}

test "Sum reduce" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var t = try Tensor.init(gpa, &.{ 2, 3 });
    defer t.deinit();
    @memcpy(t.data, &[_]f32{ 1, 2, 3, 4, 5, 6 });
    var sum = try t.sum(gpa, 1);
    defer sum.deinit();
    try testing.expectEqual(@as(f32, 6.0), sum.data[0]);
    try testing.expectEqual(@as(f32, 15.0), sum.data[1]);
}

test "Broadcast" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var t1 = try Tensor.init(gpa, &.{3});
    defer t1.deinit();
    @memcpy(t1.data, &[_]f32{ 1, 2, 3 });
    var b_t1 = try t1.broadcast(gpa, &.{ 2, 3 });
    defer b_t1.deinit();
    try testing.expectEqual(@as(f32, 1.0), b_t1.data[0]);
}

test "Softmax" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var t = try Tensor.init(gpa, &.{3});
    defer t.deinit();
    @memcpy(t.data, &[_]f32{ 1, 2, 3 });
    try t.softmax(0);
    try testing.expectApproxEqAbs(@as(f32, 0.0900), t.data[0], 1e-3);
}