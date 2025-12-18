const std = @import("std");
const cuda = @import("cuda_bindings.zig");
const futhark = @import("futhark_bindings.zig");

pub const FutharkContext = opaque {
    pub fn init() !*FutharkContext {
        const cfg = futhark.futhark_context_config_new();
        if (cfg == null) return error.FutharkConfigFailed;
        defer futhark.futhark_context_config_free(cfg);

        futhark.futhark_context_config_set_device(cfg, 0);
        futhark.futhark_context_config_set_default_group_size(cfg, 256);
        futhark.futhark_context_config_set_default_num_groups(cfg, 128);
        futhark.futhark_context_config_set_default_tile_size(cfg, 32);

        const ctx = futhark.futhark_context_new(cfg);
        if (ctx == null) return error.FutharkContextFailed;

        if (futhark.futhark_context_sync(ctx) != 0) {
            futhark.futhark_context_free(ctx);
            return error.FutharkSyncFailed;
        }

        return @ptrCast(ctx);
    }

    pub fn deinit(self: *FutharkContext) void {
        const ctx: *futhark.struct_futhark_context = @ptrCast(self);
        futhark.futhark_context_free(ctx);
    }

    pub fn sync(self: *FutharkContext) !void {
        const ctx: *futhark.struct_futhark_context = @ptrCast(self);
        if (futhark.futhark_context_sync(ctx) != 0) {
            return error.FutharkSyncFailed;
        }
    }

    pub fn getDevicePointer(self: *FutharkContext, array: *FutharkArray2DF16) !*anyopaque {
        const ctx: *futhark.struct_futhark_context = @ptrCast(self);
        const arr: *futhark.struct_futhark_f16_2d = @ptrCast(array);

        const raw_ptr = futhark.futhark_values_raw_f16_2d(ctx, arr);
        if (raw_ptr == null) {
            return error.NullDevicePointer;
        }

        return @ptrCast(raw_ptr);
    }
};

pub const PinnedMemory = struct {
    ptr: *anyopaque,
    size: usize,

    pub fn alloc(size: usize) !PinnedMemory {
        var ptr: ?*anyopaque = null;
        const err = cuda.cudaHostAlloc(@as(**anyopaque, @ptrCast(&ptr)), size, cuda.cudaHostAllocDefault);
        if (err != cuda.cudaSuccess) {
            return error.CudaHostAllocFailed;
        }

        return PinnedMemory{
            .ptr = ptr.?,
            .size = size,
        };
    }

    pub fn free(self: *PinnedMemory) void {
        _ = cuda.cudaFreeHost(self.ptr);
    }

    pub fn asSlice(self: *PinnedMemory, comptime T: type) []T {
        const count = self.size / @sizeOf(T);
        return @as([*]T, @ptrCast(@alignCast(self.ptr)))[0..count];
    }
};

pub const FutharkArray2DF16 = opaque {
    pub fn new(ctx: *FutharkContext, data: []const []const f16) !*FutharkArray2DF16 {
        const fctx: *futhark.struct_futhark_context = @ptrCast(ctx);
        const rows = @as(i64, @intCast(data.len));
        const cols = @as(i64, @intCast(data[0].len));

        var flat_data = std.ArrayList(f16).init(std.heap.page_allocator);
        defer flat_data.deinit();

        for (data) |row| {
            try flat_data.appendSlice(row);
        }

        const arr = futhark.futhark_new_f16_2d(fctx, @ptrCast(flat_data.items.ptr), rows, cols);
        if (arr == null) return error.FutharkArrayNewFailed;

        return @ptrCast(arr);
    }

    pub fn newFromFlat(ctx: *FutharkContext, flat_data: []const f16, rows: usize, cols: usize) !*FutharkArray2DF16 {
        const fctx: *futhark.struct_futhark_context = @ptrCast(ctx);
        const r = @as(i64, @intCast(rows));
        const c_val = @as(i64, @intCast(cols));

        const arr = futhark.futhark_new_f16_2d(fctx, @ptrCast(@constCast(flat_data.ptr)), r, c_val);
        if (arr == null) return error.FutharkArrayNewFailed;

        return @ptrCast(arr);
    }

    pub fn newZeros(ctx: *FutharkContext, rows: usize, cols: usize) !*FutharkArray2DF16 {
        const fctx: *futhark.struct_futhark_context = @ptrCast(ctx);
        const r = @as(i64, @intCast(rows));
        const c_val = @as(i64, @intCast(cols));

        const zeros = try std.heap.page_allocator.alloc(f16, rows * cols);
        defer std.heap.page_allocator.free(zeros);
        @memset(zeros, 0);

        const arr = futhark.futhark_new_f16_2d(fctx, @ptrCast(zeros.ptr), r, c_val);
        if (arr == null) return error.FutharkArrayNewFailed;

        return @ptrCast(arr);
    }

    pub fn free(self: *FutharkArray2DF16, ctx: *FutharkContext) void {
        const fctx: *futhark.struct_futhark_context = @ptrCast(ctx);
        const arr: *futhark.struct_futhark_f16_2d = @ptrCast(self);
        _ = futhark.futhark_free_f16_2d(fctx, arr);
    }

    pub fn values(self: *FutharkArray2DF16, ctx: *FutharkContext, allocator: std.mem.Allocator) ![][]f16 {
        const fctx: *futhark.struct_futhark_context = @ptrCast(ctx);
        const arr: *futhark.struct_futhark_f16_2d = @ptrCast(self);

        var dims: [2]i64 = undefined;
        _ = futhark.futhark_shape_f16_2d(fctx, arr, &dims);
        const rows = @as(usize, @intCast(dims[0]));
        const cols = @as(usize, @intCast(dims[1]));

        var flat = try allocator.alloc(f16, rows * cols);
        defer allocator.free(flat);

        if (futhark.futhark_values_f16_2d(fctx, arr, @ptrCast(flat.ptr)) != 0) {
            return error.FutharkValuesFailed;
        }

        var result = try allocator.alloc([]f16, rows);
        var i: usize = 0;
        while (i < rows) : (i += 1) {
            result[i] = try allocator.alloc(f16, cols);
            @memcpy(result[i], flat[i * cols .. (i + 1) * cols]);
        }

        return result;
    }
};

pub const RSFAccelerator = struct {
    ctx: *FutharkContext,
    weights_s: *FutharkArray2DF16,
    weights_t: *FutharkArray2DF16,
    velocity_s: *FutharkArray2DF16,
    velocity_t: *FutharkArray2DF16,
    model_dim: usize,

    pub fn init(model_dim: usize) !RSFAccelerator {
        const ctx = try FutharkContext.init();
        errdefer ctx.deinit();

        const weights_s = try FutharkArray2DF16.newZeros(ctx, model_dim, model_dim);
        errdefer weights_s.free(ctx);

        const weights_t = try FutharkArray2DF16.newZeros(ctx, model_dim, model_dim);
        errdefer weights_t.free(ctx);

        const velocity_s = try FutharkArray2DF16.newZeros(ctx, model_dim, model_dim);
        errdefer velocity_s.free(ctx);

        const velocity_t = try FutharkArray2DF16.newZeros(ctx, model_dim, model_dim);
        errdefer velocity_t.free(ctx);

        return RSFAccelerator{
            .ctx = ctx,
            .weights_s = weights_s,
            .weights_t = weights_t,
            .velocity_s = velocity_s,
            .velocity_t = velocity_t,
            .model_dim = model_dim,
        };
    }

    pub fn deinit(self: *RSFAccelerator) void {
        self.velocity_t.free(self.ctx);
        self.velocity_s.free(self.ctx);
        self.weights_t.free(self.ctx);
        self.weights_s.free(self.ctx);
        self.ctx.deinit();
    }

    pub fn forward(self: *RSFAccelerator, input: *FutharkArray2DF16) !*FutharkArray2DF16 {
        const fctx: *futhark.struct_futhark_context = @ptrCast(self.ctx);
        const inp: *futhark.struct_futhark_f16_2d = @ptrCast(input);
        const ws: *futhark.struct_futhark_f16_2d = @ptrCast(self.weights_s);
        const wt: *futhark.struct_futhark_f16_2d = @ptrCast(self.weights_t);

        var output: ?*futhark.struct_futhark_f16_2d = null;
        const result = futhark.futhark_entry_rsf_forward(fctx, @as(**futhark.struct_futhark_f16_2d, @ptrCast(&output)), inp, ws, wt);

        if (result != 0) {
            return error.FutharkForwardFailed;
        }

        return @ptrCast(output.?);
    }

    pub fn trainingStep(
        self: *RSFAccelerator,
        inputs: *FutharkArray2DF16,
        targets: *FutharkArray2DF16,
        learning_rate: f16,
        momentum: f16,
    ) !f16 {
        const fctx: *futhark.struct_futhark_context = @ptrCast(self.ctx);
        const inp: *futhark.struct_futhark_f16_2d = @ptrCast(inputs);
        const tgt: *futhark.struct_futhark_f16_2d = @ptrCast(targets);
        const ws: *futhark.struct_futhark_f16_2d = @ptrCast(self.weights_s);
        const wt: *futhark.struct_futhark_f16_2d = @ptrCast(self.weights_t);
        const vs: *futhark.struct_futhark_f16_2d = @ptrCast(self.velocity_s);
        const vt: *futhark.struct_futhark_f16_2d = @ptrCast(self.velocity_t);

        var new_ws: ?*futhark.struct_futhark_f16_2d = null;
        var new_wt: ?*futhark.struct_futhark_f16_2d = null;
        var new_vs: ?*futhark.struct_futhark_f16_2d = null;
        var new_vt: ?*futhark.struct_futhark_f16_2d = null;
        var loss: u16 = undefined;

        const lr_bits: u16 = @bitCast(learning_rate);
        const momentum_bits: u16 = @bitCast(momentum);

        const result = futhark.futhark_entry_training_step(
            fctx,
            @as(**futhark.struct_futhark_f16_2d, @ptrCast(&new_ws)),
            @as(**futhark.struct_futhark_f16_2d, @ptrCast(&new_wt)),
            @as(**futhark.struct_futhark_f16_2d, @ptrCast(&new_vs)),
            @as(**futhark.struct_futhark_f16_2d, @ptrCast(&new_vt)),
            &loss,
            inp,
            tgt,
            ws,
            wt,
            vs,
            vt,
            lr_bits,
            momentum_bits,
        );

        if (result != 0) {
            return error.FutharkTrainingStepFailed;
        }

        self.weights_s.free(self.ctx);
        self.weights_t.free(self.ctx);
        self.velocity_s.free(self.ctx);
        self.velocity_t.free(self.ctx);

        self.weights_s = @ptrCast(new_ws.?);
        self.weights_t = @ptrCast(new_wt.?);
        self.velocity_s = @ptrCast(new_vs.?);
        self.velocity_t = @ptrCast(new_vt.?);

        const loss_f16: f16 = @bitCast(loss);
        return loss_f16;
    }

    pub fn scaleWeightsInplace(self: *RSFAccelerator, scale_factor: f16) !void {
        const fctx: *futhark.struct_futhark_context = @ptrCast(self.ctx);
        const ws: *futhark.struct_futhark_f16_2d = @ptrCast(self.weights_s);
        const wt: *futhark.struct_futhark_f16_2d = @ptrCast(self.weights_t);

        const scale_f32: f32 = @floatCast(scale_factor);

        const result_s = futhark.futhark_entry_scale_weights_inplace(
            fctx,
            ws,
            scale_f32,
        );

        if (result_s != 0) {
            return error.FutharkScaleWeightsFailed;
        }

        const result_t = futhark.futhark_entry_scale_weights_inplace(
            fctx,
            wt,
            scale_f32,
        );

        if (result_t != 0) {
            return error.FutharkScaleWeightsFailed;
        }
    }

    pub fn getWeightsSDevicePointer(self: *RSFAccelerator) !*anyopaque {
        return try self.ctx.getDevicePointer(self.weights_s);
    }

    pub fn getWeightsTDevicePointer(self: *RSFAccelerator) !*anyopaque {
        return try self.ctx.getDevicePointer(self.weights_t);
    }

    pub fn sync(self: *RSFAccelerator) !void {
        try self.ctx.sync();
    }
};