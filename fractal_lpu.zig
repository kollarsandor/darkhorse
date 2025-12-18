
const std = @import("std");
const Allocator = std.mem.Allocator;
const SSRG = @import("../../g/core_relational/nsir_core.zig").SelfSimilarRelationalGraph;
const MemoryArbiter = @import("../rtl/MemoryArbiter.hs");
const SSISearch = @import("../rtl/SSISearch.hs");

pub const FractalDimensionConfig = struct {
    hausdorff_dim: f64,
    box_counting_levels: usize,
    min_tile_size: usize,
    max_tile_size: usize,
    coherence_threshold: f64,
};

pub const FractalTile = struct {
    level: usize,
    base_addr: u64,
    size: usize,
    children: []?*FractalTile,
    arbiter_id: u32,
    compute_units: []ComputeUnit,
    coherence: f64,
    entanglement_map: std.AutoHashMap(u64, f64),
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, level: usize, base: u64, size: usize, coherence: f64) !Self {
        const num_children: usize = if (level > 0) 4 else 0;
        var children = try allocator.alloc(?*FractalTile, num_children);
        @memset(children, null);
        const num_cu: usize = 1 << @min(level, 6);
        var compute_units = try allocator.alloc(ComputeUnit, num_cu);
        var i: usize = 0;
        while (i < num_cu) : (i += 1) {
            compute_units[i] = ComputeUnit.init(i, base + i * (size / num_cu));
        }
        return Self{
            .level = level,
            .base_addr = base,
            .size = size,
            .children = children,
            .arbiter_id = @intCast(level),
            .compute_units = compute_units,
            .coherence = coherence,
            .entanglement_map = std.AutoHashMap(u64, f64).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        for (self.children) |child_opt| {
            if (child_opt) |child| {
                child.deinit();
                self.allocator.destroy(child);
            }
        }
        self.allocator.free(self.children);
        self.allocator.free(self.compute_units);
        self.entanglement_map.deinit();
    }

    pub fn subdivide(self: *Self, config: FractalDimensionConfig) !void {
        if (self.size <= config.min_tile_size) return;
        if (self.level >= config.box_counting_levels) return;
        const child_size = self.size / 4;
        if (child_size < config.min_tile_size) return;
        var idx: usize = 0;
        while (idx < 4) : (idx += 1) {
            const child_base = self.base_addr + idx * child_size;
            const child_coherence = self.coherence * 0.9;
            const child_ptr = try self.allocator.create(FractalTile);
            child_ptr.* = try FractalTile.init(self.allocator, self.level + 1, child_base, child_size, child_coherence);
            self.children[idx] = child_ptr;
        }
    }

    pub fn mapSSRGNode(self: *Self, node_hash: u64, weight: f64) !void {
        try self.entanglement_map.put(node_hash, weight);
        const cu_idx = @mod(node_hash, @as(u64, self.compute_units.len));
        self.compute_units[@intCast(cu_idx)].pending_ops += 1;
    }

    pub fn balanceLoad(self: *Self) void {
        if (self.compute_units.len < 2) return;
        var total_ops: u64 = 0;
        for (self.compute_units) |cu| total_ops += cu.pending_ops;
        const avg = total_ops / self.compute_units.len;
        for (self.compute_units) |*cu| {
            if (cu.pending_ops > avg * 2) cu.pending_ops = avg * 2;
        }
    }

    pub fn executeFixedPoint(self: *Self, input: []const i32, output: []i32) void {
        const num_cu = self.compute_units.len;
        const chunk_size = input.len / num_cu;
        var cu_idx: usize = 0;
        while (cu_idx < num_cu) : (cu_idx += 1) {
            const start = cu_idx * chunk_size;
            const end = if (cu_idx == num_cu - 1) input.len else (cu_idx + 1) * chunk_size;
            var i = start;
            while (i < end) : (i += 1) {
                const scaled = @as(i64, input[i]) * @as(i64, @intFromFloat(self.coherence * 65536.0));
                output[i] = @intCast(@divTrunc(scaled, 65536));
            }
            self.compute_units[cu_idx].pending_ops = 0;
        }
    }
};

pub const ComputeUnit = struct {
    id: usize,
    base_addr: u64,
    pending_ops: u64,

    pub fn init(id: usize, base: u64) ComputeUnit {
        return .{ .id = id, .base_addr = base, .pending_ops = 0 };
    }
};

pub const FractalLPU = struct {
    root_tile: *FractalTile,
    config: FractalDimensionConfig,
    total_memory: usize,
    allocator: Allocator,

    const Self = @This();

    pub fn init(allocator: Allocator, total_mem: usize, hausdorff: f64) !Self {
        const config = FractalDimensionConfig{
            .hausdorff_dim = hausdorff,
            .box_counting_levels = 4,
            .min_tile_size = 4096,
            .max_tile_size = total_mem,
            .coherence_threshold = 0.7,
        };
        const root = try allocator.create(FractalTile);
        root.* = try FractalTile.init(allocator, 0, 0, total_mem, 1.0);
        return Self{
            .root_tile = root,
            .config = config,
            .total_memory = total_mem,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Self) void {
        self.root_tile.deinit();
        self.allocator.destroy(self.root_tile);
    }

    pub fn buildFromSSRG(self: *Self, ssrg: *const SSRG) !void {
        try self.root_tile.subdivide(self.config);
        for (self.root_tile.children) |child_opt| {
            if (child_opt) |child| try child.subdivide(self.config);
        }
        var node_iter = ssrg.nodes.iterator();
        while (node_iter.next()) |entry| {
            const node_hash = std.hash.Wyhash.hash(0, entry.key_ptr.*);
            const weight = entry.value_ptr.quantum_state.magnitude();
            try self.mapNodeToTile(self.root_tile, node_hash, weight);
        }
        self.balanceAllTiles(self.root_tile);
    }

    fn mapNodeToTile(self: *Self, tile: *FractalTile, hash: u64, weight: f64) !void {
        try tile.mapSSRGNode(hash, weight);
        if (tile.children.len > 0 and weight > self.config.coherence_threshold) {
            const child_idx = @mod(hash >> 16, @as(u64, tile.children.len));
            if (tile.children[@intCast(child_idx)]) |child| {
                try self.mapNodeToTile(child, hash, weight * 0.9);
            }
        }
    }

    fn balanceAllTiles(self: *Self, tile: *FractalTile) void {
        _ = self;
        tile.balanceLoad();
        for (tile.children) |child_opt| {
            if (child_opt) |child| self.balanceAllTiles(child);
        }
    }

    pub fn processFixedPointBatch(self: *Self, inputs: []const i32, outputs: []i32) void {
        self.root_tile.executeFixedPoint(inputs, outputs);
    }
};
