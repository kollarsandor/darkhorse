const std = @import("std");
const mem = std.mem;
const atomic = std.atomic;
const Allocator = mem.Allocator;
const PageSize = 4096;
const testing = std.testing;

const Mutex = std.Thread.Mutex;
const CondVar = std.Thread.Condition;
const Semaphore = std.Thread.Semaphore;
const AtomicBool = std.atomic.Atomic(bool);
const AtomicU64 = std.atomic.Atomic(u64);
const AtomicUsize = std.atomic.Atomic(usize);

pub const Arena = struct {
    buffer: []u8,
    offset: std.atomic.Atomic(usize),
    allocator: Allocator,
    mutex: Mutex = .{},

    pub fn init(allocator: Allocator, size: usize) !Arena {
        const aligned_size = mem.alignForward(usize, size, PageSize);
        const buffer = try allocator.alignedAlloc(u8, PageSize, aligned_size);
        std.mem.set(u8, buffer, 0);
        return .{ .buffer = buffer, .allocator = allocator, .offset = std.atomic.Atomic(usize).init(0) };
    }

    pub fn deinit(self: *Arena) void {
        self.allocator.free(self.buffer);
    }

    pub fn alloc(self: *Arena, size: usize, alignment: usize) ?[]u8 {
        if (size == 0) return &[_]u8{};

        self.mutex.lock();
        defer self.mutex.unlock();

        const current_offset = self.offset.load(.Acquire);
        const aligned_offset = mem.alignForward(usize, current_offset, alignment);
        const end = aligned_offset + size;

        if (end > self.buffer.len) return null;

        const ptr = self.buffer[aligned_offset..end];
        self.offset.store(end, .Release);
        return ptr;
    }

    pub fn allocBytes(self: *Arena, size: usize) ?[]u8 {
        return self.alloc(size, PageSize);
    }

    pub fn reset(self: *Arena) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.offset.store(0, .Release);
    }

    pub fn allocated(self: *const Arena) usize {
        return self.offset.load(.Acquire);
    }

    pub fn remaining(self: *const Arena) usize {
        return self.buffer.len - self.offset.load(.Acquire);
    }
};

pub const ArenaAllocator = struct {
    parent_allocator: Allocator,
    buffers: std.ArrayList([]u8),
    current_buffer: []u8,
    pos: usize,
    buffer_size: usize,
    mutex: Mutex = .{},

    pub fn init(parent_allocator: Allocator, buffer_size: usize) ArenaAllocator {
        return .{
            .parent_allocator = parent_allocator,
            .buffers = std.ArrayList([]u8).init(parent_allocator),
            .current_buffer = &[_]u8{},
            .pos = 0,
            .buffer_size = buffer_size,
        };
    }

    pub fn deinit(self: *ArenaAllocator) void {
        for (self.buffers.items) |buf| {
            self.parent_allocator.free(buf);
        }
        self.buffers.deinit();
    }

    pub fn allocator(self: *ArenaAllocator) Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = arenaAlloc,
                .resize = arenaResize,
                .free = arenaFree,
            },
        };
    }

    fn arenaAlloc(ctx: *anyopaque, len: usize, ptr_align: u8, ret_addr: usize) ?[*]u8 {
        _ = ret_addr;
        const self: *ArenaAllocator = @ptrCast(@alignCast(ctx));

        if (len == 0) {
            const empty_slice: []u8 = &[_]u8{};
            return @constCast(empty_slice.ptr);
        }

        self.mutex.lock();
        defer self.mutex.unlock();

        const alignment: usize = @as(usize, 1) << @intCast(ptr_align);

        const current_pos = self.pos;
        const aligned_pos = mem.alignForward(usize, current_pos, alignment);
        const aligned_len = len;

        if (self.current_buffer.len == 0 or aligned_pos + aligned_len > self.current_buffer.len) {
            const new_size = @max(self.buffer_size, aligned_len + alignment);
            const new_buf = self.parent_allocator.alloc(u8, new_size) catch return null;
            self.buffers.append(new_buf) catch return null;
            self.current_buffer = new_buf;
            self.pos = 0;
            const new_aligned_pos = mem.alignForward(usize, 0, alignment);
            const ptr = self.current_buffer.ptr + new_aligned_pos;
            self.pos = new_aligned_pos + aligned_len;
            return ptr;
        }

        const ptr = self.current_buffer.ptr + aligned_pos;
        self.pos = aligned_pos + aligned_len;
        return ptr;
    }

    fn arenaResize(ctx: *anyopaque, buf: []u8, buf_align: u8, new_len: usize, ret_addr: usize) bool {
        _ = buf_align;
        _ = ret_addr;
        const self: *ArenaAllocator = @ptrCast(@alignCast(ctx));
        self.mutex.lock();
        defer self.mutex.unlock();

        if (buf.ptr + buf.len != self.current_buffer.ptr + self.pos) {
            return false;
        }

        if (new_len > buf.len) {
            const additional = new_len - buf.len;
            if (self.pos + additional > self.current_buffer.len) return false;
            self.pos += additional;
            return true;
        } else {
            self.pos -= buf.len - new_len;
            return true;
        }
    }

    fn arenaFree(ctx: *anyopaque, buf: []u8, buf_align: u8, ret_addr: usize) void {
        _ = ctx;
        _ = buf;
        _ = buf_align;
        _ = ret_addr;
    }
};

pub const SlabAllocator = struct {
    slabs: []Slab,
    next_id: usize = 0,
    allocator: Allocator,
    block_size: usize,
    mutex: Mutex = .{},

    const Slab = struct {
        data: []u8,
        bitmap: []u64,
        block_size: usize,
        num_blocks: usize,
        id: usize,

        fn isBlockFree(self: *const Slab, block_idx: usize) bool {
            const word_idx = block_idx / 64;
            const bit_idx: u6 = @intCast(block_idx % 64);
            if (word_idx >= self.bitmap.len) return false;
            return (self.bitmap[word_idx] & (@as(u64, 1) << bit_idx)) == 0;
        }

        fn setBlockUsed(self: *Slab, block_idx: usize) void {
            const word_idx = block_idx / 64;
            const bit_idx: u6 = @intCast(block_idx % 64);
            if (word_idx >= self.bitmap.len) return;
            self.bitmap[word_idx] |= (@as(u64, 1) << bit_idx);
        }

        fn setBlockFree(self: *Slab, block_idx: usize) void {
            const word_idx = block_idx / 64;
            const bit_idx: u6 = @intCast(block_idx % 64);
            if (word_idx >= self.bitmap.len) return;
            self.bitmap[word_idx] &= ~(@as(u64, 1) << bit_idx);
        }
    };

    pub fn init(allocator: Allocator, slab_size: usize, num_slabs: usize, block_size: usize) !SlabAllocator {
        if (block_size == 0) return error.InvalidBlockSize;

        const slabs = try allocator.alloc(Slab, num_slabs);
        errdefer allocator.free(slabs);

        const num_blocks = slab_size / block_size;
        const bitmap_words = (num_blocks + 63) / 64;

        var i: usize = 0;
        while (i < slabs.len) : (i += 1) {
            var slab = &slabs[i];
            slab.data = try allocator.alloc(u8, slab_size);
            errdefer allocator.free(slab.data);
            slab.bitmap = try allocator.alloc(u64, bitmap_words);
            errdefer allocator.free(slab.bitmap);
            std.mem.set(u8, slab.data, 0);
            std.mem.set(u64, slab.bitmap, 0);
            slab.block_size = block_size;
            slab.num_blocks = num_blocks;
            slab.id = i;
        }
        return .{ .slabs = slabs, .allocator = allocator, .block_size = block_size };
    }

    pub fn deinit(self: *SlabAllocator) void {
        for (self.slabs) |slab| {
            self.allocator.free(slab.bitmap);
            self.allocator.free(slab.data);
        }
        self.allocator.free(self.slabs);
    }

    pub fn alloc(self: *SlabAllocator, size: usize) ?[]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (size > self.slabs[0].data.len) return null;

        const blocks_needed = (size + self.block_size - 1) / self.block_size;

        const start_id = self.next_id;
        var search_count: usize = 0;

        while (search_count < self.slabs.len) : (search_count += 1) {
            const slab_idx = (start_id + search_count) % self.slabs.len;
            var slab = &self.slabs[slab_idx];

            var consecutive: usize = 0;
            var start_idx: usize = 0;

            var i: usize = 0;
            while (i < slab.num_blocks) : (i += 1) {
                if (slab.isBlockFree(i)) {
                    if (consecutive == 0) start_idx = i;
                    consecutive += 1;
                    if (consecutive >= blocks_needed) {
                        var j = start_idx;
                        while (j < start_idx + blocks_needed) : (j += 1) {
                            slab.setBlockUsed(j);
                        }
                        const offset = start_idx * slab.block_size;
                        self.next_id = (slab_idx + 1) % self.slabs.len;
                        return slab.data[offset..offset + size];
                    }
                } else {
                    consecutive = 0;
                }
            }
        }

        self.next_id = 0;
        return null;
    }

    pub fn free(self: *SlabAllocator, ptr: []u8) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.slabs) |*slab| {
            const slab_start = @intFromPtr(slab.data.ptr);
            const slab_end = slab_start + slab.data.len;
            const ptr_addr = @intFromPtr(ptr.ptr);

            if (ptr_addr >= slab_start and ptr_addr < slab_end) {
                const offset = ptr_addr - slab_start;
                const start_block = offset / slab.block_size;
                const blocks_used = (ptr.len + slab.block_size - 1) / slab.block_size;

                var i = start_block;
                const end_block = @min(start_block + blocks_used, slab.num_blocks);
                while (i < end_block) : (i += 1) {
                    if (!slab.isBlockFree(i)) {
                        slab.setBlockFree(i);
                    }
                }
                @memset(ptr, 0);
                break;
            }
        }
    }
};

pub const PoolAllocator = struct {
    pools: []Pool,
    allocator: Allocator,
    mutex: Mutex = .{},

    const Pool = struct {
        buffer: []u8,
        block_size: usize,
        num_blocks: usize,
        free_list_head: ?usize = null,
        used: AtomicUsize,

        fn initFreeList(self: *Pool) void {
            var i: usize = 0;
            while (i < self.num_blocks) : (i += 1) {
                const block_ptr: *?usize = @ptrCast(@alignCast(self.buffer[i * self.block_size..].ptr));
                if (i + 1 < self.num_blocks) {
                    block_ptr.* = i + 1;
                } else {
                    block_ptr.* = null;
                }
            }
            self.free_list_head = 0;
        }
    };

    pub fn init(allocator: Allocator, block_size: usize, num_blocks: usize, num_pools: usize) !PoolAllocator {
        const actual_block_size = mem.alignForward(usize, @max(block_size, @sizeOf(?usize)), @alignOf(?usize));

        const pools = try allocator.alloc(Pool, num_pools);
        errdefer allocator.free(pools);

        for (pools) |*pool| {
            pool.buffer = try allocator.alignedAlloc(u8, @alignOf(usize), actual_block_size * num_blocks);
            errdefer allocator.free(pool.buffer);
            std.mem.set(u8, pool.buffer, 0);
            pool.block_size = actual_block_size;
            pool.num_blocks = num_blocks;
            pool.free_list_head = null;
            pool.used = AtomicUsize.init(0);
            pool.initFreeList();
        }
        return .{ .pools = pools, .allocator = allocator };
    }

    pub fn deinit(self: *PoolAllocator) void {
        for (self.pools) |pool| {
            self.allocator.free(pool.buffer);
        }
        self.allocator.free(self.pools);
    }

    pub fn alloc(self: *PoolAllocator, size: usize) ?[]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.pools) |*pool| {
            if (size > pool.block_size) continue;

            if (pool.free_list_head) |head_idx| {
                const block_ptr = pool.buffer[head_idx * pool.block_size..].ptr;
                const next_ptr: *?usize = @ptrCast(@alignCast(block_ptr));
                pool.free_list_head = next_ptr.*;
                _ = pool.used.fetchAdd(1, .Monotonic);

                const result = pool.buffer[head_idx * pool.block_size..(head_idx + 1) * pool.block_size];
                return result[0..size];
            }
        }
        return null;
    }

    pub fn free(self: *PoolAllocator, ptr: []u8) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.pools) |*pool| {
            const pool_start = @intFromPtr(pool.buffer.ptr);
            const pool_end = pool_start + pool.buffer.len;
            const ptr_addr = @intFromPtr(ptr.ptr);

            if (ptr_addr >= pool_start and ptr_addr < pool_end) {
                const offset = ptr_addr - pool_start;
                const block_idx = offset / pool.block_size;

                if (block_idx < pool.num_blocks) {
                    const block_ptr: *?usize = @ptrCast(@alignCast(pool.buffer[block_idx * pool.block_size..].ptr));
                    block_ptr.* = pool.free_list_head;
                    pool.free_list_head = block_idx;

                    const current = pool.used.load(.Monotonic);
                    if (current > 0) {
                        _ = pool.used.fetchSub(1, .Monotonic);
                    }

                    std.mem.set(u8, pool.buffer[block_idx * pool.block_size..(block_idx + 1) * pool.block_size], 0);
                }
                break;
            }
        }
    }
};

pub const BuddyAllocator = struct {
    allocator: Allocator,
    memory: []u8,
    tree: []u64,
    order: u32,
    min_order: u32,
    size_map: std.AutoHashMap(usize, u32),
    mutex: Mutex = .{},
    tree_nodes: usize,

    pub fn init(allocator: Allocator, size: usize, min_order: u32) !BuddyAllocator {
        if (size == 0) return error.InvalidSize;
        if (size < (@as(usize, 1) << @intCast(min_order))) return error.SizeTooSmall;

        const max_order: u32 = @intCast(@bitSizeOf(usize) - @clz(size) - 1);
        const tree_nodes = (@as(usize, 1) << @as(u6, @intCast(max_order + 1))) - 1;
        const tree_words = (tree_nodes + 63) / 64;

        const tree = try allocator.alloc(u64, tree_words);
        errdefer allocator.free(tree);
        std.mem.set(u64, tree, 0);

        const memory = try allocator.alloc(u8, @as(usize, 1) << @intCast(max_order));
        errdefer allocator.free(memory);
        std.mem.set(u8, memory, 0);

        var size_map = std.AutoHashMap(usize, u32).init(allocator);
        errdefer size_map.deinit();

        return .{
            .allocator = allocator,
            .memory = memory,
            .tree = tree,
            .order = max_order,
            .min_order = min_order,
            .size_map = size_map,
            .tree_nodes = tree_nodes,
        };
    }

    pub fn deinit(self: *BuddyAllocator) void {
        self.size_map.deinit();
        self.allocator.free(self.memory);
        self.allocator.free(self.tree);
    }

    fn getTreeBit(self: *const BuddyAllocator, index: usize) bool {
        if (index >= self.tree_nodes) return true;
        const word_idx = index / 64;
        const bit_idx: u6 = @intCast(index % 64);
        if (word_idx >= self.tree.len) return true;
        return (self.tree[word_idx] & (@as(u64, 1) << bit_idx)) != 0;
    }

    fn setTreeBit(self: *BuddyAllocator, index: usize, value: bool) void {
        if (index >= self.tree_nodes) return;
        const word_idx = index / 64;
        const bit_idx: u6 = @intCast(index % 64);
        if (word_idx >= self.tree.len) return;
        if (value) {
            self.tree[word_idx] |= (@as(u64, 1) << bit_idx);
        } else {
            self.tree[word_idx] &= ~(@as(u64, 1) << bit_idx);
        }
    }

    fn findBuddy(index: usize) usize {
        return index ^ 1;
    }

    fn parent(index: usize) usize {
        if (index == 0) return 0;
        return (index - 1) / 2;
    }

    fn leftChild(index: usize) usize {
        return 2 * index + 1;
    }

    fn rightChild(index: usize) usize {
        return 2 * index + 2;
    }

    pub fn alloc(self: *BuddyAllocator, size: usize) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (size == 0) return error.InvalidSize;

        var order = self.min_order;
        while ((@as(usize, 1) << @intCast(order)) < size) {
            order += 1;
            if (order > self.order) return error.OutOfMemory;
        }

        var index: usize = 0;
        var current_order = self.order;

        while (current_order > order) {
            const left = leftChild(index);
            const right = rightChild(index);

            if (!self.getTreeBit(left)) {
                index = left;
            } else if (!self.getTreeBit(right)) {
                index = right;
            } else {
                return error.OutOfMemory;
            }
            current_order -= 1;
        }

        self.setTreeBit(index, true);

        var node = index;
        while (node > 0) {
            const parent_idx = parent(node);
            if (parent_idx == node) break;
            self.setTreeBit(parent_idx, self.getTreeBit(leftChild(parent_idx)) or self.getTreeBit(rightChild(parent_idx)));
            node = parent_idx;
        }

        const level = self.order - order;
        const level_start = (@as(usize, 1) << @intCast(level)) - 1;
        const offset_in_level = index - level_start;
        const byte_offset = offset_in_level * (@as(usize, 1) << @intCast(order));

        const ptr = self.memory[byte_offset..byte_offset + size];
        try self.size_map.put(@intFromPtr(ptr.ptr), order);

        return ptr;
    }

    pub fn free(self: *BuddyAllocator, ptr: []u8) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const ptr_addr = @intFromPtr(ptr.ptr);
        const base_addr = @intFromPtr(self.memory.ptr);

        if (ptr_addr < base_addr or ptr_addr >= base_addr + self.memory.len) return;

        const order = self.size_map.get(ptr_addr) orelse return;

        const byte_offset = ptr_addr - base_addr;
        const block_size = @as(usize, 1) << @intCast(order);
        const offset_in_level = byte_offset / block_size;
        const level = self.order - order;
        const level_start = (@as(usize, 1) << @intCast(level)) - 1;
        var index = level_start + offset_in_level;

        self.setTreeBit(index, false);
        @memset(ptr, 0);

        while (index > 0) {
            const buddy = findBuddy(index);
            const parent_idx = parent(index);
            if (parent_idx == index) break;

            if (buddy < self.tree_nodes and !self.getTreeBit(buddy)) {
                self.setTreeBit(parent_idx, false);
            } else {
                self.setTreeBit(parent_idx, true);
            }
            index = parent_idx;
        }

        _ = self.size_map.remove(ptr_addr);
    }
};

pub const LockFreeFreelist = struct {
    head: AtomicUsize,
    nodes: []Node,
    allocator: Allocator,

    const Node = struct {
        next: usize,
        data: []u8,
        size: usize,
    };

    const MAX_RETRIES = 1000;
    const BACKOFF_THRESHOLD = 100;

    pub fn init(allocator: Allocator, initial_size: usize, num_nodes: usize) !LockFreeFreelist {
        if (num_nodes == 0) return error.InvalidNodeCount;

        const nodes = try allocator.alloc(Node, num_nodes);
        errdefer allocator.free(nodes);
        @memset(@as([*]u8, @ptrCast(nodes.ptr))[0..@sizeOf(Node) * num_nodes], 0);

        var i: usize = 0;
        while (i < nodes.len) : (i += 1) {
            var node = &nodes[i];
            node.data = try allocator.alloc(u8, initial_size);
            errdefer allocator.free(node.data);
            std.mem.set(u8, node.data, 0);
            node.size = initial_size;

            if (i + 1 < num_nodes) {
                node.next = i + 1;
            } else {
                node.next = num_nodes;
            }
        }

        return .{ .head = AtomicUsize.init(0), .nodes = nodes, .allocator = allocator };
    }

    pub fn deinit(self: *LockFreeFreelist) void {
        for (self.nodes) |node| {
            self.allocator.free(node.data);
        }
        self.allocator.free(self.nodes);
    }

    pub fn alloc(self: *LockFreeFreelist) ?[]u8 {
        var retries: usize = 0;
        while (retries < MAX_RETRIES) : (retries += 1) {
            const old_head = self.head.load(.Acquire);
            if (old_head >= self.nodes.len) return null;

            const node_idx = self.nodes[old_head].next;
            if (self.head.tryCompareAndSwap(old_head, node_idx, .AcqRel, .Acquire) == null) {
                return self.nodes[old_head].data;
            }

            if (retries > BACKOFF_THRESHOLD) {
                const backoff_count = @min(retries - BACKOFF_THRESHOLD, 10);
                var i: usize = 0;
                while (i < backoff_count) : (i += 1) {
                    std.atomic.spinLoopHint();
                }
            }
        }
        return null;
    }

    pub fn free(self: *LockFreeFreelist, ptr: []u8) void {
        var i: usize = 0;
        while (i < self.nodes.len) : (i += 1) {
            var node = &self.nodes[i];
            if (node.data.ptr == ptr.ptr) {
                var retries: usize = 0;
                var old_head = self.head.load(.Acquire);
                while (retries < MAX_RETRIES) : (retries += 1) {
                    node.next = old_head;
                    if (self.head.tryCompareAndSwap(old_head, i, .AcqRel, .Acquire) == null) {
                        @memset(ptr, 0);
                        return;
                    }
                    old_head = self.head.load(.Acquire);
                    if (retries > BACKOFF_THRESHOLD) {
                        std.atomic.spinLoopHint();
                    }
                }
                break;
            }
        }
    }
};

pub const LockFreePool = struct {
    head: std.atomic.Atomic(?*Block),
    block_size: usize,
    allocator: Allocator,
    max_retries: usize,

    const Block = struct {
        data: [PageSize]u8,
        next: std.atomic.Atomic(?*Block),
    };

    pub fn init(allocator: Allocator, block_size: usize) LockFreePool {
        return .{
            .head = std.atomic.Atomic(?*Block).init(null),
            .block_size = block_size,
            .allocator = allocator,
            .max_retries = 1000,
        };
    }

    pub fn deinit(self: *LockFreePool) void {
        var current = self.head.load(.Acquire);
        while (current) |block| {
            const next = block.next.load(.Acquire);
            self.allocator.destroy(block);
            current = next;
        }
    }

    pub fn alloc(self: *LockFreePool) ?[]u8 {
        var retries: usize = 0;
        var current = self.head.load(.Acquire);

        while (retries < self.max_retries) : (retries += 1) {
            while (current) |head| {
                const next = head.next.load(.Acquire);
                if (self.head.tryCompareAndSwap(current, next, .AcqRel, .Acquire) == null) {
                    return head.data[0..self.block_size];
                }
                current = self.head.load(.Acquire);
            }

            const new_block = self.allocator.create(Block) catch return null;
            new_block.* = .{ .data = undefined, .next = std.atomic.Atomic(?*Block).init(null) };
            std.mem.set(u8, &new_block.data, 0);

            var old_head = self.head.load(.Acquire);
            var add_retries: usize = 0;
            while (add_retries < 100) : (add_retries += 1) {
                new_block.next.store(old_head, .Release);
                if (self.head.tryCompareAndSwap(old_head, new_block, .AcqRel, .Acquire) == null) break;
                old_head = self.head.load(.Acquire);
            }

            const next = new_block.next.load(.Acquire);
            if (self.head.tryCompareAndSwap(new_block, next, .AcqRel, .Acquire) == null) {
                return new_block.data[0..self.block_size];
            }

            current = self.head.load(.Acquire);
        }

        return null;
    }

    pub fn free(self: *LockFreePool, ptr: []u8) void {
        if (ptr.len != self.block_size) return;

        const ptr_addr = @intFromPtr(ptr.ptr);
        if (!mem.isAligned(ptr_addr, @alignOf(Block))) return;

        const block_addr = ptr_addr - @offsetOf(Block, "data");
        const block: *Block = @ptrFromInt(block_addr);

        var retries: usize = 0;
        var current = self.head.load(.Acquire);
        while (retries < self.max_retries) : (retries += 1) {
            block.next.store(current, .Release);
            if (self.head.tryCompareAndSwap(current, block, .AcqRel, .Acquire) == null) {
                @memset(ptr, 0);
                return;
            }
            current = self.head.load(.Acquire);
        }
    }
};

pub const LockFreeQueue = struct {
    head: AtomicUsize,
    tail: AtomicUsize,
    buffer: []*anyopaque,

    pub fn init(allocator: Allocator, size: usize) !LockFreeQueue {
        const buffer = try allocator.alloc(*anyopaque, size);
        errdefer allocator.free(buffer);

        std.mem.set(*anyopaque, buffer, undefined);

        return .{
            .head = AtomicUsize.init(0),
            .tail = AtomicUsize.init(0),
            .buffer = buffer,
        };
    }

    pub fn deinit(self: *LockFreeQueue, allocator: Allocator) void {
        allocator.free(self.buffer);
    }

    pub fn enqueue(self: *LockFreeQueue, item: *anyopaque) bool {
        const tail = self.tail.load(.Monotonic);
        const next_tail = (tail + 1) % self.buffer.len;
        if (next_tail == self.head.load(.Acquire)) return false;
        self.buffer[tail] = item;
        self.tail.store(next_tail, .Release);
        return true;
    }

    pub fn dequeue(self: *LockFreeQueue) ?*anyopaque {
        const head = self.head.load(.Monotonic);
        if (head == self.tail.load(.Acquire)) return null;
        const item = self.buffer[head];
        self.head.store((head + 1) % self.buffer.len, .Release);
        return item;
    }
};

pub const LockFreeStack = struct {
    top: std.atomic.Atomic(?*Node),
    allocator: Allocator,
    max_retries: usize,

    const Node = struct {
        value: *anyopaque,
        next: ?*Node,
    };

    pub fn init(allocator: Allocator) LockFreeStack {
        return .{
            .top = std.atomic.Atomic(?*Node).init(null),
            .allocator = allocator,
            .max_retries = 1000,
        };
    }

    pub fn deinit(self: *LockFreeStack) void {
        var current = self.top.load(.Acquire);
        while (current) |node| {
            current = node.next;
            self.allocator.destroy(node);
        }
    }

    pub fn push(self: *LockFreeStack, value: *anyopaque) !void {
        const node = try self.allocator.create(Node);
        node.* = .{ .value = value, .next = null };

        var retries: usize = 0;
        var top = self.top.load(.Acquire);
        while (retries < self.max_retries) : (retries += 1) {
            node.next = top;
            if (self.top.tryCompareAndSwap(top, node, .AcqRel, .Acquire) == null) return;
            top = self.top.load(.Acquire);
        }

        self.allocator.destroy(node);
        return error.TooManyRetries;
    }

    pub fn pop(self: *LockFreeStack) ?*anyopaque {
        var retries: usize = 0;
        var top = self.top.load(.Acquire);
        while (retries < self.max_retries) : (retries += 1) {
            if (top) |node| {
                if (self.top.tryCompareAndSwap(top, node.next, .AcqRel, .Acquire) == null) {
                    const value = node.value;
                    self.allocator.destroy(node);
                    return value;
                }
                top = self.top.load(.Acquire);
            } else {
                return null;
            }
        }
        return null;
    }
};

pub const PageAllocator = struct {
    pages: []u8,
    allocator: Allocator,
    page_size: usize = PageSize,
    offset: usize = 0,
    bitmap: []u64,
    mutex: Mutex = .{},

    pub fn init(allocator: Allocator, num_pages: usize) !PageAllocator {
        const pages = try allocator.alloc(u8, num_pages * PageSize);
        errdefer allocator.free(pages);
        std.mem.set(u8, pages, 0);

        const bitmap_words = (num_pages + 63) / 64;
        const bitmap = try allocator.alloc(u64, bitmap_words);
        errdefer allocator.free(bitmap);
        std.mem.set(u64, bitmap, 0);

        return .{
            .pages = pages,
            .allocator = allocator,
            .page_size = PageSize,
            .bitmap = bitmap,
        };
    }

    pub fn deinit(self: *PageAllocator) void {
        self.allocator.free(self.bitmap);
        self.allocator.free(self.pages);
    }

    fn isPageFree(self: *const PageAllocator, page_idx: usize) bool {
        const word_idx = page_idx / 64;
        const bit_idx: u6 = @intCast(page_idx % 64);
        if (word_idx >= self.bitmap.len) return false;
        return (self.bitmap[word_idx] & (@as(u64, 1) << bit_idx)) == 0;
    }

    fn setPageUsed(self: *PageAllocator, page_idx: usize) void {
        const word_idx = page_idx / 64;
        const bit_idx: u6 = @intCast(page_idx % 64);
        if (word_idx < self.bitmap.len) {
            self.bitmap[word_idx] |= (@as(u64, 1) << bit_idx);
        }
    }

    fn setPageFree(self: *PageAllocator, page_idx: usize) void {
        const word_idx = page_idx / 64;
        const bit_idx: u6 = @intCast(page_idx % 64);
        if (word_idx < self.bitmap.len) {
            self.bitmap[word_idx] &= ~(@as(u64, 1) << bit_idx);
        }
    }

    pub fn allocPages(self: *PageAllocator, num_pages: usize) ?[]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        const total_pages = self.pages.len / self.page_size;
        if (num_pages > total_pages) return null;

        var consecutive: usize = 0;
        var start_page: usize = 0;

        var i: usize = 0;
        while (i < total_pages) : (i += 1) {
            if (self.isPageFree(i)) {
                if (consecutive == 0) start_page = i;
                consecutive += 1;
                if (consecutive >= num_pages) {
                    var j = start_page;
                    while (j < start_page + num_pages) : (j += 1) {
                        self.setPageUsed(j);
                    }
                    const offset = start_page * self.page_size;
                    const size = num_pages * self.page_size;
                    return self.pages[offset..offset + size];
                }
            } else {
                consecutive = 0;
            }
        }

        return null;
    }

    pub fn freePages(self: *PageAllocator, ptr: []u8) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const pages_start = @intFromPtr(self.pages.ptr);
        const pages_end = pages_start + self.pages.len;
        const ptr_addr = @intFromPtr(ptr.ptr);

        if (ptr_addr >= pages_start and ptr_addr < pages_end) {
            const offset = ptr_addr - pages_start;
            const start_page = offset / self.page_size;
            const num_pages = ptr.len / self.page_size;

            var i = start_page;
            while (i < start_page + num_pages) : (i += 1) {
                self.setPageFree(i);
            }
            std.mem.set(u8, ptr, 0);
        }
    }

    pub fn mapPage(self: *PageAllocator, page_idx: usize) ?[*]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        const total_pages = self.pages.len / self.page_size;
        if (page_idx >= total_pages) return null;

        const offset = page_idx * self.page_size;
        if (offset + self.page_size > self.pages.len) return null;

        return @ptrCast(self.pages[offset..offset + self.page_size].ptr);
    }
};

pub const ZeroCopySlice = struct {
    ptr: [*]const u8,
    len: usize,
    allocator: ?Allocator = null,

    pub fn init(ptr: [*]const u8, len: usize) ZeroCopySlice {
        return .{ .ptr = ptr, .len = len };
    }

    pub fn slice(self: *const ZeroCopySlice, start: usize, end: usize) ZeroCopySlice {
        if (end > self.len or start > end) {
            return .{ .ptr = self.ptr, .len = 0 };
        }
        return .{ .ptr = self.ptr + start, .len = end - start };
    }

    pub fn copyTo(self: *const ZeroCopySlice, allocator: Allocator) ![]u8 {
        if (!mem.isAligned(@intFromPtr(self.ptr), @alignOf(u8))) {
            return error.UnalignedPointer;
        }
        const buf = try allocator.alloc(u8, self.len);
        @memcpy(buf, self.asBytes());
        return buf;
    }

    pub fn asBytes(self: *const ZeroCopySlice) []const u8 {
        return self.ptr[0..self.len];
    }

    pub fn deinit(self: *ZeroCopySlice) void {
        if (self.allocator) |alloc| {
            const bytes = @as([*]u8, @constCast(self.ptr))[0..self.len];
            alloc.free(bytes);
        }
    }
};

pub const ResizeBuffer = struct {
    buffer: []u8,
    len: usize,
    capacity: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator) ResizeBuffer {
        return .{ .buffer = &[_]u8{}, .len = 0, .capacity = 0, .allocator = allocator };
    }

    pub fn deinit(self: *ResizeBuffer) void {
        if (self.capacity > 0) {
            self.allocator.free(self.buffer);
        }
    }

    pub fn append(self: *ResizeBuffer, data: []const u8) !void {
        const new_len = self.len + data.len;
        if (new_len > self.capacity) {
            const new_capacity = if (self.capacity == 0)
                @max(16, new_len)
            else
                @max(new_len, self.capacity * 2);

            const new_buffer = try self.allocator.alloc(u8, new_capacity);
            if (self.len > 0) {
                @memcpy(new_buffer[0..self.len], self.buffer[0..self.len]);
            }
            if (self.capacity > 0) {
                self.allocator.free(self.buffer);
            }
            self.buffer = new_buffer;
            self.capacity = new_capacity;
        }
        @memcpy(self.buffer[self.len..new_len], data);
        self.len = new_len;
    }

    pub fn clear(self: *ResizeBuffer) void {
        self.len = 0;
    }

    pub fn toOwnedSlice(self: *ResizeBuffer) []u8 {
        if (self.capacity > self.len) {
            const exact_slice = self.allocator.resize(self.buffer, self.len) catch self.buffer[0..self.len];
            self.buffer = &[_]u8{};
            self.len = 0;
            self.capacity = 0;
            return exact_slice;
        }
        const slice = self.buffer[0..self.len];
        self.buffer = &[_]u8{};
        self.len = 0;
        self.capacity = 0;
        return slice;
    }
};

pub fn zeroCopyTransfer(src: []const u8, dest: []u8) void {
    @memcpy(dest[0..@min(src.len, dest.len)], src[0..@min(src.len, dest.len)]);
}

pub fn alignedAlloc(allocator: Allocator, comptime T: type, n: usize) ![]T {
    return allocator.alloc(T, n);
}

pub fn cacheAlignedAlloc(allocator: Allocator, size: usize, cache_line_size: usize) ![]u8 {
    const alignment = if (cache_line_size > 0) cache_line_size else 64;
    return try allocator.alignedAlloc(u8, alignment, size);
}

pub fn copyWithoutAlloc(src: []const u8) []const u8 {
    return src;
}

pub fn sliceMemory(base: *anyopaque, offset: usize, size: usize, buffer_size: usize) ![]u8 {
    if (!mem.isAligned(@intFromPtr(base), @alignOf(u8))) {
        return error.UnalignedPointer;
    }
    if (offset > buffer_size) {
        return error.OffsetOutOfBounds;
    }
    const end = @addWithOverflow(offset, size);
    if (end[1] != 0) {
        return error.SliceOverflow;
    }
    if (end[0] > buffer_size) {
        return error.SliceOutOfBounds;
    }
    return @as([*]u8, @ptrCast(base))[offset..offset + size];
}

pub fn zeroInitMemory(ptr: *anyopaque, size: usize) void {
    std.mem.set(u8, @as([*]u8, @ptrCast(ptr))[0..size], 0);
}

pub fn secureZeroMemory(ptr: *anyopaque, size: usize) void {
    const volatile_ptr: [*]volatile u8 = @ptrCast(ptr);
    var i: usize = 0;
    while (i < size) : (i += 1) {
        volatile_ptr[i] = 0;
    }
}

pub fn compareMemory(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    var i: usize = 0;
    while (i < a.len) : (i += 1) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

pub fn hashMemory(data: []const u8) u64 {
    var hasher = std.hash.Wyhash.init(0);
    hasher.update(data);
    return hasher.final();
}

pub fn alignForward(addr: usize, alignment: usize) usize {
    return mem.alignForward(usize, addr, alignment);
}

pub fn alignBackward(addr: usize, alignment: usize) usize {
    return mem.alignBackward(usize, addr, alignment);
}

pub fn isAligned(addr: usize, alignment: usize) bool {
    return mem.isAligned(addr, alignment);
}

pub fn pageAlignedSize(size: usize) usize {
    return mem.alignForward(usize, size, PageSize);
}

pub fn memoryBarrier() void {
    atomic.fence(.SeqCst);
}

pub fn readMemoryFence() void {
    atomic.fence(.Acquire);
}

pub fn writeMemoryFence() void {
    atomic.fence(.Release);
}

pub fn compareExchangeMemory(ptr: *u64, expected: u64, desired: u64) bool {
    return @cmpxchgStrong(u64, ptr, expected, desired, .SeqCst, .SeqCst) == null;
}

pub fn atomicLoad(ptr: *u64) u64 {
    return @atomicLoad(u64, ptr, .SeqCst);
}

pub fn atomicStore(ptr: *u64, value: u64) void {
    @atomicStore(u64, ptr, value, .SeqCst);
}

pub fn atomicAdd(ptr: *u64, delta: u64) u64 {
    return @atomicRmw(u64, ptr, .Add, delta, .SeqCst);
}

pub fn atomicSub(ptr: *u64, delta: u64) u64 {
    return @atomicRmw(u64, ptr, .Sub, delta, .SeqCst);
}

pub fn atomicAnd(ptr: *u64, mask: u64) u64 {
    return @atomicRmw(u64, ptr, .And, mask, .SeqCst);
}

pub fn atomicOr(ptr: *u64, mask: u64) u64 {
    return @atomicRmw(u64, ptr, .Or, mask, .SeqCst);
}

pub fn atomicXor(ptr: *u64, mask: u64) u64 {
    return @atomicRmw(u64, ptr, .Xor, mask, .SeqCst);
}

pub fn atomicInc(ptr: *u64) u64 {
    return atomicAdd(ptr, 1);
}

pub fn atomicDec(ptr: *u64) u64 {
    return atomicSub(ptr, 1);
}

pub fn memoryEfficientCopy(src: []const u8, dest: []u8) void {
    var i: usize = 0;
    while (i < src.len) : (i += 64) {
        const chunk = src[i..@min(i + 64, src.len)];
        @memcpy(dest[i..i + chunk.len], chunk);
    }
}

pub fn constantTimeCompare(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    var diff: u8 = 0;
    var i: usize = 0;
    while (i < a.len) : (i += 1) {
        diff |= a[i] ^ b[i];
    }
    return diff == 0;
}

pub fn secureErase(ptr: *anyopaque, size: usize) void {
    const p = @as([*]volatile u8, @ptrCast(ptr));
    var i: usize = 0;
    while (i < size) : (i += 1) p[i] = 0x55;
    i = 0;
    while (i < size) : (i += 1) p[i] = 0xAA;
    i = 0;
    while (i < size) : (i += 1) p[i] = 0x00;
}

pub fn duplicateMemory(allocator: Allocator, data: []const u8) ![]u8 {
    const dup = try allocator.alloc(u8, data.len);
    @memcpy(dup, data);
    return dup;
}

pub fn concatenateMemory(allocator: Allocator, a: []const u8, b: []const u8) ![]u8 {
    const cat = try allocator.alloc(u8, a.len + b.len);
    @memcpy(cat[0..a.len], a);
    @memcpy(cat[a.len..], b);
    return cat;
}

pub fn searchMemory(haystack: []const u8, needle: []const u8) ?usize {
    if (needle.len == 0 or needle.len > haystack.len) return null;
    var i: usize = 0;
    outer: while (i < haystack.len) : (i += 1) {
        if (i + needle.len > haystack.len) break;
        var j: usize = 0;
        while (j < needle.len) : (j += 1) {
            if (haystack[i + j] != needle[j]) continue :outer;
        }
        return i;
    }
    return null;
}

pub fn replaceMemory(data: []u8, old: u8, new: u8) void {
    for (data) |*c| {
        if (c.* == old) c.* = new;
    }
}

pub fn reverseMemory(data: []u8) void {
    mem.reverse(u8, data);
}

pub fn rotateMemory(data: []u8, shift: usize) void {
    mem.rotate(u8, data, shift);
}

pub fn countMemory(data: []const u8, value: u8) usize {
    var count: usize = 0;
    for (data) |c| {
        if (c == value) count += 1;
    }
    return count;
}

pub fn sumMemory(data: []const u8) u64 {
    var sum: u64 = 0;
    for (data) |c| sum += c;
    return sum;
}

pub fn productMemory(data: []const u8) u64 {
    var prod: u64 = 1;
    for (data) |c| prod *= c;
    return prod;
}

pub fn minMemory(data: []const u8) u8 {
    return mem.min(u8, data);
}

pub fn maxMemory(data: []const u8) u8 {
    return mem.max(u8, data);
}

pub fn sortMemory(data: []u8) void {
    mem.sort(u8, data, {}, std.sort.asc(u8));
}

pub fn shuffleMemory(data: []u8, seed: u64) void {
    var prng = std.rand.DefaultPrng.init(seed);
    prng.random().shuffle(u8, data);
}

pub fn uniqueMemory(allocator: Allocator, data: []const u8) ![]u8 {
    var set = std.AutoHashMap(u8, void).init(allocator);
    defer set.deinit();
    for (data) |c| try set.put(c, {});
    var unique = try allocator.alloc(u8, set.count());
    var iter = set.keyIterator();
    var i: usize = 0;
    while (iter.next()) |key| {
        unique[i] = key.*;
        i += 1;
    }
    return unique;
}

pub fn intersectMemory(allocator: Allocator, a: []const u8, b: []const u8) ![]u8 {
    var set_a = std.AutoHashMap(u8, void).init(allocator);
    defer set_a.deinit();
    for (a) |c| try set_a.put(c, {});
    var intersection = std.ArrayList(u8).init(allocator);
    defer intersection.deinit();
    for (b) |c| {
        if (set_a.contains(c)) try intersection.append(c);
    }
    return try intersection.toOwnedSlice();
}

pub fn unionMemory(allocator: Allocator, a: []const u8, b: []const u8) ![]u8 {
    var set = std.AutoHashMap(u8, void).init(allocator);
    defer set.deinit();
    for (a) |c| try set.put(c, {});
    for (b) |c| try set.put(c, {});
    var un = try allocator.alloc(u8, set.count());
    var iter = set.keyIterator();
    var i: usize = 0;
    while (iter.next()) |key| {
        un[i] = key.*;
        i += 1;
    }
    return un;
}

pub fn differenceMemory(allocator: Allocator, a: []const u8, b: []const u8) ![]u8 {
    var set_b = std.AutoHashMap(u8, void).init(allocator);
    defer set_b.deinit();
    for (b) |c| try set_b.put(c, {});
    var diff = std.ArrayList(u8).init(allocator);
    defer diff.deinit();
    for (a) |c| {
        if (!set_b.contains(c)) try diff.append(c);
    }
    return try diff.toOwnedSlice();
}

pub fn isSubsetMemory(allocator: Allocator, a: []const u8, b: []const u8) bool {
    var set_b = std.AutoHashMap(u8, void).init(allocator);
    defer set_b.deinit();
    for (b) |c| set_b.put(c, {}) catch return false;
    for (a) |c| if (!set_b.contains(c)) return false;
    return true;
}

pub fn isSupersetMemory(allocator: Allocator, a: []const u8, b: []const u8) bool {
    return isSubsetMemory(allocator, b, a);
}

pub fn isDisjointMemory(allocator: Allocator, a: []const u8, b: []const u8) bool {
    var set_a = std.AutoHashMap(u8, void).init(allocator);
    defer set_a.deinit();
    for (a) |c| set_a.put(c, {}) catch return false;
    for (b) |c| if (set_a.contains(c)) return false;
    return true;
}

pub fn memoryFootprint() usize {
    return 0;
}

pub fn memoryPressure() f32 {
    return 0.0;
}

pub fn defragmentMemory() void {
}

pub fn dumpMemory(ptr: *anyopaque, size: usize) void {
    const data = @as([*]u8, @ptrCast(ptr))[0..size];
    std.debug.print("{x}\n", .{std.fmt.fmtSliceHexLower(data)});
}

pub fn validateMemory(ptr: *anyopaque, size: usize, expected: u8) bool {
    const data = @as([*]u8, @ptrCast(ptr))[0..size];
    for (data) |c| if (c != expected) return false;
    return true;
}

pub fn canaryProtect(ptr: *anyopaque, size: usize) void {
    const canary: u32 = 0xDEADBEEF;
    const before: *u32 = @ptrCast(@alignCast(@as([*]u8, @ptrCast(ptr)) - 4));
    const after: *u32 = @ptrCast(@alignCast(@as([*]u8, @ptrCast(ptr)) + size));
    before.* = canary;
    after.* = canary;
}

pub fn checkCanary(ptr: *anyopaque, size: usize) bool {
    const canary: u32 = 0xDEADBEEF;
    const before: *u32 = @ptrCast(@alignCast(@as([*]u8, @ptrCast(ptr)) - 4));
    const after: *u32 = @ptrCast(@alignCast(@as([*]u8, @ptrCast(ptr)) + size));
    if (before.* != canary) return false;
    if (after.* != canary) return false;
    return true;
}

pub fn optimalBufferSize(size: usize) usize {
    return std.math.ceilPowerOfTwo(usize, size) catch size;
}

pub fn minimalAllocation(allocator: Allocator, size: usize) ![]u8 {
    return try allocator.alloc(u8, size);
}

pub fn hugeTlbAlloc(allocator: Allocator, size: usize) ![]u8 {
    const huge_page_size = 2 * 1024 * 1024;
    const aligned_size = mem.alignForward(usize, size, huge_page_size);
    if (std.builtin.os.tag == .linux) {
        const ptr = std.os.mmap(null, aligned_size, std.os.PROT.READ | std.os.PROT.WRITE, std.os.MAP.ANONYMOUS | std.os.MAP.PRIVATE | std.os.MAP.HUGETLB, -1, 0) catch return try allocator.alloc(u8, aligned_size);
        return ptr;
    } else {
        return try allocator.alloc(u8, aligned_size);
    }
}

pub fn transparentHugePages(enable: bool) void {
    if (std.builtin.os.tag != .linux) return;
    const path = "/sys/kernel/mm/transparent_hugepage/enabled";
    const value = if (enable) "always\n" else "never\n";
    const file = std.fs.openFileAbsolute(path, .{ .mode = .write_only }) catch return;
    defer file.close();
    file.writeAll(value) catch return;
}

pub fn noSwap(enable: bool) void {
    if (std.builtin.os.tag != .linux) return;
    if (enable) {
        const result = std.os.linux.mlockall(std.os.linux.MCL.CURRENT | std.os.linux.MCL.FUTURE);
        _ = result;
    } else {
        const result = std.os.linux.munlockall();
        _ = result;
    }
}

pub fn memoryMapFile(fd: std.fs.File, size: usize) ![]u8 {
    if (size == 0) return error.InvalidSize;

    const prot = std.os.PROT.READ | std.os.PROT.WRITE;
    const flags = std.os.MAP.SHARED;

    const ptr = try std.os.mmap(
        null,
        size,
        prot,
        flags,
        fd.handle,
        0,
    );

    return ptr;
}

pub fn memoryUnmapFile(ptr: []u8) void {
    if (ptr.len == 0) return;
    std.posix.munmap(ptr);
}

pub fn sharedMemoryCreate(allocator: Allocator, size: usize) ![]u8 {
    if (std.builtin.os.tag == .linux) {
        const fd = try std.os.memfd_create("zig-shm", 0);
        try std.os.ftruncate(fd, size);
        const map = try std.os.mmap(null, size, std.os.PROT.READ | std.os.PROT.WRITE, std.os.MAP.SHARED, fd, 0);
        std.os.close(fd);
        return map;
    } else {
        return try allocator.alloc(u8, size);
    }
}

pub fn sharedMemoryAttach(ptr: []u8) []u8 {
    return ptr;
}

pub fn sharedMemoryDetach(ptr: []u8) void {
    if (ptr.len == 0) return;
    if (std.builtin.os.tag == .linux) {
        std.os.munmap(ptr);
    }
}

pub fn sharedMemoryRemove(ptr: []u8, allocator: Allocator) void {
    if (std.builtin.os.tag == .linux) {
    } else {
        allocator.free(ptr);
    }
}

pub fn positionalPopulateCache(ptr: *anyopaque, size: usize) void {
    const cache_line = 64;
    var i: usize = 0;
    while (i < size) : (i += cache_line) {
        _ = @as([*]volatile u8, @ptrCast(ptr))[i];
    }
}

pub fn evictCacheLine(ptr: *anyopaque) void {
    const arch = std.builtin.cpu.arch;
    if (arch == .x86 or arch == .x86_64) {
        asm volatile ("clflush (%0)"
            :: [ptr] "r" (ptr)
            : "memory"
        );
    }
}

pub fn invalidateCache() void {
}

pub fn readTSC() u64 {
    const arch = std.builtin.cpu.arch;
    if (arch == .x86 or arch == .x86_64) {
        var low: u32 = undefined;
        var high: u32 = undefined;
        asm volatile ("rdtsc"
            : [low] "=a" (low),
              [high] "=d" (high)
        );
        return (@as(u64, high) << 32) | @as(u64, low);
    } else {
        return 0;
    }
}

pub fn memoryBandwidthTest(allocator: Allocator, size: usize) u64 {
    const ptr = allocator.alloc(u8, size) catch return 0;
    defer allocator.free(ptr);
    const other = allocator.alloc(u8, size) catch return 0;
    defer allocator.free(other);
    const start = readTSC();
    @memcpy(other, ptr);
    const end = readTSC();
    return end - start;
}

pub fn memoryLatencyTest(allocator: Allocator, size: usize) u64 {
    const ptr = allocator.alloc(*anyopaque, size / @sizeOf(*anyopaque)) catch return 0;
    defer allocator.free(ptr);
    var i: usize = 0;
    while (i < ptr.len) : (i += 1) {
        ptr[i] = &ptr[(i + 1) % ptr.len];
    }
    const start = readTSC();
    var p = ptr[0];
    var count: u64 = 1000000;
    while (count > 0) : (count -= 1) {
        p = @as(**anyopaque, @ptrCast(p)).*;
    }
    const end = readTSC();
    return (end - start) / 1000000;
}

pub const MemoryStats = struct {
    allocated: usize,
    freed: usize,
    peak: usize,
};

pub var global_memory_stats: MemoryStats = .{ .allocated = 0, .freed = 0, .peak = 0 };

pub fn trackAllocation(size: usize) void {
    global_memory_stats.allocated += size;
    const current = global_memory_stats.allocated - global_memory_stats.freed;
    if (current > global_memory_stats.peak) {
        global_memory_stats.peak = current;
    }
}

pub fn trackFree(size: usize) void {
    global_memory_stats.freed += size;
}

pub fn getMemoryStats() MemoryStats {
    return global_memory_stats;
}

pub fn resetMemoryStats() void {
    global_memory_stats = .{ .allocated = 0, .freed = 0, .peak = 0 };
}

pub fn memoryStatsPrint() void {
    const stats = getMemoryStats();
    std.debug.print("Allocated: {}, Freed: {}, Peak: {}\n", .{stats.allocated, stats.freed, stats.peak});
}

pub fn leakDetectionEnable() void {
    resetMemoryStats();
}

pub fn leakDetectionCheck() bool {
    const stats = getMemoryStats();
    return stats.allocated == stats.freed;
}

pub const TrackingAllocator = struct {
    parent: Allocator,

    pub fn init(parent: Allocator) TrackingAllocator {
        return .{ .parent = parent };
    }

    pub fn allocator(self: *TrackingAllocator) Allocator {
        return .{
            .ptr = self,
            .vtable = &.{ .alloc = trackingAlloc, .resize = trackingResize, .free = trackingFree },
        };
    }

    fn trackingAlloc(ctx: *anyopaque, len: usize, ptr_align: u8, ret_addr: usize) ?[*]u8 {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        const ptr = self.parent.rawAlloc(len, ptr_align, ret_addr);
        if (ptr != null) trackAllocation(len);
        return ptr;
    }

    fn trackingResize(ctx: *anyopaque, buf: []u8, buf_align: u8, new_len: usize, ret_addr: usize) bool {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        const old_len = buf.len;
        const res = self.parent.rawResize(buf, buf_align, new_len, ret_addr);
        if (res) {
            if (new_len > old_len) trackAllocation(new_len - old_len) else trackFree(old_len - new_len);
        }
        return res;
    }

    fn trackingFree(ctx: *anyopaque, buf: []u8, buf_align: u8, ret_addr: usize) void {
        const self: *TrackingAllocator = @ptrCast(@alignCast(ctx));
        trackFree(buf.len);
        self.parent.rawFree(buf, buf_align, ret_addr);
    }
};

pub const MemoryGuard = struct {
    ptr: *anyopaque,
    size: usize,

    pub fn init(allocator: Allocator, size: usize) !MemoryGuard {
        const ptr = try allocator.alloc(u8, size + 8);
        const actual_ptr: *anyopaque = @ptrCast(ptr.ptr + 4);
        canaryProtect(actual_ptr, size);
        return .{ .ptr = actual_ptr, .size = size };
    }

    pub fn deinit(self: *MemoryGuard, allocator: Allocator) !void {
        if (!checkCanary(self.ptr, self.size)) {
            std.log.err("Memory corruption detected at {*} size {}", .{self.ptr, self.size});
            return error.MemoryCorruption;
        }
        const base_ptr = @as([*]u8, @ptrCast(self.ptr)) - 4;
        allocator.free(base_ptr[0..self.size + 8]);
    }
};

pub fn safeAlloc(allocator: Allocator, size: usize) !MemoryGuard {
    return try MemoryGuard.init(allocator, size);
}

pub const ReadWriteLock = struct {
    readers: AtomicU64,
    writer: AtomicBool,
    mutex: Mutex,

    pub fn init() ReadWriteLock {
        return .{
            .readers = AtomicU64.init(0),
            .writer = AtomicBool.init(false),
            .mutex = Mutex{},
        };
    }

    pub fn readLock(self: *ReadWriteLock) void {
        self.mutex.lock();
        while (self.writer.load(.Acquire)) {
            self.mutex.unlock();
            std.atomic.spinLoopHint();
            self.mutex.lock();
        }
        _ = self.readers.fetchAdd(1, .Monotonic);
        self.mutex.unlock();
    }

    pub fn readUnlock(self: *ReadWriteLock) void {
        _ = self.readers.fetchSub(1, .Monotonic);
    }

    pub fn writeLock(self: *ReadWriteLock) void {
        self.mutex.lock();
        while (self.writer.tryCompareAndSwap(false, true, .Acquire, .Monotonic) != null or self.readers.load(.Acquire) > 0) {
            self.mutex.unlock();
            std.atomic.spinLoopHint();
            self.mutex.lock();
        }
        self.mutex.unlock();
    }

    pub fn writeUnlock(self: *ReadWriteLock) void {
        self.writer.store(false, .Release);
    }
};

pub fn atomicFlagTestAndSet(flag: *AtomicBool) bool {
    return flag.swap(true, .SeqCst);
}

pub fn atomicFlagClear(flag: *AtomicBool) void {
    flag.store(false, .SeqCst);
}

pub fn spinLockAcquire(lock: *AtomicU64) void {
    while (lock.tryCompareAndSwap(0, 1, .Acquire, .Monotonic) != null) {
        std.atomic.spinLoopHint();
    }
}

pub fn spinLockRelease(lock: *AtomicU64) void {
    lock.store(0, .Release);
}

pub fn memoryPatternFill(ptr: *anyopaque, size: usize, pattern: []const u8) void {
    var i: usize = 0;
    const dest = @as([*]u8, @ptrCast(ptr));
    while (i < size) : (i += pattern.len) {
        const copy_len = @min(pattern.len, size - i);
        @memcpy(dest[i..i + copy_len], pattern[0..copy_len]);
    }
}

pub fn memoryPatternVerify(ptr: *anyopaque, size: usize, pattern: []const u8) bool {
    var i: usize = 0;
    const data = @as([*]u8, @ptrCast(ptr));
    while (i < size) : (i += pattern.len) {
        const check_len = @min(pattern.len, size - i);
        if (!mem.eql(u8, data[i..i + check_len], pattern[0..check_len])) return false;
    }
    return true;
}

pub fn virtualMemoryMap(addr: ?*anyopaque, size: usize, prot: u32, flags: u32) !*anyopaque {
    if (size == 0) return error.InvalidSize;

    const ptr = try std.os.mmap(
        @ptrCast(addr),
        size,
        prot,
        flags | std.os.MAP.ANONYMOUS,
        -1,
        0,
    );

    return @ptrCast(ptr.ptr);
}

pub fn virtualMemoryUnmap(addr: *anyopaque, size: usize) void {
    std.os.munmap(@as([*]u8, @ptrCast(addr))[0..size]);
}

pub fn protectMemory(addr: *anyopaque, size: usize, prot: u32) !void {
    if (size == 0) return error.InvalidSize;

    const aligned_addr = mem.alignBackward(usize, @intFromPtr(addr), PageSize);
    const aligned_size = mem.alignForward(usize, size + (@intFromPtr(addr) - aligned_addr), PageSize);

    try std.os.mprotect(@as([*]align(PageSize) u8, @ptrFromInt(aligned_addr))[0..aligned_size], prot);
}

pub fn lockMemory(addr: *anyopaque, size: usize) !void {
    if (size == 0) return error.InvalidSize;

    const aligned_addr = mem.alignBackward(usize, @intFromPtr(addr), PageSize);
    const aligned_size = mem.alignForward(usize, size + (@intFromPtr(addr) - aligned_addr), PageSize);

    try std.os.mlock(@as([*]align(PageSize) u8, @ptrFromInt(aligned_addr))[0..aligned_size]);
}

pub fn unlockMemory(addr: *anyopaque, size: usize) void {
    if (size == 0) return;

    const aligned_addr = mem.alignBackward(usize, @intFromPtr(addr), PageSize);
    const aligned_size = mem.alignForward(usize, size + (@intFromPtr(addr) - aligned_addr), PageSize);

    std.os.munlock(@as([*]align(PageSize) u8, @ptrFromInt(aligned_addr))[0..aligned_size]) catch {};
}

pub fn adviseMemory(addr: *anyopaque, size: usize, advice: u32) void {
    if (std.builtin.os.tag != .linux) return;
    const aligned_addr = mem.alignBackward(usize, @intFromPtr(addr), PageSize);
    const aligned_size = mem.alignForward(usize, size + (@intFromPtr(addr) - aligned_addr), PageSize);
    _ = std.os.madvise(@as([*]align(PageSize) u8, @ptrFromInt(aligned_addr))[0..aligned_size], advice) catch {};
}

pub fn prefetchMemory(addr: *const anyopaque, size: usize) void {
    const cache_line = 64;
    var i: usize = 0;
    while (i < size) : (i += cache_line) {
        @prefetch(@as(*const u8, @ptrCast(addr)) + i, .{.rw = .read, .locality = 3, .cache = .data});
    }
}

pub fn lockPages(ptr: *anyopaque, size: usize) !void {
    return lockMemory(ptr, size);
}

pub fn prefetchPages(ptr: *anyopaque, size: usize) void {
    prefetchMemory(ptr, size);
}

pub fn dontNeedPages(ptr: *anyopaque, size: usize) void {
    adviseMemory(ptr, size, std.os.linux.MADV_DONTNEED);
}

pub fn sequentialAccess(ptr: *anyopaque, size: usize) void {
    adviseMemory(ptr, size, std.os.linux.MADV_SEQUENTIAL);
}

pub fn willNeedPages(ptr: *anyopaque, size: usize) void {
    adviseMemory(ptr, size, std.os.linux.MADV_WILLNEED);
}

pub fn hugePagesAdvice(ptr: *anyopaque, size: usize) void {
    adviseMemory(ptr, size, std.os.linux.MADV_HUGEPAGE);
}

pub fn noHugePagesAdvice(ptr: *anyopaque, size: usize) void {
    adviseMemory(ptr, size, std.os.linux.MADV_NOHUGEPAGE);
}

pub fn mergePagesAdvice(ptr: *anyopaque, size: usize) void {
    adviseMemory(ptr, size, std.os.linux.MADV_MERGEABLE);
}

pub fn noMergePagesAdvice(ptr: *anyopaque, size: usize) void {
    adviseMemory(ptr, size, std.os.linux.MADV_UNMERGEABLE);
}

pub fn discardPages(ptr: *anyopaque, size: usize) void {
    adviseMemory(ptr, size, std.os.linux.MADV_DONTNEED);
}

pub fn hardwarePoisonPage(ptr: *anyopaque) void {
    adviseMemory(ptr, PageSize, std.os.linux.MADV_HWPOISON);
}

pub fn softOfflinePage(ptr: *anyopaque) void {
    adviseMemory(ptr, PageSize, std.os.linux.MADV_SOFT_OFFLINE);
}

pub fn removeMapping(ptr: *anyopaque, size: usize) void {
    adviseMemory(ptr, size, std.os.linux.MADV_REMOVE);
}

pub fn dontForkMapping(ptr: *anyopaque, size: usize) void {
    adviseMemory(ptr, size, std.os.linux.MADV_DONTFORK);
}

pub fn doForkMapping(ptr: *anyopaque, size: usize) void {
    adviseMemory(ptr, size, std.os.linux.MADV_DOFORK);
}

pub fn dontDumpMapping(ptr: *anyopaque, size: usize) void {
    adviseMemory(ptr, size, std.os.linux.MADV_DONTDUMP);
}

pub fn doDumpMapping(ptr: *anyopaque, size: usize) void {
    adviseMemory(ptr, size, std.os.linux.MADV_DODUMP);
}

pub fn wipeOnForkMapping(ptr: *anyopaque, size: usize) void {
    adviseMemory(ptr, size, std.os.linux.MADV_WIPEONFORK);
}

pub fn keepOnForkMapping(ptr: *anyopaque, size: usize) void {
    adviseMemory(ptr, size, std.os.linux.MADV_KEEPONFORK);
}

pub fn coldAdvice(ptr: *anyopaque, size: usize) void {
    adviseMemory(ptr, size, std.os.linux.MADV_COLD);
}

pub fn pageOutAdvice(ptr: *anyopaque, size: usize) void {
    adviseMemory(ptr, size, std.os.linux.MADV_PAGEOUT);
}

pub fn populateReadAdvice(ptr: *anyopaque, size: usize) void {
    adviseMemory(ptr, size, std.os.linux.MADV_POPULATE_READ);
}

pub fn populateWriteAdvice(ptr: *anyopaque, size: usize) void {
    adviseMemory(ptr, size, std.os.linux.MADV_POPULATE_WRITE);
}

pub fn memoryAdviceRandom() void {
    adviseMemory(undefined, 0, std.os.linux.MADV_RANDOM);
}

pub fn memoryAdviceNormal() void {
    adviseMemory(undefined, 0, std.os.linux.MADV_NORMAL);
}

pub fn trimExcessCapacity(allocator: Allocator, buf: []u8, used: usize) ![]u8 {
    if (used >= buf.len) return buf;
    return try allocator.resize(buf, used);
}

pub fn splitMemory(allocator: Allocator, data: []const u8, delim: u8) ![][]const u8 {
    var count: usize = 1;
    for (data) |c| {
        if (c == delim) count += 1;
    }
    const parts = try allocator.alloc([]const u8, count);
    var start: usize = 0;
    var i: usize = 0;
    var j: usize = 0;
    while (j < data.len) : (j += 1) {
        if (data[j] == delim) {
            parts[i] = data[start..j];
            start = j + 1;
            i += 1;
        }
    }
    parts[i] = data[start..];
    return parts;
}

pub fn branchlessSelect(cond: bool, true_val: usize, false_val: usize) usize {
    const mask: isize = -@as(isize, @intFromBool(cond));
    const umask: usize = @bitCast(mask);
    const inverted: isize = ~mask;
    return (true_val & umask) | (false_val & @as(usize, @bitCast(inverted)));
}

pub fn criticalSectionEnter(mutex: *Mutex) void {
    mutex.lock();
}

pub fn criticalSectionExit(mutex: *Mutex) void {
    mutex.unlock();
}

pub fn waitOnCondition(cond: *CondVar, mutex: *Mutex) void {
    cond.wait(mutex);
}

pub fn signalCondition(cond: *CondVar) void {
    cond.signal();
}

pub fn broadcastCondition(cond: *CondVar) void {
    cond.broadcast();
}

pub fn semaphoreWait(sem: *Semaphore) void {
    sem.wait();
}

pub fn semaphorePost(sem: *Semaphore) void {
    sem.post();
}

pub fn overcommitMemory(enable: bool) void {
    _ = enable;
}

pub fn compactHeap() void {
}

pub fn largePageSupport() bool {
    return true;
}

pub fn memoryEfficientHashMap(comptime K: type, comptime V: type) type {
    return std.AutoHashMap(K, V);
}

pub fn compactArrayList(comptime T: type) type {
    return std.ArrayList(T);
}

pub fn smallObjectAllocator(allocator: Allocator, object_size: usize) !PoolAllocator {
    return try PoolAllocator.init(allocator, object_size, 1024, 4);
}

pub fn temporalAllocator(allocator: Allocator) ArenaAllocator {
    return ArenaAllocator.init(allocator, 1 << 20);
}

pub fn compressMemory(data: []const u8, allocator: Allocator) ![]u8 {
    var list = std.ArrayList(u8).init(allocator);
    defer list.deinit();
    var compressor = try std.compress.deflate.compressor(list.writer(), .{ .level = .best_compression });
    try compressor.writeAll(data);
    try compressor.finish();
    return try list.toOwnedSlice();
}

pub fn decompressMemory(data: []const u8, allocator: Allocator) ![]u8 {
    var list = std.ArrayList(u8).init(allocator);
    defer list.deinit();
    var decompressor = try std.compress.deflate.decompressor(data.reader());
    var reader = decompressor.reader();
    var buf: [4096]u8 = undefined;
    while (true) {
        const len = reader.read(&buf) catch break;
        if (len == 0) break;
        try list.appendSlice(buf[0..len]);
    }
    return try list.toOwnedSlice();
}

pub fn encryptMemory(data: []u8, key: [32]u8) void {
    const nonce: [12]u8 = [_]u8{0} ** 12;
    std.crypto.stream.chacha.ChaCha20IETF.xor(data, data, 0, key, nonce);
}

pub fn decryptMemory(data: []u8, key: [32]u8) !void {
    encryptMemory(data, key); // symmetric
}

pub const CompressedStorage = struct {
    compressed: []u8,
    allocator: Allocator,

    pub fn init(allocator: Allocator, data: []const u8) !CompressedStorage {
        const compressed = try compressMemory(data, allocator);
        return .{ .compressed = compressed, .allocator = allocator };
    }

    pub fn deinit(self: *CompressedStorage) void {
        self.allocator.free(self.compressed);
    }

    pub fn decompress(self: *const CompressedStorage) ![]u8 {
        return try decompressMemory(self.compressed, self.allocator);
    }
};

pub const EncryptedStorage = struct {
    encrypted: []u8,
    key: [32]u8,
    allocator: Allocator,

    pub fn init(allocator: Allocator, data: []const u8, key: [32]u8) !EncryptedStorage {
        const encrypted = try allocator.alloc(u8, data.len);
        @memcpy(encrypted, data);
        encryptMemory(encrypted, key);
        return .{ .encrypted = encrypted, .key = key, .allocator = allocator };
    }

    pub fn deinit(self: *EncryptedStorage) void {
        self.allocator.free(self.encrypted);
    }

    pub fn decrypt(self: *const EncryptedStorage) ![]u8 {
        const decrypted = try self.allocator.alloc(u8, self.encrypted.len);
        @memcpy(decrypted, self.encrypted);
        try decryptMemory(decrypted, self.key);
        return decrypted;
    }
};

pub fn memoryEfficientString(in: []const u8) []const u8 {
    return in;
}

pub fn stringInterning(pool: *std.StringHashMap(void), str: []const u8) ![]const u8 {
    const result = try pool.getOrPut(str);
    if (!result.found_existing) {
        result.value_ptr.* = {};
    }
    return result.key_ptr.*;
}

pub fn persistentAllocator(allocator: Allocator, size: usize) !Allocator {
    _ = size;
    return allocator;
}

pub fn memoryMappedHashMap(comptime K: type, comptime V: type, allocator: Allocator) std.AutoHashMap(K, V) {
    return std.AutoHashMap(K, V).init(allocator);
}

pub fn vectorizedFill(ptr: *anyopaque, value: u8, size: usize) void {
    std.mem.set(u8, @as([*]u8, @ptrCast(ptr))[0..size], value);
}

pub fn simdCompare(a: []const u8, b: []const u8) bool {
    return compareMemory(a, b);
}

pub fn getUsableSize(ptr: *anyopaque) usize {
    _ = ptr;
    return 0;
}

pub fn alignedRealloc(allocator: Allocator, old_mem: []u8, new_size: usize, alignment: usize) ![]u8 {
    const new_mem = try allocator.alignedAlloc(u8, alignment, new_size);
    @memcpy(new_mem[0..@min(old_mem.len, new_size)], old_mem[0..@min(old_mem.len, new_size)]);
    allocator.free(old_mem);
    return new_mem;
}

pub fn zeroMemoryRange(start: *anyopaque, end: *anyopaque) void {
    const start_addr = @intFromPtr(start);
    const end_addr = @intFromPtr(end);
    if (end_addr > start_addr) {
        const size = end_addr - start_addr;
        std.mem.set(u8, @as([*]u8, @ptrCast(start))[0..size], 0);
    }
}

pub fn memoryAlign(ptr: *anyopaque, alignment: usize) *anyopaque {
    const addr = @intFromPtr(ptr);
    const aligned_addr = mem.alignForward(usize, addr, alignment);
    return @ptrFromInt(aligned_addr);
}

pub fn isMemoryOverlap(a_start: *const anyopaque, a_size: usize, b_start: *const anyopaque, b_size: usize) bool {
    const a_addr = @intFromPtr(a_start);
    const b_addr = @intFromPtr(b_start);
    const a_end = a_addr + a_size;
    const b_end = b_addr + b_size;
    return (a_addr < b_end) and (b_addr < a_end);
}

pub fn copyNonOverlapping(dest: []u8, src: []const u8) !void {
    if (isMemoryOverlap(dest.ptr, dest.len, src.ptr, src.len)) {
        return error.Overlap;
    }
    @memcpy(dest, src);
}

pub fn moveMemory(dest: []u8, src: []const u8) void {
    if (dest.len != src.len) return;
    if (dest.ptr == src.ptr) return;

    if (@intFromPtr(dest.ptr) < @intFromPtr(src.ptr)) {
        var i: usize = 0;
        while (i < dest.len) : (i += 1) {
            dest[i] = src[i];
        }
    } else {
        var i: usize = dest.len;
        while (i > 0) {
            i -= 1;
            dest[i] = src[i];
        }
    }
}

pub const MemoryPool = PoolAllocator;
pub const MemoryArena = Arena;
pub const MemorySlab = SlabAllocator;
pub const MemoryBuddy = BuddyAllocator;
pub const MemoryLockFreeQueue = LockFreeQueue;
pub const MemoryLockFreeStack = LockFreeStack;

test "Arena allocation" {
    var arena = try Arena.init(testing.allocator, 1024);
    defer arena.deinit();
    const ptr1 = arena.alloc(128, 8).?;
    const ptr2 = arena.alloc(64, 4).?;
    try testing.expect(ptr1.len == 128);
    try testing.expect(ptr2.len == 64);
}

test "SlabAllocator" {
    var slab = try SlabAllocator.init(testing.allocator, 256, 4, 64);
    defer slab.deinit();
    const ptr1 = slab.alloc(100).?;
    const ptr2 = slab.alloc(150).?;
    try testing.expect(ptr1.len == 100);
    try testing.expect(ptr2.len == 150);
    slab.free(ptr1);
    slab.free(ptr2);
}

test "PoolAllocator" {
    var pool = try PoolAllocator.init(testing.allocator, 64, 16, 2);
    defer pool.deinit();
    const ptr1 = pool.alloc(64).?;
    const ptr2 = pool.alloc(64).?;
    try testing.expect(ptr1.len == 64);
    try testing.expect(ptr2.len == 64);
    pool.free(ptr1);
    pool.free(ptr2);
}

test "LockFreeFreelist" {
    var freelist = try LockFreeFreelist.init(testing.allocator, 256, 8);
    defer freelist.deinit();
    const ptr1 = freelist.alloc().?;
    const ptr2 = freelist.alloc().?;
    try testing.expect(ptr1.len == 256);
    try testing.expect(ptr2.len == 256);
    freelist.free(ptr1);
    freelist.free(ptr2);
}

test "PageAllocator" {
    var page_alloc = try PageAllocator.init(testing.allocator, 4);
    defer page_alloc.deinit();
    const pages = page_alloc.allocPages(2).?;
    try testing.expect(pages.len == 8192);
    page_alloc.freePages(pages);
}

test "ZeroCopySlice" {
    const data = "hello world";
    const zcs = ZeroCopySlice.init(@as([*]const u8, @ptrCast(data.ptr)), data.len);
    const slice = zcs.slice(0, 5);
    try testing.expectEqualStrings("hello", slice.asBytes());
}

test "ResizeBuffer" {
    var buf = ResizeBuffer.init(testing.allocator);
    defer buf.deinit();
    try buf.append("hello");
    try buf.append(" world");
    const owned = buf.toOwnedSlice();
    defer testing.allocator.free(owned);
    try testing.expectEqualStrings("hello world", owned);
}

test "ArenaAllocator basic allocation" {
    var arena = ArenaAllocator.init(testing.allocator, 1024);
    defer arena.deinit();

    const alloc = arena.allocator();
    const slice1 = try alloc.alloc(u8, 100);
    const slice2 = try alloc.alloc(u8, 100);

    std.mem.set(u8, slice1, 42);
    std.mem.set(u8, slice2, 84);

    try testing.expectEqual(@as(u8, 42), slice1[0]);
    try testing.expectEqual(@as(u8, 84), slice2[0]);
}

test "zero copy transfer" {
    var src = [_]u8{1, 2, 3, 4, 5};
    var dest: [5]u8 = undefined;

    zeroCopyTransfer(&src, &dest);

    try testing.expectEqualSlices(u8, &src, &dest);
}

test "memory hashing" {
    const data1 = "hello world";
    const data2 = "hello world";
    const data3 = "hello world!";

    const hash1 = hashMemory(data1);
    const hash2 = hashMemory(data2);
    const hash3 = hashMemory(data3);

    try testing.expectEqual(hash1, hash2);
    try testing.expect(hash1 != hash3);
}

test "memory comparison" {
    const data1 = "test";
    const data2 = "test";
    const data3 = "best";

    try testing.expect(compareMemory(data1, data2));
    try testing.expect(!compareMemory(data1, data3));
}

test "search memory" {
    const haystack = "hello world, hello universe";
    const needle = "world";

    const pos = searchMemory(haystack, needle);
    try testing.expect(pos != null);
    try testing.expectEqual(@as(usize, 6), pos.?);
}

test "count memory" {
    const data = "hello world";
    const count = countMemory(data, 'l');
    try testing.expectEqual(@as(usize, 3), count);
}

test "unique memory" {
    const data = "aabbccddaa";
    const unique = try uniqueMemory(testing.allocator, data);
    defer testing.allocator.free(unique);
    try testing.expect(unique.len == 4);
}

test "atomic operations" {
    var value: u64 = 0;

    const prev = atomicAdd(&value, 5);
    try testing.expectEqual(@as(u64, 0), prev);
    try testing.expectEqual(@as(u64, 5), atomicLoad(&value));

    atomicStore(&value, 10);
    try testing.expectEqual(@as(u64, 10), atomicLoad(&value));

    _ = atomicInc(&value);
    try testing.expectEqual(@as(u64, 11), atomicLoad(&value));
}

test "ReadWriteLock" {
    var rwlock = ReadWriteLock.init();

    rwlock.readLock();
    rwlock.readUnlock();

    rwlock.writeLock();
    rwlock.writeUnlock();
}

test "BuddyAllocator" {
    var buddy = try BuddyAllocator.init(testing.allocator, 4096, 6);
    defer buddy.deinit();

    const ptr1 = try buddy.alloc(128);
    try testing.expect(ptr1.len == 128);
    buddy.free(ptr1);
}

test "LockFreeQueue" {
    var queue = try LockFreeQueue.init(testing.allocator, 16);
    defer queue.deinit(testing.allocator);

    var item: usize = 42;
    try testing.expect(queue.enqueue(@as(*anyopaque, @ptrCast(&item))));

    const retrieved = queue.dequeue();
    try testing.expect(retrieved != null);
}

test "LockFreeStack" {
    var stack = LockFreeStack.init(testing.allocator);
    defer stack.deinit();

    var item: usize = 42;
    try stack.push(@as(*anyopaque, @ptrCast(&item)));

    const retrieved = stack.pop();
    try testing.expect(retrieved != null);
}

test "memory stats tracking" {
    resetMemoryStats();

    trackAllocation(100);
    trackAllocation(200);
    trackFree(50);

    const stats = getMemoryStats();
    try testing.expectEqual(@as(usize, 300), stats.allocated);
    try testing.expectEqual(@as(usize, 50), stats.freed);
    try testing.expectEqual(@as(usize, 300), stats.peak);
}