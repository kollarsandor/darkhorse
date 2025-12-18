const std = @import("std");
const fs = std.fs;
const mem = std.mem;
const math = std.math;
const crypto = std.crypto;
const builtin = @import("builtin");
const Allocator = mem.Allocator;
const types = @import("types.zig");
const PRNG = types.PRNG;

fn generateRuntimeSeed() u64 {
    const timestamp: u64 = @intCast(@max(0, std.time.milliTimestamp()));
    const pid: u64 = if (builtin.os.tag != .windows) @intCast(std.os.linux.getpid()) else 0;
    var entropy_buf: [8]u8 = undefined;
    std.crypto.random.bytes(&entropy_buf);
    const entropy = mem.readInt(u64, &entropy_buf, .Little);
    return timestamp ^ (pid << 32) ^ entropy;
}

pub const MMAP = struct {
    file: fs.File,
    buffer: ?[]align(mem.page_size) u8,
    allocator: Allocator,
    is_writable: bool,

    pub fn open(allocator: Allocator, path: []const u8, mode: fs.File.OpenFlags) !MMAP {
        const file = try fs.cwd().openFile(path, mode);
        errdefer file.close();
        
        const stat = try file.stat();
        if (stat.size < 0) return error.InvalidFileSize;
        const size_u64: u64 = @intCast(stat.size);
        if (size_u64 > math.maxInt(usize)) return error.FileTooLarge;
        var size: usize = @intCast(size_u64);
        
        const is_writable = mode.mode == .read_write or mode.mode == .write_only;
        const is_append = if (mode.mode == .read_write) true else false;
        
        const prot_flags = if (is_writable or is_append)
            std.posix.PROT.READ | std.posix.PROT.WRITE
        else
            std.posix.PROT.READ;
        
        if (size == 0) {
            if (is_writable) {
                const page_size = mem.page_size;
                try file.setEndPos(page_size);
                const zeros = try allocator.alloc(u8, page_size);
                defer allocator.free(zeros);
                @memset(zeros, 0);
                try file.pwriteAll(zeros, 0);
                size = page_size;
            } else {
                return error.FileIsEmpty;
            }
        }
        
        const aligned_size = std.mem.alignForward(usize, size, mem.page_size);
        
        const map_type = if (is_writable) .SHARED else .PRIVATE;
        
        const buffer = try std.posix.mmap(
            null,
            aligned_size,
            prot_flags,
            .{ .TYPE = map_type },
            file.handle,
            0
        );
        
        return .{ 
            .file = file, 
            .buffer = buffer, 
            .allocator = allocator,
            .is_writable = is_writable,
        };
    }

    pub fn close(self: *MMAP) void {
        if (self.buffer) |buf| {
            std.posix.munmap(buf);
            self.buffer = null;
        }
        self.file.close();
    }

    pub fn read(self: *const MMAP, offset: usize, len: usize) ![]const u8 {
        const buf = self.buffer orelse return error.BufferNotMapped;
        if (offset >= buf.len) return error.OutOfBounds;
        if (offset > buf.len - len) {
            const end = buf.len;
            return buf[offset..end];
        }
        return buf[offset..offset + len];
    }

    pub fn write(self: *MMAP, offset: usize, data: []const u8, sync_mode: enum { sync, nosync }) !void {
        const buf = self.buffer orelse return error.BufferNotMapped;
        if (offset > buf.len) return error.OutOfBounds;
        if (offset + data.len > buf.len) return error.OutOfBounds;
        @memcpy(buf[offset..offset + data.len], data);
        const should_sync = sync_mode == .sync;
        try std.posix.msync(buf, .{ .SYNC = should_sync });
    }

    pub fn append(self: *MMAP, data: []const u8) !void {
        const buf = self.buffer orelse return error.BufferNotMapped;
        const old_size = buf.len;
        const new_size = old_size + data.len;
        
        const lock = try self.file.lock(.exclusive);
        defer self.file.unlock();
        _ = lock;
        
        std.posix.munmap(buf);
        self.buffer = null;
        
        const stat = try self.file.stat();
        const current_size: usize = @intCast(@max(0, stat.size));
        const extend_size = @max(new_size, current_size + data.len);
        
        const old_end = current_size;
        const new_end = extend_size;
        if (new_end > old_end) {
            const zero_len = new_end - old_end;
            const zeros = try self.allocator.alloc(u8, zero_len);
            defer self.allocator.free(zeros);
            @memset(zeros, 0);
            try self.file.setEndPos(new_end);
            try self.file.pwriteAll(zeros, old_end);
        }
        
        try self.file.pwriteAll(data, old_size);
        
        const aligned_size = std.mem.alignForward(usize, extend_size, mem.page_size);
        
        self.buffer = try std.posix.mmap(
            null,
            aligned_size,
            std.posix.PROT.READ | std.posix.PROT.WRITE,
            .{ .TYPE = .SHARED },
            self.file.handle,
            0
        );
    }
};

pub const DurableWriter = struct {
    file: fs.File,
    buffer: [8192]u8,
    pos: usize = 0,
    allocator: Allocator,
    flush_depth: usize = 0,

    pub fn init(allocator: Allocator, path: []const u8, enable_sync: bool) !DurableWriter {
        const flags: fs.File.CreateFlags = if (enable_sync) 
            .{ .truncate = true, .mode = 0o666 }
        else
            .{ .truncate = true };
        const file = try fs.cwd().createFile(path, flags);
        if (enable_sync) {
            if (builtin.os.tag != .windows) {
                const fd = file.handle;
                const O_SYNC = if (@hasDecl(std.posix, "O")) std.posix.O.SYNC else 0;
                _ = std.posix.fcntl(fd, std.posix.F.SETFL, O_SYNC) catch |err| return err;
            }
        }
        return .{ 
            .file = file, 
            .allocator = allocator, 
            .buffer = mem.zeroes([8192]u8),
        };
    }

    pub fn deinit(self: *DurableWriter) void {
        self.flush() catch |err| {
            std.debug.print("Warning: flush failed during deinit: {}\n", .{err});
        };
        self.file.close();
    }

    pub fn write(self: *DurableWriter, data: []const u8) !void {
        if (data.len == 0) return;
        
        if (self.pos == self.buffer.len) {
            try self.flush();
        }
        
        if (data.len >= self.buffer.len - self.pos) {
            if (self.pos > 0) {
                try self.flush();
            }
            if (data.len >= self.buffer.len) {
                var written: usize = 0;
                while (written < data.len) {
                    const n = try self.file.write(data[written..]);
                    if (n == 0) return error.EndOfStream;
                    written += n;
                }
                return;
            }
        }
        
        const space = self.buffer.len - self.pos;
        const to_copy = @min(data.len, space);
        @memcpy(self.buffer[self.pos..self.pos + to_copy], data[0..to_copy]);
        self.pos += to_copy;
        
        if (to_copy < data.len) {
            try self.flush();
            const remaining = data[to_copy..];
            @memcpy(self.buffer[0..remaining.len], remaining);
            self.pos = remaining.len;
        }
    }

    pub fn flush(self: *DurableWriter) !void {
        if (self.flush_depth > 10) {
            return error.RecursionDepthExceeded;
        }
        self.flush_depth += 1;
        defer self.flush_depth -= 1;
        
        if (self.pos > 0) {
            var written: usize = 0;
            while (written < self.pos) {
                const n = try self.file.write(self.buffer[written..self.pos]);
                if (n == 0) return error.EndOfStream;
                written += n;
            }
            self.pos = 0;
        }
    }

    pub fn writeAll(self: *DurableWriter, data: []const u8) !void {
        var written: usize = 0;
        while (written < data.len) {
            const chunk = data[written..];
            try self.write(chunk);
            written = data.len;
        }
        try self.flush();
    }
};

pub const BufferedReader = struct {
    file: fs.File,
    buffer: [8192]u8,
    pos: usize = 0,
    limit: usize = 0,
    allocator: Allocator,
    max_read_bytes: usize,

    pub fn init(allocator: Allocator, path: []const u8) !BufferedReader {
        const file = try fs.cwd().openFile(path, .{});
        return .{ 
            .file = file, 
            .allocator = allocator, 
            .buffer = mem.zeroes([8192]u8),
            .max_read_bytes = 100 * 1024 * 1024,
        };
    }

    pub fn deinit(self: *BufferedReader) void {
        self.file.close();
    }

    pub fn read(self: *BufferedReader, buf: []u8) !usize {
        if (buf.len == 0) return 0;
        
        var total: usize = 0;
        while (total < buf.len) {
            if (self.pos < self.limit) {
                const avail = @min(self.limit - self.pos, buf.len - total);
                @memcpy(buf[total..total + avail], self.buffer[self.pos..self.pos + avail]);
                self.pos += avail;
                total += avail;
            } else {
                const n = try self.file.read(self.buffer[0..]);
                self.limit = n;
                self.pos = 0;
                if (n == 0) break;
            }
        }
        return total;
    }

    pub fn readUntil(self: *BufferedReader, delim: u8, allocator: Allocator) ![]u8 {
        var list = std.ArrayList(u8).init(allocator);
        errdefer list.deinit();
        
        while (list.items.len < self.max_read_bytes) {
            if (self.pos < self.limit) {
                const chunk = self.buffer[self.pos..self.limit];
                if (mem.indexOfScalar(u8, chunk, delim)) |idx| {
                    try list.appendSlice(chunk[0..idx + 1]);
                    self.pos += idx + 1;
                    return list.toOwnedSlice();
                } else {
                    try list.appendSlice(chunk);
                    self.pos = self.limit;
                }
            } else {
                const n = try self.file.read(self.buffer[0..]);
                self.limit = n;
                self.pos = 0;
                if (n == 0) return list.toOwnedSlice();
            }
        }
        return error.MaxBytesExceeded;
    }

    pub fn peek(self: *BufferedReader) !?u8 {
        if (self.pos < self.limit) return self.buffer[self.pos];
        const n = try self.file.read(self.buffer[0..]);
        self.limit = n;
        self.pos = 0;
        if (n == 0) return null;
        return self.buffer[0];
    }
};

pub const BufferedWriter = struct {
    file: fs.File,
    buffer: []u8,
    pos: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator, file: fs.File, buffer_size: usize) !BufferedWriter {
        const buffer = try allocator.allocWithOptions(u8, buffer_size, null, null);
        errdefer allocator.free(buffer);
        @memset(buffer, 0);
        return .{
            .file = file,
            .buffer = buffer,
            .pos = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *BufferedWriter) void {
        self.flush() catch |err| {
            std.debug.print("Warning: flush failed during deinit: {}\n", .{err});
        };
        self.allocator.free(self.buffer);
    }

    pub fn writeByte(self: *BufferedWriter, byte: u8) !void {
        if (self.pos >= self.buffer.len) {
            try self.flush();
        }
        self.buffer[self.pos] = byte;
        self.pos += 1;
    }

    pub fn writeBytes(self: *BufferedWriter, data: []const u8) !void {
        if (data.len == 0) return;
        
        if (data.len >= self.buffer.len - self.pos) {
            if (self.pos > 0) {
                try self.flush();
            }
            if (data.len >= self.buffer.len) {
                const chunk_size = 65536;
                var written: usize = 0;
                while (written < data.len) {
                    const to_write = @min(chunk_size, data.len - written);
                    try self.file.writeAll(data[written..written + to_write]);
                    written += to_write;
                }
                return;
            }
        }
        
        var written: usize = 0;
        while (written < data.len) {
            const available = self.buffer.len - self.pos;
            const to_write = @min(available, data.len - written);
            
            @memcpy(self.buffer[self.pos..self.pos + to_write], data[written..written + to_write]);
            self.pos += to_write;
            written += to_write;
            
            if (self.pos >= self.buffer.len) {
                try self.flush();
            }
        }
    }

    pub fn flush(self: *BufferedWriter) !void {
        if (self.pos > 0) {
            try self.file.writeAll(self.buffer[0..self.pos]);
            self.pos = 0;
        }
    }
};

pub fn stableHash(data: []const u8, seed: u64) u64 {
    var hasher = std.hash.Wyhash.init(seed);
    hasher.update(data);
    const h = hasher.final();
    const mixed = h ^ (h >> 33);
    const mul1 = mixed *% 0xff51afd7ed558ccd;
    const xor1 = mul1 ^ (mul1 >> 33);
    const mul2 = xor1 *% 0xc4ceb9fe1a85ec53;
    return mul2 ^ (mul2 >> 33);
}

pub fn hash64(data: []const u8) u64 {
    const seed = generateRuntimeSeed();
    var hasher = std.hash.Wyhash.init(seed);
    hasher.update(data);
    const h = hasher.final();
    const mixed = h ^ (h >> 33);
    const mul1 = mixed *% 0xff51afd7ed558ccd;
    const xor1 = mul1 ^ (mul1 >> 33);
    const mul2 = xor1 *% 0xc4ceb9fe1a85ec53;
    return mul2 ^ (mul2 >> 33);
}

pub fn hash32(data: []const u8) u32 {
    const h64 = hash64(data);
    const mixed = h64 ^ (h64 >> 32);
    return @truncate(mixed);
}

pub fn pathJoin(allocator: Allocator, parts: []const []const u8) ![]u8 {
    if (parts.len == 0) return try allocator.alloc(u8, 0);
    
    const separator: u8 = if (builtin.os.tag == .windows) '\\' else '/';
    
    var total_len: usize = 0;
    var non_empty_count: usize = 0;
    var starts_with_slash = false;
    
    var i: usize = 0;
    while (i < parts.len) : (i += 1) {
        const part = parts[i];
        if (part.len > 0) {
            if (i == 0 and part[0] == '/') {
                starts_with_slash = true;
            }
            for (part) |c| {
                if (c == '/' or c == '\\') {
                    if (builtin.os.tag != .windows and c == '\\') {
                        return error.InvalidPathCharacter;
                    }
                }
            }
            total_len += part.len;
            non_empty_count += 1;
        }
    }
    
    if (non_empty_count == 0) return try allocator.alloc(u8, 0);
    
    const sep_count = if (non_empty_count > 1) non_empty_count - 1 else 0;
    total_len += sep_count;
    
    const path = try allocator.alloc(u8, total_len);
    errdefer allocator.free(path);
    
    var pos: usize = 0;
    var is_first = true;
    for (parts) |part| {
        if (part.len == 0) continue;
        
        if (!is_first) {
            path[pos] = separator;
            pos += 1;
        }
        
        var skip_leading_slash = false;
        if (is_first and starts_with_slash and part.len > 0 and part[0] == '/') {
            skip_leading_slash = false;
        }
        
        const src = if (skip_leading_slash and part.len > 0 and part[0] == '/') 
            part[1..] 
        else 
            part;
        
        @memcpy(path[pos..pos + src.len], src);
        pos += src.len;
        is_first = false;
    }
    return path;
}

pub fn pathExists(path: []const u8) bool {
    fs.cwd().access(path, .{ .mode = .read_only }) catch |err| {
        _ = err;
        return false;
    };
    return true;
}

pub fn createDirRecursive(allocator: Allocator, path: []const u8) !void {
    if (path.len == 0) return;
    
    const separator = if (builtin.os.tag == .windows) '\\' else '/';
    
    var it = if (builtin.os.tag == .windows)
        mem.splitAny(u8, path, "/\\")
    else
        mem.splitScalar(u8, path, separator);
    
    var current_list = std.ArrayList(u8).init(allocator);
    defer current_list.deinit();
    
    var first = true;
    while (it.next()) |part| {
        if (part.len == 0) {
            if (first and path[0] == separator) {
                try current_list.append(separator);
            }
            first = false;
            continue;
        }
        
        if (current_list.items.len > 0 and current_list.items[current_list.items.len - 1] != separator) {
            try current_list.append(separator);
        }
        try current_list.appendSlice(part);
        
        fs.cwd().makeDir(current_list.items) catch |err| {
            if (err == error.PathAlreadyExists) {
                first = false;
                continue;
            }
            return err;
        };
        first = false;
    }
}

pub fn readFile(allocator: Allocator, path: []const u8) ![]u8 {
    const file = try fs.cwd().openFile(path, .{});
    defer file.close();
    const stat = try file.stat();
    if (stat.size < 0) return error.InvalidFileSize;
    const size_u64: u64 = @intCast(stat.size);
    if (size_u64 > math.maxInt(usize)) return error.FileTooLarge;
    const size: usize = @intCast(size_u64);
    if (size == 0) return try allocator.alloc(u8, 0);
    const buf = try allocator.alloc(u8, size);
    errdefer allocator.free(buf);
    const bytes_read = try file.readAll(buf);
    if (bytes_read != size) return error.UnexpectedEndOfFile;
    return buf;
}

pub const WriteFileOptions = struct {
    create_backup: bool = false,
};

pub fn writeFile(path: []const u8, data: []const u8) !void {
    return writeFileWithOptions(path, data, .{});
}

pub fn writeFileWithOptions(path: []const u8, data: []const u8, options: WriteFileOptions) !void {
    if (options.create_backup) {
        if (pathExists(path)) {
            const backup_path = try std.fmt.allocPrint(std.heap.page_allocator, "{s}.bak", .{path});
            defer std.heap.page_allocator.free(backup_path);
            fs.cwd().copyFile(path, fs.cwd(), backup_path, .{}) catch |err| {
                std.debug.print("Warning: backup creation failed: {}\n", .{err});
            };
        }
    }
    const file = try fs.cwd().createFile(path, .{});
    defer file.close();
    try file.writeAll(data);
}

pub fn appendFile(path: []const u8, data: []const u8) !void {
    const file = fs.cwd().openFile(path, .{ .mode = .read_write }) catch |err| {
        if (err == error.FileNotFound) {
            const new_file = try fs.cwd().createFile(path, .{});
            defer new_file.close();
            try new_file.writeAll(data);
            return;
        }
        return err;
    };
    defer file.close();
    try file.seekFromEnd(0);
    try file.writeAll(data);
}

pub fn deleteFile(path: []const u8) !void {
    const stat = fs.cwd().statFile(path) catch |err| {
        return err;
    };
    if (stat.kind == .directory) {
        return fs.cwd().deleteTree(path);
    }
    try fs.cwd().deleteFile(path);
}

pub const CopyProgress = struct {
    bytes_copied: usize,
    total_bytes: usize,
};

pub fn copyFile(src: []const u8, dst: []const u8, allocator: Allocator) !void {
    return copyFileWithProgress(src, dst, allocator, null);
}

pub fn copyFileWithProgress(
    src: []const u8, 
    dst: []const u8, 
    allocator: Allocator,
    progress_callback: ?*const fn(CopyProgress) void
) !void {
    const src_file = try fs.cwd().openFile(src, .{});
    defer src_file.close();
    
    const dst_file = try fs.cwd().createFile(dst, .{});
    defer dst_file.close();
    
    const stat = try src_file.stat();
    const total_size: usize = @intCast(@max(0, stat.size));
    
    const chunk_size = 65536;
    const buffer = try allocator.alloc(u8, chunk_size);
    defer allocator.free(buffer);
    
    var bytes_copied: usize = 0;
    while (true) {
        const n = try src_file.read(buffer);
        if (n == 0) break;
        try dst_file.writeAll(buffer[0..n]);
        bytes_copied += n;
        if (progress_callback) |cb| {
            cb(.{ .bytes_copied = bytes_copied, .total_bytes = total_size });
        }
    }
}

pub fn moveFile(old: []const u8, new: []const u8) !void {
    fs.cwd().rename(old, new) catch |err| {
        if (err == error.RenameAcrossMountPoints or err == error.NotSameFileSystem) {
            try copyFile(old, new, std.heap.page_allocator);
            try deleteFile(old);
            return;
        }
        return err;
    };
}

pub fn listDir(allocator: Allocator, path: []const u8) ![][]u8 {
    var dir = try fs.cwd().openDir(path, .{ .iterate = true });
    defer dir.close();
    
    var list = std.ArrayList([]u8).init(allocator);
    errdefer {
        for (list.items) |item| allocator.free(item);
        list.deinit();
    }
    
    var iter = dir.iterate();
    while (try iter.next()) |entry| {
        if (mem.eql(u8, entry.name, ".") or mem.eql(u8, entry.name, "..")) {
            continue;
        }
        const name = try allocator.dupe(u8, entry.name);
        errdefer allocator.free(name);
        try list.append(name);
    }
    return list.toOwnedSlice();
}

pub fn createDir(path: []const u8) !void {
    try createDirRecursive(std.heap.page_allocator, path);
}

pub fn removeDir(path: []const u8) !void {
    try fs.cwd().deleteTree(path);
}

pub fn removeFile(path: []const u8) !void {
    try fs.cwd().deleteFile(path);
}

pub fn renameFile(old: []const u8, new: []const u8) !void {
    const exists = pathExists(old);
    if (!exists) return error.FileNotFound;
    try fs.cwd().rename(old, new);
}

pub fn getFileSize(path: []const u8) !usize {
    const stat = try fs.cwd().statFile(path);
    if (stat.size < 0) return error.InvalidFileSize;
    const size_u64: u64 = @intCast(stat.size);
    if (size_u64 > math.maxInt(usize)) return error.FileTooLarge;
    return @intCast(size_u64);
}

pub fn isDir(path: []const u8) bool {
    var dir = fs.cwd().openDir(path, .{}) catch return false;
    dir.close();
    return true;
}

pub inline fn toLittleEndian(comptime T: type, value: T) T {
    return switch (comptime builtin.target.cpu.arch.endian()) {
        .Little => value,
        .Big => @byteSwap(value),
    };
}

pub inline fn fromLittleEndian(comptime T: type, bytes: *const [@sizeOf(T)]u8) T {
    return mem.readIntLittle(T, bytes);
}

pub inline fn toBigEndian(comptime T: type, value: T) T {
    return switch (comptime builtin.target.cpu.arch.endian()) {
        .Little => @byteSwap(value),
        .Big => value,
    };
}

pub inline fn fromBigEndian(comptime T: type, bytes: *const [@sizeOf(T)]u8) T {
    return mem.readIntBig(T, bytes);
}

pub fn sequentialWrite(allocator: Allocator, path: []const u8, data: []const []const u8) !void {
    const file = try fs.cwd().createFile(path, .{});
    defer file.close();
    
    const buffer_size = 65536;
    var writer = try BufferedWriter.init(allocator, file, buffer_size);
    defer writer.deinit();
    
    for (data) |chunk| {
        try writer.writeBytes(chunk);
    }
    try writer.flush();
}

pub fn sequentialRead(allocator: Allocator, path: []const u8, chunk_callback: *const fn([]const u8) anyerror!void) !void {
    const file = try fs.cwd().openFile(path, .{});
    defer file.close();
    
    const chunk_size = 65536;
    const buffer = try allocator.alloc(u8, chunk_size);
    defer allocator.free(buffer);
    
    while (true) {
        const n = try file.read(buffer);
        if (n == 0) break;
        try chunk_callback(buffer[0..n]);
    }
}

pub fn atomicWrite(allocator: Allocator, path: []const u8, data: []const u8) !void {
    const temp_path = try std.fmt.allocPrint(allocator, "{s}.tmp", .{path});
    defer allocator.free(temp_path);
    
    const file = try fs.cwd().createFile(temp_path, .{});
    errdefer {
        file.close();
        fs.cwd().deleteFile(temp_path) catch |err| {
            std.debug.print("Warning: failed to delete temp file: {}\n", .{err});
        };
    }
    
    try file.writeAll(data);
    try file.sync();
    file.close();
    
    try fs.cwd().rename(temp_path, path);
}

pub fn compareFiles(allocator: Allocator, path1: []const u8, path2: []const u8) !bool {
    const data1 = readFile(allocator, path1) catch |err| {
        if (err == error.FileNotFound) return false;
        return err;
    };
    defer allocator.free(data1);
    
    const data2 = readFile(allocator, path2) catch |err| {
        if (err == error.FileNotFound) return false;
        return err;
    };
    defer allocator.free(data2);
    
    if (data1.len != data2.len) return false;
    return mem.eql(u8, data1, data2);
}

test "MMAP open and close" {
    const gpa = std.testing.allocator;
    const temp_path = "test_mmap.bin";
    
    const file = try fs.cwd().createFile(temp_path, .{});
    try file.writeAll("test data for mmap");
    file.close();
    defer fs.cwd().deleteFile(temp_path) catch |err| {
        std.debug.print("Warning: failed to delete test file: {}\n", .{err});
    };
    
    var mmap = try MMAP.open(gpa, temp_path, .{ .mode = .read_only });
    defer mmap.close();
    
    const content = try mmap.read(0, 9);
    try std.testing.expectEqualStrings("test data", content);
}

test "DurableWriter with sync" {
    var gpa = std.testing.allocator;
    var writer = try DurableWriter.init(gpa, "test_durable.txt", false);
    defer writer.deinit();
    defer fs.cwd().deleteFile("test_durable.txt") catch |err| {
        std.debug.print("Warning: failed to delete test file: {}\n", .{err});
    };
    try writer.writeAll("hello world");
    const content = try readFile(gpa, "test_durable.txt");
    defer gpa.free(content);
    try std.testing.expectEqualStrings("hello world", content);
}

test "BufferedReader zero init" {
    const file = try fs.cwd().createFile("test_buffered.txt", .{});
    defer file.close();
    try file.writeAll("line1\nline2\nline3");
    defer fs.cwd().deleteFile("test_buffered.txt") catch |err| {
        std.debug.print("Warning: failed to delete test file: {}\n", .{err});
    };
    var gpa = std.testing.allocator;
    var reader = try BufferedReader.init(gpa, "test_buffered.txt");
    defer reader.deinit();
    
    const line1 = try reader.readUntil('\n', gpa);
    defer gpa.free(line1);
    try std.testing.expectEqualStrings("line1\n", line1);
    
    const line2 = try reader.readUntil('\n', gpa);
    defer gpa.free(line2);
    try std.testing.expectEqualStrings("line2\n", line2);
    
    const line3 = try reader.readUntil('\n', gpa);
    defer gpa.free(line3);
    try std.testing.expectEqualStrings("line3", line3);
}

test "Stable hash mixing" {
    const data = "test";
    const seed: u64 = 12345;
    const hash1 = stableHash(data, seed);
    const hash2 = stableHash(data, seed);
    const hash3 = stableHash(data, 67890);
    
    try std.testing.expectEqual(hash1, hash2);
    try std.testing.expect(hash1 != hash3);
}

test "Path join with leading slash" {
    var gpa = std.testing.allocator;
    const path1 = try pathJoin(gpa, &.{ "/a", "b", "c" });
    defer gpa.free(path1);
    
    const path2 = try pathJoin(gpa, &.{ "a", "b", "c" });
    defer gpa.free(path2);
    
    try std.testing.expect(path1[0] == '/');
}

test "Atomic write" {
    var gpa = std.testing.allocator;
    try atomicWrite(gpa, "test_atomic.txt", "data");
    defer fs.cwd().deleteFile("test_atomic.txt") catch |err| {
        std.debug.print("Warning: failed to delete test file: {}\n", .{err});
    };
    const content = try readFile(gpa, "test_atomic.txt");
    defer gpa.free(content);
    try std.testing.expectEqualStrings("data", content);
}
