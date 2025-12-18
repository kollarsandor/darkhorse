const std = @import("std");
const mem = std.mem;
const math = std.math;
const Allocator = mem.Allocator;
const types = @import("../core/types.zig");
const BitSet = types.BitSet;
const PRNG = types.PRNG;
const Tensor = @import("../core/tensor.zig").Tensor;
const SSI = @import("../index/ssi.zig").SSI;
const stableHash = @import("../core/io.zig").stableHash;
const Error = types.Error;

pub const Ranker = struct {
    ngram_weights: []f32,
    lsh_hash_params: []u64,
    num_hash_functions: usize,
    num_ngrams: usize,
    seed: u64,
    allocator: Allocator,

    pub fn init(allocator: Allocator, num_ngrams: usize, num_hash_funcs: usize, seed: u64) !Ranker {
        const weights = try allocator.alloc(f32, num_ngrams);
        var i: usize = 0;
        while (i < weights.len) : (i += 1) {
            const decay = 1.0 / @as(f32, @floatFromInt(i + 1));
            weights[i] = decay;
        }
        
        const hash_params = try allocator.alloc(u64, num_hash_funcs * 2);
        i = 0;
        while (i < num_hash_funcs) : (i += 1) {
            hash_params[i * 2] = seed +% (i *% 0x9e3779b97f4a7c15);
            hash_params[i * 2 + 1] = seed +% ((i + 1) *% 0x517cc1b727220a95);
        }
        
        return .{
            .ngram_weights = weights,
            .lsh_hash_params = hash_params,
            .num_hash_functions = num_hash_funcs,
            .num_ngrams = num_ngrams,
            .seed = seed,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Ranker) void {
        self.allocator.free(self.ngram_weights);
        self.allocator.free(self.lsh_hash_params);
    }

    pub fn scoreSequence(self: *const Ranker, tokens: []const u32, ssi: *const SSI) !f32 {
        var ngram_score: f32 = 0.0;
        var gram: usize = 1;
        while (gram < @min(self.num_ngrams, tokens.len + 1)) : (gram += 1) {
            if (tokens.len < gram) continue;
            var start: usize = 0;
            while (start < tokens.len - gram + 1) : (start += 1) {
                const ngram = tokens[start .. start + gram];
                const h = stableHash(mem.sliceAsBytes(ngram), self.seed);
                if (ssi.getSegment(h)) |s| {
                    const weight_idx = @min(gram - 1, self.ngram_weights.len - 1);
                    ngram_score += self.ngram_weights[weight_idx] * s.score;
                }
            }
        }
        
        const diversity_score = self.computeTokenDiversity(tokens);
        const proximity = self.anchorProximity(tokens, ssi);
        
        var raw_score = ngram_score + 0.3 * diversity_score + 0.3 * proximity;
        raw_score = math.clamp(raw_score, 0.0, 100.0);
        return raw_score / 100.0;
    }

    pub fn scoreSequenceWithQuery(self: *const Ranker, tokens: []const u32, query: []const u32, ssi: *const SSI) !f32 {
        const base_score = try self.scoreSequence(tokens, ssi);
        
        const token_overlap = self.computeTokenOverlap(tokens, query);
        const jaccard = self.computeJaccardSimilarity(tokens, query);
        
        const combined_score = base_score * 0.4 + token_overlap * 0.3 + jaccard * 0.3;
        return math.clamp(combined_score, 0.0, 1.0);
    }

    fn computeTokenDiversity(self: *const Ranker, tokens: []const u32) f32 {
        if (tokens.len == 0) return 0.0;
        
        var unique_tokens = std.AutoHashMap(u32, void).init(self.allocator);
        defer unique_tokens.deinit();
        
        for (tokens) |token| {
            unique_tokens.put(token, {}) catch continue;
        }
        
        const unique_count = unique_tokens.count();
        const diversity = @as(f32, @floatFromInt(unique_count)) / @as(f32, @floatFromInt(tokens.len));
        
        return diversity;
    }

    fn computeTokenOverlap(self: *const Ranker, tokens: []const u32, query: []const u32) f32 {
        _ = self;
        if (tokens.len == 0 or query.len == 0) return 0.0;
        
        var overlap: usize = 0;
        for (tokens) |token| {
            for (query) |qtoken| {
                if (token == qtoken) {
                    overlap += 1;
                    break;
                }
            }
        }
        
        const max_len = @max(tokens.len, query.len);
        return @as(f32, @floatFromInt(overlap)) / @as(f32, @floatFromInt(max_len));
    }

    fn computeJaccardSimilarity(self: *const Ranker, tokens: []const u32, query: []const u32) f32 {
        if (tokens.len == 0 and query.len == 0) return 1.0;
        if (tokens.len == 0 or query.len == 0) return 0.0;
        
        var intersection: usize = 0;
        var union_size: usize = 0;
        
        var seen = std.AutoHashMap(u32, u8).init(self.allocator);
        defer seen.deinit();
        
        for (tokens) |token| {
            seen.put(token, 1) catch continue;
        }
        
        for (query) |qtoken| {
            if (seen.get(qtoken)) |val| {
                if (val == 1) {
                    seen.put(qtoken, 2) catch continue;
                }
            } else {
                seen.put(qtoken, 1) catch continue;
            }
        }
        
        var it = seen.iterator();
        while (it.next()) |entry| {
            union_size += 1;
            if (entry.value_ptr.* == 2) {
                intersection += 1;
            }
        }
        
        if (union_size == 0) return 0.0;
        return @as(f32, @floatFromInt(intersection)) / @as(f32, @floatFromInt(union_size));
    }

    fn anchorProximity(self: *const Ranker, tokens: []const u32, ssi: *const SSI) f32 {
        _ = self;
        var anchors: usize = 0;
        var total_dist: f32 = 0.0;
        var i: usize = 0;
        while (i < tokens.len) : (i += 1) {
            const pos: u64 = @intCast(i);
            if (ssi.getSegment(pos)) |s| {
                if (s.anchor_hash != 0) {
                    anchors += 1;
                    const i_u64: u64 = @intCast(i);
                    const raw_dist: u64 = if (i_u64 > s.position) i_u64 - s.position else s.position - i_u64;
                    const clamped_dist: u64 = @min(raw_dist, std.math.maxInt(u32));
                    const dist: f32 = @floatFromInt(clamped_dist);
                    total_dist += dist;
                }
            }
        }
        if (anchors == 0) return 0.0;
        const denominator: u64 = math.mul(u64, @as(u64, anchors), @as(u64, tokens.len)) catch std.math.maxInt(u64);
        const denom_clamped: u64 = @min(denominator, std.math.maxInt(u32));
        const denom_f32: f32 = @floatFromInt(@max(denom_clamped, 1));
        return 1.0 - math.clamp(total_dist / denom_f32, 0.0, 1.0);
    }

    pub fn rankCandidates(self: *const Ranker, candidates: []types.RankedSegment, ssi: *const SSI, allocator: Allocator) !void {
        var scores = try allocator.alloc(f32, candidates.len);
        defer allocator.free(scores);
        var i: usize = 0;
        while (i < candidates.len) : (i += 1) {
            scores[i] = try self.scoreSequence(candidates[i].tokens, ssi);
        }
        self.normalizeScores(scores);
        var indices = try allocator.alloc(usize, candidates.len);
        defer allocator.free(indices);
        i = 0;
        while (i < candidates.len) : (i += 1) {
            indices[i] = i;
        }
        const Context = struct {
            scores: []const f32,
            pub fn lessThan(ctx: @This(), a: usize, b: usize) bool {
                return ctx.scores[a] > ctx.scores[b];
            }
        };
        std.mem.sort(usize, indices, Context{ .scores = scores }, Context.lessThan);
        var sorted = try allocator.alloc(types.RankedSegment, candidates.len);
        defer allocator.free(sorted);
        i = 0;
        while (i < candidates.len) : (i += 1) {
            sorted[i] = candidates[indices[i]];
            sorted[i].score = scores[indices[i]];
        }
        i = 0;
        while (i < candidates.len) : (i += 1) {
            candidates[i] = sorted[i];
        }
    }

    pub fn rankCandidatesWithQuery(self: *const Ranker, candidates: []types.RankedSegment, query: []const u32, ssi: *const SSI, allocator: Allocator) !void {
        var scores = try allocator.alloc(f32, candidates.len);
        defer allocator.free(scores);
        var i: usize = 0;
        while (i < candidates.len) : (i += 1) {
            scores[i] = try self.scoreSequenceWithQuery(candidates[i].tokens, query, ssi);
        }
        self.normalizeScores(scores);
        var indices = try allocator.alloc(usize, candidates.len);
        defer allocator.free(indices);
        i = 0;
        while (i < candidates.len) : (i += 1) {
            indices[i] = i;
        }
        const Context = struct {
            scores: []const f32,
            pub fn lessThan(ctx: @This(), a: usize, b: usize) bool {
                return ctx.scores[a] > ctx.scores[b];
            }
        };
        std.mem.sort(usize, indices, Context{ .scores = scores }, Context.lessThan);
        var sorted = try allocator.alloc(types.RankedSegment, candidates.len);
        defer allocator.free(sorted);
        i = 0;
        while (i < candidates.len) : (i += 1) {
            sorted[i] = candidates[indices[i]];
            sorted[i].score = scores[indices[i]];
        }
        i = 0;
        while (i < candidates.len) : (i += 1) {
            candidates[i] = sorted[i];
        }
    }

    pub fn batchScore(self: *const Ranker, sequences: [][]u32, ssi: *const SSI, allocator: Allocator) ![]f32 {
        const batch_size = sequences.len;
        var scores = try allocator.alloc(f32, batch_size);
        var b: usize = 0;
        while (b < batch_size) : (b += 1) {
            scores[b] = try self.scoreSequence(sequences[b], ssi);
        }
        return scores;
    }

    pub fn topKHeap(self: *const Ranker, ssi: *const SSI, query: []u32, k: usize, allocator: Allocator) ![]types.RankedSegment {
        var heap = std.PriorityQueue(types.RankedSegment, void, struct {
            pub fn lessThan(_: void, a: types.RankedSegment, b: types.RankedSegment) std.math.Order {
                return std.math.order(a.score, b.score);
            }
        }.lessThan).init(allocator, {});
        defer heap.deinit();
        const candidates = try ssi.retrieveTopK(query, 1000, allocator);
        defer {
            var i: usize = 0;
            while (i < candidates.len) : (i += 1) {
                candidates[i].deinit(allocator);
            }
            allocator.free(candidates);
        }
        var i: usize = 0;
        while (i < candidates.len) : (i += 1) {
            const cand = candidates[i];
            const score = try self.scoreSequenceWithQuery(cand.tokens, query, ssi);
            const ranked = try types.RankedSegment.init(allocator, cand.tokens, score, cand.position, cand.anchor);
            if (heap.count() < k) {
                try heap.add(ranked);
            } else if (heap.peek()) |top| {
                if (score > top.score) {
                    var removed = heap.remove();
                    removed.deinit(allocator);
                    try heap.add(ranked);
                } else {
                    ranked.deinit(allocator);
                }
            }
        }
        var top_k = try allocator.alloc(types.RankedSegment, @min(k, heap.count()));
        i = heap.count();
        while (heap.removeOrNull()) |item| {
            if (i > 0) {
                i -= 1;
                top_k[i] = item;
            }
        }
        return top_k[0 .. heap.count()];
    }

    pub fn updateWeights(self: *Ranker, gradients: []f32) void {
        var i: usize = 0;
        while (i < @min(self.ngram_weights.len, gradients.len)) : (i += 1) {
            self.ngram_weights[i] += gradients[i] * 0.01;
            self.ngram_weights[i] = math.clamp(self.ngram_weights[i], 0.0, 1.0);
        }
    }

    pub fn minHashSignature(self: *const Ranker, tokens: []const u32) ![]u64 {
        const sig = try self.allocator.alloc(u64, self.num_hash_functions);
        var h: usize = 0;
        while (h < self.num_hash_functions) : (h += 1) {
            var min_hash: u64 = std.math.maxInt(u64);
            const seed_a = self.lsh_hash_params[h * 2];
            const seed_b = self.lsh_hash_params[h * 2 + 1];
            
            for (tokens) |token| {
                const hash_val = stableHash(mem.asBytes(&token), seed_a) ^ seed_b;
                if (hash_val < min_hash) {
                    min_hash = hash_val;
                }
            }
            sig[h] = min_hash;
        }
        return sig;
    }

    pub fn jaccardSimilarityFromSignatures(self: *const Ranker, sig1: []const u64, sig2: []const u64) f32 {
        _ = self;
        if (sig1.len != sig2.len) return 0.0;
        if (sig1.len == 0) return 0.0;
        
        var matches: usize = 0;
        var i: usize = 0;
        while (i < sig1.len) : (i += 1) {
            if (sig1[i] == sig2[i]) {
                matches += 1;
            }
        }
        return @as(f32, @floatFromInt(matches)) / @as(f32, @floatFromInt(sig1.len));
    }

    pub fn estimateJaccard(self: *const Ranker, set1: BitSet, set2: BitSet) f32 {
        _ = self;
        var intersect: usize = 0;
        var union_count: usize = 0;
        const words = @min(set1.bits.len, set2.bits.len);
        var i: usize = 0;
        while (i < words) : (i += 1) {
            intersect += @popCount(set1.bits[i] & set2.bits[i]);
            union_count += @popCount(set1.bits[i] | set2.bits[i]);
        }
        return if (union_count == 0) 0.0 else @as(f32, @floatFromInt(intersect)) / @as(f32, @floatFromInt(union_count));
    }

    pub fn vectorScore(self: *const Ranker, embedding: *const Tensor, query_emb: *const Tensor) !f32 {
        _ = self;
        if (!mem.eql(usize, embedding.shape, query_emb.shape)) return Error.ShapeMismatch;
        var dot_prod: f32 = 0.0;
        var norm_emb: f32 = 0.0;
        var norm_query: f32 = 0.0;
        const emb_data: [*]const f32 = @ptrCast(@alignCast(embedding.data.ptr));
        const query_data: [*]const f32 = @ptrCast(@alignCast(query_emb.data.ptr));
        const len = embedding.data.len / @sizeOf(f32);
        var i: usize = 0;
        while (i < len) : (i += 1) {
            const e = emb_data[i];
            const q = query_data[i];
            dot_prod += e * q;
            norm_emb += e * e;
            norm_query += q * q;
        }
        norm_emb = math.sqrt(norm_emb);
        norm_query = math.sqrt(norm_query);
        if (norm_emb == 0.0 or norm_query == 0.0) return 0.0;
        return dot_prod / (norm_emb * norm_query);
    }

    pub fn dotProductScore(self: *const Ranker, embedding: *const Tensor, query_emb: *const Tensor) !f32 {
        _ = self;
        if (!mem.eql(usize, embedding.shape, query_emb.shape)) return Error.ShapeMismatch;
        var dot_prod: f32 = 0.0;
        const emb_data: [*]const f32 = @ptrCast(@alignCast(embedding.data.ptr));
        const query_data: [*]const f32 = @ptrCast(@alignCast(query_emb.data.ptr));
        const len = embedding.data.len / @sizeOf(f32);
        var i: usize = 0;
        while (i < len) : (i += 1) {
            dot_prod += emb_data[i] * query_data[i];
        }
        return dot_prod;
    }

    pub fn weightedAverage(self: *const Ranker, scores: []f32, weights: []f32) f32 {
        _ = self;
        if (scores.len != weights.len) return 0.0;
        var num: f32 = 0.0;
        var den: f32 = 0.0;
        var i: usize = 0;
        while (i < scores.len) : (i += 1) {
            num += scores[i] * weights[i];
            den += weights[i];
        }
        return if (den == 0.0) 0.0 else num / den;
    }

    pub fn exponentialDecay(self: *const Ranker, scores: []f32, decay: f32) void {
        _ = self;
        var i: usize = 0;
        while (i < scores.len) : (i += 1) {
            scores[i] *= math.pow(f32, decay, @as(f32, @floatFromInt(i)));
        }
    }

    pub fn normalizeScores(self: *const Ranker, scores: []f32) void {
        _ = self;
        var min_score: f32 = math.inf(f32);
        var max_score: f32 = -math.inf(f32);
        var i: usize = 0;
        while (i < scores.len) : (i += 1) {
            if (scores[i] < min_score) min_score = scores[i];
            if (scores[i] > max_score) max_score = scores[i];
        }
        if (max_score == min_score or max_score == -math.inf(f32)) return;
        const range = max_score - min_score;
        i = 0;
        while (i < scores.len) : (i += 1) {
            scores[i] = (scores[i] - min_score) / range;
        }
    }

    pub fn rankByMultipleCriteria(self: *const Ranker, candidates: []types.RankedSegment, criteria: [][]f32, weights: []f32, allocator: Allocator) !void {
        _ = self;
        const num_cand = candidates.len;
        const num_crit = criteria.len;
        var combined = try allocator.alloc(f32, num_cand);
        defer allocator.free(combined);
        var c: usize = 0;
        while (c < num_cand) : (c += 1) {
            var crit_score: f32 = 0.0;
            var cr: usize = 0;
            while (cr < num_crit) : (cr += 1) {
                crit_score += criteria[cr][c] * weights[cr];
            }
            combined[c] = crit_score;
        }
        var indices = try allocator.alloc(usize, num_cand);
        defer allocator.free(indices);
        var i: usize = 0;
        while (i < num_cand) : (i += 1) {
            indices[i] = i;
        }
        const Context = struct {
            scores: []const f32,
            pub fn lessThan(ctx: @This(), a: usize, b: usize) bool {
                return ctx.scores[a] > ctx.scores[b];
            }
        };
        std.mem.sort(usize, indices, Context{ .scores = combined }, Context.lessThan);
        var sorted = try allocator.alloc(types.RankedSegment, num_cand);
        defer allocator.free(sorted);
        i = 0;
        while (i < num_cand) : (i += 1) {
            sorted[i] = candidates[indices[i]];
            sorted[i].score = combined[indices[i]];
        }
        i = 0;
        while (i < num_cand) : (i += 1) {
            candidates[i] = sorted[i];
        }
    }

    pub fn streamingRank(self: *const Ranker, stream: anytype, ssi: *const SSI, k: usize, allocator: Allocator) ![]types.RankedSegment {
        var buffer: [1024]u32 = undefined;
        var buf_len: usize = 0;
        var top_k = try allocator.alloc(types.RankedSegment, k);
        var i: usize = 0;
        while (i < k) : (i += 1) {
            top_k[i] = try types.RankedSegment.init(allocator, &.{}, 0.0, 0, false);
        }
        while (try stream.read(&buffer)) |chunk| {
            const tokens = buffer[0..chunk];
            buf_len += chunk;
            if (buf_len >= 512 and tokens.len >= 512) {
                const score = try self.scoreSequence(tokens[0..512], ssi);
                if (score > top_k[top_k.len - 1].score) {
                    top_k[top_k.len - 1].deinit(allocator);
                    top_k[top_k.len - 1] = try types.RankedSegment.init(allocator, tokens[0..512], score, 0, false);
                    const Context = struct {
                        pub fn lessThan(_: @This(), a: types.RankedSegment, b: types.RankedSegment) bool {
                            return a.score > b.score;
                        }
                    };
                    std.mem.sort(types.RankedSegment, top_k, Context{}, Context.lessThan);
                }
                buf_len = 0;
            }
        }
        return top_k;
    }

    pub fn parallelScore(self: *const Ranker, sequences: [][]u32, ssi: *const SSI, num_threads: usize) ![]f32 {
        _ = num_threads;
        const scores = try self.allocator.alloc(f32, sequences.len);
        var i: usize = 0;
        while (i < sequences.len) : (i += 1) {
            scores[i] = try self.scoreSequence(sequences[i], ssi);
        }
        return scores;
    }

    pub fn calibrateWeights(self: *Ranker, training_data: [][]u32, labels: []f32, ssi: *const SSI, epochs: usize) !void {
        var gradients = try self.allocator.alloc(f32, self.ngram_weights.len);
        defer self.allocator.free(gradients);
        var epoch: usize = 0;
        while (epoch < epochs) : (epoch += 1) {
            @memset(gradients, 0.0);
            var i: usize = 0;
            while (i < training_data.len) : (i += 1) {
                const pred = try self.scoreSequence(training_data[i], ssi);
                const err = pred - labels[i];
                var g: usize = 0;
                while (g < self.ngram_weights.len) : (g += 1) {
                    gradients[g] += err * 0.01;
                }
            }
            self.updateWeights(gradients);
        }
    }

    pub fn exportModel(self: *const Ranker, path: []const u8) !void {
        var file = try std.fs.cwd().createFile(path, .{});
        defer file.close();
        var writer = file.writer();
        try writer.writeInt(u8, 2, .Little);
        try writer.writeInt(usize, self.ngram_weights.len, .Little);
        var i: usize = 0;
        while (i < self.ngram_weights.len) : (i += 1) {
            try writer.writeAll(mem.asBytes(&self.ngram_weights[i]));
        }
        try writer.writeInt(usize, self.num_hash_functions, .Little);
        i = 0;
        while (i < self.lsh_hash_params.len) : (i += 1) {
            try writer.writeInt(u64, self.lsh_hash_params[i], .Little);
        }
        try writer.writeInt(u64, self.seed, .Little);
    }

    pub fn importModel(self: *Ranker, path: []const u8) !void {
        const file = try std.fs.cwd().openFile(path, .{});
        defer file.close();
        var reader = file.reader();
        const version = try reader.readInt(u8, .Little);
        if (version != 2) return error.InvalidVersion;
        const num_w = try reader.readInt(usize, .Little);
        if (self.ngram_weights.len != num_w) {
            self.allocator.free(self.ngram_weights);
            self.ngram_weights = try self.allocator.alloc(f32, num_w);
        }
        var i: usize = 0;
        while (i < self.ngram_weights.len) : (i += 1) {
            var bytes: [@sizeOf(f32)]u8 = undefined;
            _ = try reader.read(&bytes);
            self.ngram_weights[i] = @as(*const f32, @ptrCast(&bytes)).*;
        }
        const num_h = try reader.readInt(usize, .Little);
        self.num_hash_functions = num_h;
        if (self.lsh_hash_params.len != num_h * 2) {
            self.allocator.free(self.lsh_hash_params);
            self.lsh_hash_params = try self.allocator.alloc(u64, num_h * 2);
        }
        i = 0;
        while (i < self.lsh_hash_params.len) : (i += 1) {
            self.lsh_hash_params[i] = try reader.readInt(u64, .Little);
        }
        self.seed = try reader.readInt(u64, .Little);
    }
};

test "Ranker score" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ranker = try Ranker.init(gpa, 4, 8, 42);
    defer ranker.deinit();
    var ssi = SSI.init(gpa);
    defer ssi.deinit();
    try ssi.addSequence(&.{ 1, 2, 3 }, 0, false);
    const score = try ranker.scoreSequence(&.{ 1, 2 }, &ssi);
    try testing.expect(score >= 0.0);
}

test "MinHash signature deterministic" {
    const testing = std.testing;
    var gpa = std.testing.allocator;
    var ranker = try Ranker.init(gpa, 1, 32, 42);
    defer ranker.deinit();
    const sig1 = try ranker.minHashSignature(&.{ 1, 2, 3 });
    defer gpa.free(sig1);
    const sig2 = try ranker.minHashSignature(&.{ 1, 2, 3 });
    defer gpa.free(sig2);
    try testing.expectEqualSlices(u64, sig1, sig2);
}

test "Jaccard similarity from signatures" {
    const testing = std.testing;
    var gpa = std.testing.allocator;
    var ranker = try Ranker.init(gpa, 1, 32, 42);
    defer ranker.deinit();
    const sig1 = try ranker.minHashSignature(&.{ 1, 2, 3 });
    defer gpa.free(sig1);
    const sig2 = try ranker.minHashSignature(&.{ 1, 2, 3 });
    defer gpa.free(sig2);
    const sim = ranker.jaccardSimilarityFromSignatures(sig1, sig2);
    try testing.expectApproxEqAbs(1.0, sim, 0.01);
}

test "Token diversity" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ranker = try Ranker.init(gpa, 1, 1, 42);
    defer ranker.deinit();
    const div1 = ranker.computeTokenDiversity(&.{ 1, 1, 1, 1 });
    const div2 = ranker.computeTokenDiversity(&.{ 1, 2, 3, 4 });
    try testing.expect(div2 > div1);
}

test "Token overlap" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var ranker = try Ranker.init(gpa, 1, 1, 42);
    defer ranker.deinit();
    const overlap = ranker.computeTokenOverlap(&.{ 1, 2, 3 }, &.{ 2, 3, 4 });
    try testing.expect(overlap > 0.0 and overlap <= 1.0);
}

test "Estimate Jaccard" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var set1 = try BitSet.init(gpa, 128);
    defer set1.deinit();
    set1.set(0);
    set1.set(64);
    var set2 = try BitSet.init(gpa, 128);
    defer set2.deinit();
    set2.set(0);
    var ranker = try Ranker.init(gpa, 1, 1, 42);
    defer ranker.deinit();
    const est = ranker.estimateJaccard(set1, set2);
    try testing.expect(est >= 0.0 and est <= 1.0);
}

test "Vector cosine score" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var emb = try Tensor.init(gpa, &.{3});
    defer emb.deinit();
    emb.data[0] = 1.0;
    emb.data[1] = 0.0;
    emb.data[2] = 0.0;
    var qemb = try Tensor.init(gpa, &.{3});
    defer qemb.deinit();
    qemb.data[0] = 1.0;
    qemb.data[1] = 0.0;
    qemb.data[2] = 0.0;
    var ranker = try Ranker.init(gpa, 1, 1, 42);
    defer ranker.deinit();
    const score = try ranker.vectorScore(&emb, &qemb);
    try testing.expectApproxEqAbs(1.0, score, 0.01);
}

test "Dot product score" {
    const testing = std.testing;
    const gpa = std.testing.allocator;
    var emb = try Tensor.init(gpa, &.{3});
    defer emb.deinit();
    emb.data[0] = 1.0;
    emb.data[1] = 2.0;
    emb.data[2] = 3.0;
    var qemb = try Tensor.init(gpa, &.{3});
    defer qemb.deinit();
    qemb.data[0] = 1.0;
    qemb.data[1] = 2.0;
    qemb.data[2] = 3.0;
    var ranker = try Ranker.init(gpa, 1, 1, 42);
    defer ranker.deinit();
    const score = try ranker.dotProductScore(&emb, &qemb);
    try testing.expectApproxEqAbs(14.0, score, 0.01);
}
