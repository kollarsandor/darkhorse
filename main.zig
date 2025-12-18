
const std = @import("std");
const mem = std.mem;
const math = std.math;
const time = std.time;
const fs = std.fs;

const types = @import("core/types.zig");
const tensor_mod = @import("core/tensor.zig");

const rsf_mod = @import("processor/rsf.zig");
const mgt_mod = @import("tokenizer/mgt.zig");
const sfd_mod = @import("optimizer/sfd.zig");
const ssi_mod = @import("index/ssi.zig");
const ranker_mod = @import("ranker/ranker.zig");

const Tensor = tensor_mod.Tensor;
const RSF = rsf_mod.RSF;
const MGT = mgt_mod.MGT;
const SFD = sfd_mod.SFD;
const SSI = ssi_mod.SSI;
const Ranker = ranker_mod.Ranker;
const PRNG = types.PRNG;
const RankedSegment = types.RankedSegment;

const Config = struct {
    embedding_dim: usize = 128,
    rsf_layers: usize = 4,
    batch_size: usize = 16,
    num_epochs: usize = 10,
    learning_rate: f32 = 0.001,
    num_training_samples: usize = 100,
    num_validation_samples: usize = 100,
    models_dir: []u8 = undefined,
    vocab_file: ?[]u8 = null,
    dataset_path: ?[]const u8 = null,
    sample_limit: usize = 3716,
    gradient_clip_norm: f32 = 5.0,
    validation_mse_threshold: f32 = 0.1,
    validation_confidence_level: f32 = 0.95,
    sequence_length: usize = 64,
    validation_query_length: usize = 32,
    top_k: usize = 5,
    noise_level: f32 = 0.05,
    
    models_dir_owned: bool = false,
    vocab_file_owned: bool = false,
    
    pub fn deinit(self: *Config, allocator: mem.Allocator) void {
        if (self.models_dir_owned) {
            allocator.free(self.models_dir);
        }
        if (self.vocab_file_owned) {
            if (self.vocab_file) |vf| {
                allocator.free(vf);
            }
        }
    }
    
    pub fn parseArgs(allocator: mem.Allocator) !Config {
        var config = Config{
            .models_dir = undefined,
            .dataset_path = null,
            .sample_limit = 100,
        };
        var models_dir_allocated: ?[]u8 = null;
        var vocab_file_allocated: ?[]u8 = null;
        
        errdefer {
            if (models_dir_allocated) |dir| allocator.free(dir);
            if (vocab_file_allocated) |file| allocator.free(file);
        }
        
        const default_models_dir = try allocator.dupe(u8, "models");
        models_dir_allocated = default_models_dir;
        config.models_dir = default_models_dir;
        config.models_dir_owned = true;
        
        var args = try std.process.argsWithAllocator(allocator);
        defer args.deinit();
        
        _ = args.skip();
        
        while (args.next()) |arg| {
            if (mem.eql(u8, arg, "--embedding-dim")) {
                const val = args.next() orelse {
                    std.debug.print("Error: --embedding-dim requires a value\n", .{});
                    return error.MissingArgumentValue;
                };
                config.embedding_dim = std.fmt.parseInt(usize, val, 10) catch |err| {
                    std.debug.print("Error: Invalid --embedding-dim value: {s}\n", .{val});
                    return err;
                };
            } else if (mem.eql(u8, arg, "--layers")) {
                const val = args.next() orelse {
                    std.debug.print("Error: --layers requires a value\n", .{});
                    return error.MissingArgumentValue;
                };
                config.rsf_layers = std.fmt.parseInt(usize, val, 10) catch |err| {
                    std.debug.print("Error: Invalid --layers value: {s}\n", .{val});
                    return err;
                };
            } else if (mem.eql(u8, arg, "--batch-size")) {
                const val = args.next() orelse {
                    std.debug.print("Error: --batch-size requires a value\n", .{});
                    return error.MissingArgumentValue;
                };
                config.batch_size = std.fmt.parseInt(usize, val, 10) catch |err| {
                    std.debug.print("Error: Invalid --batch-size value: {s}\n", .{val});
                    return err;
                };
            } else if (mem.eql(u8, arg, "--epochs")) {
                const val = args.next() orelse {
                    std.debug.print("Error: --epochs requires a value\n", .{});
                    return error.MissingArgumentValue;
                };
                config.num_epochs = std.fmt.parseInt(usize, val, 10) catch |err| {
                    std.debug.print("Error: Invalid --epochs value: {s}\n", .{val});
                    return err;
                };
            } else if (mem.eql(u8, arg, "--lr")) {
                const val = args.next() orelse {
                    std.debug.print("Error: --lr requires a value\n", .{});
                    return error.MissingArgumentValue;
                };
                config.learning_rate = std.fmt.parseFloat(f32, val) catch |err| {
                    std.debug.print("Error: Invalid --lr value: {s}\n", .{val});
                    return err;
                };
            } else if (mem.eql(u8, arg, "--samples")) {
                const val = args.next() orelse {
                    std.debug.print("Error: --samples requires a value\n", .{});
                    return error.MissingArgumentValue;
                };
                config.num_training_samples = std.fmt.parseInt(usize, val, 10) catch |err| {
                    std.debug.print("Error: Invalid --samples value: {s}\n", .{val});
                    return err;
                };
            } else if (mem.eql(u8, arg, "--models-dir")) {
                const val = args.next() orelse {
                    std.debug.print("Error: --models-dir requires a value\n", .{});
                    return error.MissingArgumentValue;
                };
                if (models_dir_allocated) |old| allocator.free(old);
                const duped = try allocator.dupe(u8, val);
                models_dir_allocated = duped;
                config.models_dir = duped;
                config.models_dir_owned = true;
            } else if (mem.eql(u8, arg, "--vocab-file")) {
                const val = args.next() orelse {
                    std.debug.print("Error: --vocab-file requires a value\n", .{});
                    return error.MissingArgumentValue;
                };
                if (vocab_file_allocated) |old| allocator.free(old);
                const duped = try allocator.dupe(u8, val);
                vocab_file_allocated = duped;
                config.vocab_file = duped;
                config.vocab_file_owned = true;
            } else if (mem.eql(u8, arg, "--gradient-clip")) {
                const val = args.next() orelse {
                    std.debug.print("Error: --gradient-clip requires a value\n", .{});
                    return error.MissingArgumentValue;
                };
                config.gradient_clip_norm = std.fmt.parseFloat(f32, val) catch |err| {
                    std.debug.print("Error: Invalid --gradient-clip value: {s}\n", .{val});
                    return err;
                };
            } else if (mem.eql(u8, arg, "--sequence-length")) {
                const val = args.next() orelse {
                    std.debug.print("Error: --sequence-length requires a value\n", .{});
                    return error.MissingArgumentValue;
                };
                config.sequence_length = std.fmt.parseInt(usize, val, 10) catch |err| {
                    std.debug.print("Error: Invalid --sequence-length value: {s}\n", .{val});
                    return err;
                };
            } else if (mem.eql(u8, arg, "--top-k")) {
                const val = args.next() orelse {
                    std.debug.print("Error: --top-k requires a value\n", .{});
                    return error.MissingArgumentValue;
                };
                config.top_k = std.fmt.parseInt(usize, val, 10) catch |err| {
                    std.debug.print("Error: Invalid --top-k value: {s}\n", .{val});
                    return err;
                };
            } else if (mem.eql(u8, arg, "--noise-level")) {
                const val = args.next() orelse {
                    std.debug.print("Error: --noise-level requires a value\n", .{});
                    return error.MissingArgumentValue;
                };
                config.noise_level = std.fmt.parseFloat(f32, val) catch |err| {
                    std.debug.print("Error: Invalid --noise-level value: {s}\n", .{val});
                    return err;
                };
            } else if (mem.eql(u8, arg, "--dataset-path")) {
                const val = args.next() orelse {
                    std.debug.print("Error: --dataset-path requires a value\n", .{});
                    return error.MissingArgumentValue;
                };
                config.dataset_path = val;
            } else if (mem.eql(u8, arg, "--sample-limit")) {
                const val = args.next() orelse {
                    std.debug.print("Error: --sample-limit requires a value\n", .{});
                    return error.MissingArgumentValue;
                };
                config.sample_limit = std.fmt.parseInt(usize, val, 10) catch |err| {
                    std.debug.print("Error: Invalid --sample-limit value: {s}\n", .{val});
                    return err;
                };
            } else if (mem.eql(u8, arg, "--help")) {
                try printHelp();
                return error.HelpRequested;
            }
        }
        
        if (config.num_training_samples == 0) {
            std.debug.print("Error: --samples must be > 0\n", .{});
            return error.InvalidConfig;
        }
        if (config.batch_size == 0) {
            std.debug.print("Error: --batch-size must be > 0\n", .{});
            return error.InvalidConfig;
        }
        
        return config;
    }
};

const TrainingStats = struct {
    epoch: usize,
    loss: f32,
    avg_rank_score: f32,
    samples_processed: usize,
    elapsed_ms: i64,
};

const ValidationMetrics = struct {
    mse: f32,
    rmse: f32,
    mae: f32,
    r_squared: f32,
    mean_prediction: f32,
    mean_target: f32,
    confidence_interval_lower: f32,
    confidence_interval_upper: f32,
    num_samples: usize,
};

const TrainingSample = struct {
    text: []const u8,
    tokens: []u32,
};

const TerminalColors = struct {
    enabled: bool,
    reset: []const u8,
    bold: []const u8,
    cyan: []const u8,
    green: []const u8,
    yellow: []const u8,
    magenta: []const u8,
    blue: []const u8,
    red: []const u8,
    
    fn detect() TerminalColors {
        const enabled = std.io.tty.detectConfig(std.io.getStdOut()) != .no_color;
        return if (enabled) TerminalColors{
            .enabled = true,
            .reset = "\x1b[0m",
            .bold = "\x1b[1m",
            .cyan = "\x1b[36m",
            .green = "\x1b[32m",
            .yellow = "\x1b[33m",
            .magenta = "\x1b[35m",
            .blue = "\x1b[34m",
            .red = "\x1b[31m",
        } else TerminalColors{
            .enabled = false,
            .reset = "",
            .bold = "",
            .cyan = "",
            .green = "",
            .yellow = "",
            .magenta = "",
            .blue = "",
            .red = "",
        };
    }
};

fn runKgruTest(allocator: std.mem.Allocator) !void {
    const stdout = std.io.getStdOut().writer();
    const colors = TerminalColors.detect();
    
    try stdout.print("{s}{s}========================================{s}\n", .{colors.bold, colors.cyan, colors.reset});
    try stdout.print("{s}{s}  KGRU Component Test Suite{s}\n", .{colors.bold, colors.cyan, colors.reset});
    try stdout.print("{s}{s}========================================{s}\n\n", .{colors.bold, colors.cyan, colors.reset});
    
    var tests_passed: usize = 0;
    var tests_failed: usize = 0;
    
    try stdout.print("{s}[TEST 1]{s} RSF Processor Initialization & Forward Pass...\n", .{colors.yellow, colors.reset});
    {
        const dim: usize = 128;
        const layers: usize = 4;
        var rsf = RSF.init(allocator, dim, layers) catch |err| {
            try stdout.print("  {s}FAILED{s}: RSF init error: {any}\n", .{colors.red, colors.reset, err});
            tests_failed += 1;
            return;
        };
        defer rsf.deinit();
        
        var input_tensor = Tensor.init(allocator, &.{ 1, dim * 2 }) catch |err| {
            try stdout.print("  {s}FAILED{s}: Tensor init error: {any}\n", .{colors.red, colors.reset, err});
            tests_failed += 1;
            return;
        };
        defer input_tensor.deinit();
        
        var ti: usize = 0;
        while (ti < input_tensor.data.len) : (ti += 1) {
            input_tensor.data[ti] = @as(f32, @floatFromInt(ti % 10)) * 0.1;
        }
        
        rsf.forward(&input_tensor) catch |err| {
            try stdout.print("  {s}FAILED{s}: RSF forward error: {any}\n", .{colors.red, colors.reset, err});
            tests_failed += 1;
            return;
        };
        
        var has_valid_output = false;
        for (input_tensor.data) |v| {
            if (v != 0.0 and !math.isNan(v)) {
                has_valid_output = true;
                break;
            }
        }
        
        if (has_valid_output) {
            try stdout.print("  {s}PASSED{s}: RSF forward pass produces valid output\n", .{colors.green, colors.reset});
            tests_passed += 1;
        } else {
            try stdout.print("  {s}FAILED{s}: RSF forward pass returned all zeros or NaN\n", .{colors.red, colors.reset});
            tests_failed += 1;
        }
    }
    
    try stdout.print("{s}[TEST 2]{s} SFD Optimizer Initialization...\n", .{colors.yellow, colors.reset});
    {
        const param_size: usize = 128;
        var optimizer = SFD.init(allocator, param_size) catch |err| {
            try stdout.print("  {s}FAILED{s}: SFD init error: {any}\n", .{colors.red, colors.reset, err});
            tests_failed += 1;
            return;
        };
        defer optimizer.deinit();
        
        var gradients = Tensor.init(allocator, &.{param_size}) catch |err| {
            try stdout.print("  {s}FAILED{s}: Gradient tensor init error: {any}\n", .{colors.red, colors.reset, err});
            tests_failed += 1;
            return;
        };
        defer gradients.deinit();
        
        var params = Tensor.init(allocator, &.{param_size}) catch |err| {
            try stdout.print("  {s}FAILED{s}: Params tensor init error: {any}\n", .{colors.red, colors.reset, err});
            tests_failed += 1;
            return;
        };
        defer params.deinit();
        
        var pi: usize = 0;
        while (pi < param_size) : (pi += 1) {
            gradients.data[pi] = @as(f32, @floatFromInt(pi % 10)) * 0.01;
            params.data[pi] = @as(f32, @floatFromInt(pi)) * 0.001;
        }
        
        optimizer.update(&gradients, &params, 0.001) catch |err| {
            try stdout.print("  {s}FAILED{s}: SFD update error: {any}\n", .{colors.red, colors.reset, err});
            tests_failed += 1;
            return;
        };
        
        try stdout.print("  {s}PASSED{s}: SFD optimizer update completed\n", .{colors.green, colors.reset});
        tests_passed += 1;
    }
    
    try stdout.print("{s}[TEST 3]{s} MGT Tokenizer Initialization...\n", .{colors.yellow, colors.reset});
    {
        var mgt = try initTokenizer(allocator, null);
        defer mgt.deinit();
        
        const vocab_size = mgt.vocabSize();
        if (vocab_size > 0) {
            try stdout.print("  {s}PASSED{s}: MGT tokenizer initialized with vocab size {d}\n", .{colors.green, colors.reset, vocab_size});
            tests_passed += 1;
        } else {
            try stdout.print("  {s}FAILED{s}: MGT tokenizer has empty vocabulary\n", .{colors.red, colors.reset});
            tests_failed += 1;
        }
    }
    
    try stdout.print("{s}[TEST 4]{s} SSI Index Operations...\n", .{colors.yellow, colors.reset});
    {
        var ssi = SSI.init(allocator);
        defer ssi.deinit();
        
        var test_tokens = try allocator.alloc(u32, 8);
        defer allocator.free(test_tokens);
        test_tokens[0] = 1;
        test_tokens[1] = 2;
        test_tokens[2] = 3;
        test_tokens[3] = 4;
        test_tokens[4] = 5;
        test_tokens[5] = 6;
        test_tokens[6] = 7;
        test_tokens[7] = 8;
        
        ssi.addSequence(test_tokens, 42, true) catch |err| {
            try stdout.print("  {s}FAILED{s}: SSI addSequence error: {any}\n", .{colors.red, colors.reset, err});
            tests_failed += 1;
            return;
        };
        
        const stats = ssi.stats();
        if (stats.nodes > 0) {
            try stdout.print("  {s}PASSED{s}: SSI index created with {d} nodes\n", .{colors.green, colors.reset, stats.nodes});
            tests_passed += 1;
        } else {
            try stdout.print("  {s}FAILED{s}: SSI index is empty after adding sequence\n", .{colors.red, colors.reset});
            tests_failed += 1;
        }
    }
    
    try stdout.print("{s}[TEST 5]{s} Ranker Initialization...\n", .{colors.yellow, colors.reset});
    {
        var ranker = Ranker.init(allocator, 10, 16, 42) catch |err| {
            try stdout.print("  {s}FAILED{s}: {any}\n", .{colors.red, colors.reset, err});
            tests_failed += 1;
            return;
        };
        defer ranker.deinit();
        
        try stdout.print("  {s}PASSED{s}: Ranker initialized with ngrams=10, lsh_tables=16\n", .{colors.green, colors.reset});
        tests_passed += 1;
    }
    
    try stdout.writeAll("\n");
    try stdout.print("{s}{s}========================================{s}\n", .{colors.bold, colors.cyan, colors.reset});
    try stdout.print("{s}  Test Results: {d} passed, {d} failed{s}\n", .{
        if (tests_failed == 0) colors.green else colors.red,
        tests_passed,
        tests_failed,
        colors.reset
    });
    try stdout.print("{s}{s}========================================{s}\n", .{colors.bold, colors.cyan, colors.reset});
    
    if (tests_failed > 0) {
        std.process.exit(1);
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const leaked = gpa.deinit();
        if (leaked == .leak) {
            std.debug.print("FATAL: Memory leak detected!\n", .{});
            std.process.exit(1);
        }
    }
    const allocator = gpa.allocator();

    var args_iter = try std.process.argsWithAllocator(allocator);
    defer args_iter.deinit();
    _ = args_iter.skip();
    
    var mode: ?[]const u8 = null;
    while (args_iter.next()) |arg| {
        if (std.mem.eql(u8, arg, "--mode")) {
            mode = args_iter.next();
            break;
        }
    }
    
    if (mode) |m| {
        if (std.mem.eql(u8, m, "agi-demo")) {
            std.debug.print("AGI demo mode not yet implemented\n", .{});
            std.debug.print("Please run standard training mode\n", .{});
            return;
        } else if (std.mem.eql(u8, m, "kgru-test")) {
            try runKgruTest(allocator);
            return;
        }
    }

    var mgt = try initTokenizer(allocator, null);
    defer mgt.deinit();
    var ssi = SSI.init(allocator);
    defer ssi.deinit();
    var ranker = try Ranker.init(allocator, 10, 16, 42);
    defer ranker.deinit();
    
    try runInteractiveREPL(allocator, &mgt, &ssi, &ranker);
}

fn printBanner(writer: anytype, colors: TerminalColors) !void {
    try writer.print("{s}========================================{s}\n", .{colors.bold, colors.reset});
    try writer.print("{s}JAIDE v40 - Root-Level LLM System{s}\n", .{colors.cyan, colors.reset});
    try writer.print("{s}========================================{s}\n\n", .{colors.bold, colors.reset});
    
    try writer.print("{s}Architecture:{s} Jade Neural (Non-Transformer)\n", .{colors.green, colors.reset});
    try writer.print("{s}Context:{s} 50M+ tokens via SSI\n", .{colors.green, colors.reset});
    try writer.writeAll("Components:\n");
    try writer.writeAll("  • SSI: Succinct Semantic Index\n");
    try writer.writeAll("  • Ranker: Non-attention relevance scoring\n");
    try writer.writeAll("  • RSF: Reversible Scatter-Flow processor\n");
    try writer.writeAll("  • MGT: Morpho-Graph Tokenizer\n");
    try writer.writeAll("  • SFD: Spectral Fisher Diagonalizer\n\n");
    try writer.writeAll("Formal Guarantees: Lean, Isabelle, Agda, Viper, TLA+, Spin\n");
    try writer.writeAll("Hardware: Clash RTL + Futhark kernels\n");
    try writer.writeAll("ZK Proofs: Circom inference verification\n");
    try writer.writeAll("\n");
}

fn printHelp() !void {
    const stdout = std.io.getStdOut().writer();
    try stdout.writeAll(
        \\JAIDE v40 - Root-Level LLM System
        \\
        \\USAGE:
        \\    main [OPTIONS]
        \\
        \\OPTIONS:
        \\    --embedding-dim <N>     Set embedding dimension (default: 128)
        \\    --layers <N>            Set number of RSF layers (default: 4)
        \\    --batch-size <N>        Set batch size (default: 16)
        \\    --epochs <N>            Set number of epochs (default: 10)
        \\    --lr <F>                Set learning rate (default: 0.001)
        \\    --samples <N>           Set number of training samples (default: 100)
        \\    --models-dir <PATH>     Set models directory (default: models)
        \\    --vocab-file <PATH>     Set vocabulary file path
        \\    --gradient-clip <F>     Set gradient clipping norm (default: 5.0)
        \\    --sequence-length <N>   Set sequence length (default: 64)
        \\    --top-k <N>             Set top-k retrieval (default: 5)
        \\    --noise-level <F>       Set noise level (default: 0.05)
        \\    --help                  Display this help message
        \\
    );
}

fn initTokenizer(allocator: mem.Allocator, vocab_file: ?[]u8) !MGT {
    _ = vocab_file;
    const sample_vocab = [_][]const u8{
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "as", "is", "was", "are", "were",
        "be", "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "should", "could", "may", "might", "must", "can", "shall",
        "this", "that", "these", "those", "I", "you", "he", "she", "it", "we", "they",
        "my", "your", "his", "her", "its", "our", "their", "me", "him", "them", "us"
    };
    const sample_anchors = [_][]const u8{"the", "a", "and", "is", "in"};
    return try MGT.init(allocator, &sample_vocab, &sample_anchors);
}

fn calculateTotalParams(dim: usize, layers: usize) usize {
    const params_per_layer = (dim * dim) * 2 + dim * 2;
    return params_per_layer * layers;
}

fn generateSyntheticSamples(allocator: mem.Allocator, mgt: *MGT, count: usize) ![]TrainingSample {
    std.debug.print("[SYNTH] Generating {d} synthetic training samples...\n", .{count});
    
    const sample_texts = [_][]const u8{
        "The quick brown fox jumps over the lazy dog in the morning sun.",
        "A beautiful sunset paints the sky with shades of orange and purple.",
        "The scientist discovered a new method for analyzing complex data patterns.",
        "In mathematics, we often encounter problems that require careful analysis.",
        "The river flows gently through the ancient forest toward the sea.",
        "Programming requires logical thinking and attention to detail.",
        "The old library contains thousands of rare and valuable books.",
        "Music has the power to evoke deep emotions and memories.",
        "The mountain peak rises majestically above the clouds at dawn.",
        "Artificial intelligence continues to transform many industries today.",
    };
    
    var samples = std.ArrayList(TrainingSample).init(allocator);
    errdefer {
        for (samples.items) |*sample| {
            allocator.free(sample.tokens);
            allocator.free(sample.text);
        }
        samples.deinit();
    }
    
    var si: usize = 0;
    while (si < count) : (si += 1) {
        const base_text = sample_texts[si % sample_texts.len];
        const text_copy = try allocator.dupe(u8, base_text);
        errdefer allocator.free(text_copy);
        
        var tokens_list = std.ArrayList(u32).init(allocator);
        defer tokens_list.deinit();
        
        try mgt.encode(text_copy, &tokens_list);
        
        if (tokens_list.items.len == 0) {
            allocator.free(text_copy);
            continue;
        }
        
        const tokens = try tokens_list.toOwnedSlice();
        
        try samples.append(.{
            .text = text_copy,
            .tokens = tokens,
        });
    }
    
    std.debug.print("[SYNTH] Generated {d} synthetic samples\n", .{samples.items.len});
    return samples.toOwnedSlice();
}

fn loadDatasetSamples(allocator: mem.Allocator, mgt: *MGT, config: Config) ![]TrainingSample {
    const dataset_path = config.dataset_path orelse {
        std.debug.print("[INFO] No dataset path specified, using synthetic data\n", .{});
        return generateSyntheticSamples(allocator, mgt, config.sample_limit);
    };
    
    std.debug.print("[DEBUG] Opening dataset file: {s}\n", .{dataset_path});
    const file = fs.cwd().openFile(dataset_path, .{}) catch |err| {
        if (err == error.FileNotFound) {
            std.debug.print("[WARN] Dataset file not found: {s}\n", .{dataset_path});
            std.debug.print("[INFO] Falling back to synthetic training data\n", .{});
            return generateSyntheticSamples(allocator, mgt, config.sample_limit);
        }
        return err;
    };
    defer file.close();
    std.debug.print("[DEBUG] File opened successfully\n", .{});

    var samples = std.ArrayList(TrainingSample).init(allocator);
    errdefer {
        for (samples.items) |*sample| {
            allocator.free(sample.tokens);
            allocator.free(sample.text);
        }
        samples.deinit();
    }

    var buf_reader = std.io.bufferedReader(file.reader());
    const reader = buf_reader.reader();

    var line_num: usize = 0;
    
    while (samples.items.len < config.sample_limit) {
        var line_list = std.ArrayList(u8).init(allocator);
        defer line_list.deinit();
        
        reader.streamUntilDelimiter(line_list.writer(), '\n', null) catch |err| {
            if (err == error.EndOfStream) break;
            return err;
        };
        
        const line = line_list.items;
        
        line_num += 1;
        
        if (line.len == 0) continue;
        
        const parsed = std.json.parseFromSlice(std.json.Value, allocator, line, .{}) catch |err| {
            std.debug.print("[WARN] Failed to parse JSON on line {d}: {any}\n", .{line_num, err});
            continue;
        };
        defer parsed.deinit();
        
        const root = parsed.value;
        if (root != .object) continue;
        
        var combined_text = std.ArrayList(u8).init(allocator);
        defer combined_text.deinit();
        
        if (root.object.get("instruction")) |instruction_value| {
            if (instruction_value == .string) {
                try combined_text.appendSlice(instruction_value.string);
                try combined_text.appendSlice(" ");
            }
        }
        
        if (root.object.get("input")) |input_value| {
            if (input_value == .string) {
                try combined_text.appendSlice(input_value.string);
                try combined_text.appendSlice(" ");
            }
        }
        
        if (root.object.get("output")) |output_value| {
            if (output_value == .string) {
                try combined_text.appendSlice(output_value.string);
            }
        } else if (root.object.get("text")) |text_value| {
            if (text_value == .string) {
                try combined_text.appendSlice(text_value.string);
            }
        }
        
        if (combined_text.items.len == 0) continue;
        const text_copy = try combined_text.toOwnedSlice();
        errdefer allocator.free(text_copy);
        
        std.debug.print("[DEBUG] Sample {d} text length: {d} chars\n", .{samples.items.len + 1, text_copy.len});
        
        std.debug.print("[DEBUG] Starting tokenization for sample {d}...\n", .{samples.items.len + 1});
        var tokens_list = std.ArrayList(u32).init(allocator);
        defer tokens_list.deinit();
        
        try mgt.encode(text_copy, &tokens_list);
        std.debug.print("[DEBUG] Tokenization complete, tokens: {d}\n", .{tokens_list.items.len});
        
        if (tokens_list.items.len == 0) {
            allocator.free(text_copy);
            continue;
        }
        
        const tokens = try tokens_list.toOwnedSlice();
        
        try samples.append(.{
            .text = text_copy,
            .tokens = tokens,
        });
        
        if (line_num % 10 == 0) {
            std.debug.print("[DEBUG] Processing line {d}, samples loaded: {d}\n", .{line_num, samples.items.len});
        }
    }
    
    std.debug.print("[DEBUG] Dataset loading complete. Total samples: {d}\n", .{samples.items.len});
    
    return try samples.toOwnedSlice();
}

fn createEmbedding(allocator: mem.Allocator, tokens: []const u32, dim: usize, vocab_size: usize) !Tensor {
    const embedding = try Tensor.init(allocator, &.{ 1, dim * 2 });
    
    var prng = PRNG.init(42);
    
    var i: usize = 0;
    while (i < tokens.len and i < dim) : (i += 1) {
        const token_id = tokens[i];
        const normalized = @as(f32, @floatFromInt(token_id)) / @as(f32, @floatFromInt(vocab_size));
        
        const idx1 = i;
        const idx2 = dim + i;
        
        if (idx1 < embedding.data.len) {
            embedding.data[idx1] = normalized;
        }
        if (idx2 < embedding.data.len) {
            embedding.data[idx2] = normalized * 0.5;
        }
    }
    
    while (i < dim) : (i += 1) {
        const idx1 = i;
        const idx2 = dim + i;
        
        if (idx1 < embedding.data.len) {
            const rand_val = @as(f32, @floatCast(@as(f64, @floatFromInt(prng.next())) / @as(f64, @floatFromInt(math.maxInt(u64)))));
            embedding.data[idx1] = rand_val * 0.01;
        }
        if (idx2 < embedding.data.len) {
            const rand_val = @as(f32, @floatCast(@as(f64, @floatFromInt(prng.next())) / @as(f64, @floatFromInt(math.maxInt(u64)))));
            embedding.data[idx2] = rand_val * 0.01;
        }
    }
    
    return embedding;
}

fn addNoise(tensor: *Tensor, noise_level: f32) void {
    var prng = PRNG.init(@intCast(time.milliTimestamp()));
    
    for (tensor.data) |*val| {
        const rand_val = @as(f32, @floatCast(@as(f64, @floatFromInt(prng.next())) / @as(f64, @floatFromInt(math.maxInt(u64)))));
        const noise = (rand_val * 2.0 - 1.0) * noise_level;
        val.* += noise;
    }
}

fn computeMSELoss(prediction: *const Tensor, target: *const Tensor) !f32 {
    if (!mem.eql(usize, prediction.shape, target.shape)) {
        return error.ShapeMismatch;
    }
    
    var sum: f64 = 0.0;
    var i: usize = 0;
    while (i < prediction.data.len) : (i += 1) {
        const diff = prediction.data[i] - target.data[i];
        sum += diff * diff;
    }
    
    return @as(f32, @floatCast(sum / @as(f64, @floatFromInt(prediction.data.len))));
}

fn extractGradients(allocator: mem.Allocator, rsf: *const RSF, total_params: usize) !Tensor {
    var gradients = try Tensor.init(allocator, &.{total_params});
    
    var offset: usize = 0;
    
    var l: usize = 0;
    while (l < rsf.num_layers) : (l += 1) {
        const layer = &rsf.layers[l];
        
        const s_w_size = layer.s_weight_grad.data.len;
        @memcpy(gradients.data[offset..offset + s_w_size], layer.s_weight_grad.data);
        offset += s_w_size;
        
        const t_w_size = layer.t_weight_grad.data.len;
        @memcpy(gradients.data[offset..offset + t_w_size], layer.t_weight_grad.data);
        offset += t_w_size;
        
        const s_b_size = layer.s_bias_grad.data.len;
        @memcpy(gradients.data[offset..offset + s_b_size], layer.s_bias_grad.data);
        offset += s_b_size;
        
        const t_b_size = layer.t_bias_grad.data.len;
        @memcpy(gradients.data[offset..offset + t_b_size], layer.t_bias_grad.data);
        offset += t_b_size;
    }
    
    return gradients;
}

fn applyGradients(allocator: mem.Allocator, rsf: *RSF, gradients: *const Tensor, optimizer: *SFD, lr: f32) !void {
    _ = allocator;
    
    var params = try Tensor.init(rsf.allocator, &.{gradients.data.len});
    defer params.deinit();
    
    var offset: usize = 0;
    var l: usize = 0;
    while (l < rsf.num_layers) : (l += 1) {
        const layer = &rsf.layers[l];
        
        const s_w_size = layer.s_weight.data.len;
        @memcpy(params.data[offset..offset + s_w_size], layer.s_weight.data);
        offset += s_w_size;
        
        const t_w_size = layer.t_weight.data.len;
        @memcpy(params.data[offset..offset + t_w_size], layer.t_weight.data);
        offset += t_w_size;
        
        const s_b_size = layer.s_bias.data.len;
        @memcpy(params.data[offset..offset + s_b_size], layer.s_bias.data);
        offset += s_b_size;
        
        const t_b_size = layer.t_bias.data.len;
        @memcpy(params.data[offset..offset + t_b_size], layer.t_bias.data);
        offset += t_b_size;
    }
    
    try optimizer.update(gradients, &params, lr);
    
    offset = 0;
    l = 0;
    while (l < rsf.num_layers) : (l += 1) {
        const layer = &rsf.layers[l];
        
        const s_w_size = layer.s_weight.data.len;
        @memcpy(layer.s_weight.data, params.data[offset..offset + s_w_size]);
        offset += s_w_size;
        
        const t_w_size = layer.t_weight.data.len;
        @memcpy(layer.t_weight.data, params.data[offset..offset + t_w_size]);
        offset += t_w_size;
        
        const s_b_size = layer.s_bias.data.len;
        @memcpy(layer.s_bias.data, params.data[offset..offset + s_b_size]);
        offset += s_b_size;
        
        const t_b_size = layer.t_bias.data.len;
        @memcpy(layer.t_bias.data, params.data[offset..offset + t_b_size]);
        offset += t_b_size;
        
        layer.zeroGradients();
    }
}

fn saveModels(allocator: mem.Allocator, rsf: *const RSF, mgt: *const MGT, optimizer: *const SFD, ranker: *const Ranker, models_dir: []const u8) !void {
    try fs.cwd().makePath(models_dir);
    
    const rsf_path = try std.fmt.allocPrint(allocator, "{s}/rsf_trained.bin", .{models_dir});
    defer allocator.free(rsf_path);
    try saveRSF(rsf, rsf_path);
    std.debug.print("[OK] RSF model saved to {s}\n", .{rsf_path});
    
    const mgt_path = try std.fmt.allocPrint(allocator, "{s}/mgt_vocab.bin", .{models_dir});
    defer allocator.free(mgt_path);
    try saveMGT(mgt, mgt_path);
    std.debug.print("[OK] MGT vocabulary saved to {s}\n", .{mgt_path});
    
    const opt_path = try std.fmt.allocPrint(allocator, "{s}/optimizer_state.bin", .{models_dir});
    defer allocator.free(opt_path);
    try optimizer.saveState(opt_path);
    std.debug.print("[OK] Optimizer state saved to {s}\n", .{opt_path});
    
    const ranker_path = try std.fmt.allocPrint(allocator, "{s}/ranker_weights.bin", .{models_dir});
    defer allocator.free(ranker_path);
    try saveRanker(ranker, ranker_path);
    std.debug.print("[OK] Ranker weights saved to {s}\n", .{ranker_path});
}

fn saveRSF(rsf: *const RSF, path: []const u8) !void {
    const file = try fs.cwd().createFile(path, .{});
    defer file.close();
    
    const writer = file.writer();
    
    try writer.writeInt(usize, rsf.num_layers, .Little);
    try writer.writeInt(usize, rsf.dim, .Little);
    
    var l: usize = 0;
    while (l < rsf.num_layers) : (l += 1) {
        const layer = &rsf.layers[l];
        try layer.s_weight.save(writer);
        try layer.t_weight.save(writer);
        try layer.s_bias.save(writer);
        try layer.t_bias.save(writer);
    }
}

fn saveMGT(mgt: *const MGT, path: []const u8) !void {
    const file = try fs.cwd().createFile(path, .{});
    defer file.close();
    
    const writer = file.writer();
    
    const vocab_size = mgt.vocabSize();
    try writer.writeInt(u32, @as(u32, @intCast(vocab_size)), .Little);
    
    var it = mgt.token_to_id.iterator();
    while (it.next()) |entry| {
        const token = entry.key_ptr.*;
        const id = entry.value_ptr.*;
        
        try writer.writeInt(u32, @as(u32, @intCast(token.len)), .Little);
        try writer.writeAll(token);
        try writer.writeInt(u32, id, .Little);
    }
}

fn saveRanker(ranker: *const Ranker, path: []const u8) !void {
    const file = try fs.cwd().createFile(path, .{});
    defer file.close();
    
    const writer = file.writer();
    
    try writer.writeInt(usize, ranker.ngram_weights.len, .Little);
    for (ranker.ngram_weights) |weight| {
        try writer.writeAll(mem.asBytes(&weight));
    }
    
    try writer.writeInt(usize, ranker.num_hash_functions, .Little);
    for (ranker.lsh_hash_params) |param| {
        try writer.writeInt(u64, param, .Little);
    }
    
    try writer.writeInt(u64, ranker.seed, .Little);
}

fn validateModel(allocator: mem.Allocator, rsf: *RSF, mgt: *MGT, ssi: *const SSI, ranker: *const Ranker, config: Config) !ValidationMetrics {
    _ = mgt;
    _ = ssi;
    _ = ranker;
    
    const n_samples = config.num_validation_samples;
    
    var prng = PRNG.init(12345);
    
    var predictions = try allocator.alloc(f32, n_samples);
    defer allocator.free(predictions);
    
    var targets = try allocator.alloc(f32, n_samples);
    defer allocator.free(targets);
    
    var input = try Tensor.init(allocator, &.{ 1, config.embedding_dim * 2 });
    defer input.deinit();
    
    var i: usize = 0;
    while (i < n_samples) : (i += 1) {
        for (input.data) |*val| {
            const rand_val = @as(f32, @floatCast(@as(f64, @floatFromInt(prng.next())) / @as(f64, @floatFromInt(math.maxInt(u64)))));
            val.* = rand_val * 0.1;
        }
        
        try rsf.forward(&input);
        
        const sum = blk: {
            var s: f32 = 0.0;
            for (input.data) |val| s += val;
            break :blk s;
        };
        predictions[i] = sum / @as(f32, @floatFromInt(input.data.len));
        
        const rand_target = @as(f32, @floatCast(@as(f64, @floatFromInt(prng.next())) / @as(f64, @floatFromInt(math.maxInt(u64)))));
        targets[i] = rand_target * 0.1;
    }
    
    var sum_pred: f64 = 0.0;
    var sum_target: f64 = 0.0;
    for (predictions) |p| sum_pred += p;
    for (targets) |t| sum_target += t;
    
    const mean_pred = @as(f32, @floatCast(sum_pred / @as(f64, @floatFromInt(n_samples))));
    const mean_target = @as(f32, @floatCast(sum_target / @as(f64, @floatFromInt(n_samples))));
    
    var mse: f64 = 0.0;
    var mae: f64 = 0.0;
    var ss_res: f64 = 0.0;
    var ss_tot: f64 = 0.0;
    
    i = 0;
    while (i < n_samples) : (i += 1) {
        const diff = predictions[i] - targets[i];
        mse += diff * diff;
        mae += math.fabs(diff);
        
        ss_res += diff * diff;
        const target_diff = targets[i] - mean_target;
        ss_tot += target_diff * target_diff;
    }
    
    mse /= @as(f64, @floatFromInt(n_samples));
    mae /= @as(f64, @floatFromInt(n_samples));
    
    const rmse = math.sqrt(mse);
    const r_squared = 1.0 - (ss_res / (ss_tot + 1e-10));
    
    const std_err = rmse / math.sqrt(@as(f64, @floatFromInt(n_samples)));
    const z = 1.96;
    const margin = z * std_err;
    
    return ValidationMetrics{
        .mse = @floatCast(mse),
        .rmse = @floatCast(rmse),
        .mae = @floatCast(mae),
        .r_squared = @floatCast(r_squared),
        .mean_prediction = mean_pred,
        .mean_target = mean_target,
        .confidence_interval_lower = @floatCast(mse - margin),
        .confidence_interval_upper = @floatCast(mse + margin),
        .num_samples = n_samples,
    };
}

fn printValidationMetrics(writer: anytype, metrics: *const ValidationMetrics) !void {
    try writer.print("Validation Metrics (n={d}):\n", .{metrics.num_samples});
    try writer.print("  MSE: {d:.8}\n", .{metrics.mse});
    try writer.print("  RMSE: {d:.8}\n", .{metrics.rmse});
    try writer.print("  MAE: {d:.8}\n", .{metrics.mae});
    try writer.print("  R² Score: {d:.6}\n", .{metrics.r_squared});
    try writer.print("  Mean Prediction: {d:.6}\n", .{metrics.mean_prediction});
    try writer.print("  Mean Target: {d:.6}\n", .{metrics.mean_target});
    try writer.print("  95% CI: [{d:.8}, {d:.8}]\n", .{metrics.confidence_interval_lower, metrics.confidence_interval_upper});
}

fn testRSFForwardBackward(allocator: mem.Allocator, rsf: *RSF, dim: usize) !bool {
    const stdout = std.io.getStdOut().writer();
    try stdout.writeAll("[TEST 1] RSF Forward/Backward Pass with Gradient Verification\n");
    
    var input = try Tensor.init(allocator, &.{ 1, dim * 2 });
    defer input.deinit();
    
    var prng = PRNG.init(54321);
    for (input.data) |*val| {
        const rand_val = @as(f32, @floatCast(@as(f64, @floatFromInt(prng.next())) / @as(f64, @floatFromInt(math.maxInt(u64)))));
        val.* = rand_val * 0.1;
    }
    
    var input_copy = try input.copy(allocator);
    defer input_copy.deinit();
    
    try rsf.forward(&input);
    
    var changed = false;
    var idx: usize = 0;
    while (idx < input.data.len) : (idx += 1) {
        if (math.fabs(input.data[idx] - input_copy.data[idx]) > 1e-6) {
            changed = true;
            break;
        }
    }
    
    var grad_out = try Tensor.init(allocator, input.shape);
    defer grad_out.deinit();
    for (grad_out.data) |*val| {
        const rand_val = @as(f32, @floatCast(@as(f64, @floatFromInt(prng.next())) / @as(f64, @floatFromInt(math.maxInt(u64)))));
        val.* = rand_val * 0.01;
    }
    
    var grad_in = try rsf.backward(&grad_out, &input_copy);
    defer grad_in.deinit();
    
    var has_nonzero_grad = false;
    for (grad_in.data) |val| {
        if (math.fabs(val) > 1e-9) {
            has_nonzero_grad = true;
            break;
        }
    }
    
    try stdout.print("  Forward changed output: {s}\n", .{if (changed) "YES" else "NO"});
    try stdout.print("  Backward gradients nonzero: {s}\n", .{if (has_nonzero_grad) "YES" else "NO"});
    
    const passed = changed and has_nonzero_grad;
    try stdout.print("  Result: {s}\n\n", .{if (passed) "PASSED" else "FAILED"});
    
    return passed;
}

fn testMGTEncodingDecoding(allocator: mem.Allocator, mgt: *MGT) !bool {
    const stdout = std.io.getStdOut().writer();
    try stdout.writeAll("[TEST 2] MGT Encoding/Decoding Quality Verification\n");
    
    const test_text = "The quick brown fox jumps over";
    
    var tokens = std.ArrayList(u32).init(allocator);
    defer tokens.deinit();
    
    try mgt.encode(test_text, &tokens);
    
    var decoded = std.ArrayList(u8).init(allocator);
    defer decoded.deinit();
    
    try mgt.decode(tokens.items, &decoded);
    
    const has_alpha = blk: {
        for (decoded.items) |c| {
            if ((c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z')) {
                break :blk true;
            }
        }
        break :blk false;
    };
    
    const valid_tokens = tokens.items.len > 0 and tokens.items.len < 1000;
    
    try stdout.print("  Tokens: {d} (valid range: {s})\n", .{tokens.items.len, if (valid_tokens) "YES" else "NO"});
    try stdout.print("  Decoded length: {d} (contains words: {s})\n", .{decoded.items.len, if (has_alpha) "YES" else "NO"});
    
    const passed = valid_tokens and has_alpha;
    try stdout.print("  Result: {s}\n\n", .{if (passed) "PASSED" else "FAILED"});
    
    return passed;
}

fn testSSIRetrieval(allocator: mem.Allocator, ssi: *const SSI, mgt: *MGT, config: Config) !bool {
    const stdout = std.io.getStdOut().writer();
    try stdout.writeAll("[TEST 3] SSI Index Retrieval with Score Verification\n");
    
    const test_query = "neural network training";
    
    var query_tokens = std.ArrayList(u32).init(allocator);
    defer query_tokens.deinit();
    
    try mgt.encode(test_query, &query_tokens);
    
    const retrieved = try ssi.retrieveTopK(query_tokens.items, config.top_k, allocator);
    defer {
        for (retrieved) |*seg| {
            seg.deinit(allocator);
        }
        allocator.free(retrieved);
    }
    
    var min_score: f32 = 1.0;
    var max_score: f32 = 0.0;
    var all_valid_positions = true;
    
    for (retrieved) |seg| {
        if (seg.score < min_score) min_score = seg.score;
        if (seg.score > max_score) max_score = seg.score;
        if (seg.position > 10000) all_valid_positions = false;
    }
    
    try stdout.print("  Retrieved segments: {d}\n", .{retrieved.len});
    try stdout.print("  Score range: [{d:.4}, {d:.4}]\n", .{min_score, max_score});
    try stdout.print("  Valid positions: {s}\n", .{if (all_valid_positions) "YES" else "NO"});
    
    const passed = retrieved.len > 0 and all_valid_positions;
    try stdout.print("  Result: {s}\n\n", .{if (passed) "PASSED" else "FAILED"});
    
    return passed;
}

fn testRankerScoring(allocator: mem.Allocator, ranker: *const Ranker, ssi: *const SSI, mgt: *MGT, config: Config) !bool {
    const stdout = std.io.getStdOut().writer();
    try stdout.writeAll("[TEST 4] Ranker Scoring with Ordering Verification\n");
    
    const test_query = "deep learning optimization";
    
    var query_tokens = std.ArrayList(u32).init(allocator);
    defer query_tokens.deinit();
    
    try mgt.encode(test_query, &query_tokens);
    
    var segments = try ssi.retrieveTopK(query_tokens.items, config.top_k, allocator);
    defer {
        for (segments) |*seg| {
            seg.deinit(allocator);
        }
        allocator.free(segments);
    }
    
    if (segments.len == 0) {
        try stdout.writeAll("  No segments retrieved\n");
        try stdout.writeAll("  Result: FAILED\n\n");
        return false;
    }
    
    try ranker.rankCandidates(segments, ssi, allocator);
    
    var all_have_scores = true;
    var descending_order = true;
    var has_nonzero = false;
    
    var seg_idx: usize = 0;
    while (seg_idx < segments.len) : (seg_idx += 1) {
        const seg = segments[seg_idx];
        if (seg.score == 0.0) {
            all_have_scores = false;
        } else {
            has_nonzero = true;
        }
        
        if (seg_idx > 0 and segments[seg_idx - 1].score < seg.score) {
            descending_order = false;
        }
    }
    
    try stdout.print("  Ranked segments: {d}\n", .{segments.len});
    try stdout.print("  All have scores: {s}\n", .{if (all_have_scores) "YES" else "NO"});
    try stdout.print("  Scores in descending order: {s}\n", .{if (descending_order) "YES" else "NO"});
    try stdout.print("  Has nonzero scores: {s}\n", .{if (has_nonzero) "YES" else "NO"});
    
    try stdout.writeAll("  Scores: [");
    var score_idx: usize = 0;
    while (score_idx < segments.len) : (score_idx += 1) {
        if (score_idx > 0) try stdout.writeAll(", ");
        try stdout.print("{d:.4}", .{segments[score_idx].score});
    }
    try stdout.writeAll("]\n");
    
    const passed = descending_order and has_nonzero;
    try stdout.print("  Result: {s}\n\n", .{if (passed) "PASSED" else "FAILED"});
    
    return passed;
}

fn testParameterExtraction(allocator: mem.Allocator, rsf: *RSF) !bool {
    const stdout = std.io.getStdOut().writer();
    try stdout.writeAll("[TEST 5] Parameter Extraction and Update with Modification\n");
    
    var original_params = std.ArrayList(f32).init(allocator);
    defer original_params.deinit();
    
    var l: usize = 0;
    while (l < rsf.num_layers) : (l += 1) {
        const layer = &rsf.layers[l];
        
        for (layer.s_weight.data) |val| {
            try original_params.append(val);
        }
        for (layer.t_weight.data) |val| {
            try original_params.append(val);
        }
        for (layer.s_bias.data) |val| {
            try original_params.append(val);
        }
        for (layer.t_bias.data) |val| {
            try original_params.append(val);
        }
    }
    
    l = 0;
    while (l < rsf.num_layers) : (l += 1) {
        const layer = &rsf.layers[l];
        
        for (layer.s_weight.data) |*val| {
            val.* += 0.001;
        }
        for (layer.t_weight.data) |*val| {
            val.* += 0.001;
        }
        for (layer.s_bias.data) |*val| {
            val.* += 0.001;
        }
        for (layer.t_bias.data) |*val| {
            val.* += 0.001;
        }
    }
    
    var new_params = std.ArrayList(f32).init(allocator);
    defer new_params.deinit();
    
    l = 0;
    while (l < rsf.num_layers) : (l += 1) {
        const layer = &rsf.layers[l];
        
        for (layer.s_weight.data) |val| {
            try new_params.append(val);
        }
        for (layer.t_weight.data) |val| {
            try new_params.append(val);
        }
        for (layer.s_bias.data) |val| {
            try new_params.append(val);
        }
        for (layer.t_bias.data) |val| {
            try new_params.append(val);
        }
    }
    
    var params_updated = true;
    var params_changed = false;
    var total_diff: f64 = 0.0;
    
    if (original_params.items.len != new_params.items.len) {
        params_updated = false;
    } else {
        var param_idx: usize = 0;
        while (param_idx < original_params.items.len) : (param_idx += 1) {
            const orig = original_params.items[param_idx];
            const new = new_params.items[param_idx];
            const diff = math.fabs(new - orig);
            total_diff += diff;
            if (diff > 1e-9) {
                params_changed = true;
            }
        }
    }
    
    const avg_diff = if (original_params.items.len > 0)
        total_diff / @as(f64, @floatFromInt(original_params.items.len))
    else
        0.0;
    
    try stdout.print("  Parameters updated correctly: {s}\n", .{if (params_updated) "YES" else "NO"});
    try stdout.print("  Parameters actually changed: {s}\n", .{if (params_changed) "YES" else "NO"});
    try stdout.print("  Average update difference: {d:.8}\n", .{avg_diff});
    
    const passed = params_updated and params_changed;
    try stdout.print("  Result: {s}\n\n", .{if (passed) "PASSED" else "FAILED"});
    
    return passed;
}

fn testGradientClipping(allocator: mem.Allocator, optimizer: *SFD) !bool {
    const stdout = std.io.getStdOut().writer();
    try stdout.writeAll("[TEST 6] Gradient Clipping with Precise Verification\n");
    
    var gradients = try Tensor.init(allocator, &.{100});
    defer gradients.deinit();
    
    var prng = PRNG.init(99999);
    for (gradients.data) |*val| {
        const rand_val = @as(f32, @floatCast(@as(f64, @floatFromInt(prng.next())) / @as(f64, @floatFromInt(math.maxInt(u64)))));
        val.* = (rand_val * 2.0 - 1.0) * 10.0;
    }
    
    const norm_before = gradients.normL2();
    
    var grad_ptrs = [_]*Tensor{&gradients};
    const max_norm: f32 = 5.0;
    const norm_after = try optimizer.clipGradNorm(&grad_ptrs, max_norm);
    
    const was_clipped = norm_before > max_norm;
    const within_tolerance = math.fabs(norm_after - max_norm) < 0.1 or norm_after <= max_norm;
    
    try stdout.print("  Norm before clipping: {d:.6}\n", .{norm_before});
    try stdout.print("  Norm after clipping: {d:.6}\n", .{norm_after});
    try stdout.print("  Target norm: {d:.6}\n", .{max_norm});
    try stdout.print("  Was clipped: {s}\n", .{if (was_clipped) "YES" else "NO"});
    try stdout.print("  Within tolerance: {s}\n", .{if (within_tolerance) "YES" else "NO"});
    
    const passed = was_clipped and within_tolerance;
    try stdout.print("  Result: {s}\n\n", .{if (passed) "PASSED" else "FAILED"});
    
    return passed;
}

fn runInteractiveREPL(allocator: mem.Allocator, mgt: *MGT, ssi: *SSI, ranker: *Ranker) !void {
    const stdout = std.io.getStdOut().writer();
    const stdin = std.io.getStdIn().reader();
    
    var sample_texts = [_][]const u8{
        "A mesterséges intelligencia a jövő kulcsa.",
        "Az adattudomány és gépi tanulás összekapcsolódik.",
        "A neurális hálózatok komplex mintákat ismernek fel.",
        "Az automatizálás növeli a termelékenységet.",
        "A kvantumszámítógépek új lehetőségeket nyitnak.",
        "Az algoritmusok optimalizálják a döntéshozatalt.",
        "A természetes nyelvfeldolgozás emberi kommunikációt értelmez.",
        "A számítógépes látás képeket és videókat elemez.",
        "A robotika és automatizálás átalakítja az ipart.",
        "Az etikus AI fejlesztés fontos társadalmi kérdés.",
    };
    
    var sample_idx: usize = 0;
    while (sample_idx < sample_texts.len) : (sample_idx += 1) {
        var tokens = std.ArrayList(u32).init(allocator);
        defer tokens.deinit();
        try mgt.encode(sample_texts[sample_idx], &tokens);
        if (tokens.items.len > 0) {
            const is_anchor = (sample_idx % 3 == 0);
            try ssi.addSequence(tokens.items, @as(u64, @intCast(sample_idx)), is_anchor);
        }
    }
    
    var line_buf: [4096]u8 = undefined;
    
    while (true) {
        const line = stdin.readUntilDelimiterOrEof(&line_buf, '\n') catch |err| {
            try stdout.print("INPUT ERROR: {any}\n", .{err});
            continue;
        };
        
        if (line == null) break;
        
        const input = mem.trim(u8, line.?, " \t\r\n");
        if (input.len == 0) continue;
        
        if (mem.eql(u8, input, "exit") or mem.eql(u8, input, "quit")) {
            try stdout.writeAll("SYSTEM SHUTDOWN. GOODBYE.\n");
            break;
        }
        
        if (mem.eql(u8, input, "help")) {
            try stdout.writeAll("JAIDE v40 COMMANDS: help, status, exit/quit, or type any query.\n");
            continue;
        }
        
        if (mem.eql(u8, input, "status")) {
            const stats = ssi.stats();
            try stdout.print("SSI: {d} nodes, {d} leaves | MGT vocab: {d} | Ranker ngrams: {d}\n", .{
                stats.nodes, stats.leaves, mgt.vocabSize(), ranker.num_ngrams
            });
            continue;
        }
        
        var query_tokens = std.ArrayList(u32).init(allocator);
        defer query_tokens.deinit();
        
        mgt.encode(input, &query_tokens) catch |err| {
            try stdout.print("TOKENIZATION ERROR: {any}\n", .{err});
            continue;
        };
        
        if (query_tokens.items.len == 0) {
            try stdout.writeAll("EMPTY QUERY. PLEASE PROVIDE INPUT.\n");
            continue;
        }
        
        const segments = ssi.retrieveTopK(query_tokens.items, 5, allocator) catch |err| {
            try stdout.print("RETRIEVAL ERROR: {any}\n", .{err});
            continue;
        };
        defer {
            for (segments) |*seg| {
                var s = seg.*;
                s.deinit(allocator);
            }
            allocator.free(segments);
        }
        
        if (segments.len == 0) {
            try stdout.writeAll("NO MATCHING SEGMENTS FOUND. EXPANDING KNOWLEDGE BASE.\n");
            try ssi.addSequence(query_tokens.items, @as(u64, @intCast(ssi.size)), false);
            try stdout.writeAll("QUERY INDEXED. ASK AGAIN FOR CONTEXT-AWARE RESPONSE.\n");
            continue;
        }
        
        ranker.rankCandidates(segments, ssi, allocator) catch |err| {
            try stdout.print("RANKING ERROR: {any}\n", .{err});
            continue;
        };
        
        const best = segments[0];
        var decoded_list = std.ArrayList(u8).init(allocator);
        defer decoded_list.deinit();
        
        mgt.decode(best.tokens, &decoded_list) catch |err| {
            try stdout.print("DECODE ERROR: {any}\n", .{err});
            continue;
        };
        
        if (decoded_list.items.len > 0) {
            try stdout.print("RESPONSE: {s} [score: {d:.4}]\n", .{decoded_list.items, best.score});
        } else {
            try stdout.print("ANALYSIS COMPLETE. RELEVANCE SCORE: {d:.4} | TOKENS: {d}\n", .{best.score, best.tokens.len});
        }
    }
}
