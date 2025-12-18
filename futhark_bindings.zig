pub const struct_futhark_context_config = opaque {};
pub const struct_futhark_context = opaque {};
pub const struct_futhark_f16_1d = opaque {};
pub const struct_futhark_f16_2d = opaque {};
pub const struct_futhark_f16_3d = opaque {};
pub const struct_futhark_f32_1d = opaque {};
pub const struct_futhark_f32_2d = opaque {};
pub const struct_futhark_f32_3d = opaque {};
pub const struct_futhark_u64_1d = opaque {};
pub const struct_futhark_i64_1d = opaque {};

pub extern "c" fn futhark_context_config_new() ?*struct_futhark_context_config;
pub extern "c" fn futhark_context_config_free(cfg: ?*struct_futhark_context_config) void;
pub extern "c" fn futhark_context_config_set_device(cfg: ?*struct_futhark_context_config, device: c_int) void;
pub extern "c" fn futhark_context_config_set_platform(cfg: ?*struct_futhark_context_config, platform: c_int) void;
pub extern "c" fn futhark_context_config_set_default_group_size(cfg: ?*struct_futhark_context_config, size: c_int) void;
pub extern "c" fn futhark_context_config_set_default_num_groups(cfg: ?*struct_futhark_context_config, num: c_int) void;
pub extern "c" fn futhark_context_config_set_default_tile_size(cfg: ?*struct_futhark_context_config, size: c_int) void;

pub extern "c" fn futhark_context_new(cfg: ?*struct_futhark_context_config) ?*struct_futhark_context;
pub extern "c" fn futhark_context_free(ctx: ?*struct_futhark_context) void;
pub extern "c" fn futhark_context_sync(ctx: ?*struct_futhark_context) c_int;

pub extern "c" fn futhark_new_f16_2d(ctx: *struct_futhark_context, data: [*]const f16, dim0: i64, dim1: i64) ?*struct_futhark_f16_2d;
pub extern "c" fn futhark_free_f16_2d(ctx: *struct_futhark_context, arr: ?*struct_futhark_f16_2d) c_int;
pub extern "c" fn futhark_values_f16_2d(ctx: *struct_futhark_context, arr: *struct_futhark_f16_2d, data: [*]f16) c_int;
pub extern "c" fn futhark_values_raw_f16_2d(ctx: *struct_futhark_context, arr: *struct_futhark_f16_2d) ?*anyopaque;
pub extern "c" fn futhark_shape_f16_2d(ctx: *struct_futhark_context, arr: *struct_futhark_f16_2d, dims: [*]i64) c_int;

pub extern "c" fn futhark_entry_rsf_forward(ctx: *struct_futhark_context, out: **struct_futhark_f16_2d, input: *struct_futhark_f16_2d, weights_s: *struct_futhark_f16_2d, weights_t: *struct_futhark_f16_2d) c_int;
pub extern "c" fn futhark_entry_rsf_backward(ctx: *struct_futhark_context, out: **struct_futhark_f16_2d, grad_output: *struct_futhark_f16_2d, weights: *struct_futhark_f16_2d) c_int;
pub extern "c" fn futhark_entry_scale_weights_inplace(ctx: *struct_futhark_context, weights: *struct_futhark_f16_2d, scale: f32) c_int;
pub extern "c" fn futhark_entry_training_step(
    ctx: *struct_futhark_context,
    new_weights_s: **struct_futhark_f16_2d,
    new_weights_t: **struct_futhark_f16_2d,
    new_velocity_s: **struct_futhark_f16_2d,
    new_velocity_t: **struct_futhark_f16_2d,
    loss: *u16,
    inputs: *struct_futhark_f16_2d,
    targets: *struct_futhark_f16_2d,
    weights_s: *struct_futhark_f16_2d,
    weights_t: *struct_futhark_f16_2d,
    velocity_s: *struct_futhark_f16_2d,
    velocity_t: *struct_futhark_f16_2d,
    learning_rate: u16,
    momentum: u16,
) c_int;