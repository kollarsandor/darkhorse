pub const cudaError_t = c_uint;
pub const cudaSuccess: cudaError_t = 0;
pub const cudaHostAllocDefault: c_uint = 0;

pub extern "c" fn cudaHostAlloc(ptr: **anyopaque, size: usize, flags: c_uint) cudaError_t;
pub extern "c" fn cudaFreeHost(ptr: *anyopaque) cudaError_t;
pub extern "c" fn cudaMalloc(devPtr: **anyopaque, size: usize) cudaError_t;
pub extern "c" fn cudaFree(devPtr: *anyopaque) cudaError_t;
pub extern "c" fn cudaMemcpy(dst: *anyopaque, src: *const anyopaque, count: usize, kind: c_uint) cudaError_t;
pub extern "c" fn cudaDeviceSynchronize() cudaError_t;
pub extern "c" fn cudaGetLastError() cudaError_t;
pub extern "c" fn cudaGetErrorString(err: cudaError_t) ?*const u8;

pub const cudaMemcpyHostToDevice: c_uint = 1;
pub const cudaMemcpyDeviceToHost: c_uint = 2;
pub const cudaMemcpyDeviceToDevice: c_uint = 3;