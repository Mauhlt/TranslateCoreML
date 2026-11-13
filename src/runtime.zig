//! Core ML runtime bindings (iOS 16+ / macOS 13+), through the
//! Objective-C runtime (objc.zig). Link CoreML, Foundation and libobjc.
//!
//! The Apple Neural Engine has no public API of its own: it is reached
//! only through Core ML, by loading a model with compute units
//! `.cpu_and_neural_engine` (or `.all`) and letting Core ML place each
//! operation. `computePlan` (iOS 17.4+ / macOS 14.4+, MLComputePlan) tells
//! which device Core ML chose for every op of an ML program.
//!
//!     const url = try coreml.compile(gpa, "/path/model.mlpackage", "/cache/dir"); // -> .mlmodelc path
//!     var m = try coreml.Model.load(compiled_path, .cpu_and_neural_engine);
//!     defer m.deinit();
//!     var x = try coreml.MultiArray.create(&.{ 1, 64 }, .float32);
//!     defer x.deinit();
//!     x.writeF32(input);
//!     var out = try m.predict(&.{.{ .name = "x", .array = x }});
//!     defer out.deinit();
//!     (out.array("y") orelse return error.NoOutput).readF32(result);
//!
//! Ownership: every wrapper holds one retain on its object; deinit
//! releases it. Calls that create autoreleased objects should run inside
//! an objc.AutoreleasePool (the probe and the stt engine do).
//! Errors: functions return error.CoreML and leave Core ML's NSError text
//! in `lastError()` (per thread).
const std = @import("std");
const builtin = @import("builtin");
const objc = @import("objc.zig");
const Object = objc.Object;
const sel = objc.sel;
const msgSend = objc.msgSend;
const Allocator = std.mem.Allocator;

/// MLComputeUnits.
pub const ComputeUnits = enum(isize) {
    cpu_only = 0,
    cpu_and_gpu = 1,
    all = 2,
    /// iOS 16 / macOS 13.
    cpu_and_neural_engine = 3,

    pub fn name(u: ComputeUnits) []const u8 {
        return @tagName(u);
    }
};

/// MLMultiArrayDataType.
pub const DataType = enum(isize) {
    float16 = 0x10000 | 16,
    float32 = 0x10000 | 32,
    double = 0x10000 | 64,
    int32 = 0x20000 | 32,

    pub fn size(d: DataType) usize {
        return switch (d) {
            .float16 => 2,
            .float32, .int32 => 4,
            .double => 8,
        };
    }
};

pub const Error = error{ CoreML, Unavailable, BadShape, OutOfMemory, Timeout };

threadlocal var last_error_buf: [1024]u8 = undefined;
threadlocal var last_error_len: usize = 0;

/// Core ML's message for the last error.CoreML on this thread.
pub fn lastError() []const u8 {
    return last_error_buf[0..last_error_len];
}

fn setError(what: []const u8, err: ?*Object) Error {
    const text = objc.errorText(err);
    const s = std.fmt.bufPrint(&last_error_buf, "{s}: {s}", .{ what, text }) catch last_error_buf[0..];
    last_error_len = s.len;
    return error.CoreML;
}

/// Compiles a .mlpackage (or .mlmodel) to a .mlmodelc directory and moves
/// it to `<out_dir>/<name>.mlmodelc` (replacing it). Returns that path
/// (caller frees). Core ML compiles into a temporary folder otherwise.
pub fn compile(gpa: Allocator, package_path: []const u8, out_dir: []const u8, name: []const u8) Error![]u8 {
    const pool = objc.AutoreleasePool.push();
    defer pool.pop();
    const url = objc.fileUrl(package_path) orelse return error.OutOfMemory;
    var err: ?*Object = null;
    const tmp = msgSend(?*Object, objc.getClass("MLModel"), sel("compileModelAtURL:error:"), .{ url, &err }) orelse
        return setError("compileModelAtURL", err);
    const dest = std.fmt.allocPrint(gpa, "{s}/{s}.mlmodelc", .{ out_dir, name }) catch return error.OutOfMemory;
    errdefer gpa.free(dest);
    const fm = msgSend(?*Object, objc.getClass("NSFileManager"), sel("defaultManager"), .{});
    const dest_url = objc.fileUrl(dest) orelse return error.OutOfMemory;
    _ = msgSend(objc.BOOL, fm, sel("removeItemAtURL:error:"), .{ dest_url, @as(?*?*Object, null) });
    const dir_url = objc.fileUrl(out_dir) orelse return error.OutOfMemory;
    _ = msgSend(objc.BOOL, fm, sel("createDirectoryAtURL:withIntermediateDirectories:attributes:error:"), .{ dir_url, @as(objc.BOOL, 1), @as(?*Object, null), @as(?*?*Object, null) });
    err = null;
    if (msgSend(objc.BOOL, fm, sel("moveItemAtURL:toURL:error:"), .{ tmp, dest_url, &err }) == 0)
        return setError("moving the compiled model", err);
    return dest;
}

pub const Model = struct {
    obj: *Object,

    /// Loads a compiled model (.mlmodelc) with the given compute units.
    /// On the ANE path, the first load of a model also compiles it for the
    /// Neural Engine (cached by the OS afterwards), so it can take seconds.
    pub fn load(path: []const u8, units: ComputeUnits) Error!Model {
        const pool = objc.AutoreleasePool.push();
        defer pool.pop();
        const cfg = msgSend(?*Object, msgSend(?*Object, objc.getClass("MLModelConfiguration"), sel("alloc"), .{}), sel("init"), .{}) orelse return error.OutOfMemory;
        defer objc.release(cfg);
        msgSend(void, cfg, sel("setComputeUnits:"), .{@intFromEnum(units)});
        const url = objc.fileUrl(path) orelse return error.OutOfMemory;
        var err: ?*Object = null;
        const m = msgSend(?*Object, objc.getClass("MLModel"), sel("modelWithContentsOfURL:configuration:error:"), .{ url, cfg, &err }) orelse
            return setError("modelWithContentsOfURL", err);
        return .{ .obj = objc.retain(m) };
    }

    pub fn deinit(m: *Model) void {
        objc.release(m.obj);
        m.* = undefined;
    }

    pub const Input = struct { name: []const u8, array: MultiArray };

    fn provider(inputs: []const Input) Error!*Object {
        var keys: [64]?*Object = undefined;
        var vals: [64]?*Object = undefined;
        if (inputs.len > keys.len) return error.BadShape;
        for (inputs, 0..) |in, i| {
            const k = objc.newString(in.name) orelse return error.OutOfMemory;
            keys[i] = objc.autorelease(k);
            vals[i] = msgSend(?*Object, objc.getClass("MLFeatureValue"), sel("featureValueWithMultiArray:"), .{in.array.obj});
        }
        const dict = objc.dictionary(keys[0..inputs.len], vals[0..inputs.len]);
        var err: ?*Object = null;
        const a = msgSend(?*Object, objc.getClass("MLDictionaryFeatureProvider"), sel("alloc"), .{});
        const p = msgSend(?*Object, a, sel("initWithDictionary:error:"), .{ dict, &err }) orelse
            return setError("MLDictionaryFeatureProvider", err);
        return p; // +1
    }

    /// One prediction (synchronous, on this thread).
    pub fn predict(m: *const Model, inputs: []const Input) Error!Prediction {
        const pool = objc.AutoreleasePool.push();
        defer pool.pop();
        const p = try provider(inputs);
        defer objc.release(p);
        var err: ?*Object = null;
        const out = msgSend(?*Object, m.obj, sel("predictionFromFeatures:error:"), .{ p, &err }) orelse
            return setError("prediction", err);
        return .{ .obj = objc.retain(out) };
    }

    /// iOS 18 / macOS 15: a new state object for a stateful model.
    pub fn newState(m: *const Model) Error!State {
        if (!objc.respondsTo(m.obj, sel("newState"))) return error.Unavailable;
        const s = msgSend(?*Object, m.obj, sel("newState"), .{}) orelse return error.CoreML;
        return .{ .obj = s }; // "new": +1
    }

    /// iOS 18 / macOS 15: a prediction that reads and updates `state`.
    pub fn predictWithState(m: *const Model, inputs: []const Input, state: *const State) Error!Prediction {
        const pool = objc.AutoreleasePool.push();
        defer pool.pop();
        const p = try provider(inputs);
        defer objc.release(p);
        var err: ?*Object = null;
        const out = msgSend(?*Object, m.obj, sel("predictionFromFeatures:usingState:error:"), .{ p, state.obj, &err }) orelse
            return setError("prediction (state)", err);
        return .{ .obj = objc.retain(out) };
    }
};

pub const State = struct {
    obj: *Object,
    pub fn deinit(s: *State) void {
        objc.release(s.obj);
        s.* = undefined;
    }
};

pub const Prediction = struct {
    obj: *Object,

    pub fn deinit(p: *Prediction) void {
        objc.release(p.obj);
        p.* = undefined;
    }

    /// The output called `name` as a (retained) MultiArray.
    pub fn array(p: *const Prediction, name: []const u8) ?MultiArray {
        const pool = objc.AutoreleasePool.push();
        defer pool.pop();
        const k = objc.newString(name) orelse return null;
        defer objc.release(k);
        const fv = msgSend(?*Object, p.obj, sel("featureValueForName:"), .{k}) orelse return null;
        const a = msgSend(?*Object, fv, sel("multiArrayValue"), .{}) orelse return null;
        return MultiArray.adopt(objc.retain(a));
    }
};

/// MLMultiArray. Core ML's own arrays (outputs) may have padded strides
/// (fp16 arrays backed by pixel buffers), so element access goes through
/// `strides`.
pub const MultiArray = struct {
    obj: *Object,
    dtype: DataType,
    shape_buf: [8]usize = undefined,
    strides_buf: [8]usize = undefined,
    rank: usize,

    pub fn shape(a: *const MultiArray) []const usize {
        return a.shape_buf[0..a.rank];
    }
    pub fn strides(a: *const MultiArray) []const usize {
        return a.strides_buf[0..a.rank];
    }
    pub fn count(a: *const MultiArray) usize {
        var n: usize = 1;
        for (a.shape()) |d| n *= d;
        return n;
    }
    /// True when the elements are densely packed in row-major order.
    pub fn contiguous(a: *const MultiArray) bool {
        var s: usize = 1;
        var i = a.rank;
        while (i > 0) {
            i -= 1;
            if (a.shape_buf[i] != 1 and a.strides_buf[i] != s) return false;
            s *= a.shape_buf[i];
        }
        return true;
    }

    fn adopt(obj: *Object) MultiArray {
        var a: MultiArray = .{ .obj = obj, .dtype = @enumFromInt(msgSend(isize, obj, sel("dataType"), .{})), .rank = 0 };
        const sh = msgSend(?*Object, obj, sel("shape"), .{});
        const st = msgSend(?*Object, obj, sel("strides"), .{});
        a.rank = @min(objc.arrayCount(sh), 8);
        for (0..a.rank) |i| {
            a.shape_buf[i] = @intCast(objc.numberValue(objc.arrayAt(sh, i)));
            a.strides_buf[i] = @intCast(objc.numberValue(objc.arrayAt(st, i)));
        }
        return a;
    }

    fn shapeArray(dims: []const usize) ?*Object {
        var nums: [8]?*Object = undefined;
        for (dims, 0..) |d, i| nums[i] = objc.number(@intCast(d));
        return objc.array(nums[0..dims.len]);
    }

    /// A new array owned by Core ML (zero-filled by us).
    pub fn create(dims: []const usize, dtype: DataType) Error!MultiArray {
        if (dims.len == 0 or dims.len > 8) return error.BadShape;
        const pool = objc.AutoreleasePool.push();
        defer pool.pop();
        var err: ?*Object = null;
        const a = msgSend(?*Object, objc.getClass("MLMultiArray"), sel("alloc"), .{});
        const o = msgSend(?*Object, a, sel("initWithShape:dataType:error:"), .{ shapeArray(dims), @intFromEnum(dtype), &err }) orelse
            return setError("MLMultiArray", err);
        var r = adopt(o);
        @memset(r.bytes(), 0);
        return r;
    }

    /// An array over caller memory (row-major, no copy). `data` must stay
    /// valid and unmoved for as long as Core ML may use the array.
    pub fn wrap(data: []u8, dims: []const usize, dtype: DataType) Error!MultiArray {
        if (dims.len == 0 or dims.len > 8) return error.BadShape;
        var n: usize = dtype.size();
        for (dims) |d| n *= d;
        if (data.len < n) return error.BadShape;
        const pool = objc.AutoreleasePool.push();
        defer pool.pop();
        var st: [8]usize = undefined;
        var s: usize = 1;
        var i = dims.len;
        while (i > 0) {
            i -= 1;
            st[i] = s;
            s *= dims[i];
        }
        var err: ?*Object = null;
        const a = msgSend(?*Object, objc.getClass("MLMultiArray"), sel("alloc"), .{});
        const o = msgSend(?*Object, a, sel("initWithDataPointer:shape:dataType:strides:deallocator:error:"), .{
            @as(*anyopaque, data.ptr), shapeArray(dims), @intFromEnum(dtype), shapeArray(st[0..dims.len]), @as(?*anyopaque, null), &err,
        }) orelse return setError("MLMultiArray wrap", err);
        return adopt(o);
    }

    pub fn deinit(a: *MultiArray) void {
        objc.release(a.obj);
        a.* = undefined;
    }

    /// The array's memory, spanning the last element (with padding).
    pub fn bytes(a: *const MultiArray) []u8 {
        const p: [*]u8 = @ptrCast(msgSend(?*anyopaque, a.obj, sel("dataPointer"), .{}) orelse return &.{});
        var last: usize = 0;
        for (a.shape(), a.strides()) |d, s| last += (d - 1) * s;
        return p[0 .. (last + 1) * a.dtype.size()];
    }

    /// Element offset (in elements) of flat row-major index `flat`.
    fn offsetOf(a: *const MultiArray, flat: usize) usize {
        var f = flat;
        var off: usize = 0;
        var i = a.rank;
        while (i > 0) {
            i -= 1;
            off += (f % a.shape_buf[i]) * a.strides_buf[i];
            f /= a.shape_buf[i];
        }
        return off;
    }

    /// Copies row-major f32 values in (converting to the array's type).
    pub fn writeF32(a: *MultiArray, src: []const f32) void {
        const b = a.bytes();
        const n = @min(src.len, a.count());
        const dense = a.contiguous();
        for (src[0..n], 0..) |v, i| {
            const o = if (dense) i else a.offsetOf(i);
            switch (a.dtype) {
                .float32 => std.mem.bytesAsSlice(f32, b)[o] = v,
                .float16 => std.mem.bytesAsSlice(f16, b)[o] = @floatCast(v),
                .double => std.mem.bytesAsSlice(f64, b)[o] = v,
                .int32 => std.mem.bytesAsSlice(i32, b)[o] = @intFromFloat(v),
            }
        }
    }

    /// Copies the values out as row-major f32.
    pub fn readF32(a: *const MultiArray, dst: []f32) void {
        const b = a.bytes();
        const n = @min(dst.len, a.count());
        const dense = a.contiguous();
        for (dst[0..n], 0..) |*v, i| {
            const o = if (dense) i else a.offsetOf(i);
            v.* = switch (a.dtype) {
                .float32 => std.mem.bytesAsSlice(f32, b)[o],
                .float16 => @floatCast(std.mem.bytesAsSlice(f16, b)[o]),
                .double => @floatCast(std.mem.bytesAsSlice(f64, b)[o]),
                .int32 => @floatFromInt(std.mem.bytesAsSlice(i32, b)[o]),
            };
        }
    }

    /// Copies raw fp16 values (row-major) in; the array must be float16.
    pub fn writeF16(a: *MultiArray, src: []const f16) void {
        std.debug.assert(a.dtype == .float16);
        const d = std.mem.bytesAsSlice(f16, a.bytes());
        if (a.contiguous()) {
            @memcpy(d[0..src.len], src);
        } else for (src, 0..) |v, i| {
            d[a.offsetOf(i)] = v;
        }
    }

    /// Copies the values out as raw fp16 (row-major); the array must be float16.
    pub fn readF16(a: *const MultiArray, dst: []f16) void {
        std.debug.assert(a.dtype == .float16);
        const s = std.mem.bytesAsSlice(f16, a.bytes());
        if (a.contiguous()) {
            @memcpy(dst, s[0..dst.len]);
        } else for (dst, 0..) |*v, i| {
            v.* = s[a.offsetOf(i)];
        }
    }
};

// ------------------------------------------------------------ devices

pub const Device = enum { cpu, gpu, neural_engine, unknown };

fn deviceOf(obj: ?*Object) Device {
    const n = objc.className(obj);
    if (std.mem.indexOf(u8, n, "NeuralEngine") != null) return .neural_engine;
    if (std.mem.indexOf(u8, n, "GPU") != null) return .gpu;
    if (std.mem.indexOf(u8, n, "CPU") != null) return .cpu;
    return .unknown;
}

/// What MLModel.availableComputeDevices reports (iOS 17 / macOS 14), as
/// flags; null if the OS is older.
pub const Devices = struct { cpu: bool = false, gpu: bool = false, neural_engine: bool = false, ane_cores: i64 = 0 };

pub fn availableDevices() ?Devices {
    const pool = objc.AutoreleasePool.push();
    defer pool.pop();
    const cls = objc.lookUpClass("MLModel") orelse return null;
    const has = msgSend(objc.BOOL, cls, sel("respondsToSelector:"), .{sel("availableComputeDevices")}) != 0;
    if (!has) return null;
    const list = msgSend(?*Object, cls, sel("availableComputeDevices"), .{});
    var r: Devices = .{};
    for (0..objc.arrayCount(list)) |i| {
        const d = objc.arrayAt(list, i);
        switch (deviceOf(d)) {
            .cpu => r.cpu = true,
            .gpu => r.gpu = true,
            .neural_engine => {
                r.neural_engine = true;
                if (objc.respondsTo(d, sel("totalCoreCount"))) r.ane_cores = msgSend(isize, d, sel("totalCoreCount"), .{});
            },
            .unknown => {},
        }
    }
    return r;
}

/// Where Core ML placed one ML program operation.
pub const OpPlacement = struct {
    op: []const u8, // operator name, e.g. "ios17.linear"
    output: []const u8, // first output name
    device: Device, // preferred device
    supported: struct { cpu: bool = false, gpu: bool = false, neural_engine: bool = false } = .{},
    cost: f64 = 0, // estimated relative cost (0..1), if reported
};

pub const Plan = struct {
    arena: std.heap.ArenaAllocator,
    ops: []OpPlacement,

    pub fn deinit(p: *Plan) void {
        p.arena.deinit();
    }

    /// Ops whose preferred device is `d` (const ops have no placement and are skipped).
    pub fn countOn(p: *const Plan, d: Device) usize {
        var n: usize = 0;
        for (p.ops) |o| if (o.device == d) {
            n += 1;
        };
        return n;
    }
};

const PlanWait = struct {
    var sema: objc.Semaphore = undefined;
    var plan: ?*Object = null;
    var err: ?*Object = null;
    var lock: std.atomic.Mutex = .unlocked;

    fn done(_: *const objc.Block, plan_obj: ?*Object, err_obj: ?*Object) callconv(.c) void {
        plan = if (plan_obj) |o| objc.retain(o) else null;
        err = if (err_obj) |o| objc.retain(o) else null;
        sema.signal();
    }
    var block: objc.Block = undefined;
};

/// MLComputePlan for a compiled model (iOS 17.4+ / macOS 14.4+): the
/// device Core ML prefers for every operation of the main function.
/// Waits up to `timeout_ns` for Core ML's completion handler (call it off
/// the main thread on iOS). One call at a time per process.
pub fn computePlan(gpa: Allocator, compiled_path: []const u8, units: ComputeUnits, timeout_ns: u64) Error!Plan {
    const cls = objc.lookUpClass("MLComputePlan") orelse return error.Unavailable;
    while (!PlanWait.lock.tryLock()) std.atomic.spinLoopHint();
    defer PlanWait.lock.unlock();
    const pool = objc.AutoreleasePool.push();
    defer pool.pop();

    const cfg = msgSend(?*Object, msgSend(?*Object, objc.getClass("MLModelConfiguration"), sel("alloc"), .{}), sel("init"), .{}) orelse return error.OutOfMemory;
    defer objc.release(cfg);
    msgSend(void, cfg, sel("setComputeUnits:"), .{@intFromEnum(units)});
    const url = objc.fileUrl(compiled_path) orelse return error.OutOfMemory;

    PlanWait.sema = objc.Semaphore.init();
    defer PlanWait.sema.deinit();
    PlanWait.plan = null;
    PlanWait.err = null;
    PlanWait.block = objc.Block.global(@ptrCast(&PlanWait.done));
    msgSend(void, cls, sel("loadContentsOfURL:configuration:completionHandler:"), .{ url, cfg, @as(*const objc.Block, &PlanWait.block) });
    // A late handler after a timeout would write the globals; keep the
    // semaphore alive in that case (leak it) rather than signal freed memory.
    if (!PlanWait.sema.wait(timeout_ns)) {
        PlanWait.sema = objc.Semaphore.init();
        return error.Timeout;
    }
    const plan = PlanWait.plan orelse return setError("MLComputePlan", PlanWait.err);
    defer objc.release(plan);
    if (PlanWait.err) |e| objc.release(e);

    var out: Plan = .{ .arena = .init(gpa), .ops = &.{} };
    errdefer out.arena.deinit();
    const a = out.arena.allocator();
    var list: std.ArrayList(OpPlacement) = .empty;

    const structure = msgSend(?*Object, plan, sel("modelStructure"), .{});
    const program = msgSend(?*Object, structure, sel("program"), .{}) orelse return error.Unavailable;
    const functions = msgSend(?*Object, program, sel("functions"), .{});
    const main_key = objc.newString("main") orelse return error.OutOfMemory;
    defer objc.release(main_key);
    const func = objc.dictionaryGet(functions, main_key) orelse return error.Unavailable;
    const blk = msgSend(?*Object, func, sel("block"), .{});
    const ops = msgSend(?*Object, blk, sel("operations"), .{});
    for (0..objc.arrayCount(ops)) |i| {
        const op = objc.arrayAt(ops, i);
        const usage = msgSend(?*Object, plan, sel("computeDeviceUsageForMLProgramOperation:"), .{op}) orelse continue;
        var pl: OpPlacement = .{
            .op = a.dupe(u8, objc.stringBytes(msgSend(?*Object, op, sel("operatorName"), .{}))) catch return error.OutOfMemory,
            .output = "",
            .device = deviceOf(msgSend(?*Object, usage, sel("preferredComputeDevice"), .{})),
        };
        const outs = msgSend(?*Object, op, sel("outputs"), .{});
        if (objc.arrayCount(outs) > 0)
            pl.output = a.dupe(u8, objc.stringBytes(msgSend(?*Object, objc.arrayAt(outs, 0), sel("name"), .{}))) catch return error.OutOfMemory;
        const sup = msgSend(?*Object, usage, sel("supportedComputeDevices"), .{});
        for (0..objc.arrayCount(sup)) |j| switch (deviceOf(objc.arrayAt(sup, j))) {
            .cpu => pl.supported.cpu = true,
            .gpu => pl.supported.gpu = true,
            .neural_engine => pl.supported.neural_engine = true,
            .unknown => {},
        };
        if (msgSend(?*Object, plan, sel("estimatedCostForMLProgramOperation:"), .{op})) |cost|
            pl.cost = msgSend(f64, cost, sel("weight"), .{});
        list.append(a, pl) catch return error.OutOfMemory;
    }
    out.ops = list.items;
    return out;
}

/// Every binding, for compile-only checks (`zig build apple-check`
/// references this so unused functions are analysed too).
pub const all_decls = .{
    compile,                 lastError,             Model.load,          Model.deinit,
    Model.predict,           Model.newState,        Model.predictWithState, State.deinit,
    Prediction.deinit,       Prediction.array,      MultiArray.create,   MultiArray.wrap,
    MultiArray.deinit,       MultiArray.bytes,      MultiArray.writeF32, MultiArray.readF32,
    MultiArray.writeF16,     MultiArray.readF16,    availableDevices,    computePlan,
};
