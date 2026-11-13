//! Reference interpreter for mil.Builder programs, on the host CPU: runs
//! the same op list that is serialized into the .mlpackage, with the MIL
//! semantics as this package understands them, to produce the expected
//! outputs a Core ML run is compared against (and to test model graphs on
//! machines without Core ML).
//!
//! Values are computed in f32. With `emulate_fp16` (default), every value
//! whose MIL type is fp16 is rounded to fp16 when it is produced (inputs,
//! consts and op outputs), which models the storage precision of an fp16
//! Core ML program; accumulation inside an op stays f32. int32/bool values
//! are carried as f32 (exact below 2^24).
//!
//! Single-threaded, no SIMD intrinsics beyond @Vector; meant for
//! correctness, at a few GFLOP/s. Intermediate tensors are freed after
//! their last use.
const std = @import("std");
const Allocator = std.mem.Allocator;
const mil = @import("mil.zig");
const Builder = mil.Builder;
const V = mil.V;

pub const Tensor = struct {
    shape: []usize,
    data: []f32,

    pub fn count(t: Tensor) usize {
        return t.data.len;
    }
    pub fn deinit(t: *Tensor, gpa: Allocator) void {
        gpa.free(t.shape);
        gpa.free(t.data);
        t.* = undefined;
    }
};

pub const Named = struct { name: []const u8, shape: []const usize, data: []const f32 };

pub const Error = error{ OutOfMemory, BadInput, MissingInput, Unsupported, BadShape };

pub const Interp = struct {
    gpa: Allocator,
    b: *const Builder,
    emulate_fp16: bool = true,
    vals: []?Tensor,
    /// Index of the last op reading each value (outputs never die).
    last_use: []u32,
    /// Constants converted to f32 once (kept across runs).
    consts: []?[]f32,
    /// iOS 18 states by value id (persist across runs until `resetStates`).
    states: std.AutoHashMapUnmanaged(u32, Tensor) = .empty,
    /// Set by `run`: ops executed, and the name of the op that failed.
    failed_op: []const u8 = "",

    pub fn init(gpa: Allocator, b: *const Builder) !Interp {
        const n = b.values.items.len;
        const vals = try gpa.alloc(?Tensor, n);
        @memset(vals, null);
        const last = try gpa.alloc(u32, n);
        @memset(last, 0);
        const consts = try gpa.alloc(?[]f32, n);
        @memset(consts, null);
        for (b.ops.items, 0..) |o, i| for (o.params) |p| switch (p.arg) {
            .one => |v| last[v.id] = @intCast(i),
            .many => |vs| for (vs) |v| {
                last[v.id] = @intCast(i);
            },
        };
        for (b.outputs.items) |v| last[v.id] = std.math.maxInt(u32);
        return .{ .gpa = gpa, .b = b, .vals = vals, .last_use = last, .consts = consts };
    }

    pub fn deinit(it: *Interp) void {
        it.clearValues();
        for (it.consts) |c| if (c) |d| it.gpa.free(d);
        it.gpa.free(it.consts);
        it.gpa.free(it.vals);
        it.gpa.free(it.last_use);
        it.resetStates();
        it.states.deinit(it.gpa);
        it.* = undefined;
    }

    pub fn resetStates(it: *Interp) void {
        var i = it.states.valueIterator();
        while (i.next()) |t| t.deinit(it.gpa);
        it.states.clearRetainingCapacity();
    }

    fn clearValues(it: *Interp) void {
        for (it.vals) |*v| if (v.*) |*t| {
            t.deinit(it.gpa);
            v.* = null;
        };
    }

    /// Sets a state's contents (zero-initialized otherwise).
    pub fn setState(it: *Interp, name: []const u8, data: []const f32) Error!void {
        for (it.b.inputs.items) |v| {
            const inf = it.b.info(v);
            if (inf.kind != .state or !std.mem.eql(u8, inf.name, name)) continue;
            if (data.len != inf.count()) return error.BadInput;
            const t = try it.newTensor(inf.shape);
            @memcpy(t.data, data);
            if (it.emulate_fp16) roundF16(t.data);
            if (it.states.fetchRemove(v.id)) |old| {
                var o = old.value;
                o.deinit(it.gpa);
            }
            try it.states.put(it.gpa, v.id, t);
            return;
        }
        return error.MissingInput;
    }

    pub fn state(it: *Interp, name: []const u8) ?Tensor {
        for (it.b.inputs.items) |v| {
            const inf = it.b.info(v);
            if (inf.kind == .state and std.mem.eql(u8, inf.name, name)) return it.states.get(v.id);
        }
        return null;
    }

    /// Runs the program. Outputs stay readable through `output` until the
    /// next run.
    pub fn run(it: *Interp, inputs: []const Named) Error!void {
        it.clearValues();
        const b = it.b;
        for (b.inputs.items) |v| {
            const inf = b.info(v);
            if (inf.kind == .state) {
                if (!it.states.contains(v.id)) {
                    const t = try it.newTensor(inf.shape);
                    @memset(t.data, 0);
                    try it.states.put(it.gpa, v.id, t);
                }
                continue;
            }
            const in = for (inputs) |x| {
                if (std.mem.eql(u8, x.name, inf.name)) break x;
            } else return error.MissingInput;
            if (!std.mem.eql(usize, in.shape, inf.shape) or in.data.len != inf.count()) return error.BadInput;
            const t = try it.newTensor(inf.shape);
            @memcpy(t.data, in.data);
            if (inf.dtype == .fp16 and it.emulate_fp16) roundF16(t.data);
            it.vals[v.id] = t;
        }
        for (b.ops.items, 0..) |o, i| {
            it.failed_op = o.name;
            try it.exec(o);
            // free inputs whose last use was this op
            for (o.params) |p| switch (p.arg) {
                .one => |v| it.release(v, i),
                .many => |vs| for (vs) |v| it.release(v, i),
            };
        }
        it.failed_op = "";
    }

    fn release(it: *Interp, v: V, op_index: usize) void {
        if (it.last_use[v.id] != op_index) return;
        if (it.vals[v.id]) |*t| {
            t.deinit(it.gpa);
            it.vals[v.id] = null;
        }
    }

    pub fn output(it: *const Interp, name: []const u8) ?Tensor {
        for (it.b.outputs.items) |v| {
            if (std.mem.eql(u8, it.b.info(v).name, name)) return it.vals[v.id];
        }
        return null;
    }

    fn newTensor(it: *Interp, shape: []const usize) Error!Tensor {
        var n: usize = 1;
        for (shape) |d| n *= d;
        const s = try it.gpa.dupe(usize, shape);
        errdefer it.gpa.free(s);
        return .{ .shape = s, .data = try it.gpa.alloc(f32, n) };
    }

    /// The f32 view of a value (const data converted once).
    fn val(it: *Interp, v: V) Error![]const f32 {
        if (it.vals[v.id]) |t| return t.data;
        const inf = it.b.info(v);
        switch (inf.kind) {
            .constant => |d| {
                if (it.consts[v.id]) |c| return c;
                const out = try it.gpa.alloc(f32, d.len());
                switch (d) {
                    .fp16 => |x| for (out, x) |*o, h| {
                        o.* = @floatCast(h);
                    },
                    .fp32 => |x| {
                        @memcpy(out, x);
                        if (inf.dtype == .fp16 and it.emulate_fp16) roundF16(out);
                    },
                    .int32 => |x| for (out, x) |*o, n| {
                        o.* = @floatFromInt(n);
                    },
                    .bool => |x| for (out, x) |*o, n| {
                        o.* = if (n) 1 else 0;
                    },
                    .string => out[0] = 0,
                }
                it.consts[v.id] = out;
                return out;
            },
            .state => return (it.states.get(v.id) orelse return error.MissingInput).data,
            else => return error.MissingInput,
        }
    }

    fn arg(it: *Interp, o: mil.Op, name: []const u8) Error![]const f32 {
        const a = o.param(name) orelse return error.Unsupported;
        return it.val(a.one);
    }
    fn argV(o: mil.Op, name: []const u8) ?V {
        const a = o.param(name) orelse return null;
        return a.one;
    }
    fn constString(it: *Interp, o: mil.Op, name: []const u8) []const u8 {
        const v = argV(o, name) orelse return "";
        return switch (it.b.info(v).kind) {
            .constant => |d| switch (d) {
                .string => |s| s,
                else => "",
            },
            else => "",
        };
    }
    fn constInts(it: *Interp, o: mil.Op, name: []const u8) []const i32 {
        const v = argV(o, name) orelse return &.{};
        return switch (it.b.info(v).kind) {
            .constant => |d| switch (d) {
                .int32 => |s| s,
                else => &.{},
            },
            else => &.{},
        };
    }
    fn constBool(it: *Interp, o: mil.Op, name: []const u8) bool {
        const v = argV(o, name) orelse return false;
        return switch (it.b.info(v).kind) {
            .constant => |d| switch (d) {
                .bool => |s| s[0],
                else => false,
            },
            else => false,
        };
    }

    fn exec(it: *Interp, o: mil.Op) Error!void {
        const ty = o.type;
        if (std.mem.eql(u8, ty, "const")) return; // read lazily
        if (std.mem.eql(u8, ty, "write_state")) {
            const st = argV(o, "input").?;
            const d = try it.arg(o, "data");
            const t = it.states.getPtr(st.id) orelse return error.MissingInput;
            @memcpy(t.data, d);
            if (it.emulate_fp16) roundF16(t.data);
            return;
        }
        const out_v = o.outputs[0];
        const out_inf = it.b.info(out_v);
        var out = try it.newTensor(out_inf.shape);
        errdefer out.deinit(it.gpa);
        const y = out.data;
        const xv = argV(o, "x");
        const xs = if (xv) |v| it.b.shape(v) else &[_]usize{};

        if (isBinary(ty)) |f| {
            const yv = argV(o, "y").?;
            broadcast(y, out_inf.shape, try it.val(xv.?), xs, try it.val(yv), it.b.shape(yv), f);
        } else if (std.mem.eql(u8, ty, "linear")) {
            const x = try it.val(xv.?);
            const wv = argV(o, "weight").?;
            const w = try it.val(wv);
            const n_out = it.b.shape(wv)[0];
            const n_in = it.b.shape(wv)[1];
            const rows = x.len / n_in;
            gemmNT(y, x, w, rows, n_out, n_in);
            if (argV(o, "bias")) |bv| {
                const bias = try it.val(bv);
                for (0..rows) |r| for (y[r * n_out ..][0..n_out], bias) |*a, c| {
                    a.* += c;
                };
            }
        } else if (std.mem.eql(u8, ty, "matmul")) {
            try it.matmul(o, y, out_inf.shape);
        } else if (std.mem.eql(u8, ty, "layer_norm")) {
            const x = try it.val(xv.?);
            const d = xs[xs.len - 1];
            const axes = it.constInts(o, "axes");
            if (axes.len != 1 or (axes[0] != -1 and axes[0] != @as(i32, @intCast(xs.len - 1)))) return error.Unsupported;
            const g = if (argV(o, "gamma")) |v| try it.val(v) else null;
            const be = if (argV(o, "beta")) |v| try it.val(v) else null;
            const eps = (try it.arg(o, "epsilon"))[0];
            var r: usize = 0;
            while (r < x.len) : (r += d) {
                const row = x[r..][0..d];
                var mean: f64 = 0;
                for (row) |v| mean += v;
                mean /= @floatFromInt(d);
                var vs: f64 = 0;
                for (row) |v| vs += (v - mean) * (v - mean);
                vs /= @floatFromInt(d);
                const inv: f32 = @floatCast(1.0 / @sqrt(vs + eps));
                const m32: f32 = @floatCast(mean);
                for (y[r..][0..d], row, 0..) |*a, v, j| {
                    var t = (v - m32) * inv;
                    if (g) |gg| t *= gg[j];
                    if (be) |bb| t += bb[j];
                    a.* = t;
                }
            }
        } else if (std.mem.eql(u8, ty, "softmax")) {
            const x = try it.val(xv.?);
            const ax = normAxis((try it.arg(o, "axis"))[0], xs.len);
            const oi = outerInner(xs, ax);
            for (0..oi.outer) |a| for (0..oi.inner) |c| {
                const base = a * oi.n * oi.inner + c;
                var mx: f32 = -std.math.inf(f32);
                for (0..oi.n) |k| mx = @max(mx, x[base + k * oi.inner]);
                var s: f32 = 0;
                for (0..oi.n) |k| {
                    const e = @exp(x[base + k * oi.inner] - mx);
                    y[base + k * oi.inner] = e;
                    s += e;
                }
                for (0..oi.n) |k| y[base + k * oi.inner] /= s;
            };
        } else if (std.mem.eql(u8, ty, "gelu")) {
            const x = try it.val(xv.?);
            const mode = it.constString(o, "mode");
            for (y, x) |*a, v| a.* = if (std.mem.eql(u8, mode, "TANH_APPROXIMATION"))
                0.5 * v * (1 + std.math.tanh(0.7978845608028654 * (v + 0.044715 * v * v * v)))
            else if (std.mem.eql(u8, mode, "SIGMOID_APPROXIMATION"))
                v / (1 + @exp(-1.702 * v))
            else
                0.5 * v * (1 + erf(v * 0.7071067811865476));
        } else if (unaryFn(ty)) |f| {
            const x = try it.val(xv.?);
            for (y, x) |*a, v| a.* = f(v);
        } else if (std.mem.eql(u8, ty, "log")) {
            const x = try it.val(xv.?);
            const eps = (try it.arg(o, "epsilon"))[0];
            for (y, x) |*a, v| a.* = @log(v + eps);
        } else if (std.mem.eql(u8, ty, "cast")) {
            const x = try it.val(xv.?);
            const to = out_inf.dtype;
            for (y, x) |*a, v| a.* = switch (to) {
                .int32 => @trunc(v),
                .bool => if (v != 0) 1 else 0,
                else => v,
            };
        } else if (std.mem.eql(u8, ty, "reshape") or std.mem.eql(u8, ty, "identity") or std.mem.eql(u8, ty, "read_state")) {
            const x = if (std.mem.eql(u8, ty, "read_state")) try it.val(argV(o, "input").?) else try it.val(xv.?);
            @memcpy(y, x);
        } else if (std.mem.eql(u8, ty, "transpose")) {
            const x = try it.val(xv.?);
            const perm = it.constInts(o, "perm");
            transposeNd(y, x, xs, perm);
        } else if (std.mem.eql(u8, ty, "concat")) {
            const vs = o.param("values").?.many;
            const ax = normAxis((try it.arg(o, "axis"))[0], out_inf.shape.len);
            const oi = outerInner(out_inf.shape, ax);
            var off: usize = 0;
            for (vs) |v| {
                const d = try it.val(v);
                const n = it.b.shape(v)[ax];
                for (0..oi.outer) |a| {
                    @memcpy(y[(a * oi.n + off) * oi.inner ..][0 .. n * oi.inner], d[a * n * oi.inner ..][0 .. n * oi.inner]);
                }
                off += n;
            }
        } else if (std.mem.eql(u8, ty, "slice_by_size")) {
            const x = try it.val(xv.?);
            const begin = it.constInts(o, "begin");
            sliceNd(y, out_inf.shape, x, xs, begin);
        } else if (std.mem.eql(u8, ty, "conv")) {
            try it.conv(o, y, out_inf.shape);
        } else if (std.mem.startsWith(u8, ty, "reduce_") and !std.mem.eql(u8, ty, "reduce_argmax")) {
            const x = try it.val(xv.?);
            const axes = it.constInts(o, "axes");
            reduceNd(it.gpa, ty, y, x, xs, axes) catch |e| return if (e == error.OutOfMemory) error.OutOfMemory else error.Unsupported;
        } else if (std.mem.eql(u8, ty, "reduce_argmax")) {
            const x = try it.val(xv.?);
            const ax = normAxis((try it.arg(o, "axis"))[0], xs.len);
            const oi = outerInner(xs, ax);
            for (0..oi.outer) |a| for (0..oi.inner) |c| {
                const base = a * oi.n * oi.inner + c;
                var best: usize = 0;
                for (1..oi.n) |k| {
                    if (x[base + k * oi.inner] > x[base + best * oi.inner]) best = k;
                }
                y[a * oi.inner + c] = @floatFromInt(best);
            };
        } else if (std.mem.eql(u8, ty, "gather")) {
            const x = try it.val(xv.?);
            const idx = try it.arg(o, "indices");
            const ax = normAxis((try it.arg(o, "axis"))[0], xs.len);
            const oi = outerInner(xs, ax);
            for (0..oi.outer) |a| for (idx, 0..) |fi, j| {
                const k: usize = @intFromFloat(fi);
                if (k >= oi.n) return error.BadInput;
                @memcpy(y[(a * idx.len + j) * oi.inner ..][0..oi.inner], x[(a * oi.n + k) * oi.inner ..][0..oi.inner]);
            };
        } else {
            std.log.err("ref: op type {s} not implemented", .{ty});
            return error.Unsupported;
        }
        if (out_inf.dtype == .fp16 and it.emulate_fp16) roundF16(y);
        it.vals[out_v.id] = out;
    }

    fn matmul(it: *Interp, o: mil.Op, y: []f32, out_shape: []const usize) Error!void {
        const xv = argV(o, "x").?;
        const yv = argV(o, "y").?;
        const tx = it.constBool(o, "transpose_x");
        const ty = it.constBool(o, "transpose_y");
        const xs = it.b.shape(xv);
        const ys = it.b.shape(yv);
        const xd = try it.val(xv);
        const yd = try it.val(yv);
        const r = out_shape.len;
        const m = out_shape[r - 2];
        const n = out_shape[r - 1];
        const k = if (tx) xs[xs.len - 2] else xs[xs.len - 1];
        const batch_shape = out_shape[0 .. r - 2];
        var nb: usize = 1;
        for (batch_shape) |d| nb *= d;
        // A (m x k) row-major and B as (n x k) row-major ("NT" form) per batch.
        const a_tmp = try it.gpa.alloc(f32, if (tx) m * k else 0);
        defer it.gpa.free(a_tmp);
        const b_tmp = try it.gpa.alloc(f32, if (ty) 0 else n * k);
        defer it.gpa.free(b_tmp);
        var idx = try it.gpa.alloc(usize, batch_shape.len);
        defer it.gpa.free(idx);
        @memset(idx, 0);
        for (0..nb) |bi| {
            const xo = batchOffset(idx, batch_shape, xs[0 .. xs.len - 2]) * m * k;
            const yo = batchOffset(idx, batch_shape, ys[0 .. ys.len - 2]) * n * k;
            var a = xd[xo..][0 .. m * k];
            if (tx) { // stored k x m
                for (0..k) |p| for (0..m) |i| {
                    a_tmp[i * k + p] = a[p * m + i];
                };
                a = a_tmp;
            }
            var bm = yd[yo..][0 .. n * k];
            if (!ty) { // stored k x n -> n x k
                for (0..k) |p| for (0..n) |j| {
                    b_tmp[j * k + p] = bm[p * n + j];
                };
                bm = b_tmp;
            }
            gemmNT(y[bi * m * n ..][0 .. m * n], a, bm, m, n, k);
            // next batch index
            var d = batch_shape.len;
            while (d > 0) {
                d -= 1;
                idx[d] += 1;
                if (idx[d] < batch_shape[d]) break;
                idx[d] = 0;
            }
        }
    }

    fn conv(it: *Interp, o: mil.Op, y: []f32, out_shape: []const usize) Error!void {
        const xv = argV(o, "x").?;
        const wv = argV(o, "weight").?;
        const xs = it.b.shape(xv);
        const ws = it.b.shape(wv);
        if (xs.len != 3) return error.Unsupported;
        const x = try it.val(xv);
        const w = try it.val(wv);
        const stride: usize = @intCast(it.constInts(o, "strides")[0]);
        const pad = it.constInts(o, "pad");
        const pl: usize = @intCast(pad[0]);
        const c_in = xs[1];
        const width = xs[2];
        const c_out = ws[0];
        const kk = ws[2];
        const w_out = out_shape[2];
        const col = try it.gpa.alloc(f32, w_out * c_in * kk);
        defer it.gpa.free(col);
        const tmp = try it.gpa.alloc(f32, w_out * c_out);
        defer it.gpa.free(tmp);
        const bias = if (argV(o, "bias")) |bv| try it.val(bv) else null;
        for (0..xs[0]) |nb| {
            const xb = x[nb * c_in * width ..][0 .. c_in * width];
            for (0..w_out) |t| for (0..c_in) |ci| for (0..kk) |k| {
                const pos = @as(isize, @intCast(t * stride + k)) - @as(isize, @intCast(pl));
                col[(t * c_in + ci) * kk + k] = if (pos >= 0 and pos < width) xb[ci * width + @as(usize, @intCast(pos))] else 0;
            };
            gemmNT(tmp, col, w, w_out, c_out, c_in * kk);
            const yb = y[nb * c_out * w_out ..][0 .. c_out * w_out];
            for (0..c_out) |co| for (0..w_out) |t| {
                yb[co * w_out + t] = tmp[t * c_out + co] + if (bias) |bb| bb[co] else 0;
            };
        }
    }
};

/// Rounds every value to the nearest fp16 (in place).
pub fn roundF16(xs: []f32) void {
    for (xs) |*x| x.* = @floatCast(@as(f16, @floatCast(x.*)));
}

/// C (m x n) = A (m x k) · Bᵀ with B stored n x k; both row-major.
pub fn gemmNT(c: []f32, a: []const f32, b: []const f32, m: usize, n: usize, k: usize) void {
    const L = 8;
    const Vec = @Vector(L, f32);
    const kv = k / L * L;
    var i: usize = 0;
    while (i < m) : (i += 1) {
        const ar = a[i * k ..][0..k];
        var j: usize = 0;
        // 4 output columns at a time share the loads of A's row.
        while (j + 4 <= n) : (j += 4) {
            const b0 = b[j * k ..][0..k];
            const b1 = b[(j + 1) * k ..][0..k];
            const b2 = b[(j + 2) * k ..][0..k];
            const b3 = b[(j + 3) * k ..][0..k];
            var s0: Vec = @splat(0);
            var s1: Vec = @splat(0);
            var s2: Vec = @splat(0);
            var s3: Vec = @splat(0);
            var p: usize = 0;
            while (p < kv) : (p += L) {
                const av: Vec = ar[p..][0..L].*;
                s0 += av * @as(Vec, b0[p..][0..L].*);
                s1 += av * @as(Vec, b1[p..][0..L].*);
                s2 += av * @as(Vec, b2[p..][0..L].*);
                s3 += av * @as(Vec, b3[p..][0..L].*);
            }
            var t0 = @reduce(.Add, s0);
            var t1 = @reduce(.Add, s1);
            var t2 = @reduce(.Add, s2);
            var t3 = @reduce(.Add, s3);
            while (p < k) : (p += 1) {
                t0 += ar[p] * b0[p];
                t1 += ar[p] * b1[p];
                t2 += ar[p] * b2[p];
                t3 += ar[p] * b3[p];
            }
            c[i * n + j] = t0;
            c[i * n + j + 1] = t1;
            c[i * n + j + 2] = t2;
            c[i * n + j + 3] = t3;
        }
        while (j < n) : (j += 1) {
            const br = b[j * k ..][0..k];
            var s: Vec = @splat(0);
            var p: usize = 0;
            while (p < kv) : (p += L) s += @as(Vec, ar[p..][0..L].*) * @as(Vec, br[p..][0..L].*);
            var t = @reduce(.Add, s);
            while (p < k) : (p += 1) t += ar[p] * br[p];
            c[i * n + j] = t;
        }
    }
}

fn batchOffset(idx: []const usize, out_batch: []const usize, in_batch: []const usize) usize {
    // in_batch is right-aligned against out_batch; broadcast dims (1) stay 0.
    var off: usize = 0;
    const lead = out_batch.len - in_batch.len;
    for (in_batch, 0..) |d, i| {
        const ix = if (d == 1) 0 else idx[lead + i];
        off = off * d + ix;
    }
    return off;
}

const OuterInner = struct { outer: usize, n: usize, inner: usize };
fn outerInner(shape: []const usize, axis: usize) OuterInner {
    var r: OuterInner = .{ .outer = 1, .n = shape[axis], .inner = 1 };
    for (shape[0..axis]) |d| r.outer *= d;
    for (shape[axis + 1 ..]) |d| r.inner *= d;
    return r;
}

fn normAxis(a: f32, r: usize) usize {
    const i: isize = @intFromFloat(a);
    return @intCast(if (i < 0) i + @as(isize, @intCast(r)) else i);
}

const BinFn = *const fn (f32, f32) f32;
fn isBinary(ty: []const u8) ?BinFn {
    const F = struct {
        fn add(a: f32, b: f32) f32 {
            return a + b;
        }
        fn sub(a: f32, b: f32) f32 {
            return a - b;
        }
        fn mul(a: f32, b: f32) f32 {
            return a * b;
        }
        fn div(a: f32, b: f32) f32 {
            return a / b;
        }
        fn max(a: f32, b: f32) f32 {
            return @max(a, b);
        }
        fn min(a: f32, b: f32) f32 {
            return @min(a, b);
        }
        fn pow(a: f32, b: f32) f32 {
            return std.math.pow(f32, a, b);
        }
    };
    const table = .{
        .{ "add", F.add }, .{ "sub", F.sub },     .{ "mul", F.mul },     .{ "real_div", F.div },
        .{ "maximum", F.max }, .{ "minimum", F.min }, .{ "pow", F.pow },
    };
    inline for (table) |e| if (std.mem.eql(u8, ty, e[0])) return e[1];
    return null;
}

const UnFn = *const fn (f32) f32;
fn unaryFn(ty: []const u8) ?UnFn {
    const F = struct {
        fn sigmoid(x: f32) f32 {
            return 1 / (1 + @exp(-x));
        }
        fn tanh(x: f32) f32 {
            return std.math.tanh(x);
        }
        fn exp(x: f32) f32 {
            return @exp(x);
        }
        fn sqrt(x: f32) f32 {
            return @sqrt(x);
        }
        fn rsqrt(x: f32) f32 {
            return 1 / @sqrt(x);
        }
        fn relu(x: f32) f32 {
            return @max(x, 0);
        }
        fn abs(x: f32) f32 {
            return @abs(x);
        }
        fn erf_(x: f32) f32 {
            return erf(x);
        }
    };
    const table = .{
        .{ "sigmoid", F.sigmoid }, .{ "tanh", F.tanh }, .{ "exp", F.exp }, .{ "sqrt", F.sqrt },
        .{ "rsqrt", F.rsqrt },     .{ "relu", F.relu }, .{ "abs", F.abs }, .{ "erf", F.erf_ },
    };
    inline for (table) |e| if (std.mem.eql(u8, ty, e[0])) return e[1];
    return null;
}

/// erf(x), max error about 1.2e-7 (Numerical Recipes erfc Chebyshev fit).
pub fn erf(x: f32) f32 {
    const z: f64 = @abs(@as(f64, x));
    const t = 1.0 / (1.0 + 0.5 * z);
    const r = t * @exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
        t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
        t * (-0.82215223 + t * 0.17087277)))))))));
    const v: f32 = @floatCast(1.0 - r);
    return if (x >= 0) v else -v;
}

fn broadcast(y: []f32, out_shape: []const usize, a: []const f32, as: []const usize, b: []const f32, bs: []const usize, f: BinFn) void {
    if (a.len == y.len and b.len == y.len) {
        for (y, a, b) |*o, p, q| o.* = f(p, q);
        return;
    }
    if (a.len == y.len and b.len == 1) {
        for (y, a) |*o, p| o.* = f(p, b[0]);
        return;
    }
    if (b.len == y.len and a.len == 1) {
        for (y, b) |*o, q| o.* = f(a[0], q);
        return;
    }
    // a matches y, b broadcasts along leading dims (bias rows).
    if (a.len == y.len and bs.len <= out_shape.len) {
        var trailing = true;
        const lead = out_shape.len - bs.len;
        var first_real: usize = 0;
        while (first_real < bs.len and bs[first_real] == 1) first_real += 1;
        for (bs[first_real..], out_shape[lead + first_real ..]) |d, o| if (d != o) {
            trailing = false;
        };
        if (trailing and b.len > 0 and y.len % b.len == 0) {
            var i: usize = 0;
            while (i < y.len) : (i += b.len) for (y[i..][0..b.len], a[i..][0..b.len], b) |*o, p, q| {
                o.* = f(p, q);
            };
            return;
        }
    }
    // general case
    const r = out_shape.len;
    var sa: [8]usize = @splat(0);
    var sb: [8]usize = @splat(0);
    strides(&sa, as, r);
    strides(&sb, bs, r);
    var idx: [8]usize = @splat(0);
    for (y) |*o| {
        var ia: usize = 0;
        var ib: usize = 0;
        for (0..r) |d| {
            ia += idx[d] * sa[d];
            ib += idx[d] * sb[d];
        }
        o.* = f(a[ia], b[ib]);
        var d = r;
        while (d > 0) {
            d -= 1;
            idx[d] += 1;
            if (idx[d] < out_shape[d]) break;
            idx[d] = 0;
        }
    }
}

/// Element strides of `shape` right-aligned in rank `r`, 0 for broadcast dims.
fn strides(out: *[8]usize, shape: []const usize, r: usize) void {
    var s: usize = 1;
    var i = shape.len;
    while (i > 0) {
        i -= 1;
        out[r - shape.len + i] = if (shape[i] == 1) 0 else s;
        s *= shape[i];
    }
}

fn transposeNd(y: []f32, x: []const f32, xs: []const usize, perm: []const i32) void {
    const r = xs.len;
    var in_strides: [8]usize = undefined;
    var s: usize = 1;
    var i = r;
    while (i > 0) {
        i -= 1;
        in_strides[i] = s;
        s *= xs[i];
    }
    var out_shape: [8]usize = undefined;
    var st: [8]usize = undefined;
    for (0..r) |d| {
        const pd: usize = @intCast(perm[d]);
        out_shape[d] = xs[pd];
        st[d] = in_strides[pd];
    }
    // innermost output dim as the fast loop
    var idx: [8]usize = @splat(0);
    const inner = out_shape[r - 1];
    const inner_st = st[r - 1];
    var o: usize = 0;
    while (o < y.len) : (o += inner) {
        var base: usize = 0;
        for (0..r - 1) |d| base += idx[d] * st[d];
        for (0..inner) |k| y[o + k] = x[base + k * inner_st];
        var d = r - 1;
        while (d > 0) {
            d -= 1;
            idx[d] += 1;
            if (idx[d] < out_shape[d]) break;
            idx[d] = 0;
        }
    }
}

fn sliceNd(y: []f32, ys: []const usize, x: []const f32, xs: []const usize, begin: []const i32) void {
    const r = xs.len;
    var in_strides: [8]usize = undefined;
    var s: usize = 1;
    var i = r;
    while (i > 0) {
        i -= 1;
        in_strides[i] = s;
        s *= xs[i];
    }
    var idx: [8]usize = @splat(0);
    const inner = ys[r - 1];
    var o: usize = 0;
    while (o < y.len) : (o += inner) {
        var base: usize = 0;
        for (0..r) |d| base += (idx[d] + @as(usize, @intCast(begin[d]))) * in_strides[d];
        @memcpy(y[o..][0..inner], x[base..][0..inner]);
        if (r == 1) break;
        var d = r - 1;
        while (d > 0) {
            d -= 1;
            idx[d] += 1;
            if (idx[d] < ys[d]) break;
            idx[d] = 0;
        }
    }
}

fn reduceNd(gpa: Allocator, ty: []const u8, y: []f32, x: []const f32, xs: []const usize, axes: []const i32) !void {
    const r = xs.len;
    var reduced: [8]bool = @splat(false);
    for (axes) |a| reduced[normAxis(@floatFromInt(a), r)] = true;
    const Kind = enum { max, sum, mean, lse };
    const k: Kind = if (std.mem.eql(u8, ty, "reduce_max")) .max else if (std.mem.eql(u8, ty, "reduce_sum")) .sum else if (std.mem.eql(u8, ty, "reduce_mean")) .mean else if (std.mem.eql(u8, ty, "reduce_log_sum_exp")) .lse else return error.Unsupported;
    // output strides over the kept dims
    var os: [8]usize = @splat(0);
    var s: usize = 1;
    var i = r;
    while (i > 0) {
        i -= 1;
        if (!reduced[i]) {
            os[i] = s;
            s *= xs[i];
        }
    }
    const init: f32 = if (k == .max) -std.math.inf(f32) else 0;
    @memset(y, init);
    var mx: []f32 = &.{};
    // log-sum-exp: two passes (max, then sum of exp)
    var idx: [8]usize = @splat(0);
    if (k == .lse) {
        // first pass: max into y
        iterate(x, xs, &idx, &os, y, .max);
        mx = try gpa.dupe(f32, y);
        defer gpa.free(mx);
        @memset(y, 0);
        idx = @splat(0);
        for (x) |v| {
            var o: usize = 0;
            for (0..r) |d| o += idx[d] * os[d];
            y[o] += @exp(v - mx[o]);
            advance(&idx, xs);
        }
        for (y, mx) |*a, m| a.* = @log(a.*) + m;
        return;
    }
    if (k == .max) iterate(x, xs, &idx, &os, y, .max) else iterate(x, xs, &idx, &os, y, .sum);
    if (k == .mean) {
        const n: f32 = @floatFromInt(x.len / y.len);
        for (y) |*a| a.* /= n;
    }
}

fn iterate(x: []const f32, xs: []const usize, idx: *[8]usize, os: *const [8]usize, y: []f32, comptime mode: enum { max, sum }) void {
    for (x) |v| {
        var o: usize = 0;
        for (0..xs.len) |d| o += idx[d] * os[d];
        y[o] = if (mode == .max) @max(y[o], v) else y[o] + v;
        advance(idx, xs);
    }
}

fn advance(idx: *[8]usize, shape: []const usize) void {
    var d = shape.len;
    while (d > 0) {
        d -= 1;
        idx[d] += 1;
        if (idx[d] < shape[d]) return;
        idx[d] = 0;
    }
}

// ------------------------------------------------------------------- tests

const testing = std.testing;

test "linear + bias, broadcast add, softmax" {
    var b = try Builder.init(testing.allocator, .{});
    defer b.deinit();
    const x = try b.input("x", .fp32, &.{ 2, 3 });
    const w = try b.constFloat(.fp32, &.{ 2, 3 }, &.{ 1, 0, 0, 0, 1, 1 });
    const bias = try b.constFloat(.fp32, &.{2}, &.{ 0.5, -0.5 });
    const y = try b.linear(x, w, bias);
    const s = try b.softmax(y, -1);
    try b.output("y", y);
    try b.output("s", s);
    var it = try Interp.init(testing.allocator, &b);
    defer it.deinit();
    try it.run(&.{.{ .name = "x", .shape = &.{ 2, 3 }, .data = &.{ 1, 2, 3, 4, 5, 6 } }});
    const yo = it.output("y").?;
    try testing.expectEqualSlices(f32, &.{ 1.5, 4.5, 4.5, 10.5 }, yo.data);
    const so = it.output("s").?;
    try testing.expectApproxEqAbs(@as(f32, 1.0), so.data[0] + so.data[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 1.0 / (1.0 + @exp(3.0))), so.data[0], 1e-6);
}

test "matmul with transposes and batch broadcast" {
    var b = try Builder.init(testing.allocator, .{});
    defer b.deinit();
    const x = try b.input("x", .fp32, &.{ 2, 2, 3 }); // batch 2
    const yv = try b.input("y", .fp32, &.{ 1, 4, 3 }); // broadcast batch
    const z = try b.matmul(x, yv, false, true); // [2][2][4]
    try testing.expectError(error.BadShape, b.matmul(yv, x, true, true)); // [3x4]·[3x2]: k mismatch
    try b.output("z", z);
    var it = try Interp.init(testing.allocator, &b);
    defer it.deinit();
    const xd = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const yd = [_]f32{ 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1 };
    try it.run(&.{ .{ .name = "x", .shape = &.{ 2, 2, 3 }, .data = &xd }, .{ .name = "y", .shape = &.{ 1, 4, 3 }, .data = &yd } });
    const zo = it.output("z").?;
    try testing.expectEqualSlices(f32, &.{ 1, 2, 3, 6, 4, 5, 6, 15, 7, 8, 9, 24, 10, 11, 12, 33 }, zo.data);
}

test "conv1d, transpose, concat, slice, reductions, gather, argmax" {
    var b = try Builder.init(testing.allocator, .{ .target = .ios17 });
    defer b.deinit();
    const x = try b.input("x", .fp32, &.{ 1, 1, 5 });
    const w = try b.constFloat(.fp32, &.{ 2, 1, 3 }, &.{ 1, 1, 1, 0, 1, 0 });
    const c = try b.conv1d(x, w, null, 2, .{ 1, 1 }); // width (5+2-3)/2+1 = 3
    try b.output("c", c);
    const t = try b.transpose(c, &.{ 0, 2, 1 }); // [1][3][2]
    try b.output("t", t);
    const cc = try b.concat(&.{ t, t }, 2); // [1][3][4]
    try b.output("cc", try b.sliceBySize(cc, &.{ 0, 1, 1 }, &.{ 1, 2, 2 }));
    try b.output("mx", try b.reduce("reduce_max", c, &.{2}, false));
    try b.output("lse", try b.reduce("reduce_log_sum_exp", c, &.{2}, true));
    const table = try b.constFloat(.fp32, &.{ 3, 2 }, &.{ 0, 1, 10, 11, 20, 21 });
    const idx = try b.constI32Owned(&.{2}, &.{ 2, 0 });
    try b.output("g", try b.gather(table, idx, 0));
    try b.output("am", try b.argmax(c, 2, false));
    var it = try Interp.init(testing.allocator, &b);
    defer it.deinit();
    try it.run(&.{.{ .name = "x", .shape = &.{ 1, 1, 5 }, .data = &.{ 1, 2, 3, 4, 5 } }});
    // padded: 0 1 2 3 4 5 0; windows at 0,2,4: [0 1 2] [2 3 4] [4 5 0]
    try testing.expectEqualSlices(f32, &.{ 3, 9, 9, 1, 3, 5 }, it.output("c").?.data);
    try testing.expectEqualSlices(f32, &.{ 3, 1, 9, 3, 9, 5 }, it.output("t").?.data);
    try testing.expectEqualSlices(f32, &.{ 3, 9, 5, 9 }, it.output("cc").?.data);
    try testing.expectEqualSlices(f32, &.{ 9, 5 }, it.output("mx").?.data);
    const lse = it.output("lse").?.data;
    try testing.expectApproxEqAbs(@as(f32, @log(@exp(3.0) + 2 * @exp(9.0))), lse[0], 1e-4);
    try testing.expectEqualSlices(f32, &.{ 20, 21, 0, 1 }, it.output("g").?.data);
    try testing.expectEqualSlices(f32, &.{ 1, 2 }, it.output("am").?.data);
}

test "layer norm, gelu, fp16 rounding" {
    var b = try Builder.init(testing.allocator, .{});
    defer b.deinit();
    const x = try b.input("x", .fp32, &.{ 1, 4 });
    const h = try b.cast(x, .fp16);
    const g = try b.constFloat(.fp16, &.{4}, &.{ 1, 1, 1, 1 });
    const be = try b.constFloat(.fp16, &.{4}, &.{ 0, 0, 0, 0 });
    const n = try b.layerNorm(h, g, be, 1e-5);
    try b.output("n", try b.cast(n, .fp32));
    try b.output("e", try b.cast(try b.gelu(h, .exact), .fp32));
    var it = try Interp.init(testing.allocator, &b);
    defer it.deinit();
    try it.run(&.{.{ .name = "x", .shape = &.{ 1, 4 }, .data = &.{ 1, 2, 3, 4.0001 } }});
    const nd = it.output("n").?.data;
    try testing.expectApproxEqAbs(@as(f32, -1.3416), nd[0], 1e-3);
    const e = it.output("e").?.data;
    try testing.expectApproxEqAbs(@as(f32, 0.8413), e[0], 1e-3);
    // fp16: 4.0001 rounds to 4, and the gelu result is rounded to fp16
    const g4: f32 = 4.0 * 0.5 * (1 + erf(4.0 * 0.7071067811865476));
    try testing.expectEqual(@as(f32, @floatCast(@as(f16, @floatCast(g4)))), e[3]);
}

test "erf" {
    try testing.expectApproxEqAbs(@as(f32, 0.8427007), erf(1.0), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, -0.5204999), erf(-0.5), 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0), erf(0), 1e-7);
}
