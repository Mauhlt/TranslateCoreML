//! Builder for Core ML "ML Program" (MIL) models: a single `main` function
//! made op by op, with static shapes tracked here so every op's output
//! type is known (Core ML checks them), then serialized to the Model
//! protobuf (spec.zig) plus the weight blob file (blob.zig).
//!
//!     var b = try mil.Builder.init(gpa, .{ .target = .ios17 });
//!     defer b.deinit();
//!     const x = try b.input("x", .fp32, &.{ 1, 64 });
//!     const h = try b.cast(x, .fp16);
//!     const y = try b.linear(h, try b.constF16(&.{ 32, 64 }, w), try b.constF16(&.{32}, bias));
//!     try b.output("y", try b.cast(y, .fp32));
//!     try package.write(gpa, io, dir, "model.mlpackage", &b);
//!
//! Op names and parameters follow the MIL op set (coremltools
//! converters/mil/mil/ops/defs/iOS15..iOS18, BSD-3-Clause; re-derived, not
//! copied). Every parameter that has a non-null default is written
//! explicitly, as coremltools does. Values are immutable SSA: each op
//! helper returns its output value. The same graph can be run on the host
//! by ref.zig (a reference interpreter) to get expected outputs.
//!
//! Data passed to the const helpers is borrowed, not copied: it must stay
//! alive until the builder is serialized (the `*Owned` helpers copy into
//! the builder's arena).
const std = @import("std");
const Allocator = std.mem.Allocator;
const pb = @import("pb.zig");
const spec = @import("spec.zig");
const blob = @import("blob.zig");

pub const Target = spec.Target;

pub const DType = enum {
    fp16,
    fp32,
    int32,
    bool,
    string,

    pub fn mil(d: DType) spec.DataType {
        return switch (d) {
            .fp16 => .float16,
            .fp32 => .float32,
            .int32 => .int32,
            .bool => .bool,
            .string => .string,
        };
    }
    /// Model input/output feature type (MLMultiArray data type).
    pub fn array(d: DType) ?spec.ArrayDataType {
        return switch (d) {
            .fp16 => .float16,
            .fp32 => .float32,
            .int32 => .int32,
            else => null,
        };
    }
    pub fn name(d: DType) []const u8 {
        return switch (d) {
            .fp16 => "fp16",
            .fp32 => "fp32",
            .int32 => "int32",
            .bool => "bool",
            .string => "string",
        };
    }
    pub fn isFloat(d: DType) bool {
        return d == .fp16 or d == .fp32;
    }
};

/// Constant data (borrowed).
pub const Data = union(enum) {
    fp16: []const f16,
    fp32: []const f32,
    int32: []const i32,
    bool: []const bool,
    string: []const u8,

    pub fn len(d: Data) usize {
        return switch (d) {
            .string => 1,
            inline else => |s| s.len,
        };
    }
    pub fn dtype(d: Data) DType {
        return switch (d) {
            .fp16 => .fp16,
            .fp32 => .fp32,
            .int32 => .int32,
            .bool => .bool,
            .string => .string,
        };
    }
};

/// A value of the program (an SSA variable).
pub const V = struct { id: u32 };

pub const Kind = union(enum) {
    input,
    /// iOS 18 state (MLState): a function input of state type.
    state,
    output_of_op: u32,
    constant: Data,
};

pub const Info = struct {
    name: []const u8,
    dtype: DType,
    shape: []const usize,
    kind: Kind,

    pub fn count(i: Info) usize {
        var n: usize = 1;
        for (i.shape) |d| n *= d;
        return n;
    }
};

pub const Arg = union(enum) {
    one: V,
    /// Variadic parameter (concat's `values`).
    many: []const V,
};

pub const Param = struct { name: []const u8, arg: Arg };

pub const Op = struct {
    type: []const u8,
    name: []const u8,
    params: []const Param,
    outputs: []const V,

    pub fn param(op: Op, n: []const u8) ?Arg {
        for (op.params) |p| if (std.mem.eql(u8, p.name, n)) return p.arg;
        return null;
    }
};

pub const Error = error{ OutOfMemory, BadShape, BadType, BadName };

pub const Options = struct {
    target: Target = .ios17,
    /// Consts with at least this many elements (fp16/fp32) go to the weight
    /// file; smaller ones are immediate values (coremltools uses 10).
    blob_threshold: usize = 10,
    /// Metadata.shortDescription / author (optional).
    description: []const u8 = "",
    author: []const u8 = "",
};

pub const GeluMode = enum {
    exact,
    tanh_approximation,
    sigmoid_approximation,

    pub fn str(m: GeluMode) []const u8 {
        return switch (m) {
            .exact => "EXACT",
            .tanh_approximation => "TANH_APPROXIMATION",
            .sigmoid_approximation => "SIGMOID_APPROXIMATION",
        };
    }
};

pub const Builder = struct {
    gpa: Allocator,
    arena_state: *std.heap.ArenaAllocator,
    arena: Allocator,
    opts: Options,
    values: std.ArrayList(Info) = .empty,
    ops: std.ArrayList(Op) = .empty,
    inputs: std.ArrayList(V) = .empty,
    outputs: std.ArrayList(V) = .empty,
    n_auto: u32 = 0,

    pub fn init(gpa: Allocator, opts: Options) !Builder {
        const a = try gpa.create(std.heap.ArenaAllocator);
        a.* = .init(gpa);
        return .{ .gpa = gpa, .arena_state = a, .arena = a.allocator(), .opts = opts };
    }

    pub fn deinit(b: *Builder) void {
        b.arena_state.deinit();
        b.gpa.destroy(b.arena_state);
        b.* = undefined;
    }

    pub fn info(b: *const Builder, v: V) Info {
        return b.values.items[v.id];
    }
    pub fn shape(b: *const Builder, v: V) []const usize {
        return b.values.items[v.id].shape;
    }
    pub fn dtype(b: *const Builder, v: V) DType {
        return b.values.items[v.id].dtype;
    }
    pub fn rank(b: *const Builder, v: V) usize {
        return b.values.items[v.id].shape.len;
    }

    fn checkName(n: []const u8) Error!void {
        if (n.len == 0) return error.BadName;
        for (n, 0..) |c, i| {
            const ok = std.ascii.isAlphabetic(c) or c == '_' or (i > 0 and (std.ascii.isDigit(c) or c == '@'));
            if (!ok) return error.BadName;
        }
    }

    fn autoName(b: *Builder, prefix: []const u8) Error![]const u8 {
        b.n_auto += 1;
        return std.fmt.allocPrint(b.arena, "{s}_{d}", .{ prefix, b.n_auto });
    }

    fn addValue(b: *Builder, name: []const u8, dt: DType, shp: []const usize, kind: Kind) Error!V {
        const id: u32 = @intCast(b.values.items.len);
        try b.values.append(b.arena, .{ .name = name, .dtype = dt, .shape = try b.arena.dupe(usize, shp), .kind = kind });
        return .{ .id = id };
    }

    // ------------------------------------------------------------ inputs

    /// A model input (an MLMultiArray of `dt`: fp16, fp32 or int32).
    pub fn input(b: *Builder, name: []const u8, dt: DType, shp: []const usize) Error!V {
        try checkName(name);
        if (dt.array() == null) return error.BadType;
        const v = try b.addValue(try b.arena.dupe(u8, name), dt, shp, .input);
        try b.inputs.append(b.arena, v);
        return v;
    }

    /// An iOS 18 state (fp16): read with `readState`, written with `writeState`.
    pub fn state(b: *Builder, name: []const u8, dt: DType, shp: []const usize) Error!V {
        try checkName(name);
        if (b.opts.target != .ios18) return error.BadType;
        if (dt != .fp16) return error.BadType; // Core ML states are fp16
        const v = try b.addValue(try b.arena.dupe(u8, name), dt, shp, .state);
        try b.inputs.append(b.arena, v);
        return v;
    }

    /// Makes `v` a model output called `name` (an identity op renames it
    /// if needed).
    pub fn output(b: *Builder, name: []const u8, v: V) Error!void {
        try checkName(name);
        const i = b.info(v);
        if (i.dtype.array() == null) return error.BadType;
        const out = if (std.mem.eql(u8, i.name, name) and i.kind == .output_of_op)
            v
        else
            try b.opNamed("identity", name, &.{.{ .name = "x", .arg = .{ .one = v } }}, i.dtype, i.shape);
        try b.outputs.append(b.arena, out);
    }

    // ------------------------------------------------------------ consts

    pub fn constant(b: *Builder, shp: []const usize, data: Data) Error!V {
        var n: usize = 1;
        for (shp) |d| n *= d;
        if (data.len() != n) return error.BadShape;
        const name = try b.autoName("c");
        const v = try b.addValue(name, data.dtype(), shp, .{ .constant = data });
        // The const op that defines it.
        const outs = try b.arena.alloc(V, 1);
        outs[0] = v;
        try b.ops.append(b.arena, .{ .type = "const", .name = name, .params = &.{}, .outputs = outs });
        return v;
    }
    pub fn constF16(b: *Builder, shp: []const usize, d: []const f16) Error!V {
        return b.constant(shp, .{ .fp16 = d });
    }
    pub fn constF32(b: *Builder, shp: []const usize, d: []const f32) Error!V {
        return b.constant(shp, .{ .fp32 = d });
    }
    pub fn constI32(b: *Builder, shp: []const usize, d: []const i32) Error!V {
        return b.constant(shp, .{ .int32 = d });
    }
    /// A copy of `d` converted to `dt` (fp16 or fp32), owned by the builder.
    pub fn constFloat(b: *Builder, dt: DType, shp: []const usize, d: []const f32) Error!V {
        return switch (dt) {
            .fp32 => b.constant(shp, .{ .fp32 = try b.arena.dupe(f32, d) }),
            .fp16 => blk: {
                const h = try b.arena.alloc(f16, d.len);
                for (h, d) |*o, x| o.* = @floatCast(x);
                break :blk b.constant(shp, .{ .fp16 = h });
            },
            else => error.BadType,
        };
    }
    pub fn constI32Owned(b: *Builder, shp: []const usize, d: []const i32) Error!V {
        return b.constant(shp, .{ .int32 = try b.arena.dupe(i32, d) });
    }
    /// A rank-1 int32 tensor (shapes, axes, permutations).
    pub fn ints(b: *Builder, d: []const i32) Error!V {
        return b.constI32Owned(&.{d.len}, d);
    }
    pub fn intsFromUsize(b: *Builder, d: []const usize) Error!V {
        const x = try b.arena.alloc(i32, d.len);
        for (x, d) |*o, v| o.* = @intCast(v);
        return b.constant(&.{d.len}, .{ .int32 = x });
    }
    pub fn scalarFloat(b: *Builder, dt: DType, x: f32) Error!V {
        return b.constFloat(dt, &.{}, &.{x});
    }
    pub fn scalarI32(b: *Builder, x: i32) Error!V {
        return b.constI32Owned(&.{}, &.{x});
    }
    pub fn scalarBool(b: *Builder, x: bool) Error!V {
        const d = try b.arena.alloc(bool, 1);
        d[0] = x;
        return b.constant(&.{}, .{ .bool = d });
    }
    pub fn string(b: *Builder, s: []const u8) Error!V {
        return b.constant(&.{}, .{ .string = try b.arena.dupe(u8, s) });
    }

    // ------------------------------------------------------------ ops

    fn opNamed(b: *Builder, ty: []const u8, out_name: []const u8, params: []const Param, dt: DType, shp: []const usize) Error!V {
        const op_index: u32 = @intCast(b.ops.items.len);
        const name = try b.arena.dupe(u8, out_name);
        const v = try b.addValue(name, dt, shp, .{ .output_of_op = op_index });
        const outs = try b.arena.alloc(V, 1);
        outs[0] = v;
        try b.ops.append(b.arena, .{ .type = ty, .name = name, .params = try b.arena.dupe(Param, params), .outputs = outs });
        return v;
    }

    fn op(b: *Builder, ty: []const u8, params: []const Param, dt: DType, shp: []const usize) Error!V {
        return b.opNamed(ty, try b.autoName(ty), params, dt, shp);
    }

    fn p(name: []const u8, v: V) Param {
        return .{ .name = name, .arg = .{ .one = v } };
    }

    /// Element-wise binary op with numpy broadcasting: add, sub, mul,
    /// real_div, maximum, minimum, pow.
    pub fn binary(b: *Builder, ty: []const u8, x: V, y: V) Error!V {
        if (b.dtype(x) != b.dtype(y)) return error.BadType;
        const s = try broadcastShape(b.arena, b.shape(x), b.shape(y));
        return b.op(ty, &.{ p("x", x), p("y", y) }, b.dtype(x), s);
    }
    pub fn add(b: *Builder, x: V, y: V) Error!V {
        return b.binary("add", x, y);
    }
    pub fn sub(b: *Builder, x: V, y: V) Error!V {
        return b.binary("sub", x, y);
    }
    pub fn mul(b: *Builder, x: V, y: V) Error!V {
        return b.binary("mul", x, y);
    }
    pub fn div(b: *Builder, x: V, y: V) Error!V {
        return b.binary("real_div", x, y);
    }
    pub fn maximum(b: *Builder, x: V, y: V) Error!V {
        return b.binary("maximum", x, y);
    }
    pub fn minimum(b: *Builder, x: V, y: V) Error!V {
        return b.binary("minimum", x, y);
    }
    /// x (op) scalar in x's type.
    pub fn addScalar(b: *Builder, x: V, s: f32) Error!V {
        return b.add(x, try b.scalarFloat(b.dtype(x), s));
    }
    pub fn mulScalar(b: *Builder, x: V, s: f32) Error!V {
        return b.mul(x, try b.scalarFloat(b.dtype(x), s));
    }
    pub fn maxScalar(b: *Builder, x: V, s: f32) Error!V {
        return b.maximum(x, try b.scalarFloat(b.dtype(x), s));
    }

    /// Element-wise unary op without parameters: sigmoid, tanh, exp, sqrt,
    /// relu, erf, abs, identity, ...
    pub fn unary(b: *Builder, ty: []const u8, x: V) Error!V {
        return b.op(ty, &.{p("x", x)}, b.dtype(x), b.shape(x));
    }
    pub fn sigmoid(b: *Builder, x: V) Error!V {
        return b.unary("sigmoid", x);
    }
    /// log(x + epsilon).
    pub fn log(b: *Builder, x: V, epsilon: f32) Error!V {
        return b.op("log", &.{ p("x", x), p("epsilon", try b.scalarFloat(b.dtype(x), epsilon)) }, b.dtype(x), b.shape(x));
    }

    /// y = x · weightᵀ + bias; weight [out][in] (const), x [..., in].
    pub fn linear(b: *Builder, x: V, weight: V, bias: ?V) Error!V {
        const xs = b.shape(x);
        const ws = b.shape(weight);
        if (ws.len != 2 or xs.len == 0 or xs[xs.len - 1] != ws[1]) return error.BadShape;
        if (b.info(weight).kind != .constant) return error.BadType;
        const s = try b.arena.dupe(usize, xs);
        s[s.len - 1] = ws[0];
        var params: [3]Param = .{ p("x", x), p("weight", weight), undefined };
        var n: usize = 2;
        if (bias) |bv| {
            if (b.shape(bv).len != 1 or b.shape(bv)[0] != ws[0]) return error.BadShape;
            params[2] = p("bias", bv);
            n = 3;
        }
        return b.op("linear", params[0..n], b.dtype(x), s);
    }

    /// Batched matrix product over the last two dims (batch dims broadcast).
    pub fn matmul(b: *Builder, x: V, y: V, transpose_x: bool, transpose_y: bool) Error!V {
        const xs = b.shape(x);
        const ys = b.shape(y);
        if (xs.len < 2 or ys.len < 2) return error.BadShape;
        const m = if (transpose_x) xs[xs.len - 1] else xs[xs.len - 2];
        const kx = if (transpose_x) xs[xs.len - 2] else xs[xs.len - 1];
        const ky = if (transpose_y) ys[ys.len - 1] else ys[ys.len - 2];
        const n = if (transpose_y) ys[ys.len - 2] else ys[ys.len - 1];
        if (kx != ky) return error.BadShape;
        const batch = try broadcastShape(b.arena, xs[0 .. xs.len - 2], ys[0 .. ys.len - 2]);
        const s = try b.arena.alloc(usize, batch.len + 2);
        @memcpy(s[0..batch.len], batch);
        s[batch.len] = m;
        s[batch.len + 1] = n;
        return b.op("matmul", &.{
            p("x", x),                                      p("y", y),
            p("transpose_x", try b.scalarBool(transpose_x)), p("transpose_y", try b.scalarBool(transpose_y)),
        }, b.dtype(x), s);
    }

    /// Normalizes over the last `n_axes` dims, then gamma/beta (shape of those dims).
    pub fn layerNorm(b: *Builder, x: V, gamma: ?V, beta: ?V, epsilon: f32) Error!V {
        const r: i32 = @intCast(b.rank(x));
        var params: [5]Param = undefined;
        var n: usize = 0;
        params[n] = p("x", x);
        n += 1;
        params[n] = p("axes", try b.ints(&.{r - 1}));
        n += 1;
        if (gamma) |g| {
            params[n] = p("gamma", g);
            n += 1;
        }
        if (beta) |be| {
            params[n] = p("beta", be);
            n += 1;
        }
        params[n] = p("epsilon", try b.scalarFloat(b.dtype(x), epsilon));
        n += 1;
        return b.op("layer_norm", params[0..n], b.dtype(x), b.shape(x));
    }

    pub fn softmax(b: *Builder, x: V, axis: i32) Error!V {
        return b.op("softmax", &.{ p("x", x), p("axis", try b.scalarI32(axis)) }, b.dtype(x), b.shape(x));
    }

    pub fn gelu(b: *Builder, x: V, mode: GeluMode) Error!V {
        return b.op("gelu", &.{ p("x", x), p("mode", try b.string(mode.str())) }, b.dtype(x), b.shape(x));
    }

    pub fn reshape(b: *Builder, x: V, new_shape: []const usize) Error!V {
        var n: usize = 1;
        for (new_shape) |d| n *= d;
        if (n != b.info(x).count()) return error.BadShape;
        return b.op("reshape", &.{ p("x", x), p("shape", try b.intsFromUsize(new_shape)) }, b.dtype(x), new_shape);
    }

    pub fn transpose(b: *Builder, x: V, perm: []const usize) Error!V {
        const xs = b.shape(x);
        if (perm.len != xs.len) return error.BadShape;
        const s = try b.arena.alloc(usize, xs.len);
        for (s, perm) |*o, a| {
            if (a >= xs.len) return error.BadShape;
            o.* = xs[a];
        }
        return b.op("transpose", &.{ p("x", x), p("perm", try b.intsFromUsize(perm)) }, b.dtype(x), s);
    }

    pub fn concat(b: *Builder, vals: []const V, axis: usize) Error!V {
        if (vals.len == 0) return error.BadShape;
        const s0 = b.shape(vals[0]);
        if (axis >= s0.len) return error.BadShape;
        const s = try b.arena.dupe(usize, s0);
        s[axis] = 0;
        for (vals) |v| {
            const vs = b.shape(v);
            if (vs.len != s0.len or b.dtype(v) != b.dtype(vals[0])) return error.BadShape;
            for (vs, s0, 0..) |d, d0, i| if (i != axis and d != d0) return error.BadShape;
            s[axis] += vs[axis];
        }
        return b.op("concat", &.{
            .{ .name = "values", .arg = .{ .many = try b.arena.dupe(V, vals) } },
            p("axis", try b.scalarI32(@intCast(axis))),
            p("interleave", try b.scalarBool(false)),
        }, b.dtype(vals[0]), s);
    }

    /// x[begin[i] .. begin[i] + size[i]] along every axis.
    pub fn sliceBySize(b: *Builder, x: V, begin: []const usize, size: []const usize) Error!V {
        const xs = b.shape(x);
        if (begin.len != xs.len or size.len != xs.len) return error.BadShape;
        for (xs, begin, size) |d, o, n| if (o + n > d or n == 0) return error.BadShape;
        return b.op("slice_by_size", &.{ p("x", x), p("begin", try b.intsFromUsize(begin)), p("size", try b.intsFromUsize(size)) }, b.dtype(x), size);
    }

    /// 1-D convolution: x [N][C_in][W], weight [C_out][C_in][K] (const),
    /// bias [C_out], explicit padding (pad_type "custom").
    pub fn conv1d(b: *Builder, x: V, weight: V, bias: ?V, stride: usize, pad: [2]usize) Error!V {
        const xs = b.shape(x);
        const ws = b.shape(weight);
        if (xs.len != 3 or ws.len != 3 or xs[1] != ws[1]) return error.BadShape;
        const k = ws[2];
        const w_out = (xs[2] + pad[0] + pad[1] - k) / stride + 1;
        var params: [8]Param = undefined;
        var n: usize = 0;
        params[n] = p("x", x);
        n += 1;
        params[n] = p("weight", weight);
        n += 1;
        if (bias) |bv| {
            params[n] = p("bias", bv);
            n += 1;
        }
        params[n] = p("strides", try b.ints(&.{@intCast(stride)}));
        n += 1;
        params[n] = p("pad_type", try b.string("custom"));
        n += 1;
        params[n] = p("pad", try b.ints(&.{ @intCast(pad[0]), @intCast(pad[1]) }));
        n += 1;
        params[n] = p("dilations", try b.ints(&.{1}));
        n += 1;
        params[n] = p("groups", try b.scalarI32(1));
        n += 1;
        return b.op("conv", params[0..n], b.dtype(x), &.{ xs[0], ws[0], w_out });
    }

    pub fn cast(b: *Builder, x: V, to: DType) Error!V {
        if (b.dtype(x) == to) return x;
        return b.op("cast", &.{ p("x", x), p("dtype", try b.string(to.name())) }, to, b.shape(x));
    }

    /// reduce_max / reduce_sum / reduce_mean / reduce_log_sum_exp over `axes`.
    pub fn reduce(b: *Builder, ty: []const u8, x: V, axes: []const usize, keep_dims: bool) Error!V {
        const xs = b.shape(x);
        var out: std.ArrayList(usize) = .empty;
        for (xs, 0..) |d, i| {
            const reduced = std.mem.indexOfScalar(usize, axes, i) != null;
            if (!reduced) try out.append(b.arena, d) else if (keep_dims) try out.append(b.arena, 1);
        }
        return b.op(ty, &.{ p("x", x), p("axes", try b.intsFromUsize(axes)), p("keep_dims", try b.scalarBool(keep_dims)) }, b.dtype(x), out.items);
    }

    /// Index of the largest value along `axis` (int32).
    pub fn argmax(b: *Builder, x: V, axis: usize, keep_dims: bool) Error!V {
        const xs = b.shape(x);
        var out: std.ArrayList(usize) = .empty;
        for (xs, 0..) |d, i| {
            if (i != axis) try out.append(b.arena, d) else if (keep_dims) try out.append(b.arena, 1);
        }
        var params: [4]Param = .{ p("x", x), p("axis", try b.scalarI32(@intCast(axis))), p("keep_dims", try b.scalarBool(keep_dims)), undefined };
        var n: usize = 3;
        if (b.opts.target != .ios16) {
            params[3] = p("output_dtype", try b.string("int32"));
            n = 4;
        }
        return b.op("reduce_argmax", params[0..n], .int32, out.items);
    }

    /// Rows of `x` (along axis 0) picked by int32 `indices`.
    pub fn gather(b: *Builder, x: V, indices: V, axis: usize) Error!V {
        if (b.dtype(indices) != .int32) return error.BadType;
        const xs = b.shape(x);
        const is = b.shape(indices);
        var out: std.ArrayList(usize) = .empty;
        try out.appendSlice(b.arena, xs[0..axis]);
        try out.appendSlice(b.arena, is);
        try out.appendSlice(b.arena, xs[axis + 1 ..]);
        var params: [5]Param = .{ p("x", x), p("indices", indices), p("axis", try b.scalarI32(@intCast(axis))), p("batch_dims", try b.scalarI32(0)), undefined };
        var n: usize = 4;
        if (b.opts.target != .ios16) {
            params[4] = p("validate_indices", try b.scalarBool(false));
            n = 5;
        }
        return b.op("gather", params[0..n], b.dtype(x), out.items);
    }

    /// iOS 18: the current contents of a state.
    pub fn readState(b: *Builder, st: V) Error!V {
        if (b.info(st).kind != .state) return error.BadType;
        return b.op("read_state", &.{p("input", st)}, b.dtype(st), b.shape(st));
    }

    /// iOS 18: replaces a state's contents (no output value).
    pub fn writeState(b: *Builder, st: V, data: V) Error!void {
        if (b.info(st).kind != .state) return error.BadType;
        if (!std.mem.eql(usize, b.shape(st), b.shape(data)) or b.dtype(data) != b.dtype(st)) return error.BadShape;
        const name = try b.autoName("write_state");
        try b.ops.append(b.arena, .{ .type = "write_state", .name = name, .params = try b.arena.dupe(Param, &.{ p("input", st), p("data", data) }), .outputs = &.{} });
    }

    // ------------------------------------------------------------ serialize

    /// The Model protobuf (Data/com.apple.CoreML/model.mlmodel), appending
    /// large consts to `weights`.
    pub fn serialize(b: *const Builder, gpa: Allocator, weights: *blob.Writer) ![]u8 {
        var w = pb.Writer.init(gpa);
        errdefer w.deinit();
        const t = b.opts.target;
        try w.int32(spec.Model.specification_version, t.specVersion());

        // description
        {
            const d = try w.begin(spec.Model.description);
            for (b.inputs.items) |v| {
                const i = b.info(v);
                if (i.kind == .state) continue;
                try spec.featureDescription(&w, spec.ModelDescription.input, i.name, i.dtype.array().?, i.shape, false);
            }
            for (b.outputs.items) |v| {
                const i = b.info(v);
                try spec.featureDescription(&w, spec.ModelDescription.output, i.name, i.dtype.array().?, i.shape, false);
            }
            for (b.inputs.items) |v| {
                const i = b.info(v);
                if (i.kind != .state) continue;
                try spec.featureDescription(&w, spec.ModelDescription.state, i.name, i.dtype.array().?, i.shape, true);
            }
            if (b.opts.description.len > 0 or b.opts.author.len > 0) {
                const md = try w.begin(spec.ModelDescription.metadata);
                try w.string(spec.Metadata.short_description, b.opts.description);
                try w.string(spec.Metadata.author, b.opts.author);
                try w.end(md);
            }
            try w.end(d);
        }

        // mlProgram
        const prog = try w.begin(spec.Model.ml_program);
        try w.int64(spec.Program.version, 1);
        const fe = try w.begin(spec.Program.functions);
        try w.string(spec.MapEntry.key, "main");
        const func = try w.begin(spec.MapEntry.value);
        for (b.inputs.items) |v| {
            const i = b.info(v);
            try spec.namedValueType(&w, spec.Function.inputs, i.name, i.dtype.mil(), i.shape, i.kind == .state);
        }
        try w.string(spec.Function.opset, t.opset());
        const be = try w.begin(spec.Function.block_specializations);
        try w.string(spec.MapEntry.key, t.opset());
        const blk = try w.begin(spec.MapEntry.value);
        for (b.outputs.items) |v| try w.bytesAlways(spec.Block.outputs, b.info(v).name);
        for (b.ops.items) |o| try b.writeOp(&w, o, weights);
        try w.end(blk);
        try w.end(be);
        try w.end(func);
        try w.end(fe);
        try w.end(prog);
        return w.toOwnedSlice();
    }

    fn writeOp(b: *const Builder, w: *pb.Writer, o: Op, weights: *blob.Writer) !void {
        const m = try w.begin(spec.Block.operations);
        try w.string(spec.Operation.type, o.type);
        for (o.params) |prm| {
            const e = try w.begin(spec.Operation.inputs);
            try w.string(spec.MapEntry.key, prm.name);
            const arg = try w.begin(spec.MapEntry.value);
            switch (prm.arg) {
                .one => |v| try bindName(w, b.info(v).name),
                .many => |vs| for (vs) |v| try bindName(w, b.info(v).name),
            }
            try w.end(arg);
            try w.end(e);
        }
        for (o.outputs) |v| {
            const i = b.info(v);
            try spec.namedValueType(w, spec.Operation.outputs, i.name, i.dtype.mil(), i.shape, false);
        }
        // attributes: name (and val for const)
        {
            const e = try w.begin(spec.Operation.attributes);
            try w.string(spec.MapEntry.key, "name");
            try spec.stringValue(w, spec.MapEntry.value, o.name);
            try w.end(e);
        }
        if (std.mem.eql(u8, o.type, "const")) {
            const i = b.info(o.outputs[0]);
            const e = try w.begin(spec.Operation.attributes);
            try w.string(spec.MapEntry.key, "val");
            try b.writeConstValue(w, i, weights);
            try w.end(e);
        }
        try w.end(m);
    }

    fn writeConstValue(b: *const Builder, w: *pb.Writer, i: Info, weights: *blob.Writer) !void {
        const data = i.kind.constant;
        const f = spec.MapEntry.value;
        const big = data.len() >= b.opts.blob_threshold;
        switch (data) {
            .fp16 => |d| if (big) {
                try spec.blobValue(w, f, .float16, i.shape, blob.file_name, try weights.addF16(d));
            } else try spec.immediateValue(w, f, .float16, i.shape, .{ .halfs = d }),
            .fp32 => |d| if (big) {
                try spec.blobValue(w, f, .float32, i.shape, blob.file_name, try weights.addF32(d));
            } else try spec.immediateValue(w, f, .float32, i.shape, .{ .floats = d }),
            .int32 => |d| try spec.immediateValue(w, f, .int32, i.shape, .{ .ints = d }),
            .bool => |d| try spec.immediateValue(w, f, .bool, i.shape, .{ .bools = d }),
            .string => |s| try spec.immediateValue(w, f, .string, i.shape, .{ .strings = &.{s} }),
        }
    }
};

fn bindName(w: *pb.Writer, name: []const u8) !void {
    const bd = try w.begin(spec.Argument.arguments);
    try w.bytesAlways(spec.Argument.Binding.name, name);
    try w.end(bd);
}

/// numpy broadcasting of two shapes.
pub fn broadcastShape(a: Allocator, x: []const usize, y: []const usize) Error![]usize {
    const n = @max(x.len, y.len);
    const out = try a.alloc(usize, n);
    errdefer a.free(out);
    for (0..n) |i| {
        const dx = if (i < n - x.len) 1 else x[i - (n - x.len)];
        const dy = if (i < n - y.len) 1 else y[i - (n - y.len)];
        if (dx != dy and dx != 1 and dy != 1) return error.BadShape;
        out[i] = @max(dx, dy);
    }
    return out;
}

// ------------------------------------------------------------------- tests

const testing = std.testing;

test "shape inference" {
    var b = try Builder.init(testing.allocator, .{});
    defer b.deinit();
    const x = try b.input("x", .fp16, &.{ 2, 5, 8 });
    const w = try b.constFloat(.fp16, &.{ 3, 8 }, &@as([24]f32, @splat(0.5)));
    const y = try b.linear(x, w, null);
    try testing.expectEqualSlices(usize, &.{ 2, 5, 3 }, b.shape(y));
    const q = try b.matmul(x, x, false, true);
    try testing.expectEqualSlices(usize, &.{ 2, 5, 5 }, b.shape(q));
    const t = try b.transpose(x, &.{ 1, 0, 2 });
    try testing.expectEqualSlices(usize, &.{ 5, 2, 8 }, b.shape(t));
    const c = try b.concat(&.{ x, x }, 1);
    try testing.expectEqualSlices(usize, &.{ 2, 10, 8 }, b.shape(c));
    const r = try b.reduce("reduce_max", x, &.{ 1, 2 }, true);
    try testing.expectEqualSlices(usize, &.{ 2, 1, 1 }, b.shape(r));
    const am = try b.argmax(x, 2, false);
    try testing.expectEqualSlices(usize, &.{ 2, 5 }, b.shape(am));
    try testing.expectEqual(DType.int32, b.dtype(am));
    const s = try b.sliceBySize(x, &.{ 0, 1, 0 }, &.{ 2, 3, 8 });
    try testing.expectEqualSlices(usize, &.{ 2, 3, 8 }, b.shape(s));
    const cx = try b.input("a", .fp16, &.{ 1, 80, 100 });
    const cw = try b.constFloat(.fp16, &.{ 16, 80, 3 }, &@as([16 * 80 * 3]f32, @splat(0)));
    const cv = try b.conv1d(cx, cw, null, 2, .{ 1, 1 });
    try testing.expectEqualSlices(usize, &.{ 1, 16, 50 }, b.shape(cv));
    try testing.expectError(error.BadShape, b.matmul(x, w, false, false));
    try testing.expectError(error.BadName, b.input("1x", .fp16, &.{1}));
}

test "broadcast" {
    const s = try broadcastShape(testing.allocator, &.{ 4, 1, 3 }, &.{ 5, 1 });
    defer testing.allocator.free(s);
    try testing.expectEqualSlices(usize, &.{ 4, 5, 3 }, s);
    try testing.expectError(error.BadShape, broadcastShape(testing.allocator, &.{3}, &.{4}));
}
