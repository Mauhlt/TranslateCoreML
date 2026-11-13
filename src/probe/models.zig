//! The probe models (Phase 1), built with mil.Builder, plus their inputs
//! and host reference outputs:
//!
//!   matmul         y = x · W + b      x [4][64] fp32 in/out, fp16 inside
//!   matmul_fp16io  the same with fp16 model inputs/outputs (iOS 16+)
//!   block          one pre-norm transformer block at whisper-tiny width:
//!                  x [1][128][384] -> LayerNorm -> q/k/v (linear) -> 6 heads
//!                  of 64 (reshape/transpose) -> softmax(q·kᵀ/8)·v -> out
//!                  linear -> +x -> LayerNorm -> 1536 GELU -> 384 -> +x
//!
//! Weights come from a fixed-seed PRNG, so the host generator and the
//! device report always agree on what was computed.
const std = @import("std");
const Allocator = std.mem.Allocator;
const coreml = @import("coreml");
const mil = coreml.mil;
const ref = coreml.ref;

pub const names = [_][]const u8{ "matmul", "matmul_fp16io", "block" };

pub const Built = struct {
    b: mil.Builder,
    arena: std.heap.ArenaAllocator,
    inputs: []ref.Named,

    pub fn deinit(x: *Built) void {
        x.b.deinit();
        x.arena.deinit();
    }
};

fn randoms(a: Allocator, rng: std.Random, n: usize, scale: f32) ![]f32 {
    const out = try a.alloc(f32, n);
    for (out) |*v| v.* = (rng.float(f32) * 2 - 1) * scale;
    return out;
}

pub fn build(gpa: Allocator, name: []const u8, target: mil.Target) !Built {
    var x: Built = .{ .b = try mil.Builder.init(gpa, .{ .target = target, .description = "TranslateCoreML probe" }), .arena = .init(gpa), .inputs = &.{} };
    errdefer x.deinit();
    const a = x.arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x5eed_c0de);
    const rng = prng.random();
    const b = &x.b;
    if (std.mem.eql(u8, name, "matmul") or std.mem.eql(u8, name, "matmul_fp16io")) {
        const io: mil.DType = if (std.mem.eql(u8, name, "matmul")) .fp32 else .fp16;
        const in = try b.input("x", io, &.{ 4, 64 });
        const h = try b.cast(in, .fp16);
        const w = try b.constFloat(.fp16, &.{ 64, 32 }, try randoms(a, rng, 64 * 32, 0.125));
        const bias = try b.constFloat(.fp16, &.{32}, try randoms(a, rng, 32, 0.5));
        const y = try b.add(try b.matmul(h, w, false, false), bias);
        try b.output("y", try b.cast(y, io));
        x.inputs = try a.dupe(ref.Named, &.{.{ .name = "x", .shape = &.{ 4, 64 }, .data = try randoms(a, rng, 4 * 64, 1.0) }});
    } else if (std.mem.eql(u8, name, "block")) {
        const T = 128;
        const D = 384;
        const H = 6;
        const F = 4 * D;
        const in = try b.input("x", .fp32, &.{ 1, T, D });
        const xh = try b.cast(in, .fp16);
        const sd: f32 = 1.0 / @sqrt(@as(f32, D));
        const sf: f32 = 1.0 / @sqrt(@as(f32, F));
        const ones = try a.alloc(f32, D);
        for (ones) |*v| v.* = 1.0 + (rng.float(f32) - 0.5) * 0.2;
        var h = try b.layerNorm(xh, try b.constFloat(.fp16, &.{D}, ones), try b.constFloat(.fp16, &.{D}, try randoms(a, rng, D, 0.1)), 1e-5);
        // attention scale 1/sqrt(64) folded into the query weights, as stt does
        const q = try b.linear(h, try b.constFloat(.fp16, &.{ D, D }, try randoms(a, rng, D * D, sd / 8.0)), try b.constFloat(.fp16, &.{D}, try randoms(a, rng, D, 0.1 / 8.0)));
        const k = try b.linear(h, try b.constFloat(.fp16, &.{ D, D }, try randoms(a, rng, D * D, sd)), null);
        const v = try b.linear(h, try b.constFloat(.fp16, &.{ D, D }, try randoms(a, rng, D * D, sd)), try b.constFloat(.fp16, &.{D}, try randoms(a, rng, D, 0.1)));
        const qh = try b.transpose(try b.reshape(q, &.{ 1, T, H, 64 }), &.{ 0, 2, 1, 3 });
        const kh = try b.transpose(try b.reshape(k, &.{ 1, T, H, 64 }), &.{ 0, 2, 1, 3 });
        const vh = try b.transpose(try b.reshape(v, &.{ 1, T, H, 64 }), &.{ 0, 2, 1, 3 });
        const p = try b.softmax(try b.matmul(qh, kh, false, true), -1); // [1][H][T][T]
        const o = try b.reshape(try b.transpose(try b.matmul(p, vh, false, false), &.{ 0, 2, 1, 3 }), &.{ 1, T, D });
        const ao = try b.linear(o, try b.constFloat(.fp16, &.{ D, D }, try randoms(a, rng, D * D, sd)), try b.constFloat(.fp16, &.{D}, try randoms(a, rng, D, 0.1)));
        const x1 = try b.add(xh, ao);
        h = try b.layerNorm(x1, try b.constFloat(.fp16, &.{D}, ones), try b.constFloat(.fp16, &.{D}, try randoms(a, rng, D, 0.1)), 1e-5);
        const f = try b.gelu(try b.linear(h, try b.constFloat(.fp16, &.{ F, D }, try randoms(a, rng, F * D, sd)), try b.constFloat(.fp16, &.{F}, try randoms(a, rng, F, 0.1))), .exact);
        const f2 = try b.linear(f, try b.constFloat(.fp16, &.{ D, F }, try randoms(a, rng, D * F, sf)), try b.constFloat(.fp16, &.{D}, try randoms(a, rng, D, 0.1)));
        try b.output("y", try b.cast(try b.add(x1, f2), .fp32));
        x.inputs = try a.dupe(ref.Named, &.{.{ .name = "x", .shape = &.{ 1, T, D }, .data = try randoms(a, rng, T * D, 1.0) }});
    } else return error.UnknownProbeModel;
    return x;
}

/// Runs the model on the host (fp16-emulating or plain f32) and returns
/// copies of its outputs.
pub fn reference(gpa: Allocator, x: *const Built, emulate_fp16: bool) ![]ref.Named {
    var it = try ref.Interp.init(gpa, &x.b);
    defer it.deinit();
    it.emulate_fp16 = emulate_fp16;
    try it.run(x.inputs);
    const outs = try gpa.alloc(ref.Named, x.b.outputs.items.len);
    for (x.b.outputs.items, outs) |v, *o| {
        const inf = x.b.info(v);
        const t = it.output(inf.name).?;
        o.* = .{ .name = try gpa.dupe(u8, inf.name), .shape = try gpa.dupe(usize, t.shape), .data = try gpa.dupe(f32, t.data) };
    }
    return outs;
}

pub fn freeNamed(gpa: Allocator, xs: []ref.Named) void {
    for (xs) |x| {
        gpa.free(x.name);
        gpa.free(x.shape);
        gpa.free(x.data);
    }
    gpa.free(xs);
}

// ---------------------------------------------------------------- ref files

/// `<name>.ref`: the inputs and expected outputs of a probe model.
///   "CMLREF01", u32 count, then per tensor:
///   u8 role (0 input, 1 expected output fp16-emulated, 2 output f32 math),
///   u16 name length, name, u8 rank, u32 dims[rank], f32 data (little-endian)
pub const Role = enum(u8) { input = 0, expected = 1, exact = 2 };
pub const Entry = struct { role: Role, t: ref.Named };

pub fn writeRef(gpa: Allocator, entries: []const Entry) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    try out.appendSlice(gpa, "CMLREF01");
    try appendInt(&out, gpa, u32, @intCast(entries.len));
    for (entries) |e| {
        try out.append(gpa, @intFromEnum(e.role));
        try appendInt(&out, gpa, u16, @intCast(e.t.name.len));
        try out.appendSlice(gpa, e.t.name);
        try out.append(gpa, @intCast(e.t.shape.len));
        for (e.t.shape) |d| try appendInt(&out, gpa, u32, @intCast(d));
        for (e.t.data) |v| try appendInt(&out, gpa, u32, @bitCast(v));
    }
    return out.toOwnedSlice(gpa);
}

fn appendInt(out: *std.ArrayList(u8), gpa: Allocator, comptime I: type, v: I) !void {
    var buf: [@sizeOf(I)]u8 = undefined;
    std.mem.writeInt(I, &buf, v, .little);
    try out.appendSlice(gpa, &buf);
}

/// Parses a ref file; the result borrows names from `bytes` and owns shapes/data.
pub fn readRef(gpa: Allocator, bytes: []const u8) ![]Entry {
    if (bytes.len < 12 or !std.mem.eql(u8, bytes[0..8], "CMLREF01")) return error.BadRefFile;
    var pos: usize = 8;
    const n = try takeInt(bytes, &pos, u32);
    var list: std.ArrayList(Entry) = .empty;
    errdefer {
        for (list.items) |e| {
            gpa.free(e.t.shape);
            gpa.free(e.t.data);
        }
        list.deinit(gpa);
    }
    for (0..n) |_| {
        if (pos >= bytes.len) return error.BadRefFile;
        const role = std.enums.fromInt(Role, bytes[pos]) orelse return error.BadRefFile;
        pos += 1;
        const nl = try takeInt(bytes, &pos, u16);
        if (pos + nl + 1 > bytes.len) return error.BadRefFile;
        const name = bytes[pos..][0..nl];
        pos += nl;
        const rank = bytes[pos];
        pos += 1;
        const shape = try gpa.alloc(usize, rank);
        errdefer gpa.free(shape);
        var count: usize = 1;
        for (shape) |*d| {
            d.* = try takeInt(bytes, &pos, u32);
            count *= d.*;
        }
        const data = try gpa.alloc(f32, count);
        errdefer gpa.free(data);
        for (data) |*v| v.* = @bitCast(try takeInt(bytes, &pos, u32));
        try list.append(gpa, .{ .role = role, .t = .{ .name = name, .shape = shape, .data = data } });
    }
    return list.toOwnedSlice(gpa);
}

pub fn freeRef(gpa: Allocator, es: []Entry) void {
    for (es) |e| {
        gpa.free(e.t.shape);
        gpa.free(e.t.data);
    }
    gpa.free(es);
}

fn takeInt(bytes: []const u8, pos: *usize, comptime I: type) !I {
    if (pos.* + @sizeOf(I) > bytes.len) return error.BadRefFile;
    defer pos.* += @sizeOf(I);
    return std.mem.readInt(I, bytes[pos.*..][0..@sizeOf(I)], .little);
}

/// Max absolute error and max error relative to the reference's largest
/// magnitude.
pub const Diff = struct { max_abs: f32, rel: f32 };

pub fn diff(got: []const f32, want: []const f32) Diff {
    var m: f32 = 0;
    var scale: f32 = 0;
    for (got, want) |g, w| {
        m = @max(m, @abs(g - w));
        scale = @max(scale, @abs(w));
    }
    if (got.len != want.len) return .{ .max_abs = std.math.inf(f32), .rel = std.math.inf(f32) };
    return .{ .max_abs = m, .rel = if (scale > 0) m / scale else m };
}

// ------------------------------------------------------------------- tests

test "probe models build, serialize and run on the host" {
    const gpa = std.testing.allocator;
    for (names) |n| {
        var x = try build(gpa, n, .ios17);
        defer x.deinit();
        var f = try coreml.package.build(gpa, &x.b);
        defer f.deinit(gpa);
        var sum = try coreml.decode.decode(gpa, f.model);
        defer sum.deinit();
        try std.testing.expectEqual(x.b.ops.items.len, sum.ops.len);
        const e16 = try reference(gpa, &x, true);
        defer freeNamed(gpa, e16);
        const e32 = try reference(gpa, &x, false);
        defer freeNamed(gpa, e32);
        const d = diff(e16[0].data, e32[0].data);
        // fp16 storage costs about 3 decimal digits, not more
        try std.testing.expect(d.rel < 1e-2);
        try std.testing.expect(d.rel > 0);
    }
}

test "matmul probe matches a hand computation" {
    const gpa = std.testing.allocator;
    var x = try build(gpa, "matmul", .ios17);
    defer x.deinit();
    const out = try reference(gpa, &x, false);
    defer freeNamed(gpa, out);
    // y[0][0] = Σ_p x[0][p] · W[p][0] + b[0], with W, b as stored (fp16)
    const consts = x.b.values.items;
    var w: []const f16 = &.{};
    var bias: []const f16 = &.{};
    for (consts) |c| switch (c.kind) {
        .constant => |d| switch (d) {
            .fp16 => |h| if (h.len == 64 * 32) {
                w = h;
            } else if (h.len == 32) {
                bias = h;
            },
            else => {},
        },
        else => {},
    };
    var s: f32 = bias[0];
    for (0..64) |p| s += @as(f32, @floatCast(@as(f16, @floatCast(x.inputs[0].data[p])))) * @as(f32, @floatCast(w[p * 32]));
    try std.testing.expectApproxEqAbs(s, out[0].data[0], 1e-4);
}

test "ref file round trip" {
    const gpa = std.testing.allocator;
    const bytes = try writeRef(gpa, &.{
        .{ .role = .input, .t = .{ .name = "x", .shape = &.{ 2, 2 }, .data = &.{ 1, 2, 3, 4 } } },
        .{ .role = .expected, .t = .{ .name = "y", .shape = &.{1}, .data = &.{-0.5} } },
    });
    defer gpa.free(bytes);
    const es = try readRef(gpa, bytes);
    defer freeRef(gpa, es);
    try std.testing.expectEqual(@as(usize, 2), es.len);
    try std.testing.expectEqualStrings("y", es[1].t.name);
    try std.testing.expectEqual(Role.expected, es[1].role);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, es[0].t.data);
    try std.testing.expectError(error.BadRefFile, readRef(gpa, bytes[0 .. bytes.len - 1]));
}
