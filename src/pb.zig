//! Minimal protocol buffers wire format (proto3): an encoder for writing
//! Core ML model specs and a field-level decoder used by the tests to check
//! what the encoder wrote. No schema compiler: messages are written field
//! by field (see spec.zig for the Core ML messages).
//!
//! Wire format (https://protobuf.dev/programming-guides/encoding/):
//!   key = (field_number << 3) | wire_type, as a varint
//!   wire types: 0 varint, 1 fixed64, 2 length-delimited, 5 fixed32
//!   int32/int64 negative values are 10-byte two's complement varints
//!   (not zig-zag; that is sint32/sint64, unused here); repeated scalars
//!   are packed (one length-delimited field) by default in proto3.
//! Nested messages are length-delimited: `begin` writes the key and a
//! one-byte length placeholder, `end` patches the real (minimal) varint
//! length in, moving the body if the length needs more bytes, so the
//! output is canonical (what protoc-generated code writes).
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const WireType = enum(u3) {
    varint = 0,
    fixed64 = 1,
    len = 2,
    fixed32 = 5,
    _,
};

pub const Writer = struct {
    gpa: Allocator,
    buf: std.ArrayList(u8) = .empty,

    pub fn init(gpa: Allocator) Writer {
        return .{ .gpa = gpa };
    }

    pub fn deinit(w: *Writer) void {
        w.buf.deinit(w.gpa);
        w.* = undefined;
    }

    pub fn toOwnedSlice(w: *Writer) ![]u8 {
        return w.buf.toOwnedSlice(w.gpa);
    }

    pub fn bytesWritten(w: *const Writer) []const u8 {
        return w.buf.items;
    }

    pub fn rawVarint(w: *Writer, value: u64) !void {
        var v = value;
        while (v >= 0x80) : (v >>= 7) try w.buf.append(w.gpa, @as(u8, @truncate(v)) | 0x80);
        try w.buf.append(w.gpa, @truncate(v));
    }

    pub fn key(w: *Writer, field: u32, wt: WireType) !void {
        std.debug.assert(field > 0 and field < (1 << 29));
        try w.rawVarint((@as(u64, field) << 3) | @intFromEnum(wt));
    }

    // Scalar fields. proto3 does not write a field that has its default
    // value (0, false, ""), except inside a oneof, where presence matters:
    // the `*Always` variants write it anyway.

    pub fn uint64(w: *Writer, field: u32, v: u64) !void {
        if (v != 0) try w.uint64Always(field, v);
    }
    pub fn uint64Always(w: *Writer, field: u32, v: u64) !void {
        try w.key(field, .varint);
        try w.rawVarint(v);
    }
    pub fn int64(w: *Writer, field: u32, v: i64) !void {
        if (v != 0) try w.uint64Always(field, @bitCast(v));
    }
    /// int32 and enums: negative values sign-extend to 64 bits (10 bytes).
    pub fn int32(w: *Writer, field: u32, v: i32) !void {
        if (v != 0) try w.uint64Always(field, @bitCast(@as(i64, v)));
    }
    pub fn int32Always(w: *Writer, field: u32, v: i32) !void {
        try w.uint64Always(field, @bitCast(@as(i64, v)));
    }
    pub fn boolean(w: *Writer, field: u32, v: bool) !void {
        if (v) try w.uint64Always(field, 1);
    }
    pub fn float32(w: *Writer, field: u32, v: f32) !void {
        if (@as(u32, @bitCast(v)) == 0) return;
        try w.key(field, .fixed32);
        try w.raw32(@bitCast(v));
    }
    pub fn float64(w: *Writer, field: u32, v: f64) !void {
        if (@as(u64, @bitCast(v)) == 0) return;
        try w.key(field, .fixed64);
        try w.raw64(@bitCast(v));
    }
    pub fn raw32(w: *Writer, v: u32) !void {
        var b: [4]u8 = undefined;
        std.mem.writeInt(u32, &b, v, .little);
        try w.buf.appendSlice(w.gpa, &b);
    }
    pub fn raw64(w: *Writer, v: u64) !void {
        var b: [8]u8 = undefined;
        std.mem.writeInt(u64, &b, v, .little);
        try w.buf.appendSlice(w.gpa, &b);
    }

    pub fn string(w: *Writer, field: u32, s: []const u8) !void {
        if (s.len != 0) try w.bytesAlways(field, s);
    }
    /// A length-delimited field written even when empty (oneof members,
    /// repeated string elements).
    pub fn bytesAlways(w: *Writer, field: u32, s: []const u8) !void {
        try w.key(field, .len);
        try w.rawVarint(s.len);
        try w.buf.appendSlice(w.gpa, s);
    }

    // Packed repeated scalars (written even when empty only if `always`).

    pub fn packedFloat32(w: *Writer, field: u32, vs: []const f32) !void {
        if (vs.len == 0) return;
        try w.key(field, .len);
        try w.rawVarint(vs.len * 4);
        for (vs) |v| try w.raw32(@bitCast(v));
    }
    pub fn packedInt32(w: *Writer, field: u32, vs: []const i32) !void {
        if (vs.len == 0) return;
        const m = try w.begin(field);
        for (vs) |v| try w.rawVarint(@bitCast(@as(i64, v)));
        try w.end(m);
    }
    pub fn packedInt64(w: *Writer, field: u32, vs: []const i64) !void {
        if (vs.len == 0) return;
        const m = try w.begin(field);
        for (vs) |v| try w.rawVarint(@bitCast(v));
        try w.end(m);
    }
    pub fn packedBool(w: *Writer, field: u32, vs: []const bool) !void {
        if (vs.len == 0) return;
        try w.key(field, .len);
        try w.rawVarint(vs.len);
        for (vs) |v| try w.buf.append(w.gpa, @intFromBool(v));
    }

    /// Start of a nested message (or any length-delimited field built in
    /// place). Pass the result to `end`.
    pub const Mark = struct { len_at: usize };

    pub fn begin(w: *Writer, field: u32) !Mark {
        try w.key(field, .len);
        try w.buf.append(w.gpa, 0);
        return .{ .len_at = w.buf.items.len - 1 };
    }

    pub fn end(w: *Writer, m: Mark) !void {
        const body_start = m.len_at + 1;
        const n = w.buf.items.len - body_start;
        var tmp: [10]u8 = undefined;
        const k = encodeVarint(&tmp, n);
        if (k > 1) {
            // Make room for the longer length and move the body up.
            try w.buf.appendNTimes(w.gpa, 0, k - 1);
            const items = w.buf.items;
            std.mem.copyBackwards(u8, items[body_start + k - 1 ..][0..n], items[body_start..][0..n]);
        }
        @memcpy(w.buf.items[m.len_at..][0..k], tmp[0..k]);
    }
};

/// Writes `v` as a varint into `out`; returns the number of bytes.
pub fn encodeVarint(out: *[10]u8, value: u64) usize {
    var v = value;
    var i: usize = 0;
    while (v >= 0x80) : (v >>= 7) {
        out[i] = @as(u8, @truncate(v)) | 0x80;
        i += 1;
    }
    out[i] = @truncate(v);
    return i + 1;
}

// ---------------------------------------------------------------- decoding

pub const Field = struct {
    number: u32,
    wire: WireType,
    /// varint / fixed32 / fixed64 payload.
    int: u64 = 0,
    /// length-delimited payload.
    bytes: []const u8 = &.{},

    pub fn asI64(f: Field) i64 {
        return @bitCast(f.int);
    }
    pub fn asF32(f: Field) f32 {
        return @bitCast(@as(u32, @truncate(f.int)));
    }
};

pub const DecodeError = error{ Truncated, BadVarint, BadWireType };

/// Iterates over the top-level fields of one message.
pub const Reader = struct {
    buf: []const u8,
    pos: usize = 0,

    pub fn init(buf: []const u8) Reader {
        return .{ .buf = buf };
    }

    pub fn varint(r: *Reader) DecodeError!u64 {
        var v: u64 = 0;
        var shift: u7 = 0;
        while (true) {
            if (r.pos >= r.buf.len) return error.Truncated;
            const b = r.buf[r.pos];
            r.pos += 1;
            if (shift >= 64 or (shift == 63 and b > 1)) return error.BadVarint;
            v |= @as(u64, b & 0x7f) << @intCast(shift);
            if (b & 0x80 == 0) return v;
            shift += 7;
        }
    }

    fn take(r: *Reader, n: usize) DecodeError![]const u8 {
        if (r.buf.len - r.pos < n) return error.Truncated;
        defer r.pos += n;
        return r.buf[r.pos..][0..n];
    }

    pub fn next(r: *Reader) DecodeError!?Field {
        if (r.pos >= r.buf.len) return null;
        const k = try r.varint();
        const number: u32 = std.math.cast(u32, k >> 3) orelse return error.BadVarint;
        const wire: WireType = @enumFromInt(@as(u3, @truncate(k)));
        var f: Field = .{ .number = number, .wire = wire };
        switch (wire) {
            .varint => f.int = try r.varint(),
            .fixed64 => f.int = std.mem.readInt(u64, (try r.take(8))[0..8], .little),
            .fixed32 => f.int = std.mem.readInt(u32, (try r.take(4))[0..4], .little),
            .len => {
                const n = try r.varint();
                f.bytes = try r.take(std.math.cast(usize, n) orelse return error.Truncated);
            },
            else => return error.BadWireType,
        }
        return f;
    }

    /// The first field `number` of the message, or null.
    pub fn find(buf: []const u8, number: u32) DecodeError!?Field {
        var r = Reader.init(buf);
        while (try r.next()) |f| if (f.number == number) return f;
        return null;
    }

    /// Number of occurrences of field `number`.
    pub fn count(buf: []const u8, number: u32) DecodeError!usize {
        var r = Reader.init(buf);
        var n: usize = 0;
        while (try r.next()) |f| {
            if (f.number == number) n += 1;
        }
        return n;
    }

    /// The `i`th occurrence of field `number`.
    pub fn nth(buf: []const u8, number: u32, i: usize) DecodeError!?Field {
        var r = Reader.init(buf);
        var n: usize = 0;
        while (try r.next()) |f| if (f.number == number) {
            if (n == i) return f;
            n += 1;
        };
        return null;
    }
};

/// Decodes a packed repeated varint field body into `out` (as i64).
pub fn unpackVarints(gpa: Allocator, body: []const u8) ![]i64 {
    var out: std.ArrayList(i64) = .empty;
    errdefer out.deinit(gpa);
    var r = Reader.init(body);
    while (r.pos < body.len) try out.append(gpa, @bitCast(try r.varint()));
    return out.toOwnedSlice(gpa);
}

// ------------------------------------------------------------------- tests

const testing = std.testing;

test "golden bytes from the protobuf encoding guide" {
    var w = Writer.init(testing.allocator);
    defer w.deinit();
    // message Test1 { int32 a = 1; } with a = 150  ->  08 96 01
    try w.int32(1, 150);
    try testing.expectEqualSlices(u8, &.{ 0x08, 0x96, 0x01 }, w.bytesWritten());

    w.buf.clearRetainingCapacity();
    // message Test2 { string b = 2; } with b = "testing"
    try w.string(2, "testing");
    try testing.expectEqualSlices(u8, &.{ 0x12, 0x07, 0x74, 0x65, 0x73, 0x74, 0x69, 0x6e, 0x67 }, w.bytesWritten());

    w.buf.clearRetainingCapacity();
    // message Test3 { Test1 c = 3; } with c.a = 150  ->  1a 03 08 96 01
    const m = try w.begin(3);
    try w.int32(1, 150);
    try w.end(m);
    try testing.expectEqualSlices(u8, &.{ 0x1a, 0x03, 0x08, 0x96, 0x01 }, w.bytesWritten());

    w.buf.clearRetainingCapacity();
    // message Test5 { repeated int32 f = 6 [packed]; } with 3, 270, 86942
    //   -> 32 06 03 8e 02 9e a7 05
    try w.packedInt32(6, &.{ 3, 270, 86942 });
    try testing.expectEqualSlices(u8, &.{ 0x32, 0x06, 0x03, 0x8e, 0x02, 0x9e, 0xa7, 0x05 }, w.bytesWritten());
}

test "negative int32 is a 10-byte varint" {
    var w = Writer.init(testing.allocator);
    defer w.deinit();
    try w.int32(1, -1);
    try testing.expectEqualSlices(u8, &.{ 0x08, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01 }, w.bytesWritten());
    var r = Reader.init(w.bytesWritten());
    const f = (try r.next()).?;
    try testing.expectEqual(@as(i64, -1), f.asI64());
}

test "proto3 defaults are not written" {
    var w = Writer.init(testing.allocator);
    defer w.deinit();
    try w.int32(1, 0);
    try w.string(2, "");
    try w.boolean(3, false);
    try w.float32(4, 0);
    try testing.expectEqual(@as(usize, 0), w.bytesWritten().len);
    try w.int32Always(1, 0);
    try w.bytesAlways(2, "");
    try testing.expectEqualSlices(u8, &.{ 0x08, 0x00, 0x12, 0x00 }, w.bytesWritten());
}

test "nested message longer than 127 bytes gets a 2-byte length" {
    var w = Writer.init(testing.allocator);
    defer w.deinit();
    const outer = try w.begin(1);
    const inner = try w.begin(2);
    const payload: [200]u8 = @splat('x');
    try w.string(3, &payload);
    try w.end(inner);
    try w.int32(4, 7);
    try w.end(outer);

    var r = Reader.init(w.bytesWritten());
    const f1 = (try r.next()).?;
    try testing.expectEqual(@as(u32, 1), f1.number);
    try testing.expect((try r.next()) == null);
    // inner: 2 + 200 bytes of string field + key/len = 203 -> 2-byte length
    const f2 = (try Reader.find(f1.bytes, 2)).?;
    const f3 = (try Reader.find(f2.bytes, 3)).?;
    try testing.expectEqualSlices(u8, &payload, f3.bytes);
    try testing.expectEqual(@as(u64, 7), (try Reader.find(f1.bytes, 4)).?.int);
    // canonical: outer length is exactly the body length
    try testing.expectEqual(@as(usize, 0x0a), w.bytesWritten()[0]);
    try testing.expectEqual(w.bytesWritten().len - 3, f1.bytes.len);
}

test "packed floats and fixed fields round-trip" {
    var w = Writer.init(testing.allocator);
    defer w.deinit();
    try w.packedFloat32(1, &.{ 1.5, -2.25 });
    try w.float32(2, 3.0);
    try w.float64(3, -0.5);
    var r = Reader.init(w.bytesWritten());
    const p = (try r.next()).?;
    try testing.expectEqual(@as(usize, 8), p.bytes.len);
    try testing.expectEqual(@as(f32, -2.25), @as(f32, @bitCast(std.mem.readInt(u32, p.bytes[4..8], .little))));
    try testing.expectEqual(@as(f32, 3.0), (try r.next()).?.asF32());
    try testing.expectEqual(@as(f64, -0.5), @as(f64, @bitCast((try r.next()).?.int)));
}

test "truncated input is an error" {
    var r = Reader.init(&.{ 0x12, 0x05, 0x01 });
    try testing.expectError(error.Truncated, r.next());
}
