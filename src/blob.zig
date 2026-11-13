//! The ML Program weight file (`weights/weight.bin`, "MIL blob storage"
//! version 2), re-derived from coremltools' public C++ sources
//! (mlmodel/src/MILBlob/Blob/StorageFormat.hpp, StorageWriter.cpp;
//! BSD-3-Clause). Little-endian:
//!
//!   storage_header (64 bytes) at offset 0:
//!     u32 count, u32 version = 2, 7 x u64 reserved (0)
//!   per blob, at the next 64-byte aligned offset:
//!     blob_metadata (64 bytes):
//!       u32 sentinel = 0xDEADBEEF, u32 mil_dtype (BlobDataType),
//!       u64 sizeInBytes, u64 offset (of the data: metadata offset + 64),
//!       u64 padding_size_in_bits (0 for byte-sized types), 4 x u64 reserved (0)
//!     the data, right after the metadata
//!
//! A const's Value.blobFileValue.offset is the offset of its *metadata*
//! (what StorageWriter::WriteData returns), and its fileName is
//! "@model_path/weights/weight.bin".
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const file_name = "@model_path/weights/weight.bin";
pub const alignment = 64;
pub const sentinel: u32 = 0xDEADBEEF;

/// MILBlob::Blob::BlobDataType.
pub const DataType = enum(u32) {
    float16 = 1,
    float32 = 2,
    uint8 = 3,
    int8 = 4,
    bfloat16 = 5,
    int16 = 6,
    uint16 = 7,
    int32 = 14,
    uint32 = 15,
};

pub const Writer = struct {
    gpa: Allocator,
    buf: std.ArrayList(u8) = .empty,
    count: u32 = 0,

    pub fn init(gpa: Allocator) !Writer {
        var w: Writer = .{ .gpa = gpa };
        try w.buf.appendNTimes(gpa, 0, 64);
        w.patchHeader();
        return w;
    }

    pub fn deinit(w: *Writer) void {
        w.buf.deinit(w.gpa);
        w.* = undefined;
    }

    fn patchHeader(w: *Writer) void {
        std.mem.writeInt(u32, w.buf.items[0..4], w.count, .little);
        std.mem.writeInt(u32, w.buf.items[4..8], 2, .little);
    }

    /// Appends one blob; returns the offset of its metadata (the value for
    /// BlobFileValue.offset).
    pub fn add(w: *Writer, dt: DataType, data: []const u8) !u64 {
        const pad = std.mem.alignForward(usize, w.buf.items.len, alignment) - w.buf.items.len;
        try w.buf.appendNTimes(w.gpa, 0, pad);
        const meta_at = w.buf.items.len;
        var meta: [64]u8 = @splat(0);
        std.mem.writeInt(u32, meta[0..4], sentinel, .little);
        std.mem.writeInt(u32, meta[4..8], @intFromEnum(dt), .little);
        std.mem.writeInt(u64, meta[8..16], data.len, .little);
        std.mem.writeInt(u64, meta[16..24], meta_at + 64, .little);
        try w.buf.appendSlice(w.gpa, &meta);
        try w.buf.appendSlice(w.gpa, data);
        w.count += 1;
        w.patchHeader();
        return meta_at;
    }

    pub fn addF16(w: *Writer, data: []const f16) !u64 {
        // Little-endian hosts only (x86_64, aarch64): the bytes are the file format.
        comptime std.debug.assert(@import("builtin").cpu.arch.endian() == .little);
        return w.add(.float16, std.mem.sliceAsBytes(data));
    }

    pub fn addF32(w: *Writer, data: []const f32) !u64 {
        comptime std.debug.assert(@import("builtin").cpu.arch.endian() == .little);
        return w.add(.float32, std.mem.sliceAsBytes(data));
    }

    pub fn bytes(w: *const Writer) []const u8 {
        return w.buf.items;
    }
};

pub const Blob = struct { dtype: DataType, data: []const u8 };

pub const ReadError = error{ BadBlobFile, BadBlobOffset };

/// Reads the blob whose metadata is at `offset` (validating the header and
/// metadata the way Core ML's reader would need them).
pub fn read(file: []const u8, offset: u64) ReadError!Blob {
    if (file.len < 64) return error.BadBlobFile;
    if (std.mem.readInt(u32, file[4..8], .little) != 2) return error.BadBlobFile;
    if (offset % alignment != 0 or offset < 64 or offset + 64 > file.len) return error.BadBlobOffset;
    const m = file[@intCast(offset)..][0..64];
    if (std.mem.readInt(u32, m[0..4], .little) != sentinel) return error.BadBlobOffset;
    const dt: DataType = std.enums.fromInt(DataType, std.mem.readInt(u32, m[4..8], .little)) orelse return error.BadBlobFile;
    const size = std.mem.readInt(u64, m[8..16], .little);
    const at = std.mem.readInt(u64, m[16..24], .little);
    if (at != offset + 64 or at + size > file.len) return error.BadBlobFile;
    for (m[24..64]) |b| if (b != 0) return error.BadBlobFile;
    return .{ .dtype = dt, .data = file[@intCast(at)..][0..@intCast(size)] };
}

pub fn count(file: []const u8) ReadError!u32 {
    if (file.len < 64) return error.BadBlobFile;
    return std.mem.readInt(u32, file[0..4], .little);
}

test "layout: header, aligned metadata, data" {
    var w = try Writer.init(std.testing.allocator);
    defer w.deinit();
    const a = try w.addF16(&.{ 1.0, 2.0, 3.0 }); // 6 bytes of data
    const b = try w.addF32(&.{ 1.0, -1.0 });
    try std.testing.expectEqual(@as(u64, 64), a);
    // 64 (header) + 64 (meta) + 6 (data) = 134 -> next aligned is 192
    try std.testing.expectEqual(@as(u64, 192), b);
    const f = w.bytes();
    try std.testing.expectEqual(@as(usize, 192 + 64 + 8), f.len);
    try std.testing.expectEqual(@as(u32, 2), try count(f));
    // golden header and first metadata
    try std.testing.expectEqualSlices(u8, &.{ 2, 0, 0, 0, 2, 0, 0, 0 }, f[0..8]);
    try std.testing.expectEqualSlices(u8, &.{ 0xef, 0xbe, 0xad, 0xde, 1, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0 }, f[64..88]);
    const ra = try read(f, a);
    try std.testing.expectEqual(DataType.float16, ra.dtype);
    try std.testing.expectEqualSlices(u8, &.{ 0x00, 0x3c, 0x00, 0x40, 0x00, 0x42 }, ra.data);
    const rb = try read(f, b);
    try std.testing.expectEqual(DataType.float32, rb.dtype);
    try std.testing.expectEqual(@as(f32, -1.0), @as(f32, @bitCast(std.mem.readInt(u32, rb.data[4..8], .little))));
    try std.testing.expectError(error.BadBlobOffset, read(f, 128));
}
