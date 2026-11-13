//! `.mlpackage` directories (Core ML model packages), as Core ML's
//! MLModel.compileModelAtURL: reads them:
//!
//!   <name>.mlpackage/
//!     Manifest.json
//!     Data/com.apple.CoreML/model.mlmodel        Model protobuf (spec.zig)
//!     Data/com.apple.CoreML/weights/weight.bin   blob storage (blob.zig)
//!
//! Manifest.json (format of coremltools' modelpackage/src/ModelPackage.cpp,
//! BSD-3-Clause, re-derived): fileFormatVersion "1.0.0", itemInfoEntries
//! keyed by an identifier (a UUID string) with path (relative to Data/),
//! name, author and description, and rootModelIdentifier naming the model
//! item. The identifiers here are derived from a hash of the model bytes
//! (UUID-shaped, uppercase), so the same model gives the same package.
const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const mil = @import("mil.zig");
const blob = @import("blob.zig");

pub const model_path = "Data/com.apple.CoreML/model.mlmodel";
pub const weights_path = "Data/com.apple.CoreML/weights/weight.bin";

/// The three files of a package, in memory.
pub const Files = struct {
    manifest: []u8,
    model: []u8,
    weights: []u8,

    pub fn deinit(f: *Files, gpa: Allocator) void {
        gpa.free(f.manifest);
        gpa.free(f.model);
        gpa.free(f.weights);
        f.* = undefined;
    }
};

pub fn build(gpa: Allocator, b: *const mil.Builder) !Files {
    var weights = try blob.Writer.init(gpa);
    defer weights.deinit();
    const model = try b.serialize(gpa, &weights);
    errdefer gpa.free(model);
    const manifest = try manifestJson(gpa, model);
    errdefer gpa.free(manifest);
    const wbytes = try weights.buf.toOwnedSlice(gpa);
    return .{ .manifest = manifest, .model = model, .weights = wbytes };
}

/// Writes `<dir>/<sub_path>` (replacing an existing package there).
pub fn write(gpa: Allocator, io: Io, dir: Io.Dir, sub_path: []const u8, b: *const mil.Builder) !void {
    var f = try build(gpa, b);
    defer f.deinit(gpa);
    try writeFiles(gpa, io, dir, sub_path, &f);
}

pub fn writeFiles(gpa: Allocator, io: Io, dir: Io.Dir, sub_path: []const u8, f: *const Files) !void {
    try dir.deleteTree(io, sub_path);
    const wdir = try std.fs.path.join(gpa, &.{ sub_path, "Data", "com.apple.CoreML", "weights" });
    defer gpa.free(wdir);
    try dir.createDirPath(io, wdir);
    const files = [_]struct { rel: []const u8, data: []const u8 }{
        .{ .rel = "Manifest.json", .data = f.manifest },
        .{ .rel = model_path, .data = f.model },
        .{ .rel = weights_path, .data = f.weights },
    };
    for (files) |x| {
        const path = try std.fs.path.join(gpa, &.{ sub_path, x.rel });
        defer gpa.free(path);
        try dir.writeFile(io, .{ .sub_path = path, .data = x.data });
    }
}

fn uuid(out: *[36]u8, h: u128) void {
    const hex = "0123456789ABCDEF";
    var bytes: [16]u8 = @bitCast(h);
    bytes[6] = (bytes[6] & 0x0f) | 0x40; // version 4 layout
    bytes[8] = (bytes[8] & 0x3f) | 0x80; // RFC 4122 variant
    var o: usize = 0;
    for (bytes, 0..) |byte, i| {
        if (i == 4 or i == 6 or i == 8 or i == 10) {
            out[o] = '-';
            o += 1;
        }
        out[o] = hex[byte >> 4];
        out[o + 1] = hex[byte & 15];
        o += 2;
    }
}

pub fn manifestJson(gpa: Allocator, model: []const u8) ![]u8 {
    var h1: [16]u8 = undefined;
    var h2: [16]u8 = undefined;
    var hs = std.crypto.hash.Blake3.init(.{});
    hs.update("model");
    hs.update(model);
    hs.final(&h1);
    hs = std.crypto.hash.Blake3.init(.{});
    hs.update("weights");
    hs.update(model);
    hs.final(&h2);
    var id_model: [36]u8 = undefined;
    var id_weights: [36]u8 = undefined;
    uuid(&id_model, @bitCast(h1));
    uuid(&id_weights, @bitCast(h2));
    return std.fmt.allocPrint(gpa,
        \\{{
        \\    "fileFormatVersion": "1.0.0",
        \\    "itemInfoEntries": {{
        \\        "{s}": {{
        \\            "author": "com.apple.CoreML",
        \\            "description": "CoreML Model Specification",
        \\            "name": "model.mlmodel",
        \\            "path": "com.apple.CoreML/model.mlmodel"
        \\        }},
        \\        "{s}": {{
        \\            "author": "com.apple.CoreML",
        \\            "description": "CoreML Model Weights",
        \\            "name": "weights",
        \\            "path": "com.apple.CoreML/weights"
        \\        }}
        \\    }},
        \\    "rootModelIdentifier": "{s}"
        \\}}
        \\
    , .{ &id_model, &id_weights, &id_model });
}

test "manifest is valid JSON naming the root model" {
    const m = try manifestJson(std.testing.allocator, "abc");
    defer std.testing.allocator.free(m);
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, m, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    try std.testing.expectEqualStrings("1.0.0", root.get("fileFormatVersion").?.string);
    const id = root.get("rootModelIdentifier").?.string;
    try std.testing.expectEqual(@as(usize, 36), id.len);
    const entries = root.get("itemInfoEntries").?.object;
    try std.testing.expectEqual(@as(usize, 2), entries.count());
    const item = entries.get(id).?.object;
    try std.testing.expectEqualStrings("com.apple.CoreML/model.mlmodel", item.get("path").?.string);
    try std.testing.expectEqual(@as(u8, '4'), id[14]);
}

test "package layout on disk" {
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var b = try mil.Builder.init(std.testing.allocator, .{});
    defer b.deinit();
    const x = try b.input("x", .fp32, &.{ 1, 16 });
    const w = try b.constFloat(.fp32, &.{ 4, 16 }, &@as([64]f32, @splat(0.25)));
    try b.output("y", try b.linear(x, w, null));
    try write(std.testing.allocator, io, tmp.dir, "t.mlpackage", &b);
    // and again (replaces)
    try write(std.testing.allocator, io, tmp.dir, "t.mlpackage", &b);
    const man = try tmp.dir.readFileAlloc(io, "t.mlpackage/Manifest.json", std.testing.allocator, .limited(1 << 20));
    defer std.testing.allocator.free(man);
    try std.testing.expect(std.mem.indexOf(u8, man, "com.apple.CoreML/weights") != null);
    const wb = try tmp.dir.readFileAlloc(io, "t.mlpackage/" ++ weights_path, std.testing.allocator, .limited(1 << 20));
    defer std.testing.allocator.free(wb);
    try std.testing.expectEqual(@as(u32, 1), try blob.count(wb));
    const r = try blob.read(wb, 64);
    try std.testing.expectEqual(@as(usize, 64 * 4), r.data.len);
    const mb = try tmp.dir.readFileAlloc(io, "t.mlpackage/" ++ model_path, std.testing.allocator, .limited(1 << 20));
    defer std.testing.allocator.free(mb);
    try std.testing.expect(mb.len > 50);
}
