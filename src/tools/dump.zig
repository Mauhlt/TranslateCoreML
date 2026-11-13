//! `mlmodel-dump <model.mlmodel | x.mlpackage>`: prints what decode.zig reads
//! from a Core ML model spec (description, function inputs, ops with their
//! bindings and output types, const values), for comparing packages made
//! here with ones made by other tools.
const std = @import("std");
const coreml = @import("coreml");
const decode = coreml.decode;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    var args: std.process.Args.Iterator = try .initAllocator(init.minimal.args, gpa);
    defer args.deinit();
    _ = args.next();
    const path_in = args.next() orelse {
        std.debug.print("usage: mlmodel-dump <model.mlmodel | x.mlpackage> [max ops]\n", .{});
        return error.Usage;
    };
    const max_ops = if (args.next()) |m| try std.fmt.parseInt(usize, m, 10) else std.math.maxInt(usize);
    const path = if (std.mem.endsWith(u8, std.mem.trimEnd(u8, path_in, "/\\"), ".mlpackage"))
        try std.fs.path.join(gpa, &.{ path_in, coreml.package.model_path })
    else
        try gpa.dupe(u8, path_in);
    defer gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 30));
    defer gpa.free(bytes);
    var s = try decode.decode(gpa, bytes);
    defer s.deinit();
    const p = std.debug.print;
    p("specificationVersion {d}, opset {s}\n", .{ s.spec_version, s.opset });
    for (s.inputs) |t| p("input  {s} type {d} shape {any}\n", .{ t.name, t.dtype, t.shape });
    for (s.outputs) |t| p("output {s} type {d} shape {any}\n", .{ t.name, t.dtype, t.shape });
    for (s.states) |t| p("state  {s} type {d} shape {any}\n", .{ t.name, t.dtype, t.shape });
    for (s.function_inputs) |t| p("main({s}: dtype {d} {any}{s})\n", .{ t.name, t.dtype, t.shape, if (t.is_state) " state" else "" });
    for (s.block_outputs) |o| p("block output {s}\n", .{o});
    for (s.ops[0..@min(max_ops, s.ops.len)]) |o| {
        p("{s} [{s}]", .{ o.type, o.name });
        for (o.inputs) |b| {
            p(" {s}=", .{b.param});
            for (b.names, 0..) |n, i| p("{s}{s}", .{ if (i > 0) "," else "", n });
        }
        for (o.outputs) |t| p(" -> {s}: dtype {d} {any}", .{ t.name, t.dtype, t.shape });
        switch (o.val) {
            .none => {},
            .blob => |bl| p(" = blob {s} @{d}", .{ bl.file, bl.offset }),
            .floats => |f| p(" = floats ({d} bytes)", .{f.len}),
            .halfs => |f| p(" = fp16 bytes ({d})", .{f.len}),
            .ints => |v| p(" = ints {any}", .{v[0..@min(v.len, 8)]}),
            .bools => |v| p(" = bools {any}", .{v[0..@min(v.len, 8)]}),
            .string => |v| p(" = \"{s}\"", .{v}),
        }
        p("\n", .{});
    }
    p("{d} ops\n", .{s.ops.len});
}
