//! `coreml-probe-gen <out dir> [ios16|ios17|ios18]`: writes the probe models
//! as `<name>.mlpackage` directories plus `<name>.ref` (inputs and host
//! reference outputs) into <out dir>. Runs on any host; the build calls it
//! to make the probe app's resources.
const std = @import("std");
const models = @import("models.zig");
const coreml = @import("coreml");

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    var args: std.process.Args.Iterator = try .initAllocator(init.minimal.args, gpa);
    defer args.deinit();
    _ = args.next();
    const out_path = args.next() orelse {
        std.debug.print("usage: coreml-probe-gen <out dir> [ios16|ios17|ios18]\n", .{});
        return error.Usage;
    };
    const target: coreml.mil.Target = if (args.next()) |t| std.meta.stringToEnum(coreml.mil.Target, t) orelse return error.Usage else .ios17;

    const cwd = std.Io.Dir.cwd();
    try cwd.createDirPath(io, out_path);
    var out = try cwd.openDir(io, out_path, .{});
    defer out.close(io);

    for (models.names) |name| {
        var x = try models.build(gpa, name, target);
        defer x.deinit();
        const pkg = try std.fmt.allocPrint(gpa, "{s}.mlpackage", .{name});
        defer gpa.free(pkg);
        try coreml.package.write(gpa, io, out, pkg, &x.b);

        const e16 = try models.reference(gpa, &x, true);
        defer models.freeNamed(gpa, e16);
        const e32 = try models.reference(gpa, &x, false);
        defer models.freeNamed(gpa, e32);
        var entries: std.ArrayList(models.Entry) = .empty;
        defer entries.deinit(gpa);
        for (x.inputs) |t| try entries.append(gpa, .{ .role = .input, .t = t });
        for (e16) |t| try entries.append(gpa, .{ .role = .expected, .t = t });
        for (e32) |t| try entries.append(gpa, .{ .role = .exact, .t = t });
        const bytes = try models.writeRef(gpa, entries.items);
        defer gpa.free(bytes);
        const rname = try std.fmt.allocPrint(gpa, "{s}.ref", .{name});
        defer gpa.free(rname);
        try out.writeFile(io, .{ .sub_path = rname, .data = bytes });
    }
}
