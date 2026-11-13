//! `coreml-probe` on a Mac (Apple silicon has the same Neural Engine
//! family as the iPhone):
//!
//!   coreml-probe <resources dir> [--work <dir>] [--out <report.json>] [--runs N]
//!
//! <resources dir> is what coreml-probe-gen wrote (zig-out/probe).
const std = @import("std");
const coreml = @import("coreml");
const run = @import("run.zig");

fn logLine(s: []const u8) void {
    std.debug.print("{s}\n", .{s});
}

comptime {
    for (coreml.runtime.all_decls) |f| _ = &f;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    var args: std.process.Args.Iterator = try .initAllocator(init.minimal.args, gpa);
    defer args.deinit();
    _ = args.next();
    var resources: ?[]const u8 = null;
    var work: []const u8 = "/tmp/coreml-probe";
    var out: []const u8 = "coreml-probe.json";
    var runs: usize = 10;
    while (args.next()) |a| {
        if (std.mem.eql(u8, a, "--work")) {
            work = args.next() orelse return error.Usage;
        } else if (std.mem.eql(u8, a, "--out")) {
            out = args.next() orelse return error.Usage;
        } else if (std.mem.eql(u8, a, "--runs")) {
            runs = try std.fmt.parseInt(usize, args.next() orelse return error.Usage, 10);
        } else resources = a;
    }
    const res = resources orelse {
        std.debug.print("usage: coreml-probe <resources dir> [--work <dir>] [--out <report.json>] [--runs N]\n", .{});
        return error.Usage;
    };
    try std.Io.Dir.cwd().createDirPath(io, work);
    const pool = coreml.objc.AutoreleasePool.push();
    defer pool.pop();
    const report = try run.run(gpa, io, .{ .resources = res, .work = work, .runs = runs, .log = logLine });
    defer gpa.free(report);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out, .data = report });
    std.debug.print("report: {s}\n", .{out});
}
