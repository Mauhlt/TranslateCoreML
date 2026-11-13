//! Build helper (copied from Portal src/tools/ios_zig_lib.zig) for the iOS
//! steps (`ios-check`, `ios-coreml-probe`): a copy of the running
//! Zig's lib directory with one line of std fixed, so std.Io.Threaded
//! compiles for iOS.
//!
//!   ios-zig-lib <zig lib dir> <output dir>
//!
//! Zig 0.18.0-dev.4: std/Io/Threaded.zig gives iOS (with wasi, tvOS,
//! visionOS, watchOS) a `NullFile` without an `fd` field, but
//! `getDevNullFd` (reached from `processReplace` -> `spawnDarwin`, and from
//! the default panic handler through `debug_io`) reads `null_file.fd` on
//! every Darwin target, so no iOS program using std.Io compiles. The fix
//! keeps only wasi on the fd-less struct. If a newer Zig no longer has the
//! line, the copy is made unchanged and a note says so (drop this helper
//! then). Report upstream.
//!
//! Only what an iOS build needs is copied (std, compiler_rt, the Darwin
//! libc stubs and headers): the other targets' libc sources stay behind.

const std = @import("std");
const Io = std.Io;

const broken = "    .wasi, .ios, .tvos, .visionos, .watchos => struct {\n        fn deinit(this: @This()) void {";
const fixed = "    .wasi => struct {\n        fn deinit(this: @This()) void {";

/// Paths (relative, '/'-separated) not copied.
const skip = [_][]const u8{
    "libc/glibc",      "libc/musl",    "libc/mingw",    "libc/freebsd", "libc/netbsd",
    "libc/openbsd",    "libc/wasi",    "libcxx",        "libcxxabi",    "libtsan",
    "libunwind",       "docs",         "build-web",     "lldb",
};

fn skipped(rel: []const u8) bool {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    if (rel.len > buf.len) return false;
    const p = buf[0..rel.len];
    for (rel, p) |c, *o| o.* = if (c == '\\') '/' else c;
    for (skip) |s| {
        if (std.mem.eql(u8, p, s)) return true;
        if (p.len > s.len and std.mem.startsWith(u8, p, s) and p[s.len] == '/') return true;
    }
    // libc headers: only the Darwin ones.
    if (std.mem.startsWith(u8, p, "libc/include/")) {
        const rest = p["libc/include/".len..];
        const top = rest[0 .. std.mem.indexOfScalar(u8, rest, '/') orelse rest.len];
        if (std.mem.indexOf(u8, top, "darwin") == null and std.mem.indexOf(u8, top, "macos") == null) return true;
    }
    return false;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    var args: std.process.Args.Iterator = try .initAllocator(init.minimal.args, gpa);
    defer args.deinit();
    _ = args.next();
    const from_path = args.next() orelse return error.Usage;
    const to_path = args.next() orelse return error.Usage;

    const cwd = Io.Dir.cwd();
    var from = try cwd.openDir(io, from_path, .{ .iterate = true });
    defer from.close(io);
    try cwd.createDirPath(io, to_path);
    var to = try cwd.openDir(io, to_path, .{});
    defer to.close(io);

    var walker = try from.walk(gpa);
    defer walker.deinit();
    while (try walker.next(io)) |e| {
        if (skipped(e.path)) continue;
        switch (e.kind) {
            .directory => try to.createDirPath(io, e.path),
            else => try from.copyFile(e.path, to, e.path, io, .{}),
        }
    }

    const threaded = "std/Io/Threaded.zig";
    const src = try to.readFileAlloc(io, threaded, gpa, .limited(64 << 20));
    defer gpa.free(src);
    const n = std.mem.count(u8, src, broken);
    if (n == 0) {
        std.debug.print("ios-zig-lib: std/Io/Threaded.zig no longer has the iOS NullFile line; copied unchanged (this helper may be obsolete)\n", .{});
        return;
    }
    const out = try std.mem.replaceOwned(u8, gpa, src, broken, fixed);
    defer gpa.free(out);
    try to.writeFile(io, .{ .sub_path = threaded, .data = out });
}

test "skip list" {
    try std.testing.expect(skipped("libc/glibc/abilists"));
    try std.testing.expect(skipped("libc\\musl"));
    try std.testing.expect(!skipped("libc/darwin/libSystem.tbd"));
    try std.testing.expect(!skipped("libc/include/any-darwin-any/stdio.h"));
    try std.testing.expect(skipped("libc/include/x86_64-linux-gnu/bits"));
    try std.testing.expect(!skipped("std/Io/Threaded.zig"));
    try std.testing.expect(!skipped("compiler_rt.zig"));
}
