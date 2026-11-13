//! `zig build ios-coreml-probe-sign`: signs CoreMLProbe.app for a
//! development install (run on the Mac; codesign is Apple's). Adapted from
//! Portal's src/tools/ios_sign.zig, with the same environment variables so
//! one setup serves both:
//!
//!   ios-sign <App.app> <out dir> <bundle id>
//!
//!   PORTAL_SIGN_IDENTITY        "Apple Development: Name (ABCDE12345)"
//!                               (`security find-identity -v -p codesigning`)
//!   PORTAL_PROVISIONING_PROFILE a development .mobileprovision whose app id
//!                               matches the bundle id (or a wildcard) and
//!                               that lists the iPhone
//!   PORTAL_TEAM_ID              the 10-character team id
//!
//! Copies the app to <out dir>/<App.app>, embeds the profile, writes the
//! entitlements (application id, team, get-task-allow), signs, verifies.
const std = @import("std");
const Io = std.Io;

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    var args: std.process.Args.Iterator = try .initAllocator(init.minimal.args, gpa);
    defer args.deinit();
    _ = args.next();
    const app_in = args.next() orelse return usage();
    const out_dir = args.next() orelse return usage();
    const bundle_id = args.next() orelse return usage();

    const env = init.environ_map;
    const identity = env.get("PORTAL_SIGN_IDENTITY") orelse return missing("PORTAL_SIGN_IDENTITY");
    const profile = env.get("PORTAL_PROVISIONING_PROFILE") orelse return missing("PORTAL_PROVISIONING_PROFILE");
    const team = env.get("PORTAL_TEAM_ID") orelse return missing("PORTAL_TEAM_ID");

    const cwd = Io.Dir.cwd();
    const app = try std.fs.path.join(gpa, &.{ out_dir, std.fs.path.basename(app_in) });
    defer gpa.free(app);
    try copyTree(gpa, io, app_in, app);
    const embedded = try std.fs.path.join(gpa, &.{ app, "embedded.mobileprovision" });
    defer gpa.free(embedded);
    try cwd.copyFile(profile, cwd, embedded, io, .{});

    const ent_text = try entitlements(gpa, team, bundle_id);
    defer gpa.free(ent_text);
    const ent_path = try std.fmt.allocPrint(gpa, "{s}.entitlements", .{app});
    defer gpa.free(ent_path);
    try cwd.writeFile(io, .{ .sub_path = ent_path, .data = ent_text });
    try run(io, &.{ "codesign", "--force", "--timestamp=none", "--sign", identity, "--entitlements", ent_path, app });
    try run(io, &.{ "codesign", "--verify", "--deep", "--strict", "--verbose=2", app });
    std.debug.print("ios-sign: signed {s}\n", .{app});
}

fn copyTree(gpa: std.mem.Allocator, io: Io, from_path: []const u8, to_path: []const u8) !void {
    const cwd = Io.Dir.cwd();
    var from = try cwd.openDir(io, from_path, .{ .iterate = true });
    defer from.close(io);
    try cwd.createDirPath(io, to_path);
    var to = try cwd.openDir(io, to_path, .{});
    defer to.close(io);
    var walker = try from.walk(gpa);
    defer walker.deinit();
    while (try walker.next(io)) |e| switch (e.kind) {
        .directory => try to.createDirPath(io, e.path),
        else => try from.copyFile(e.path, to, e.path, io, .{}),
    };
}

fn usage() error{Usage} {
    std.debug.print("usage: ios-sign <App.app> <out dir> <bundle id>\n", .{});
    return error.Usage;
}

fn missing(name: []const u8) error{MissingEnvironment} {
    std.debug.print("ios-sign: set {s} (see README.md, \"On the iPhone\")\n", .{name});
    return error.MissingEnvironment;
}

fn run(io: Io, argv: []const []const u8) !void {
    var child = try std.process.spawn(io, .{ .argv = argv });
    const term = try child.wait(io);
    switch (term) {
        .exited => |code| if (code == 0) return,
        else => {},
    }
    std.debug.print("ios-sign: {s} failed\n", .{argv[0]});
    return error.CommandFailed;
}

/// Development entitlements (an XML property list).
pub fn entitlements(gpa: std.mem.Allocator, team: []const u8, bundle_id: []const u8) ![]u8 {
    return std.fmt.allocPrint(gpa,
        \\<?xml version="1.0" encoding="UTF-8"?>
        \\<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        \\<plist version="1.0">
        \\<dict>
        \\    <key>application-identifier</key>
        \\    <string>{s}.{s}</string>
        \\    <key>com.apple.developer.team-identifier</key>
        \\    <string>{s}</string>
        \\    <key>get-task-allow</key>
        \\    <true/>
        \\</dict>
        \\</plist>
        \\
    , .{ team, bundle_id, team });
}

test "entitlements" {
    const t = try entitlements(std.testing.allocator, "ABCDE12345", "local.coreml.probe");
    defer std.testing.allocator.free(t);
    try std.testing.expect(std.mem.indexOf(u8, t, "<string>ABCDE12345.local.coreml.probe</string>") != null);
}
