//! CoreMLProbe.app: the probe on an iPhone. A window with a text view that
//! shows the log; the probe runs on a background queue (model compilation
//! can take longer than iOS allows at launch), writes
//! Documents/coreml-probe.json (visible in Files / Finder thanks to
//! UIFileSharingEnabled) and NSLogs every line. The models are in the app
//! bundle's `probe/` folder (from coreml-probe-gen at build time).
const std = @import("std");
const coreml = @import("coreml");
const objc = coreml.objc;
const run = @import("run.zig");
const Object = objc.Object;
const sel = objc.sel;
const msgSend = objc.msgSend;

extern fn UIApplicationMain(argc: c_int, argv: ?[*]?[*:0]u8, principal: ?*Object, delegate: ?*Object) c_int;
extern "c" fn _NSGetArgc() *c_int;
extern "c" fn _NSGetArgv() *?[*]?[*:0]u8;

const CGRect = extern struct { x: f64, y: f64, w: f64, h: f64 };

const G = struct {
    var gpa: std.mem.Allocator = undefined;
    var io: std.Io = undefined;
    var text_view: ?*Object = null;
    var window: ?*Object = null;
    var lock: std.atomic.Mutex = .unlocked;
    var text: std.ArrayList(u8) = .empty;
};

comptime {
    for (coreml.runtime.all_decls) |f| _ = &f;
}

pub fn main(init: std.process.Init) void {
    G.gpa = init.gpa;
    G.io = init.io;
    const pool = objc.AutoreleasePool.push();
    const cb = objc.ClassBuilder.begin(objc.getClass("UIResponder"), "CMLProbeAppDelegate") orelse @panic("delegate class exists");
    cb.addMethod("application:didFinishLaunchingWithOptions:", &didFinishLaunching, "c@:@@");
    _ = cb.register();
    const name = objc.newString("CMLProbeAppDelegate");
    pool.pop();
    _ = UIApplicationMain(_NSGetArgc().*, _NSGetArgv().*, null, name);
}

fn didFinishLaunching(_: *Object, _: *objc.Selector, _: ?*Object, _: ?*Object) callconv(.c) objc.BOOL {
    const screen = msgSend(?*Object, objc.getClass("UIScreen"), sel("mainScreen"), .{});
    const bounds = msgSend(CGRect, screen, sel("bounds"), .{});
    const win = msgSend(?*Object, msgSend(?*Object, objc.getClass("UIWindow"), sel("alloc"), .{}), sel("initWithFrame:"), .{bounds});
    const vc = msgSend(?*Object, msgSend(?*Object, objc.getClass("UIViewController"), sel("alloc"), .{}), sel("init"), .{});
    const tv = msgSend(?*Object, msgSend(?*Object, objc.getClass("UITextView"), sel("alloc"), .{}), sel("initWithFrame:"), .{bounds});
    msgSend(void, tv, sel("setEditable:"), .{@as(objc.BOOL, 0)});
    msgSend(void, vc, sel("setView:"), .{tv});
    msgSend(void, win, sel("setRootViewController:"), .{vc});
    msgSend(void, win, sel("makeKeyAndVisible"), .{});
    G.window = win;
    G.text_view = tv;
    logLine("CoreMLProbe: running (results also go to Documents/coreml-probe.json)");
    objc.onBackgroundAsync(null, &work);
    return 1;
}

fn logLine(s: []const u8) void {
    objc.log(s);
    while (!G.lock.tryLock()) std.atomic.spinLoopHint();
    G.text.appendSlice(G.gpa, s) catch {};
    G.text.append(G.gpa, '\n') catch {};
    G.lock.unlock();
    objc.onMainAsync(null, &showText);
}

fn showText(_: ?*anyopaque) callconv(.c) void {
    const pool = objc.AutoreleasePool.push();
    defer pool.pop();
    while (!G.lock.tryLock()) std.atomic.spinLoopHint();
    const s = objc.newString(G.text.items);
    G.lock.unlock();
    defer if (s) |x| objc.release(x);
    msgSend(void, G.text_view, sel("setText:"), .{s});
}

fn work(_: ?*anyopaque) callconv(.c) void {
    const pool = objc.AutoreleasePool.push();
    defer pool.pop();
    const gpa = G.gpa;
    const res = std.fmt.allocPrint(gpa, "{s}/probe", .{objc.bundleResourcePath()}) catch return;
    defer gpa.free(res);
    const wdir = std.fmt.allocPrint(gpa, "{s}/coreml-probe", .{objc.tempDir()}) catch return;
    defer gpa.free(wdir);
    std.Io.Dir.cwd().createDirPath(G.io, wdir) catch {};
    const report = run.run(gpa, G.io, .{ .resources = res, .work = wdir, .log = logLine }) catch |e| {
        var buf: [128]u8 = undefined;
        logLine(std.fmt.bufPrint(&buf, "probe failed: {t}", .{e}) catch "probe failed");
        return;
    };
    defer gpa.free(report);
    const out = std.fmt.allocPrint(gpa, "{s}/coreml-probe.json", .{objc.documentsDir()}) catch return;
    defer gpa.free(out);
    std.Io.Dir.cwd().writeFile(G.io, .{ .sub_path = out, .data = report }) catch {
        logLine("could not write the report");
        return;
    };
    logLine("done: Documents/coreml-probe.json");
}
