//! Objective-C runtime for iOS and macOS, with libobjc linked directly
//! (`-lobjc`) rather than dlopen'd, so the same code runs in an iPhone app
//! and a Mac command-line tool.
//!
//! The types (Object, Class, Selector, BOOL) and the comptime objc_msgSend
//! signature builder come from the owner's ../MacOS package; its OBJC.load()
//! opens AppKit (absent on iOS) and keeps its symbol table private, so the
//! symbol binding is done here, the same way the Renderer's IosObjC module
//! (src/Window/ios_objc.zig) does it. Only what Core ML needs is here:
//! selectors, classes, messages, autorelease pools, NSString / NSNumber /
//! NSArray / NSDictionary helpers, global blocks and a dispatch semaphore.
const std = @import("std");
const Mac = @import("MacOS");
pub const OBJC = Mac.OBJC;

pub const Object = OBJC.Object;
pub const Class = OBJC.Class;
pub const Selector = OBJC.Selector;
pub const BOOL = OBJC.BOOL;
pub const NSUInteger = usize;
pub const NSInteger = isize;

extern fn objc_getClass(name: [*:0]const u8) ?*Class;
extern fn objc_msgSend() void;
extern fn objc_autoreleasePoolPush() ?*anyopaque;
extern fn objc_autoreleasePoolPop(token: ?*anyopaque) void;
extern fn sel_registerName(name: [*:0]const u8) ?*Selector;
extern fn object_getClass(obj: ?*anyopaque) ?*Class;
extern fn class_getName(cls: ?*Class) [*:0]const u8;
extern fn class_respondsToSelector(cls: ?*Class, sel: *Selector) BOOL;

/// Registered selector for `name`, cached after the first call.
pub fn sel(comptime name: [:0]const u8) *Selector {
    const Cache = struct {
        var value: std.atomic.Value(usize) = .init(0);
    };
    var v = Cache.value.load(.acquire);
    if (v == 0) {
        v = @intFromPtr(sel_registerName(name.ptr) orelse @panic("sel_registerName failed: " ++ name));
        Cache.value.store(v, .release);
    }
    return @ptrFromInt(v);
}

/// Class named `name`, or null (e.g. a class newer than this OS).
pub fn lookUpClass(name: [*:0]const u8) ?*Class {
    return objc_getClass(name);
}

pub fn getClass(comptime name: [:0]const u8) *Class {
    return objc_getClass(name.ptr) orelse @panic("Objective-C class not found: " ++ name);
}

/// The class name of an object ("MLNeuralEngineComputeDevice", ...).
pub fn className(obj: ?*Object) []const u8 {
    if (obj == null) return "";
    return std.mem.span(class_getName(object_getClass(obj)));
}

pub fn respondsTo(obj: anytype, s: *Selector) bool {
    const o: ?*anyopaque = @ptrCast(obj);
    if (o == null) return false;
    return class_respondsToSelector(object_getClass(o), s) != 0;
}

/// `[target selector args...]` (non-struct returns). Argument types must
/// be concrete.
pub fn msgSend(comptime Ret: type, target: anytype, selector: *Selector, args: anytype) Ret {
    const F = OBJC.MsgSendFn(Ret, @TypeOf(args));
    const f: *const F = @ptrCast(&objc_msgSend);
    return @call(.auto, f, .{ @as(?*Object, @ptrCast(target)), selector } ++ args);
}

pub fn retain(obj: anytype) @TypeOf(obj) {
    return @ptrCast(msgSend(?*Object, obj, sel("retain"), .{}));
}
pub fn release(obj: anytype) void {
    msgSend(void, obj, sel("release"), .{});
}
pub fn autorelease(obj: anytype) @TypeOf(obj) {
    return @ptrCast(msgSend(?*Object, obj, sel("autorelease"), .{}));
}

pub const AutoreleasePool = struct {
    token: ?*anyopaque,
    pub fn push() AutoreleasePool {
        return .{ .token = objc_autoreleasePoolPush() };
    }
    pub fn pop(self: AutoreleasePool) void {
        objc_autoreleasePoolPop(self.token);
    }
};

// ------------------------------------------------------------ Foundation

const NSUTF8StringEncoding: NSUInteger = 4;

/// A new (+1) NSString from UTF-8 bytes.
pub fn newString(utf8: []const u8) ?*Object {
    const a = msgSend(?*Object, getClass("NSString"), sel("alloc"), .{}) orelse return null;
    return msgSend(?*Object, a, sel("initWithBytes:length:encoding:"), .{
        @as([*]const u8, utf8.ptr), @as(NSUInteger, utf8.len), NSUTF8StringEncoding,
    });
}

/// The UTF-8 bytes of an NSString (valid while the autorelease pool lives).
pub fn stringBytes(s: ?*Object) []const u8 {
    const p = msgSend(?[*:0]const u8, s orelse return "", sel("UTF8String"), .{}) orelse return "";
    return std.mem.span(p);
}

/// [NSNumber numberWithLongLong:] (autoreleased).
pub fn number(v: i64) ?*Object {
    return msgSend(?*Object, getClass("NSNumber"), sel("numberWithLongLong:"), .{@as(c_longlong, v)});
}

pub fn numberValue(n: ?*Object) i64 {
    return msgSend(c_longlong, n, sel("longLongValue"), .{});
}

/// [NSArray arrayWithObjects:count:] (autoreleased).
pub fn array(objs: []const ?*Object) ?*Object {
    return msgSend(?*Object, getClass("NSArray"), sel("arrayWithObjects:count:"), .{ @as([*]const ?*Object, objs.ptr), @as(NSUInteger, objs.len) });
}

pub fn arrayCount(a: ?*Object) usize {
    if (a == null) return 0;
    return msgSend(NSUInteger, a, sel("count"), .{});
}

pub fn arrayAt(a: ?*Object, i: usize) ?*Object {
    return msgSend(?*Object, a, sel("objectAtIndex:"), .{@as(NSUInteger, i)});
}

/// [NSDictionary dictionaryWithObjects:forKeys:count:] (autoreleased).
pub fn dictionary(keys: []const ?*Object, values: []const ?*Object) ?*Object {
    std.debug.assert(keys.len == values.len);
    return msgSend(?*Object, getClass("NSDictionary"), sel("dictionaryWithObjects:forKeys:count:"), .{
        @as([*]const ?*Object, values.ptr), @as([*]const ?*Object, keys.ptr), @as(NSUInteger, keys.len),
    });
}

pub fn dictionaryGet(d: ?*Object, key: ?*Object) ?*Object {
    return msgSend(?*Object, d, sel("objectForKey:"), .{key});
}

/// [NSURL fileURLWithPath:] for a UTF-8 path (autoreleased).
pub fn fileUrl(path: []const u8) ?*Object {
    const s = newString(path) orelse return null;
    defer release(s);
    return msgSend(?*Object, getClass("NSURL"), sel("fileURLWithPath:"), .{s});
}

/// The filesystem path of a file NSURL.
pub fn urlPath(url: ?*Object) []const u8 {
    return stringBytes(msgSend(?*Object, url, sel("path"), .{}));
}

/// NSError's localizedDescription (or "" for nil).
pub fn errorText(err: ?*Object) []const u8 {
    if (err == null) return "";
    return stringBytes(msgSend(?*Object, err, sel("localizedDescription"), .{}));
}

// ------------------------------------------------------------ blocks

/// A global (non-capturing) block literal, Clang ABI: isa =
/// _NSConcreteGlobalBlock, flags = BLOCK_IS_GLOBAL. Copying such a block
/// returns it unchanged, so one in static memory stays valid. State goes
/// through globals. `invoke` is `callconv(.c) fn (*const Block, args...)`.
pub const Block = extern struct {
    isa: ?*anyopaque,
    flags: c_int,
    reserved: c_int = 0,
    invoke: *const anyopaque,
    descriptor: *const Descriptor,

    pub const Descriptor = extern struct { reserved: c_ulong = 0, size: c_ulong };
    pub const default_descriptor: Descriptor = .{ .size = @sizeOf(Block) };
    const BLOCK_IS_GLOBAL: c_int = 1 << 28;

    pub fn global(invoke: *const anyopaque) Block {
        return .{ .isa = @ptrCast(&_NSConcreteGlobalBlock), .flags = BLOCK_IS_GLOBAL, .invoke = invoke, .descriptor = &default_descriptor };
    }
};
extern var _NSConcreteGlobalBlock: [32]usize;

// ------------------------------------------------------------ libdispatch

extern "c" fn dispatch_semaphore_create(value: isize) ?*Object;
extern "c" fn dispatch_semaphore_signal(sema: ?*Object) isize;
extern "c" fn dispatch_semaphore_wait(sema: ?*Object, timeout: u64) isize;
extern "c" fn dispatch_time(when: u64, delta: i64) u64;
pub const DispatchFn = *const fn (ctx: ?*anyopaque) callconv(.c) void;
extern "c" var _dispatch_main_q: u8;
extern "c" fn dispatch_async_f(queue: *anyopaque, ctx: ?*anyopaque, work: DispatchFn) void;
extern "c" fn dispatch_get_global_queue(identifier: isize, flags: usize) ?*anyopaque;

/// A counting semaphore for waiting on a completion handler.
pub const Semaphore = struct {
    obj: ?*Object,

    pub fn init() Semaphore {
        return .{ .obj = dispatch_semaphore_create(0) };
    }
    pub fn deinit(s: Semaphore) void {
        if (s.obj) |o| release(o);
    }
    pub fn signal(s: Semaphore) void {
        _ = dispatch_semaphore_signal(s.obj);
    }
    /// Blocks until signaled or `timeout_ns` passes; false on timeout.
    pub fn wait(s: Semaphore, timeout_ns: u64) bool {
        const deadline = dispatch_time(0, @intCast(@min(timeout_ns, std.math.maxInt(i64))));
        return dispatch_semaphore_wait(s.obj, deadline) == 0;
    }
};

/// Runs `work(ctx)` later on the main queue.
pub fn onMainAsync(ctx: ?*anyopaque, work: DispatchFn) void {
    dispatch_async_f(@ptrCast(&_dispatch_main_q), ctx, work);
}

/// Runs `work(ctx)` on a background (default QoS) queue.
pub fn onBackgroundAsync(ctx: ?*anyopaque, work: DispatchFn) void {
    const q = dispatch_get_global_queue(0, 0) orelse return work(ctx);
    dispatch_async_f(q, ctx, work);
}

// ------------------------------------------------------------ classes

extern fn objc_allocateClassPair(superclass: ?*Class, name: [*:0]const u8, extra: usize) ?*Class;
extern fn objc_registerClassPair(cls: *Class) void;
extern fn class_addMethod(cls: *Class, name: *Selector, imp: OBJC.Imp, types: [*:0]const u8) BOOL;

/// Builds a subclass at runtime (an app delegate).
pub const ClassBuilder = struct {
    class: *Class,

    /// Null if a class named `name` already exists.
    pub fn begin(superclass: *Class, name: [*:0]const u8) ?ClassBuilder {
        return .{ .class = objc_allocateClassPair(superclass, name, 0) orelse return null };
    }
    /// `imp`: callconv(.c) fn (self, _cmd, args...); `types`: ObjC type encoding.
    pub fn addMethod(self: ClassBuilder, comptime name: [:0]const u8, imp: anytype, types: [*:0]const u8) void {
        _ = class_addMethod(self.class, sel(name), @ptrCast(imp), types);
    }
    pub fn register(self: ClassBuilder) *Class {
        objc_registerClassPair(self.class);
        return self.class;
    }
};

// Foundation C functions.
extern fn NSLog(format: ?*Object, ...) void;
extern fn NSTemporaryDirectory() ?*Object;
extern fn NSSearchPathForDirectoriesInDomains(directory: NSUInteger, domains: NSUInteger, expand: BOOL) ?*Object;

/// NSLog("%s", text): visible in Console.app and `devicectl ... --console`.
pub fn log(text: []const u8) void {
    var buf: [1024]u8 = undefined;
    const n = @min(text.len, buf.len - 1);
    @memcpy(buf[0..n], text[0..n]);
    buf[n] = 0;
    const fmt = newString("%s") orelse return;
    defer release(fmt);
    NSLog(fmt, @as([*:0]const u8, buf[0..n :0]));
}

/// The app's temporary folder (no trailing slash guaranteed).
pub fn tempDir() []const u8 {
    return std.mem.trimEnd(u8, stringBytes(NSTemporaryDirectory()), "/");
}

/// The app's Documents folder (NSDocumentDirectory, user domain).
pub fn documentsDir() []const u8 {
    const list = NSSearchPathForDirectoriesInDomains(9, 1, 1);
    if (arrayCount(list) == 0) return "";
    return stringBytes(arrayAt(list, 0));
}

/// [[NSBundle mainBundle] resourcePath].
pub fn bundleResourcePath() []const u8 {
    const b = msgSend(?*Object, getClass("NSBundle"), sel("mainBundle"), .{});
    return stringBytes(msgSend(?*Object, b, sel("resourcePath"), .{}));
}
