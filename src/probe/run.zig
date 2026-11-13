//! The probe itself (Apple targets): for every probe model in a resource
//! folder (`<name>.mlpackage` + `<name>.ref` from coreml-probe-gen), compile
//! it with Core ML, load it with each compute-unit setting, run it, compare
//! the outputs with the host reference, time it, and ask MLComputePlan
//! where each op runs. The result is a JSON report (and log lines).
//!
//! What proves ANE placement: in the report, "plan" for compute units
//! "cpu_and_neural_engine" lists per op `"device":"neural_engine"`, and
//! "ops_on" counts them; plus the ANE run is usually much faster than
//! cpu_only for the block model. MLComputePlan needs iOS 17.4 / macOS 14.4;
//! on older systems "plan" is "unavailable" and only timing hints remain.
const std = @import("std");
const Io = std.Io;
const Allocator = std.mem.Allocator;
const coreml = @import("coreml");
const rt = coreml.runtime;
const models = @import("models.zig");

pub const Options = struct {
    /// Folder with the .mlpackage / .ref files.
    resources: []const u8,
    /// Writable folder for compiled models.
    work: []const u8,
    runs: usize = 10,
    /// Called with each human-readable line as it is produced.
    log: *const fn ([]const u8) void,
};

const units_list = [_]rt.ComputeUnits{ .cpu_only, .cpu_and_gpu, .cpu_and_neural_engine, .all };

fn now(io: Io) i96 {
    return Io.Clock.awake.now(io).nanoseconds;
}
fn msSince(io: Io, t0: i96) f64 {
    return @as(f64, @floatFromInt(now(io) - t0)) / 1e6;
}

/// Runs everything; returns the JSON report (caller frees).
pub fn run(gpa: Allocator, io: Io, opts: Options) ![]u8 {
    var j: std.ArrayList(u8) = .empty;
    errdefer j.deinit(gpa);
    var line_buf: [512]u8 = undefined;
    const log = struct {
        fn f(o: Options, buf: []u8, comptime fmt: []const u8, args: anytype) void {
            o.log(std.fmt.bufPrint(buf, fmt, args) catch buf);
        }
    }.f;

    try j.appendSlice(gpa, "{\n  \"probe\": \"TranslateCoreML coreml-probe 1\",\n");
    if (rt.availableDevices()) |d| {
        try j.print(gpa, "  \"devices\": {{\"cpu\": {}, \"gpu\": {}, \"neural_engine\": {}, \"ane_cores\": {d}}},\n", .{ d.cpu, d.gpu, d.neural_engine, d.ane_cores });
        log(opts, &line_buf, "devices: cpu={} gpu={} neural_engine={} (ANE cores {d})", .{ d.cpu, d.gpu, d.neural_engine, d.ane_cores });
    } else try j.appendSlice(gpa, "  \"devices\": \"unavailable (needs iOS 17 / macOS 14)\",\n");
    try j.appendSlice(gpa, "  \"models\": [\n");

    for (models.names, 0..) |name, mi| {
        if (mi > 0) try j.appendSlice(gpa, ",\n");
        try j.print(gpa, "    {{\"name\": \"{s}\"", .{name});
        modelReport(gpa, io, opts, name, &j, &line_buf) catch |e| {
            try j.print(gpa, ", \"error\": \"{t}\", \"coreml\": \"{s}\"", .{ e, jsonSafe(rt.lastError()) });
            log(opts, &line_buf, "{s}: FAILED {t} {s}", .{ name, e, rt.lastError() });
        };
        try j.appendSlice(gpa, "}");
    }
    try j.appendSlice(gpa, "\n  ]\n}\n");
    return j.toOwnedSlice(gpa);
}

/// Replaces characters that would break a JSON string.
fn jsonSafe(s: []const u8) []const u8 {
    const T = struct {
        threadlocal var buf: [1024]u8 = undefined;
    };
    const n = @min(s.len, T.buf.len);
    for (s[0..n], T.buf[0..n]) |c, *o| o.* = if (c == '"' or c == '\\' or c < 0x20) '\'' else c;
    return T.buf[0..n];
}

fn modelReport(gpa: Allocator, io: Io, opts: Options, name: []const u8, j: *std.ArrayList(u8), line_buf: []u8) !void {
    const log = struct {
        fn f(o: Options, buf: []u8, comptime fmt: []const u8, args: anytype) void {
            o.log(std.fmt.bufPrint(buf, fmt, args) catch buf);
        }
    }.f;
    const ref_path = try std.fmt.allocPrint(gpa, "{s}/{s}.ref", .{ opts.resources, name });
    defer gpa.free(ref_path);
    const ref_bytes = try Io.Dir.cwd().readFileAlloc(io, ref_path, gpa, .limited(64 << 20));
    defer gpa.free(ref_bytes);
    const entries = try models.readRef(gpa, ref_bytes);
    defer models.freeRef(gpa, entries);

    const pkg = try std.fmt.allocPrint(gpa, "{s}/{s}.mlpackage", .{ opts.resources, name });
    defer gpa.free(pkg);
    var t0 = now(io);
    const compiled = try rt.compile(gpa, pkg, opts.work, name);
    defer gpa.free(compiled);
    const compile_ms = msSince(io, t0);
    try j.print(gpa, ", \"compile_ms\": {d:.1}, \"runs\": [", .{compile_ms});
    log(opts, line_buf, "{s}: compiled in {d:.1} ms", .{ name, compile_ms });

    for (units_list, 0..) |units, ui| {
        if (ui > 0) try j.appendSlice(gpa, ",");
        try j.print(gpa, "\n      {{\"units\": \"{s}\"", .{units.name()});
        unitsReport(gpa, io, opts, compiled, units, entries, j) catch |e| {
            try j.print(gpa, ", \"error\": \"{t}\", \"coreml\": \"{s}\"", .{ e, jsonSafe(rt.lastError()) });
            log(opts, line_buf, "{s} [{s}]: FAILED {t} {s}", .{ name, units.name(), e, rt.lastError() });
        };
        try j.appendSlice(gpa, "}");
    }
    try j.appendSlice(gpa, "\n    ]");

    // Placement
    t0 = now(io);
    if (rt.computePlan(gpa, compiled, .cpu_and_neural_engine, 120 * std.time.ns_per_s)) |plan_c| {
        var plan = plan_c;
        defer plan.deinit();
        const ane = plan.countOn(.neural_engine);
        const gpu = plan.countOn(.gpu);
        const cpu = plan.countOn(.cpu);
        try j.print(gpa, ", \"plan\": {{\"units\": \"cpu_and_neural_engine\", \"ops_on\": {{\"neural_engine\": {d}, \"gpu\": {d}, \"cpu\": {d}}}, \"ops\": [", .{ ane, gpu, cpu });
        for (plan.ops, 0..) |o, i| {
            if (i > 0) try j.appendSlice(gpa, ",");
            try j.print(gpa, "\n        {{\"op\": \"{s}\", \"output\": \"{s}\", \"device\": \"{t}\", \"ane_ok\": {}, \"cost\": {d:.4}}}", .{ o.op, o.output, o.device, o.supported.neural_engine, o.cost });
        }
        try j.appendSlice(gpa, "]}");
        log(opts, line_buf, "{s}: plan (cpu_and_neural_engine): {d} ops on neural_engine, {d} gpu, {d} cpu", .{ name, ane, gpu, cpu });
    } else |e| {
        try j.print(gpa, ", \"plan\": \"{t}\"", .{e});
        log(opts, line_buf, "{s}: no compute plan ({t}{s}{s})", .{ name, e, if (rt.lastError().len > 0) ": " else "", rt.lastError() });
    }
}

fn unitsReport(gpa: Allocator, io: Io, opts: Options, compiled: []const u8, units: rt.ComputeUnits, entries: []const models.Entry, j: *std.ArrayList(u8)) !void {
    var line: [512]u8 = undefined;
    var t0 = now(io);
    var m = try rt.Model.load(compiled, units);
    defer m.deinit();
    const load_ms = msSince(io, t0);

    // Inputs: the model description decides the array type (fp32 / fp16).
    var arrays: [8]rt.MultiArray = undefined;
    var inputs: [8]rt.Model.Input = undefined;
    var n_in: usize = 0;
    defer for (arrays[0..n_in]) |*a| a.deinit();
    const want_fp16 = std.mem.indexOf(u8, compiled, "fp16io") != null;
    for (entries) |e| if (e.role == .input) {
        arrays[n_in] = try rt.MultiArray.create(e.t.shape, if (want_fp16) .float16 else .float32);
        arrays[n_in].writeF32(e.t.data);
        inputs[n_in] = .{ .name = e.t.name, .array = arrays[n_in] };
        n_in += 1;
    };

    t0 = now(io);
    var first = try m.predict(inputs[0..n_in]);
    first.deinit();
    const first_ms = msSince(io, t0);
    var times: [64]f64 = undefined;
    const runs = @min(opts.runs, times.len);
    var last: ?rt.Prediction = null;
    defer if (last) |*p| p.deinit();
    for (times[0..runs]) |*t| {
        if (last) |*p| p.deinit();
        last = null;
        t0 = now(io);
        last = try m.predict(inputs[0..n_in]);
        t.* = msSince(io, t0);
    }
    std.mem.sort(f64, times[0..runs], {}, std.sort.asc(f64));
    const median = if (runs > 0) times[runs / 2] else first_ms;

    try j.print(gpa, ", \"load_ms\": {d:.1}, \"first_ms\": {d:.3}, \"median_ms\": {d:.3}", .{ load_ms, first_ms, median });
    var worst: models.Diff = .{ .max_abs = 0, .rel = 0 };
    var worst_exact: models.Diff = .{ .max_abs = 0, .rel = 0 };
    for (entries) |e| if (e.role != .input) {
        var out = (last.?.array(e.t.name)) orelse return error.MissingOutput;
        defer out.deinit();
        const got = try gpa.alloc(f32, out.count());
        defer gpa.free(got);
        out.readF32(got);
        const d = models.diff(got, e.t.data);
        const w = if (e.role == .expected) &worst else &worst_exact;
        w.max_abs = @max(w.max_abs, d.max_abs);
        w.rel = @max(w.rel, d.rel);
    };
    // fp16 compute: a relative error of a few 1e-3 is normal; 1e-2 is the bar.
    const ok = worst.rel < 1e-2;
    try j.print(gpa, ", \"vs_fp16_ref\": {{\"max_abs\": {e:.3}, \"rel\": {e:.3}}}, \"vs_f32_ref\": {{\"max_abs\": {e:.3}, \"rel\": {e:.3}}}, \"ok\": {}", .{ worst.max_abs, worst.rel, worst_exact.max_abs, worst_exact.rel, ok });
    opts.log(std.fmt.bufPrint(&line, "{s} [{s}]: load {d:.1} ms, first {d:.2} ms, median {d:.3} ms, rel err {e:.2} (vs f32 {e:.2}) {s}", .{
        std.fs.path.stem(compiled), units.name(), load_ms, first_ms, median, worst.rel, worst_exact.rel, if (ok) "OK" else "MISMATCH",
    }) catch &line);
}
