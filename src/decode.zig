//! Reads back the Model protobuf this package writes (the same subset of
//! fields, spec.zig) into a plain summary: used by the tests to check the
//! encoder round-trips, and by `mlmodel-dump` to inspect a package.
const std = @import("std");
const Allocator = std.mem.Allocator;
const pb = @import("pb.zig");
const spec = @import("spec.zig");

pub const TensorDesc = struct {
    name: []const u8,
    dtype: i32,
    shape: []i64,
    is_state: bool = false,
};

pub const ConstVal = union(enum) {
    none,
    blob: struct { file: []const u8, offset: u64 },
    floats: []const u8, // packed f32 bytes
    halfs: []const u8, // raw fp16 bytes
    ints: []i64,
    bools: []i64,
    string: []const u8,
};

pub const Binding = struct { param: []const u8, names: [][]const u8 };

pub const OpDesc = struct {
    type: []const u8,
    name: []const u8,
    inputs: []Binding,
    outputs: []TensorDesc,
    val: ConstVal = .none,
};

pub const Summary = struct {
    arena: std.heap.ArenaAllocator,
    spec_version: i64 = 0,
    inputs: []TensorDesc = &.{},
    outputs: []TensorDesc = &.{},
    states: []TensorDesc = &.{},
    opset: []const u8 = "",
    function_inputs: []TensorDesc = &.{},
    block_outputs: [][]const u8 = &.{},
    ops: []OpDesc = &.{},

    pub fn deinit(s: *Summary) void {
        s.arena.deinit();
    }

    pub fn op(s: *const Summary, name: []const u8) ?OpDesc {
        for (s.ops) |o| if (std.mem.eql(u8, o.name, name)) return o;
        return null;
    }
};

pub const Error = pb.DecodeError || error{ OutOfMemory, BadModel };

pub fn decode(gpa: Allocator, model: []const u8) Error!Summary {
    var s: Summary = .{ .arena = .init(gpa) };
    errdefer s.arena.deinit();
    const a = s.arena.allocator();
    var r = pb.Reader.init(model);
    var ins: std.ArrayList(TensorDesc) = .empty;
    var outs: std.ArrayList(TensorDesc) = .empty;
    var states: std.ArrayList(TensorDesc) = .empty;
    while (try r.next()) |f| switch (f.number) {
        spec.Model.specification_version => s.spec_version = f.asI64(),
        spec.Model.description => {
            var d = pb.Reader.init(f.bytes);
            while (try d.next()) |g| switch (g.number) {
                spec.ModelDescription.input => try ins.append(a, try feature(a, g.bytes)),
                spec.ModelDescription.output => try outs.append(a, try feature(a, g.bytes)),
                spec.ModelDescription.state => try states.append(a, try feature(a, g.bytes)),
                else => {},
            };
        },
        spec.Model.ml_program => try program(a, &s, f.bytes),
        else => {},
    };
    s.inputs = ins.items;
    s.outputs = outs.items;
    s.states = states.items;
    return s;
}

fn feature(a: Allocator, body: []const u8) Error!TensorDesc {
    var t: TensorDesc = .{ .name = "", .dtype = 0, .shape = &.{} };
    t.name = if (try pb.Reader.find(body, spec.FeatureDescription.name)) |f| f.bytes else "";
    var ft = (try pb.Reader.find(body, spec.FeatureDescription.type)) orelse return error.BadModel;
    var arr = try pb.Reader.find(ft.bytes, spec.FeatureType.multi_array_type);
    if (arr == null) {
        const st = (try pb.Reader.find(ft.bytes, spec.FeatureType.state_type)) orelse return error.BadModel;
        t.is_state = true;
        ft = st;
        arr = try pb.Reader.find(st.bytes, spec.StateFeatureType.array_type);
    }
    const at = arr orelse return error.BadModel;
    if (try pb.Reader.find(at.bytes, spec.ArrayFeatureType.shape)) |sh| t.shape = try pb.unpackVarints(a, sh.bytes);
    if (try pb.Reader.find(at.bytes, spec.ArrayFeatureType.data_type)) |dt| t.dtype = @intCast(dt.asI64());
    return t;
}

fn program(a: Allocator, s: *Summary, body: []const u8) Error!void {
    const fe = (try pb.Reader.find(body, spec.Program.functions)) orelse return error.BadModel;
    const key = (try pb.Reader.find(fe.bytes, spec.MapEntry.key)) orelse return error.BadModel;
    if (!std.mem.eql(u8, key.bytes, "main")) return error.BadModel;
    const func = (try pb.Reader.find(fe.bytes, spec.MapEntry.value)) orelse return error.BadModel;
    var fins: std.ArrayList(TensorDesc) = .empty;
    var r = pb.Reader.init(func.bytes);
    var block: []const u8 = &.{};
    while (try r.next()) |f| switch (f.number) {
        spec.Function.inputs => try fins.append(a, try namedValueType(a, f.bytes)),
        spec.Function.opset => s.opset = f.bytes,
        spec.Function.block_specializations => {
            const k = (try pb.Reader.find(f.bytes, spec.MapEntry.key)) orelse return error.BadModel;
            if (!std.mem.eql(u8, k.bytes, s.opset) and s.opset.len > 0) return error.BadModel;
            block = ((try pb.Reader.find(f.bytes, spec.MapEntry.value)) orelse return error.BadModel).bytes;
        },
        else => {},
    };
    s.function_inputs = fins.items;
    var bouts: std.ArrayList([]const u8) = .empty;
    var ops: std.ArrayList(OpDesc) = .empty;
    var br = pb.Reader.init(block);
    while (try br.next()) |f| switch (f.number) {
        spec.Block.outputs => try bouts.append(a, f.bytes),
        spec.Block.operations => try ops.append(a, try operation(a, f.bytes)),
        else => {},
    };
    s.block_outputs = bouts.items;
    s.ops = ops.items;
}

fn namedValueType(a: Allocator, body: []const u8) Error!TensorDesc {
    var t: TensorDesc = .{ .name = "", .dtype = 0, .shape = &.{} };
    t.name = if (try pb.Reader.find(body, spec.NamedValueType.name)) |f| f.bytes else "";
    const vt = (try pb.Reader.find(body, spec.NamedValueType.type)) orelse return error.BadModel;
    try valueType(a, vt.bytes, &t);
    return t;
}

fn valueType(a: Allocator, body: []const u8, t: *TensorDesc) Error!void {
    var tt = try pb.Reader.find(body, spec.ValueType.tensor_type);
    if (tt == null) {
        const st = (try pb.Reader.find(body, spec.ValueType.state_type)) orelse return error.BadModel;
        const wt = (try pb.Reader.find(st.bytes, spec.StateType.wrapped_type)) orelse return error.BadModel;
        t.is_state = true;
        tt = try pb.Reader.find(wt.bytes, spec.ValueType.tensor_type);
    }
    const tb = (tt orelse return error.BadModel).bytes;
    t.dtype = if (try pb.Reader.find(tb, spec.TensorType.data_type)) |f| @intCast(f.asI64()) else 0;
    const rank: usize = if (try pb.Reader.find(tb, spec.TensorType.rank)) |f| @intCast(f.asI64()) else 0;
    const shape = try a.alloc(i64, rank);
    var n: usize = 0;
    var r = pb.Reader.init(tb);
    while (try r.next()) |f| if (f.number == spec.TensorType.dimensions) {
        if (n >= rank) return error.BadModel;
        const c = (try pb.Reader.find(f.bytes, spec.Dimension.constant)) orelse return error.BadModel;
        shape[n] = if (try pb.Reader.find(c.bytes, spec.Dimension.constant_size)) |sz| sz.asI64() else 0;
        n += 1;
    };
    if (n != rank) return error.BadModel;
    t.shape = shape;
}

fn operation(a: Allocator, body: []const u8) Error!OpDesc {
    var o: OpDesc = .{ .type = "", .name = "", .inputs = &.{}, .outputs = &.{} };
    var ins: std.ArrayList(Binding) = .empty;
    var outs: std.ArrayList(TensorDesc) = .empty;
    var r = pb.Reader.init(body);
    while (try r.next()) |f| switch (f.number) {
        spec.Operation.type => o.type = f.bytes,
        spec.Operation.inputs => {
            const k = (try pb.Reader.find(f.bytes, spec.MapEntry.key)) orelse return error.BadModel;
            const v = (try pb.Reader.find(f.bytes, spec.MapEntry.value)) orelse return error.BadModel;
            var names: std.ArrayList([]const u8) = .empty;
            var ar = pb.Reader.init(v.bytes);
            while (try ar.next()) |bf| if (bf.number == spec.Argument.arguments) {
                const nm = (try pb.Reader.find(bf.bytes, spec.Argument.Binding.name)) orelse return error.BadModel;
                try names.append(a, nm.bytes);
            };
            try ins.append(a, .{ .param = k.bytes, .names = names.items });
        },
        spec.Operation.outputs => try outs.append(a, try namedValueType(a, f.bytes)),
        spec.Operation.attributes => {
            const k = (try pb.Reader.find(f.bytes, spec.MapEntry.key)) orelse return error.BadModel;
            const v = (try pb.Reader.find(f.bytes, spec.MapEntry.value)) orelse return error.BadModel;
            if (std.mem.eql(u8, k.bytes, "name")) {
                o.name = try scalarString(v.bytes);
            } else if (std.mem.eql(u8, k.bytes, "val")) {
                o.val = try constValue(a, v.bytes);
            }
        },
        else => {},
    };
    o.inputs = ins.items;
    o.outputs = outs.items;
    return o;
}

fn tensorValue(value: []const u8) Error!?[]const u8 {
    const iv = (try pb.Reader.find(value, spec.Value.immediate_value)) orelse return null;
    const t = (try pb.Reader.find(iv.bytes, spec.ImmediateValue.tensor)) orelse return error.BadModel;
    return t.bytes;
}

fn scalarString(value: []const u8) Error![]const u8 {
    const t = (try tensorValue(value)) orelse return error.BadModel;
    const ss = (try pb.Reader.find(t, spec.TensorValue.strings)) orelse return error.BadModel;
    return ((try pb.Reader.find(ss.bytes, spec.TensorValue.values)) orelse return error.BadModel).bytes;
}

fn constValue(a: Allocator, value: []const u8) Error!ConstVal {
    if (try pb.Reader.find(value, spec.Value.blob_file_value)) |bf| {
        const name = (try pb.Reader.find(bf.bytes, spec.BlobFileValue.file_name)) orelse return error.BadModel;
        const off = (try pb.Reader.find(bf.bytes, spec.BlobFileValue.offset)) orelse return error.BadModel;
        return .{ .blob = .{ .file = name.bytes, .offset = off.int } };
    }
    const t = (try tensorValue(value)) orelse return error.BadModel;
    var r = pb.Reader.init(t);
    const f = (try r.next()) orelse return error.BadModel;
    const inner: []const u8 = if (try pb.Reader.find(f.bytes, spec.TensorValue.values)) |x| x.bytes else &.{};
    return switch (f.number) {
        spec.TensorValue.floats => .{ .floats = inner },
        spec.TensorValue.bytes => .{ .halfs = inner },
        spec.TensorValue.ints => .{ .ints = try pb.unpackVarints(a, inner) },
        spec.TensorValue.bools => .{ .bools = try pb.unpackVarints(a, inner) },
        spec.TensorValue.strings => .{ .string = inner },
        else => error.BadModel,
    };
}

// ------------------------------------------------------------------- tests

const mil = @import("mil.zig");
const blob = @import("blob.zig");

test "round trip: builder -> protobuf -> summary" {
    const gpa = std.testing.allocator;
    var b = try mil.Builder.init(gpa, .{ .target = .ios17 });
    defer b.deinit();
    const x = try b.input("x", .fp32, &.{ 1, 64 });
    const h = try b.cast(x, .fp16);
    var wdata: [32 * 64]f32 = undefined;
    for (&wdata, 0..) |*w, i| w.* = @floatFromInt(i % 7);
    const w = try b.constFloat(.fp16, &.{ 32, 64 }, &wdata);
    const bias = try b.constFloat(.fp16, &.{4}, &.{ 1, 2, 3, 4 });
    _ = bias;
    const y = try b.linear(h, w, null);
    const s = try b.softmax(y, -1);
    try b.output("y", try b.cast(s, .fp32));

    var weights = try blob.Writer.init(gpa);
    defer weights.deinit();
    const bytes = try b.serialize(gpa, &weights);
    defer gpa.free(bytes);
    var sum = try decode(gpa, bytes);
    defer sum.deinit();

    try std.testing.expectEqual(@as(i64, 8), sum.spec_version);
    try std.testing.expectEqualStrings("CoreML7", sum.opset);
    try std.testing.expectEqual(@as(usize, 1), sum.inputs.len);
    try std.testing.expectEqualStrings("x", sum.inputs[0].name);
    try std.testing.expectEqual(@as(i32, 65568), sum.inputs[0].dtype);
    try std.testing.expectEqualSlices(i64, &.{ 1, 64 }, sum.inputs[0].shape);
    try std.testing.expectEqualStrings("y", sum.outputs[0].name);
    try std.testing.expectEqualStrings("y", sum.block_outputs[0]);
    try std.testing.expectEqualStrings("x", sum.function_inputs[0].name);
    try std.testing.expectEqual(@as(i32, 11), sum.function_inputs[0].dtype);
    try std.testing.expectEqual(b.ops.items.len, sum.ops.len);

    // every op: same type, name, bindings and output types
    for (b.ops.items, sum.ops) |bo, so| {
        try std.testing.expectEqualStrings(bo.type, so.type);
        try std.testing.expectEqualStrings(bo.name, so.name);
        try std.testing.expectEqual(bo.params.len, so.inputs.len);
        for (bo.params, so.inputs) |bp, sp| {
            try std.testing.expectEqualStrings(bp.name, sp.param);
            try std.testing.expectEqualStrings(b.info(bp.arg.one).name, sp.names[0]);
        }
        for (bo.outputs, so.outputs) |v, t| {
            const inf = b.info(v);
            try std.testing.expectEqualStrings(inf.name, t.name);
            try std.testing.expectEqual(@intFromEnum(inf.dtype.mil()), t.dtype);
            try std.testing.expectEqual(inf.shape.len, t.shape.len);
            for (inf.shape, t.shape) |d, e| try std.testing.expectEqual(@as(i64, @intCast(d)), e);
        }
    }

    // the weight went to the blob file, the small bias stayed immediate
    const wop = sum.op(b.info(w).name).?;
    try std.testing.expectEqualStrings(blob.file_name, wop.val.blob.file);
    const rb = try blob.read(weights.bytes(), wop.val.blob.offset);
    try std.testing.expectEqual(blob.DataType.float16, rb.dtype);
    try std.testing.expectEqual(@as(usize, 32 * 64 * 2), rb.data.len);
    try std.testing.expectEqual(@as(u16, @bitCast(@as(f16, 3.0))), std.mem.readInt(u16, rb.data[6..8], .little));
    var found_small = false;
    for (sum.ops) |o| if (o.val == .halfs and o.val.halfs.len == 8) {
        found_small = true;
        try std.testing.expectEqual(@as(u16, @bitCast(@as(f16, 4.0))), std.mem.readInt(u16, o.val.halfs[6..8], .little));
    };
    try std.testing.expect(found_small);
    // the cast's dtype parameter is a string const "fp16"
    const cast_op = for (sum.ops) |o| {
        if (std.mem.eql(u8, o.type, "cast")) break o;
    } else unreachable;
    const dt_name = cast_op.inputs[1].names[0];
    try std.testing.expectEqualStrings("fp16", sum.op(dt_name).?.val.string);
}

test "states are function inputs of state type and description states" {
    const gpa = std.testing.allocator;
    var b = try mil.Builder.init(gpa, .{ .target = .ios18 });
    defer b.deinit();
    const x = try b.input("x", .fp16, &.{ 1, 4 });
    const st = try b.state("cache", .fp16, &.{ 1, 4 });
    const old = try b.readState(st);
    const nw = try b.add(old, x);
    try b.writeState(st, nw);
    try b.output("y", nw);
    var weights = try blob.Writer.init(gpa);
    defer weights.deinit();
    const bytes = try b.serialize(gpa, &weights);
    defer gpa.free(bytes);
    var sum = try decode(gpa, bytes);
    defer sum.deinit();
    try std.testing.expectEqual(@as(i64, 9), sum.spec_version);
    try std.testing.expectEqual(@as(usize, 1), sum.inputs.len);
    try std.testing.expectEqual(@as(usize, 1), sum.states.len);
    try std.testing.expect(sum.states[0].is_state);
    try std.testing.expectEqual(@as(i32, 65552), sum.states[0].dtype);
    try std.testing.expectEqual(@as(usize, 2), sum.function_inputs.len);
    try std.testing.expect(sum.function_inputs[1].is_state);
    var saw_write = false;
    for (sum.ops) |o| if (std.mem.eql(u8, o.type, "write_state")) {
        saw_write = true;
        try std.testing.expectEqual(@as(usize, 0), o.outputs.len);
    };
    try std.testing.expect(saw_write);
}
