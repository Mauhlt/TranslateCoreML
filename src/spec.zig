//! The subset of the Core ML model specification that an ML Program
//! (`.mlpackage`) needs, as protobuf field numbers plus small writers.
//!
//! Provenance: field numbers and enum values were re-derived by hand from
//! Apple's public Core ML format definitions (coremltools, BSD-3-Clause,
//! mlmodel/format/Model.proto, FeatureTypes.proto and MIL.proto,
//! package CoreML.Specification / CoreML.Specification.MILSpec). No .proto
//! file is vendored and nothing is generated from them. Only the messages
//! and fields below are written; everything else keeps its proto3 default.
//!
//!   Model                      specificationVersion=1 description=2 mlProgram=502
//!   ModelDescription           input=1 output=10 state=13 metadata=100
//!   Metadata                   shortDescription=1 versionString=2 author=3 license=4 userDefined=100 (map)
//!   FeatureDescription         name=1 shortDescription=2 type=3
//!   FeatureType                multiArrayType=5 stateType=8 isOptional=1000
//!   ArrayFeatureType           shape=1 (packed int64) dataType=2 (enum ArrayDataType)
//!   StateFeatureType           arrayType=1
//!   MILSpec.Program            version=1 functions=2 (map<string,Function>) docString=3 attributes=4
//!   MILSpec.Function           inputs=1 opset=2 block_specializations=3 (map<string,Block>) attributes=4
//!   MILSpec.Block              inputs=1 outputs=2 operations=3 attributes=4
//!   MILSpec.Argument           arguments=1;  Argument.Binding  name=1 | value=2 (oneof)
//!   MILSpec.Operation          type=1 inputs=2 (map<string,Argument>) outputs=3 blocks=4 attributes=5 (map<string,Value>)
//!   MILSpec.NamedValueType     name=1 type=2
//!   MILSpec.ValueType          tensorType=1 listType=2 tupleType=3 dictionaryType=4 stateType=5 (oneof)
//!   MILSpec.TensorType         dataType=1 rank=2 dimensions=3 attributes=4
//!   MILSpec.StateType          wrappedType=1
//!   MILSpec.Dimension          constant=1 {size=1} | unknown=2 {variadic=1}
//!   MILSpec.Value              docString=1 type=2 immediateValue=3 | blobFileValue=5
//!   Value.ImmediateValue       tensor=1 tuple=2 list=3 dictionary=4 (oneof)
//!   Value.BlobFileValue        fileName=1 offset=2
//!   MILSpec.TensorValue        floats=1 ints=2 bools=3 strings=4 longInts=5 doubles=6 bytes=7 (oneof),
//!                              each Repeated* {values=1}; floats/ints/bools/longInts packed
//!   map<K,V> entries           key=1 value=2
//!
//! Enum values: MILSpec.DataType BOOL=1 STRING=2 FLOAT16=10 FLOAT32=11
//! FLOAT64=12 INT32=23 INT64=24; ArrayFeatureType.ArrayDataType
//! FLOAT32=65568 DOUBLE=65600 INT32=131104 FLOAT16=65552.
//! specificationVersion: 7 = iOS 16 / macOS 13 (opset "CoreML6"),
//! 8 = iOS 17 / macOS 14 ("CoreML7"), 9 = iOS 18 / macOS 15 ("CoreML8").
//!
//! Conventions copied from what coremltools' serializer writes (so Core ML
//! sees familiar input): fp16 immediate values go in TensorValue.bytes,
//! every operation has a "name" attribute (a scalar string Value), const
//! ops carry their data in a "val" attribute, and the function's single
//! block is stored under the opset name in block_specializations.
const std = @import("std");
const pb = @import("pb.zig");

pub const Model = struct {
    pub const specification_version = 1;
    pub const description = 2;
    pub const ml_program = 502;
};
pub const ModelDescription = struct {
    pub const input = 1;
    pub const output = 10;
    pub const state = 13;
    pub const metadata = 100;
};
pub const Metadata = struct {
    pub const short_description = 1;
    pub const version_string = 2;
    pub const author = 3;
    pub const license = 4;
    pub const user_defined = 100;
};
pub const FeatureDescription = struct {
    pub const name = 1;
    pub const short_description = 2;
    pub const @"type" = 3;
};
pub const FeatureType = struct {
    pub const multi_array_type = 5;
    pub const state_type = 8;
    pub const is_optional = 1000;
};
pub const ArrayFeatureType = struct {
    pub const shape = 1;
    pub const data_type = 2;
};
pub const StateFeatureType = struct {
    pub const array_type = 1;
};
pub const ArrayDataType = enum(i32) {
    float32 = 65568,
    double = 65600,
    int32 = 131104,
    float16 = 65552,
};

pub const Program = struct {
    pub const version = 1;
    pub const functions = 2;
    pub const doc_string = 3;
    pub const attributes = 4;
};
pub const Function = struct {
    pub const inputs = 1;
    pub const opset = 2;
    pub const block_specializations = 3;
    pub const attributes = 4;
};
pub const Block = struct {
    pub const inputs = 1;
    pub const outputs = 2;
    pub const operations = 3;
    pub const attributes = 4;
};
pub const Argument = struct {
    pub const arguments = 1;
    pub const Binding = struct {
        pub const name = 1;
        pub const value = 2;
    };
};
pub const Operation = struct {
    pub const @"type" = 1;
    pub const inputs = 2;
    pub const outputs = 3;
    pub const blocks = 4;
    pub const attributes = 5;
};
pub const NamedValueType = struct {
    pub const name = 1;
    pub const @"type" = 2;
};
pub const ValueType = struct {
    pub const tensor_type = 1;
    pub const list_type = 2;
    pub const tuple_type = 3;
    pub const dictionary_type = 4;
    pub const state_type = 5;
};
pub const DataType = enum(i32) {
    bool = 1,
    string = 2,
    float16 = 10,
    float32 = 11,
    float64 = 12,
    int32 = 23,
    int64 = 24,
};
pub const TensorType = struct {
    pub const data_type = 1;
    pub const rank = 2;
    pub const dimensions = 3;
    pub const attributes = 4;
};
pub const StateType = struct {
    pub const wrapped_type = 1;
};
pub const Dimension = struct {
    pub const constant = 1;
    pub const unknown = 2;
    pub const constant_size = 1;
};
pub const Value = struct {
    pub const doc_string = 1;
    pub const @"type" = 2;
    pub const immediate_value = 3;
    pub const blob_file_value = 5;
};
pub const ImmediateValue = struct {
    pub const tensor = 1;
    pub const tuple = 2;
    pub const list = 3;
    pub const dictionary = 4;
};
pub const BlobFileValue = struct {
    pub const file_name = 1;
    pub const offset = 2;
};
pub const TensorValue = struct {
    pub const floats = 1;
    pub const ints = 2;
    pub const bools = 3;
    pub const strings = 4;
    pub const long_ints = 5;
    pub const doubles = 6;
    pub const bytes = 7;
    /// Field of every Repeated* message.
    pub const values = 1;
};
pub const MapEntry = struct {
    pub const key = 1;
    pub const value = 2;
};

/// Model.specificationVersion and the MIL opset name for a deployment target.
pub const Target = enum {
    ios16,
    ios17,
    ios18,

    pub fn specVersion(t: Target) i32 {
        return switch (t) {
            .ios16 => 7,
            .ios17 => 8,
            .ios18 => 9,
        };
    }
    pub fn opset(t: Target) []const u8 {
        return switch (t) {
            .ios16 => "CoreML6",
            .ios17 => "CoreML7",
            .ios18 => "CoreML8",
        };
    }
};

// ----------------------------------------------------------------- writers

/// MILSpec.TensorType as field `field` of the current message.
pub fn tensorType(w: *pb.Writer, field: u32, dt: DataType, shape: []const usize) !void {
    const m = try w.begin(field);
    try w.int32(TensorType.data_type, @intFromEnum(dt));
    try w.int64(TensorType.rank, @intCast(shape.len));
    for (shape) |d| {
        const dm = try w.begin(TensorType.dimensions);
        const c = try w.begin(Dimension.constant);
        try w.uint64(Dimension.constant_size, d);
        try w.end(c);
        try w.end(dm);
    }
    try w.end(m);
}

/// MILSpec.ValueType{tensorType} as field `field`.
pub fn valueTypeTensor(w: *pb.Writer, field: u32, dt: DataType, shape: []const usize) !void {
    const m = try w.begin(field);
    try tensorType(w, ValueType.tensor_type, dt, shape);
    try w.end(m);
}

/// MILSpec.ValueType{stateType{wrappedType{tensorType}}} as field `field`.
pub fn valueTypeState(w: *pb.Writer, field: u32, dt: DataType, shape: []const usize) !void {
    const m = try w.begin(field);
    const s = try w.begin(ValueType.state_type);
    try valueTypeTensor(w, StateType.wrapped_type, dt, shape);
    try w.end(s);
    try w.end(m);
}

/// MILSpec.NamedValueType as field `field`.
pub fn namedValueType(w: *pb.Writer, field: u32, name: []const u8, dt: DataType, shape: []const usize, is_state: bool) !void {
    const m = try w.begin(field);
    try w.string(NamedValueType.name, name);
    if (is_state)
        try valueTypeState(w, NamedValueType.type, dt, shape)
    else
        try valueTypeTensor(w, NamedValueType.type, dt, shape);
    try w.end(m);
}

/// The payload of an immediate tensor value.
pub const Immediate = union(enum) {
    floats: []const f32,
    /// fp16 values (written as raw little-endian bytes, like coremltools).
    halfs: []const f16,
    ints: []const i32,
    bools: []const bool,
    strings: []const []const u8,
};

/// MILSpec.Value{type, immediateValue{tensor}} as field `field`.
pub fn immediateValue(w: *pb.Writer, field: u32, dt: DataType, shape: []const usize, v: Immediate) !void {
    const m = try w.begin(field);
    try valueTypeTensor(w, Value.type, dt, shape);
    const iv = try w.begin(Value.immediate_value);
    const t = try w.begin(ImmediateValue.tensor);
    switch (v) {
        .floats => |xs| {
            const r = try w.begin(TensorValue.floats);
            try w.packedFloat32(TensorValue.values, xs);
            try w.end(r);
        },
        .halfs => |xs| {
            const r = try w.begin(TensorValue.bytes);
            try w.key(TensorValue.values, .len);
            try w.rawVarint(xs.len * 2);
            for (xs) |x| {
                var b: [2]u8 = undefined;
                std.mem.writeInt(u16, &b, @bitCast(x), .little);
                try w.buf.appendSlice(w.gpa, &b);
            }
            try w.end(r);
        },
        .ints => |xs| {
            const r = try w.begin(TensorValue.ints);
            try w.packedInt32(TensorValue.values, xs);
            try w.end(r);
        },
        .bools => |xs| {
            const r = try w.begin(TensorValue.bools);
            try w.packedBool(TensorValue.values, xs);
            try w.end(r);
        },
        .strings => |xs| {
            const r = try w.begin(TensorValue.strings);
            for (xs) |s| try w.bytesAlways(TensorValue.values, s);
            try w.end(r);
        },
    }
    try w.end(t);
    try w.end(iv);
    try w.end(m);
}

/// A scalar string Value (op "name" attributes, string parameters).
pub fn stringValue(w: *pb.Writer, field: u32, s: []const u8) !void {
    try immediateValue(w, field, .string, &.{}, .{ .strings = &.{s} });
}

/// MILSpec.Value{type, blobFileValue{fileName, offset}} as field `field`.
pub fn blobValue(w: *pb.Writer, field: u32, dt: DataType, shape: []const usize, file_name: []const u8, offset: u64) !void {
    const m = try w.begin(field);
    try valueTypeTensor(w, Value.type, dt, shape);
    const b = try w.begin(Value.blob_file_value);
    try w.string(BlobFileValue.file_name, file_name);
    try w.uint64(BlobFileValue.offset, offset);
    try w.end(b);
    try w.end(m);
}

/// FeatureDescription{name, type{multiArrayType | stateType{arrayType}}}
/// as field `field` of ModelDescription.
pub fn featureDescription(w: *pb.Writer, field: u32, name: []const u8, dt: ArrayDataType, shape: ?[]const usize, is_state: bool) !void {
    const m = try w.begin(field);
    try w.string(FeatureDescription.name, name);
    const ft = try w.begin(FeatureDescription.type);
    const outer = if (is_state) try w.begin(FeatureType.state_type) else null;
    const at = try w.begin(if (is_state) StateFeatureType.array_type else FeatureType.multi_array_type);
    if (shape) |s| {
        var dims: [16]i64 = undefined;
        for (s, 0..) |d, i| dims[i] = @intCast(d);
        try w.packedInt64(ArrayFeatureType.shape, dims[0..s.len]);
    }
    try w.int32(ArrayFeatureType.data_type, @intFromEnum(dt));
    try w.end(at);
    if (outer) |o| try w.end(o);
    try w.end(ft);
    try w.end(m);
}

test "tensor type golden bytes" {
    var w = pb.Writer.init(std.testing.allocator);
    defer w.deinit();
    // ValueType{tensorType{dataType=FLOAT16, rank=2, dimensions=[{constant{size=2}}, {constant{size=3}}]}}
    try valueTypeTensor(&w, 1, .float16, &.{ 2, 3 });
    try std.testing.expectEqualSlices(u8, &.{
        0x0a, 0x12, // field 1 (NamedValueType.type / whatever), 18 bytes
        0x0a, 0x10, //   tensorType, 16 bytes
        0x08, 0x0a, //     dataType = 10
        0x10, 0x02, //     rank = 2
        0x1a, 0x04, 0x0a, 0x02, 0x08, 0x02, // dimensions {constant {size 2}}
        0x1a, 0x04, 0x0a, 0x02, 0x08, 0x03, // dimensions {constant {size 3}}
    }, w.bytesWritten());
}

test "scalar string value golden bytes" {
    var w = pb.Writer.init(std.testing.allocator);
    defer w.deinit();
    try stringValue(&w, 2, "x");
    try std.testing.expectEqualSlices(u8, &.{
        0x12, 0x0f, // Value (field 2), 15 bytes
        0x12, 0x04, 0x0a, 0x02, 0x08, 0x02, //   type {tensorType {dataType STRING; rank 0 omitted}}
        0x1a, 0x07, 0x0a, 0x05, 0x22, 0x03, //   immediateValue {tensor {strings {
        0x0a, 0x01, 'x', //                        values: "x" }}}
    }, w.bytesWritten());
}

test "feature description golden bytes" {
    var w = pb.Writer.init(std.testing.allocator);
    defer w.deinit();
    try featureDescription(&w, 1, "x", .float32, &.{ 1, 4 }, false);
    try std.testing.expectEqualSlices(u8, &.{
        0x0a, 0x0f, // FeatureDescription (input = 1), 15 bytes
        0x0a, 0x01, 'x', //   name
        0x1a, 0x0a, //   type (FeatureType), 10 bytes
        0x2a, 0x08, //     multiArrayType (5), 8 bytes
        0x0a, 0x02, 0x01, 0x04, //       shape packed [1, 4]
        0x10, 0xa0, 0x80, 0x04, //       dataType = 65568 (FLOAT32)
    }, w.bytesWritten());
}
