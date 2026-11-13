//! TranslateCoreML: Core ML from Zig, without Python or Xcode projects.
//!
//!   pb        protobuf wire format (encoder + field decoder)
//!   spec      the Core ML Model / MIL message subset (field numbers)
//!   mil       ML Program builder (ops, static shapes, consts)
//!   blob      weights/weight.bin (MIL blob storage v2)
//!   package   .mlpackage directories (Manifest.json + model + weights)
//!   decode    reads back what spec/mil write (tests, mlmodel-dump)
//!   ref       host reference interpreter for mil programs
//!   runtime   Core ML runtime bindings (MLModel, MLMultiArray, ...),
//!             Apple targets only (iOS / macOS), via the Objective-C runtime
const std = @import("std");
const builtin = @import("builtin");

pub const pb = @import("pb.zig");
pub const spec = @import("spec.zig");
pub const mil = @import("mil.zig");
pub const blob = @import("blob.zig");
pub const package = @import("package.zig");
pub const decode = @import("decode.zig");
pub const ref = @import("ref.zig");

/// True when the Core ML runtime bindings are available (Apple targets).
pub const has_runtime = builtin.os.tag == .ios or builtin.os.tag == .macos;
pub const runtime = if (has_runtime) @import("runtime.zig") else struct {};
pub const objc = if (has_runtime) @import("objc.zig") else struct {};

test {
    _ = pb;
    _ = spec;
    _ = mil;
    _ = blob;
    _ = package;
    _ = decode;
    _ = ref;
}
