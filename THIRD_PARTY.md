# Third-party material

TranslateCoreML's own code is under the MIT licence in [LICENSE](LICENSE). No
third-party source code is vendored or copied into this repository.

## Core ML format definitions (coremltools, BSD-3-Clause)

Field numbers, enum values, the weight-file layout, the package manifest and
the MIL op parameter lists were re-derived by hand from Apple's public
coremltools sources (<https://github.com/apple/coremltools>), BSD-3-Clause,
Copyright (c) Apple Inc. (see that repository's LICENSE.txt for the exact
notice):

- `mlmodel/format/Model.proto`, `FeatureTypes.proto`, `MIL.proto`
  (`src/spec.zig` lists the exact subset used)
- `mlmodel/src/MILBlob/Blob/StorageFormat.hpp` (`src/blob.zig`)
- `modelpackage/src/ModelPackage.cpp` (`src/package.zig`)
- `coremltools/converters/mil/mil/ops/defs/iOS15..iOS18` (`src/mil.zig`)

No `.proto` or source file from coremltools is included or generated from;
the conventions describe the file format Core ML reads. If any part is judged
a derivative of those files, coremltools' BSD-3-Clause notice applies to it.

## Apple frameworks

`src/objc.zig` and `src/runtime.zig` open libobjc, Foundation and CoreML with
dlopen at run time on iOS / macOS. Nothing from Apple's SDKs is vendored or
linked.

## Fetched dependency

- **MacOS** (<https://github.com/Mauhlt/MacOS>, pinned in `build.zig.zon`,
  lazy, Apple targets only): Objective-C runtime types. Same owner.

## Build helpers

`src/tools/ios_zig_lib.zig` copies part of the running Zig installation's
`lib` directory (Zig's standard library, MIT, Copyright (c) Zig contributors,
plus the Darwin libc headers Zig ships under their own licences) into the
build cache for iOS builds, with one line patched. Nothing from it is
committed here.
