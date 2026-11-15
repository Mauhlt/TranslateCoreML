# TranslateCoreML: Core ML from Zig

Write Core ML **ML Programs** (`.mlpackage`) in Zig and run them through
Core ML, which is the only public way onto the Apple Neural Engine. No
Python, no coremltools, no Xcode project: models are built op by op with a
MIL builder, serialized by a small protobuf encoder, and loaded through the
Objective-C runtime.

| file | what |
|---|---|
| `src/pb.zig` | protobuf wire format: encoder + field decoder |
| `src/spec.zig` | the Core ML `Model` / `MILSpec` message subset (field numbers, provenance) |
| `src/mil.zig` | ML Program builder: inputs, consts, ops, static shapes, serialization |
| `src/blob.zig` | `weights/weight.bin` (MIL blob storage v2) |
| `src/package.zig` | `.mlpackage` directories (`Manifest.json` + model + weights) |
| `src/decode.zig` | reads back what `spec`/`mil` write (tests, `mlmodel-dump`) |
| `src/ref.zig` | host reference interpreter for `mil` programs (fp16 rounding emulated) |
| `src/objc.zig` | Objective-C runtime (libobjc / Foundation / CoreML opened with dlopen) |
| `src/runtime.zig` | Core ML: compile, MLModel, MLMultiArray, predictions, MLState, MLComputePlan |
| `src/probe/` | the probe models and the probe programs (Mac CLI, iPhone app) |
| `src/tools/` | `mlmodel-dump`, `ios-zig-lib` (from Portal), `ios-sign` (from Portal) |

Depends on [MacOS](https://github.com/Mauhlt/MacOS) (Objective-C types and
the `objc_msgSend` signature builder), a lazy dependency fetched only when a
build targets iOS or macOS.

## Format provenance

Field numbers and enum values were re-derived by hand from Apple's public
Core ML format definitions in coremltools (BSD-3-Clause:
`mlmodel/format/Model.proto`, `FeatureTypes.proto`, `MIL.proto`), the
weight-file layout from `mlmodel/src/MILBlob/Blob/StorageFormat.hpp`, the
package manifest from `modelpackage/src/ModelPackage.cpp`, and op parameter
lists from `coremltools/converters/mil/mil/ops/defs/iOS15..iOS18`. Nothing
is vendored or generated from those files; `src/spec.zig` lists the exact
subset used. Conventions follow what coremltools' serializer writes (fp16
immediates as bytes, a `name` attribute on every op, `val` on consts, the
block stored under the opset name). `mlmodel-dump` reads coremltools-made
packages (checked on a Silero VAD and WhisperKit's tiny.en decoder) with
the same decoder the tests use.

Targets: `ios16` (specificationVersion 7, opset CoreML6), `ios17` (8,
CoreML7, the default), `ios18` (9, CoreML8: states).

## Build

```sh
zig build test            # unit tests + aarch64-ios / aarch64-macos compile checks
zig build probe-models    # zig-out/probe: probe .mlpackages + reference data (any host)
zig build                 # zig-out/bin/mlmodel-dump
```

## On a Mac (no iPhone needed)

Apple-silicon Macs have the same Neural Engine family. Core ML is opened at
run time, so this even cross-links from Windows (`-Dtarget=aarch64-macos`).

```sh
zig build coreml-probe
zig-out/bin/coreml-probe zig-out/probe --out probe-mac.json
```

Each probe model (`matmul`, `matmul_fp16io`, `block` = one transformer block
at whisper-tiny width) is compiled, loaded with `cpu_only`, `cpu_and_gpu`,
`cpu_and_neural_engine` and `all`, run 1 + 10 times, and compared with the
host reference (`rel err` below 1e-2 = OK; fp16 math gives a few 1e-3).

**What proves ANE placement:** the `plan` lines / the `"plan"` object
(MLComputePlan, macOS 14.4 / iOS 17.4+): `N ops on neural_engine`, and per
op `"device": "neural_engine"`. A much lower `median_ms` for
`cpu_and_neural_engine` than for `cpu_only` on `block` is the second sign.

## On the iPhone

```sh
export PORTAL_SIGN_IDENTITY="Apple Development: Name (ABCDE12345)"   # security find-identity -v -p codesigning
export PORTAL_PROVISIONING_PROFILE=~/path/to/dev.mobileprovision    # lists the iPhone; app id = bundle id or wildcard
export PORTAL_TEAM_ID=ABCDE12345
zig build ios-coreml-probe-sign -Dprobe-bundle-id=<bundle id of the profile>
xcrun devicectl list devices
xcrun devicectl device install app --device <UDID> zig-out/ios/CoreMLProbe.app
xcrun devicectl device process launch --console --device <UDID> <bundle id>
xcrun devicectl device copy from --device <UDID> --domain-type appDataContainer \
    --domain-identifier <bundle id> --source Documents/coreml-probe.json --destination .
```

The app shows the log on screen (and NSLogs it), and writes
`Documents/coreml-probe.json` (also visible in Files / Finder). iOS 17+.
