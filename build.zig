//! TranslateCoreML: write Core ML ML Programs (.mlpackage) from Zig and run
//! them through Core ML (so on the Apple Neural Engine) from Zig.
//!
//!   zig build test                 unit tests (protobuf, spec, MIL builder,
//!                                  blob, package layout, reference
//!                                  interpreter, probe models), plus
//!                                  compile-only checks for aarch64-ios and
//!                                  aarch64-macos (the Core ML bindings and
//!                                  the probe programs)
//!   zig build probe-models         zig-out/probe: the probe .mlpackages and
//!                                  their host reference data (any host)
//!   zig build coreml-probe         (on a Mac) zig-out/bin/coreml-probe and
//!                                  zig-out/probe; run:
//!                                  zig-out/bin/coreml-probe zig-out/probe
//!   zig build ios-coreml-probe     (on a Mac) zig-out/ios/CoreMLProbe.app
//!   zig build ios-coreml-probe-sign  sign it (PORTAL_SIGN_IDENTITY,
//!                                  PORTAL_PROVISIONING_PROFILE, PORTAL_TEAM_ID)
//!
//! Options: -Dios-sdk=<path> (default: xcrun --sdk iphoneos --show-sdk-path),
//! -Dprobe-bundle-id=<id> (default local.coreml.probe),
//! -Dprobe-target=ios16|ios17|ios18 (MIL opset of the probe models, default ios17).
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const probe_target = b.option([]const u8, "probe-target", "MIL target of the probe models: ios16, ios17 (default) or ios18") orelse "ios17";

    // The library module other projects import: `@import("coreml")`.
    const mod = coremlModule(b, target, optimize, true);

    const test_step = b.step("test", "Unit tests and aarch64-ios / aarch64-macos compile checks");
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = mod })).step);

    const host_coreml = coremlModule(b, b.graph.host, .ReleaseFast, false);
    const models_tests = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("src/probe/models.zig"),
        .target = b.graph.host,
        .optimize = .Debug,
        .imports = &.{.{ .name = "coreml", .module = coremlModule(b, b.graph.host, .Debug, false) }},
    }) });
    test_step.dependOn(&b.addRunArtifact(models_tests).step);

    // Probe models (host tool).
    const gen = b.addExecutable(.{ .name = "coreml-probe-gen", .root_module = b.createModule(.{
        .root_source_file = b.path("src/probe/gen.zig"),
        .target = b.graph.host,
        .optimize = .ReleaseFast,
        .imports = &.{.{ .name = "coreml", .module = host_coreml }},
    }) });
    const gen_run = b.addRunArtifact(gen);
    const probe_dir = gen_run.addOutputDirectoryArg("probe");
    gen_run.addArg(probe_target);
    const install_probe = b.addInstallDirectory(.{ .source_dir = probe_dir, .install_dir = .prefix, .install_subdir = "probe" });
    b.step("probe-models", "Write zig-out/probe (probe .mlpackages + reference data)").dependOn(&install_probe.step);

    const dump = b.addExecutable(.{ .name = "mlmodel-dump", .root_module = b.createModule(.{
        .root_source_file = b.path("src/tools/dump.zig"),
        .target = b.graph.host,
        .optimize = .ReleaseSafe,
        .imports = &.{.{ .name = "coreml", .module = host_coreml }},
    }) });
    b.installArtifact(dump);

    // Zig lib dir with the iOS std fix (see src/tools/ios_zig_lib.zig).
    const lib_tool = b.addExecutable(.{ .name = "ios-zig-lib", .root_module = b.createModule(.{
        .root_source_file = b.path("src/tools/ios_zig_lib.zig"),
        .target = b.graph.host,
        .optimize = .ReleaseSafe,
    }) });
    const lib_run = b.addRunArtifact(lib_tool);
    lib_run.addDirectoryArg(.zig_lib);
    const zig_lib = lib_run.addOutputDirectoryArg("zig-lib");

    const ios_t = b.resolveTargetQuery(.{ .cpu_arch = .aarch64, .os_tag = .ios, .os_version_min = .{ .semver = .{ .major = 17, .minor = 0, .patch = 0 } } });
    const mac_t = b.resolveTargetQuery(.{ .cpu_arch = .aarch64, .os_tag = .macos, .os_version_min = .{ .semver = .{ .major = 14, .minor = 0, .patch = 0 } } });

    // Compile checks (objects; no Apple SDK needed).
    const ios_check = b.addObject(.{ .name = "coreml-probe-ios-check", .root_module = probeModule(b, "src/probe/ios_main.zig", ios_t, .Debug), .zig_lib_dir = zig_lib });
    const mac_check = b.addObject(.{ .name = "coreml-probe-macos-check", .root_module = probeModule(b, "src/probe/macos_main.zig", mac_t, .Debug) });
    const check = b.step("apple-check", "Compile the Core ML bindings and probes for aarch64-ios and aarch64-macos (objects only)");
    check.dependOn(&ios_check.step);
    check.dependOn(&mac_check.step);
    test_step.dependOn(check);

    const sign_tool = b.addExecutable(.{ .name = "ios-sign", .root_module = b.createModule(.{
        .root_source_file = b.path("src/tools/ios_sign.zig"),
        .target = b.graph.host,
        .optimize = .ReleaseSafe,
    }) });
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = sign_tool.root_module })).step);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = lib_tool.root_module })).step);

    // macOS probe (on a Mac).
    const mac_step = b.step("coreml-probe", "On a Mac: build zig-out/bin/coreml-probe and zig-out/probe");
    if (b.graph.host.result.os.tag == .macos) {
        const pm = probeModule(b, "src/probe/macos_main.zig", b.graph.host, optimize);
        pm.linkFramework("CoreML", .{});
        pm.linkFramework("Foundation", .{});
        pm.linkSystemLibrary("objc", .{});
        const exe = b.addExecutable(.{ .name = "coreml-probe", .root_module = pm });
        mac_step.dependOn(&b.addInstallArtifact(exe, .{}).step);
        mac_step.dependOn(&install_probe.step);
    } else {
        mac_step.dependOn(&b.addFail("zig build coreml-probe runs on a Mac (Core ML); use apple-check here").step);
    }

    // iPhone app.
    addIosApp(b, optimize, zig_lib, probe_dir, sign_tool, ios_t);
}

/// Module "coreml" for `t`. On Apple targets it imports ../MacOS (the
/// Objective-C types) and needs libc.
fn coremlModule(b: *std.Build, t: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, exported: bool) *std.Build.Module {
    const opts: std.Build.Module.CreateOptions = .{
        .root_source_file = b.path("src/root.zig"),
        .target = t,
        .optimize = optimize,
    };
    const m = if (exported) b.addModule("coreml", opts) else b.createModule(opts);
    const os = t.result.os.tag;
    if (os == .ios or os == .macos) {
        m.link_libc = true;
        m.addImport("MacOS", b.dependency("MacOS", .{ .target = t, .optimize = optimize }).module("MacOS"));
    }
    return m;
}

fn probeModule(b: *std.Build, root: []const u8, t: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{
        .root_source_file = b.path(root),
        .target = t,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{.{ .name = "coreml", .module = coremlModule(b, t, optimize, false) }},
    });
}

fn addIosApp(b: *std.Build, optimize: std.builtin.OptimizeMode, zig_lib: std.Build.LazyPath, probe_dir: std.Build.LazyPath, sign_tool: *std.Build.Step.Compile, t: std.Build.ResolvedTarget) void {
    const bundle_id = b.option([]const u8, "probe-bundle-id", "CoreMLProbe.app bundle id (default local.coreml.probe; must match the provisioning profile)") orelse "local.coreml.probe";
    const env = &b.graph.environ_map;
    const sdk = b.option([]const u8, "ios-sdk", "iPhoneOS SDK path (env IOS_SDK; default: xcrun --sdk iphoneos --show-sdk-path on a Mac)") orelse env.get("IOS_SDK") orelse if (b.graph.host.result.os.tag == .macos)
        std.mem.trim(u8, b.run(&.{ "xcrun", "--sdk", "iphoneos", "--show-sdk-path" }), " \r\n")
    else
        null;
    const app_step = b.step("ios-coreml-probe", "On a Mac: build zig-out/ios/CoreMLProbe.app");
    const sign_step = b.step("ios-coreml-probe-sign", "Sign zig-out/ios/CoreMLProbe.app (env PORTAL_SIGN_IDENTITY, PORTAL_PROVISIONING_PROFILE, PORTAL_TEAM_ID)");
    const sdk_path = sdk orelse {
        const fail = b.addFail("zig build ios-coreml-probe needs the iPhoneOS SDK: run it on a Mac with Xcode, or pass -Dios-sdk=<path> / IOS_SDK");
        app_step.dependOn(&fail.step);
        sign_step.dependOn(&fail.step);
        return;
    };
    const mod = probeModule(b, "src/probe/ios_main.zig", t, optimize);
    mod.addSystemFrameworkPath(.{ .cwd_relative = b.pathJoin(&.{ sdk_path, "System/Library/Frameworks" }) });
    mod.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ sdk_path, "usr/lib" }) });
    mod.addSystemIncludePath(.{ .cwd_relative = b.pathJoin(&.{ sdk_path, "usr/include" }) });
    for ([_][]const u8{ "UIKit", "Foundation", "CoreML" }) |fw| mod.linkFramework(fw, .{});
    mod.linkSystemLibrary("objc", .{});
    const exe = b.addExecutable(.{ .name = "CoreMLProbe", .root_module = mod, .zig_lib_dir = zig_lib });
    const libc_txt = b.addWriteFiles().add("libc-ios.txt", b.fmt(
        \\include_dir={s}/usr/include
        \\sys_include_dir={s}/usr/include
        \\crt_dir={s}/usr/lib
        \\msvc_lib_dir=
        \\kernel32_lib_dir=
        \\gcc_dir=
        \\
    , .{ sdk_path, sdk_path, sdk_path }));
    exe.setLibCFile(libc_txt);

    const app = b.addWriteFiles();
    _ = app.addCopyFile(exe.getEmittedBin(), "CoreMLProbe.app/CoreMLProbe");
    _ = app.add("CoreMLProbe.app/Info.plist", infoPlist(b, bundle_id));
    _ = app.add("CoreMLProbe.app/PkgInfo", "APPL????");
    _ = app.addCopyDirectory(probe_dir, "CoreMLProbe.app/probe", .{});
    const install = b.addInstallDirectory(.{ .source_dir = app.getDirectory().path(b, "CoreMLProbe.app"), .install_dir = .prefix, .install_subdir = "ios/CoreMLProbe.app" });
    app_step.dependOn(&install.step);

    const sign = b.addRunArtifact(sign_tool);
    sign.addDirectoryArg(app.getDirectory().path(b, "CoreMLProbe.app"));
    const signed = sign.addOutputDirectoryArg("signed");
    sign.addArg(bundle_id);
    sign.has_side_effects = true;
    const install_signed = b.addInstallDirectory(.{ .source_dir = signed.path(b, "CoreMLProbe.app"), .install_dir = .prefix, .install_subdir = "ios/CoreMLProbe.app" });
    install_signed.step.dependOn(&install.step);
    sign_step.dependOn(&install_signed.step);
}

/// Info.plist (configuration text): no scene manifest (the app delegate
/// makes the window), Documents shared in Files / Finder.
fn infoPlist(b: *std.Build, bundle_id: []const u8) []const u8 {
    return b.fmt(
        \\<?xml version="1.0" encoding="UTF-8"?>
        \\<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        \\<plist version="1.0">
        \\<dict>
        \\    <key>CFBundleDevelopmentRegion</key><string>en</string>
        \\    <key>CFBundleExecutable</key><string>CoreMLProbe</string>
        \\    <key>CFBundleIdentifier</key><string>{s}</string>
        \\    <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
        \\    <key>CFBundleName</key><string>CoreMLProbe</string>
        \\    <key>CFBundleDisplayName</key><string>CoreML Probe</string>
        \\    <key>CFBundlePackageType</key><string>APPL</string>
        \\    <key>CFBundleShortVersionString</key><string>0.1.0</string>
        \\    <key>CFBundleVersion</key><string>1</string>
        \\    <key>CFBundleSupportedPlatforms</key><array><string>iPhoneOS</string></array>
        \\    <key>DTPlatformName</key><string>iphoneos</string>
        \\    <key>MinimumOSVersion</key><string>17.0</string>
        \\    <key>LSRequiresIPhoneOS</key><true/>
        \\    <key>UIDeviceFamily</key><array><integer>1</integer></array>
        \\    <key>UIRequiredDeviceCapabilities</key><array><string>arm64</string></array>
        \\    <key>UILaunchScreen</key><dict/>
        \\    <key>UIFileSharingEnabled</key><true/>
        \\    <key>LSSupportsOpeningDocumentsInPlace</key><true/>
        \\</dict>
        \\</plist>
        \\
    , .{bundle_id});
}
