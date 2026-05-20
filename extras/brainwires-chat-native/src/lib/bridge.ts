// JS-side surface of the brainwires-bridge native module.
//
// Until uniffi-generated bindings are linked into ios/android/macos/windows,
// these stubs let the UI render. Once the bridge is built and linked, replace
// the stubs with calls into the generated NativeModule (see ./bridgeNative.ts
// once it's generated).

export async function frameworkVersion(): Promise<string> {
  return 'stub (FFI not yet linked)';
}
