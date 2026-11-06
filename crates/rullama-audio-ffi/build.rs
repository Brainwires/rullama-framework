//! Give the macOS dylib a relocatable install name. rustc otherwise records the
//! absolute `target/.../deps` path, which breaks once the library is copied
//! into an app bundle and loaded from there.

fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-cdylib-link-arg=-Wl,-install_name,@rpath/librullama_audio_ffi.dylib");
    }
}
