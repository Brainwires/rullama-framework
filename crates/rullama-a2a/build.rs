fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "grpc")]
    {
        tonic_build::configure()
            .build_server(cfg!(feature = "grpc-server"))
            .build_client(cfg!(feature = "grpc-client"))
            .compile_protos(&["proto/a2a.proto"], &["proto/"])?;
    }
    Ok(())
}
