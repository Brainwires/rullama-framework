//! Re-export generated proto types from tonic-build.

#[cfg(feature = "grpc")]
/// Generated proto types for `lf.a2a.v1`.
pub mod lf_a2a_v1 {
    tonic::include_proto!("lf.a2a.v1");
}
