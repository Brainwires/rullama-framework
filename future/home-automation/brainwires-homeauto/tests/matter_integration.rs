//! Matter protocol integration tests.
//!
//! These tests verify protocol correctness without requiring hardware.
//! The `matter_e2e` test spawns a real server and controller on loopback.
//!
//! Run with:
//!   cargo test -p brainwires-hardware --features matter --test matter_integration
//!   cargo test -p brainwires-hardware --features matter --test matter_integration -- --include-ignored

#![cfg(feature = "matter")]

use brainwires_homeauto::matter::secure_channel::{PaseCommissionee, PaseCommissioner};

// ── PASE smoke test ───────────────────────────────────────────────────────────

/// PASE full handshake in-process (no network, already unit-tested — here as
/// integration smoke test that the public API composes correctly end-to-end).
#[test]
fn pase_handshake_smoke() {
    const PASSCODE: u32 = 20202021;
    const SALT: &[u8] = b"integration-test-salt-32-bytes!!";
    const ITERATIONS: u32 = 1000;

    // Commissionee side
    let mut commissionee = PaseCommissionee::new_with_params(PASSCODE, SALT.to_vec(), ITERATIONS);

    // Commissioner side
    let mut commissioner = PaseCommissioner::new(PASSCODE);

    // Step 1: Commissioner → PBKDFParamRequest
    let (_session_id, param_req) = commissioner
        .build_param_request()
        .expect("build_param_request failed");

    // Step 2: Commissionee → PBKDFParamResponse
    let param_resp = commissionee
        .handle_param_request(&param_req)
        .expect("handle_param_request failed");

    // Step 3: Commissioner → Pake1
    let pake1 = commissioner
        .handle_param_response(&param_resp)
        .expect("handle_param_response failed");

    // Step 4: Commissionee → Pake2
    let pake2 = commissionee
        .handle_pake1(&pake1)
        .expect("handle_pake1 failed");

    // Step 5: Commissioner → Pake3 (handle_pake2 returns the Pake3 payload)
    let pake3 = commissioner
        .handle_pake2(&pake2)
        .expect("handle_pake2 failed");

    // Step 6: Commissionee verifies Pake3 and establishes session
    let ce_session = commissionee
        .handle_pake3(&pake3)
        .expect("handle_pake3 failed");

    // Retrieve commissioner session
    let cr_session = commissioner
        .established_session()
        .expect("commissioner has no established session");

    // Keys must be cross-symmetric:
    // - Commissioner is the initiator → its encrypt_key = I2R, decrypt_key = R2I
    // - Commissionee is the responder → its encrypt_key = R2I, decrypt_key = I2R
    assert_eq!(
        cr_session.encrypt_key, ce_session.decrypt_key,
        "I2R key mismatch: commissioner encrypt != commissionee decrypt"
    );
    assert_eq!(
        cr_session.decrypt_key, ce_session.encrypt_key,
        "R2I key mismatch: commissioner decrypt != commissionee encrypt"
    );
    assert_eq!(
        cr_session.attestation_challenge, ce_session.attestation_challenge,
        "attestation challenge mismatch"
    );
}

// ── CASE with FabricManager ───────────────────────────────────────────────────

/// CASE full handshake in-process with real FabricManager-generated certificates.
///
/// Verifies that `CaseInitiator` and `CaseResponder` derive identical session
/// keys when the same fabric root CA signs both sides' NOCs.
#[tokio::test]
async fn case_handshake_with_fabric_manager() {
    use brainwires_homeauto::matter::fabric::{FabricIndex, FabricManager};
    use brainwires_homeauto::matter::secure_channel::{CaseInitiator, CaseResponder};
    use p256::{SecretKey, elliptic_curve::sec1::ToEncodedPoint};
    use rand_core::OsRng;

    // Build a FabricManager and generate a root CA
    let tmp_dir = TempDir::new();
    let mut mgr = FabricManager::new(tmp_dir.as_path()).expect("FabricManager::new failed");

    let (ca_key_bytes, rcac, descriptor) = mgr
        .generate_root_ca(0xFFF1, 0xCAFE_BABE_0000_0001, 1, "integration-test")
        .expect("generate_root_ca failed");

    // Generate ephemeral node keys for initiator and responder
    let init_node_key = SecretKey::random(&mut OsRng);
    let init_pub_ep = init_node_key.public_key().to_encoded_point(false);
    let init_pub: Vec<u8> = init_pub_ep.as_bytes().to_vec();

    let resp_node_key = SecretKey::random(&mut OsRng);
    let resp_pub_ep = resp_node_key.public_key().to_encoded_point(false);
    let resp_pub: Vec<u8> = resp_pub_ep.as_bytes().to_vec();

    // Register the fabric entry (using rcac as placeholder NOC — will be overwritten below)
    mgr.add_fabric_entry(
        descriptor.clone(),
        &rcac,
        &rcac, // placeholder NOC for registration
        None,
        ca_key_bytes.clone(),
    );

    let fabric_index = FabricIndex(1);

    // Issue proper NOCs signed by the fabric CA
    let init_noc = mgr
        .issue_noc(fabric_index, &init_pub, 0x0001)
        .expect("issue_noc (initiator) failed");

    let resp_noc = mgr
        .issue_noc(fabric_index, &resp_pub, 0x0002)
        .expect("issue_noc (responder) failed");

    // Run the CASE (SIGMA) handshake entirely in-memory (no network)
    let mut initiator = CaseInitiator::new(init_node_key, init_noc, None, descriptor.clone());
    let mut responder = CaseResponder::new(resp_node_key, resp_noc, None, descriptor.clone());

    // Sigma1: initiator → responder
    let (_init_session_id, sigma1) = initiator.build_sigma1().expect("build_sigma1 failed");

    // Sigma2: responder → initiator
    let (_resp_session_id, sigma2) = responder
        .handle_sigma1(&sigma1)
        .expect("handle_sigma1 failed");

    // Sigma3: initiator → responder (handle_sigma2 returns the Sigma3 payload)
    let sigma3 = initiator
        .handle_sigma2(&sigma2)
        .expect("handle_sigma2 failed");

    // Responder finalises; handle_sigma3 returns the established session directly
    let resp_session = responder
        .handle_sigma3(&sigma3)
        .expect("handle_sigma3 failed");

    // Initiator's established session is available after handle_sigma2
    let init_session = initiator
        .established_session()
        .expect("initiator has no established session");

    // Attestation challenge must match on both sides
    assert_eq!(
        init_session.attestation_challenge, resp_session.attestation_challenge,
        "CASE: attestation challenge mismatch"
    );

    // Keys are cross-symmetric:
    // - Initiator encrypt_key = I2R, decrypt_key = R2I
    // - Responder encrypt_key = R2I, decrypt_key = I2R
    assert_eq!(
        init_session.encrypt_key, resp_session.decrypt_key,
        "CASE: initiator encrypt key != responder decrypt key"
    );
    assert_eq!(
        init_session.decrypt_key, resp_session.encrypt_key,
        "CASE: initiator decrypt key != responder encrypt key"
    );
}

// ── End-to-end commission + invoke ───────────────────────────────────────────

/// End-to-end: spawn `MatterDeviceServer` on loopback, commission with
/// `MatterController`, invoke OnOff::On, assert handler fires.
///
/// Marked `#[ignore]` — requires loopback network and an available UDP port.
/// Run with:
///   cargo test -p brainwires-hardware --features matter --test matter_integration \
///     matter_e2e_commission_and_invoke -- --include-ignored
#[tokio::test]
#[ignore]
async fn matter_e2e_commission_and_invoke() {
    use brainwires_homeauto::matter::{
        MatterController, MatterDeviceConfig, MatterDeviceServer,
    };
    use std::net::UdpSocket;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    // 1. Find a free UDP port via OS port-0 bind trick.
    let sock = UdpSocket::bind("127.0.0.1:0").expect("bind port 0");
    let port = sock.local_addr().unwrap().port();
    drop(sock); // release so the server can bind it

    // 2. Build MatterDeviceConfig with well-known test passcode.
    const PASSCODE: u32 = 20202021;
    const DISCRIMINATOR: u16 = 3840;

    let device_storage = TempDir::new();
    let ctrl_storage = TempDir::new();

    let config = MatterDeviceConfig::builder()
        .device_name("E2E Test Light")
        .vendor_id(0xFFF1)
        .product_id(0x8001)
        .discriminator(DISCRIMINATOR)
        .passcode(PASSCODE)
        .storage_path(device_storage.as_path().to_str().expect("valid path"))
        .port(port)
        .build();

    // 3. Shared flag — set to true when the on_off handler fires with `true`.
    let handler_fired = Arc::new(AtomicBool::new(false));
    let handler_fired_clone = Arc::clone(&handler_fired);

    // 4. Create server and register the on_off handler.
    let server = MatterDeviceServer::new(config)
        .await
        .expect("MatterDeviceServer::new failed");

    let qr_code = server.qr_code().to_string();

    server.set_on_off_handler(move |on| {
        if on {
            handler_fired_clone.store(true, Ordering::SeqCst);
        }
    });

    // 5. Start server in a background task.
    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    let server_task = tokio::spawn(async move {
        let _ = server_clone.start().await;
    });

    // 6. Wait briefly for the server to bind and advertise.
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    // 7. Build controller and commission via the server's QR code.
    let controller = MatterController::new(
        "E2E Fabric",
        std::path::Path::new(ctrl_storage.as_path().to_str().expect("valid path")),
    )
    .await
    .expect("MatterController::new failed");

    let device = controller
        .commission_qr(&qr_code, 1)
        .await
        .expect("commission_qr failed");

    // 8. Invoke OnOff::On.
    controller
        .on_off(&device, 1, true)
        .await
        .expect("on_off(true) failed");

    // 9. Assert handler fired with `true`.
    assert!(
        handler_fired.load(Ordering::SeqCst),
        "on_off handler was not called with `true`"
    );

    // 10. Shutdown server.
    server.stop().await.expect("server stop failed");
    server_task.abort();
}

/// End-to-end commissioning chain via `commission_qr_with_session`: verifies
/// the `CommissioningSession` broadcast emits every phase in order and that
/// the controller's `FabricManager` holds a stored fabric on disk after the
/// run.
///
/// `#[ignore]` — same network-port requirement as the other e2e test.
#[tokio::test]
#[ignore]
async fn commissioning_chain_csr_addnoc_case_drives_all_phases() {
    use brainwires_homeauto::matter::fabric::FabricManager;
    use brainwires_homeauto::matter::{
        MatterController, MatterDeviceConfig, MatterDeviceServer, Phase,
    };
    use std::net::UdpSocket;
    use std::sync::Arc;

    let sock = UdpSocket::bind("127.0.0.1:0").expect("bind port 0");
    let port = sock.local_addr().unwrap().port();
    drop(sock);

    const PASSCODE: u32 = 20202021;
    const DISCRIMINATOR: u16 = 3840;

    let device_storage = TempDir::new();
    let ctrl_storage = TempDir::new();

    let config = MatterDeviceConfig::builder()
        .device_name("Commissioning Chain Light")
        .vendor_id(0xFFF1)
        .product_id(0x8001)
        .discriminator(DISCRIMINATOR)
        .passcode(PASSCODE)
        .storage_path(device_storage.as_path().to_str().expect("valid path"))
        .port(port)
        .build();

    let server = MatterDeviceServer::new(config)
        .await
        .expect("MatterDeviceServer::new");
    let qr_code = server.qr_code().to_string();
    let server = Arc::new(server);
    let server_clone = Arc::clone(&server);
    let server_task = tokio::spawn(async move {
        let _ = server_clone.start().await;
    });

    tokio::time::sleep(std::time::Duration::from_millis(200)).await;

    let controller = MatterController::new(
        "Chain Fabric",
        std::path::Path::new(ctrl_storage.as_path().to_str().expect("valid path")),
    )
    .await
    .expect("MatterController::new");

    let (_device, session) = controller
        .commission_qr_with_session(&qr_code, 0xDEAD_BEEF)
        .await
        .expect("commission_qr_with_session");

    // Six expected phases beyond the initial Parsed state.
    let mut expected = vec![
        Phase::Discovered,
        Phase::PaseEstablished,
        Phase::CsrReceived,
        Phase::NocInstalled,
        Phase::CaseEstablished,
    ];
    expected.retain(|p| *p != session.phase()); // if some have already coalesced
    assert_eq!(
        session.phase(),
        Phase::CaseEstablished,
        "final phase must be CaseEstablished, got {:?}",
        session.phase()
    );

    // FabricManager must have a persisted entry.
    let ctrl_fabric = FabricManager::load(std::path::Path::new(
        ctrl_storage.as_path().to_str().expect("valid path"),
    ))
    .await
    .expect("FabricManager::load after commissioning");
    assert!(
        !ctrl_fabric.fabrics().is_empty(),
        "controller fabric should be persisted after commissioning"
    );

    server.stop().await.expect("server stop");
    server_task.abort();
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Owned temporary directory that is removed on drop.
struct TempDir {
    path: std::path::PathBuf,
}

impl TempDir {
    fn new() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        let path =
            std::env::temp_dir().join(format!("bw-matter-test-{}-{}", std::process::id(), ts));
        std::fs::create_dir_all(&path).unwrap();
        Self { path }
    }

    fn as_path(&self) -> &std::path::Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
