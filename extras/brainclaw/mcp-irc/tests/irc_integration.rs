//! Integration test: spin up a minimal IRC server on a loopback TCP
//! port, connect the adapter via [`brainwires_irc_channel::irc_client::run`],
//! send a PRIVMSG line from the "server" side, and assert the adapter
//! forwards it to the gateway sink.

use std::sync::Arc;
use std::time::Duration;

use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{broadcast, mpsc};

use brainwires_irc_channel::config::IrcConfig;
use brainwires_irc_channel::irc_client::{IrcChannel, run};
use brainwires_network::channels::{ChannelEvent, MessageContent};

async fn spawn_stub(script: Vec<&'static str>) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    tokio::spawn(async move {
        loop {
            let (socket, _peer) = match listener.accept().await {
                Ok(x) => x,
                Err(_) => return,
            };
            let script = script.clone();
            tokio::spawn(handle_client(socket, script));
        }
    });
    addr
}

async fn handle_client(socket: tokio::net::TcpStream, script: Vec<&'static str>) {
    let (reader, mut writer) = socket.into_split();
    let mut reader = BufReader::new(reader).lines();
    let mut welcomed = false;
    let mut joined_emitted = false;

    loop {
        let read = tokio::time::timeout(Duration::from_secs(5), reader.next_line()).await;
        let line = match read {
            Ok(Ok(Some(l))) => l,
            _ => break,
        };
        let upper = line.to_uppercase();
        if upper.starts_with("CAP LS") {
            writer.write_all(b":test.server CAP * LS :\r\n").await.ok();
        } else if upper.starts_with("CAP REQ") {
            writer.write_all(b":test.server CAP * NAK :\r\n").await.ok();
        } else if upper.starts_with("NICK") || upper.starts_with("USER") {
            if !welcomed {
                welcomed = true;
                writer
                    .write_all(b":test.server 001 testbot :Welcome to the test IRC network\r\n")
                    .await
                    .ok();
                writer
                    .write_all(b":test.server 376 testbot :End of MOTD\r\n")
                    .await
                    .ok();
            }
        } else if upper.starts_with("JOIN ") && !joined_emitted {
            joined_emitted = true;
            writer
                .write_all(b":testbot!testbot@host JOIN :#test\r\n")
                .await
                .ok();
            writer
                .write_all(b":test.server 353 testbot = #test :@testbot alice bob\r\n")
                .await
                .ok();
            writer
                .write_all(b":test.server 366 testbot #test :End of /NAMES list.\r\n")
                .await
                .ok();
            for msg in &script {
                writer.write_all(msg.as_bytes()).await.ok();
            }
            writer.flush().await.ok();
        }
    }
}

async fn wait_for_listener(addr: &str) {
    for _ in 0..40 {
        if TcpStream::connect(addr).await.is_ok() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

fn irc_cfg(host: &str, port: u16) -> IrcConfig {
    IrcConfig {
        server: host.to_string(),
        port,
        use_tls: false,
        nick: "testbot".into(),
        username: "testbot".into(),
        realname: "Test Bot".into(),
        sasl_password: None,
        channels: vec!["#test".into()],
        message_prefix: "brainclaw: ".into(),
        gateway_url: String::new(),
        gateway_token: None,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn adapter_forwards_prefixed_channel_message() {
    let addr = spawn_stub(vec![
        ":alice!alice@host PRIVMSG #test :brainclaw: hello there\r\n",
    ])
    .await;
    wait_for_listener(&addr).await;
    let (host, port_str) = addr.split_once(':').unwrap();
    let port: u16 = port_str.parse().unwrap();

    let cfg = irc_cfg(host, port);
    let channel = Arc::new(IrcChannel::new(cfg.server.clone()));
    let (event_tx, mut event_rx) = mpsc::channel(8);
    let (_shutdown_tx, shutdown_rx) = broadcast::channel::<()>(1);
    tokio::spawn(async move {
        let _ = run(cfg, channel, event_tx, shutdown_rx).await;
    });

    let evt = tokio::time::timeout(Duration::from_secs(10), event_rx.recv())
        .await
        .expect("timeout waiting for forwarded event")
        .expect("channel sender dropped");
    match evt {
        ChannelEvent::MessageReceived(m) => {
            assert_eq!(m.author, "alice");
            assert_eq!(m.conversation.channel_id, "#test");
            match m.content {
                MessageContent::Text(t) => assert_eq!(t, "hello there"),
                _ => panic!(),
            }
        }
        _ => panic!("unexpected event"),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn adapter_ignores_unprefixed_channel_message() {
    let addr = spawn_stub(vec![":alice!alice@host PRIVMSG #test :just chatting\r\n"]).await;
    wait_for_listener(&addr).await;
    let (host, port_str) = addr.split_once(':').unwrap();
    let port: u16 = port_str.parse().unwrap();

    let cfg = irc_cfg(host, port);
    let channel = Arc::new(IrcChannel::new(cfg.server.clone()));
    let (event_tx, mut event_rx) = mpsc::channel(8);
    let (_shutdown_tx, shutdown_rx) = broadcast::channel::<()>(1);
    tokio::spawn(async move {
        let _ = run(cfg, channel, event_tx, shutdown_rx).await;
    });

    let result = tokio::time::timeout(Duration::from_millis(3000), event_rx.recv()).await;
    assert!(result.is_err(), "expected no event, got {result:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn adapter_forwards_private_message() {
    let addr = spawn_stub(vec![":bob!bob@host PRIVMSG testbot :ping\r\n"]).await;
    wait_for_listener(&addr).await;
    let (host, port_str) = addr.split_once(':').unwrap();
    let port: u16 = port_str.parse().unwrap();

    let cfg = irc_cfg(host, port);
    let channel = Arc::new(IrcChannel::new(cfg.server.clone()));
    let (event_tx, mut event_rx) = mpsc::channel(8);
    let (_shutdown_tx, shutdown_rx) = broadcast::channel::<()>(1);
    tokio::spawn(async move {
        let _ = run(cfg, channel, event_tx, shutdown_rx).await;
    });

    let evt = tokio::time::timeout(Duration::from_secs(10), event_rx.recv())
        .await
        .expect("timeout")
        .expect("dropped");
    match evt {
        ChannelEvent::MessageReceived(m) => {
            assert_eq!(m.author, "bob");
            assert_eq!(m.conversation.channel_id, "pm:bob");
            match m.content {
                MessageContent::Text(t) => assert_eq!(t, "ping"),
                _ => panic!(),
            }
        }
        _ => panic!("unexpected"),
    }
}
