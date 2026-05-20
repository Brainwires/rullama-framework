//! Session Server
//!
//! Manages a TUI session running in a PTY.
//! Accepts client connections via Unix socket and proxies I/O.

use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, bail};
use chrono::Local;
use nix::sys::signal::{Signal, kill};
use nix::sys::wait::{WaitStatus, waitpid};
use nix::unistd::Pid;
use nix::unistd::{ForkResult, fork, setsid};

/// Log to a file (since daemon has no stdout)
fn daemon_log(session_id: &str, msg: &str) {
    if let Ok(sessions_dir) = crate::config::PlatformPaths::sessions_dir() {
        let log_path = sessions_dir.join(format!("{}.log", session_id));
        if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&log_path) {
            let _ = writeln!(f, "[{}] {}", Local::now().format("%H:%M:%S"), msg);
        }
    }
}

/// Helper to close a raw fd using libc
fn close_fd(fd: RawFd) {
    unsafe {
        libc::close(fd);
    }
}

/// Helper to dup2 using libc
fn dup2_fd(src: RawFd, dst: RawFd) -> Result<()> {
    if unsafe { libc::dup2(src, dst) } < 0 {
        bail!("dup2 failed: {}", std::io::Error::last_os_error());
    }
    Ok(())
}

/// Open a PTY pair using libc
fn openpty_libc() -> Result<(RawFd, RawFd)> {
    let mut master: libc::c_int = 0;
    let mut slave: libc::c_int = 0;
    let ret = unsafe {
        libc::openpty(
            &mut master,
            &mut slave,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
        )
    };
    if ret < 0 {
        bail!("openpty failed: {}", std::io::Error::last_os_error());
    }
    Ok((master, slave))
}

/// Session server that manages a TUI in a PTY
pub struct SessionServer {
    session_id: String,
    socket_path: PathBuf,
    master_fd: Option<OwnedFd>,
    child_pid: Option<Pid>,
    running: Arc<AtomicBool>,
}

impl SessionServer {
    /// Create a new session server
    pub fn new(session_id: String) -> Result<Self> {
        let socket_path = super::get_session_socket_path(&session_id)?;

        // Ensure parent directory exists
        if let Some(parent) = socket_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Remove stale socket
        if socket_path.exists() {
            std::fs::remove_file(&socket_path)?;
        }

        Ok(Self {
            session_id,
            socket_path,
            master_fd: None,
            child_pid: None,
            running: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Start the session server
    ///
    /// This spawns the TUI in a PTY and starts accepting client connections.
    /// The server runs until the TUI exits or receives a shutdown signal.
    pub fn run(&mut self, tui_args: Vec<String>) -> Result<i32> {
        daemon_log(&self.session_id, "SessionServer::run() starting");

        // Open a PTY pair
        let (master_fd_raw, slave_fd) = openpty_libc()?;
        daemon_log(
            &self.session_id,
            &format!("PTY opened: master={}, slave={}", master_fd_raw, slave_fd),
        );
        let master_fd = unsafe { OwnedFd::from_raw_fd(master_fd_raw) };

        // Set initial PTY size - use reasonable defaults
        // The TUI needs a non-zero size to render properly
        let mut ws: libc::winsize = unsafe { std::mem::zeroed() };
        ws.ws_col = 120;
        ws.ws_row = 40;
        ws.ws_xpixel = 0;
        ws.ws_ypixel = 0;
        unsafe {
            libc::ioctl(master_fd_raw, libc::TIOCSWINSZ, &ws);
        }
        daemon_log(&self.session_id, "PTY size set to 120x40");

        // Fork to create the TUI process
        daemon_log(&self.session_id, "Forking TUI process...");
        match unsafe { fork() }.context("Failed to fork")? {
            ForkResult::Child => {
                // Child process - runs the TUI
                drop(master_fd); // Close master in child

                // Create new session and set controlling terminal
                setsid().context("Failed to setsid")?;

                // Set slave as stdin/stdout/stderr
                dup2_fd(slave_fd, 0)?;
                dup2_fd(slave_fd, 1)?;
                dup2_fd(slave_fd, 2)?;

                if slave_fd > 2 {
                    close_fd(slave_fd);
                }

                // Set controlling terminal
                unsafe {
                    libc::ioctl(0, libc::TIOCSCTTY as libc::c_ulong, 0);
                }

                // Exec the TUI
                let exe = std::env::current_exe()?;
                let mut args = vec!["chat".to_string(), "--tui".to_string()];
                args.extend(tui_args);

                let c_exe = std::ffi::CString::new(exe.to_string_lossy().as_bytes())?;
                let c_args: Vec<std::ffi::CString> = std::iter::once(Ok(c_exe.clone()))
                    .chain(args.iter().map(|a| std::ffi::CString::new(a.as_bytes())))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| {
                        anyhow::anyhow!("Invalid argument string (contains null byte): {}", e)
                    })?;
                let c_args_ptrs: Vec<*const libc::c_char> = c_args
                    .iter()
                    .map(|a| a.as_ptr())
                    .chain(std::iter::once(std::ptr::null()))
                    .collect();

                unsafe {
                    libc::execv(c_exe.as_ptr(), c_args_ptrs.as_ptr());
                }
                // execv only returns on error
                std::process::exit(1);
            }
            ForkResult::Parent { child } => {
                // Parent process - runs the server
                close_fd(slave_fd); // Close slave in parent
                daemon_log(&self.session_id, &format!("TUI forked with PID {}", child));

                self.master_fd = Some(master_fd);
                self.child_pid = Some(child);
                self.running.store(true, Ordering::SeqCst);

                // Run the server loop
                daemon_log(&self.session_id, "Starting server loop...");
                self.server_loop()
            }
        }
    }

    /// Send terminal cleanup sequences to a client before disconnection
    /// This restores the terminal to a sane state (leaves alternate screen, disables mouse)
    fn send_terminal_cleanup(stream: &mut UnixStream, session_id: &str) {
        // Terminal cleanup sequences:
        // - Disable mouse capture: ESC [ ? 1003 l and ESC [ ? 1006 l
        // - Disable bracketed paste: ESC [ ? 2004 l
        // - Leave alternate screen: ESC [ ? 1049 l
        // - Show cursor: ESC [ ? 25 h
        let terminal_cleanup = b"\x1b[?1003l\x1b[?1006l\x1b[?2004l\x1b[?1049l\x1b[?25h";
        if let Err(e) = stream.write_all(terminal_cleanup) {
            daemon_log(
                session_id,
                &format!("Failed to send terminal cleanup: {}", e),
            );
        } else {
            daemon_log(session_id, "Sent terminal cleanup sequences to client");
        }
    }

    /// Main server loop - accepts clients and proxies I/O
    fn server_loop(&mut self) -> Result<i32> {
        daemon_log(
            &self.session_id,
            &format!("Binding socket: {}", self.socket_path.display()),
        );

        // Bind to socket
        let listener = match UnixListener::bind(&self.socket_path) {
            Ok(l) => l,
            Err(e) => {
                daemon_log(&self.session_id, &format!("Failed to bind socket: {}", e));
                return Err(e).with_context(|| {
                    format!("Failed to bind socket: {}", self.socket_path.display())
                });
            }
        };

        daemon_log(&self.session_id, "Socket bound successfully");

        // Set non-blocking for the listener
        listener.set_nonblocking(true)?;

        let master_fd = self.master_fd.as_ref().unwrap().as_raw_fd();

        // Set master FD non-blocking
        unsafe {
            let flags = libc::fcntl(master_fd, libc::F_GETFL);
            libc::fcntl(master_fd, libc::F_SETFL, flags | libc::O_NONBLOCK);
        }

        daemon_log(&self.session_id, "Entering main loop...");

        // Track connected client
        let mut client: Option<UnixStream> = None;
        let mut buf = [0u8; 4096];

        while self.running.load(Ordering::SeqCst) {
            // Check if child is still alive
            if let Some(pid) = self.child_pid {
                match waitpid(pid, Some(nix::sys::wait::WaitPidFlag::WNOHANG)) {
                    Ok(WaitStatus::Exited(_, code)) => {
                        daemon_log(&self.session_id, &format!("TUI exited with code {}", code));
                        return Ok(code);
                    }
                    Ok(WaitStatus::Signaled(_, sig, _)) => {
                        daemon_log(&self.session_id, &format!("TUI killed by signal {:?}", sig));
                        return Ok(128 + sig as i32);
                    }
                    Ok(WaitStatus::StillAlive) => {}
                    Ok(_) => {}
                    Err(_) => {}
                }
            }

            // Accept new clients
            match listener.accept() {
                Ok((mut stream, _)) => {
                    daemon_log(&self.session_id, "Client connected!");

                    // Disconnect previous client if any
                    if let Some(old) = client.take() {
                        drop(old);
                    }

                    // IMPORTANT: Do a blocking read first to get the client's window size
                    // The client sends this immediately on connect, before any other data
                    // This prevents the flicker from rendering at wrong size
                    let mut winsize_buf = [0u8; 8];
                    stream.set_read_timeout(Some(std::time::Duration::from_millis(500)))?;
                    match stream.read_exact(&mut winsize_buf) {
                        Ok(()) => {
                            // Check if it's a window size message: ESC ] W S [cols:u16] [rows:u16]
                            if winsize_buf[0] == 0x1b
                                && winsize_buf[1] == 0x5d
                                && winsize_buf[2] == 0x57
                                && winsize_buf[3] == 0x53
                            {
                                let cols = u16::from_be_bytes([winsize_buf[4], winsize_buf[5]]);
                                let rows = u16::from_be_bytes([winsize_buf[6], winsize_buf[7]]);
                                daemon_log(
                                    &self.session_id,
                                    &format!("Initial window size from client: {}x{}", cols, rows),
                                );

                                // Set PTY size
                                let mut ws: libc::winsize = unsafe { std::mem::zeroed() };
                                ws.ws_col = cols;
                                ws.ws_row = rows;
                                ws.ws_xpixel = 0;
                                ws.ws_ypixel = 0;
                                unsafe {
                                    libc::ioctl(master_fd, libc::TIOCSWINSZ, &ws);
                                }
                            } else {
                                daemon_log(
                                    &self.session_id,
                                    "First message wasn't window size, using defaults",
                                );
                                // Set default size
                                let mut ws: libc::winsize = unsafe { std::mem::zeroed() };
                                ws.ws_col = 80;
                                ws.ws_row = 24;
                                unsafe {
                                    libc::ioctl(master_fd, libc::TIOCSWINSZ, &ws);
                                }
                                // Write the data to PTY since it wasn't a window size message
                                unsafe {
                                    libc::write(
                                        master_fd,
                                        winsize_buf.as_ptr() as *const libc::c_void,
                                        8,
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            daemon_log(
                                &self.session_id,
                                &format!("Failed to read initial window size: {}", e),
                            );
                            // Set default size
                            let mut ws: libc::winsize = unsafe { std::mem::zeroed() };
                            ws.ws_col = 80;
                            ws.ws_row = 24;
                            unsafe {
                                libc::ioctl(master_fd, libc::TIOCSWINSZ, &ws);
                            }
                        }
                    }

                    // Now set non-blocking for normal operation
                    stream.set_read_timeout(None)?;
                    stream.set_nonblocking(true)?;

                    // Send terminal setup sequences directly to the client
                    // These are the escape sequences that the TUI sent when it started,
                    // but the client wasn't connected then. We need to send them now:
                    // - Enter alternate screen: ESC [ ? 1049 h
                    // - Enable mouse capture (SGR mode): ESC [ ? 1006 h and ESC [ ? 1003 h
                    // - Enable bracketed paste: ESC [ ? 2004 h
                    let terminal_setup = b"\x1b[?1049h\x1b[?1006h\x1b[?1003h\x1b[?2004h";
                    if let Err(e) = stream.write_all(terminal_setup) {
                        daemon_log(
                            &self.session_id,
                            &format!("Failed to send terminal setup: {}", e),
                        );
                    } else {
                        daemon_log(&self.session_id, "Sent terminal setup sequences to client");
                    }

                    client = Some(stream);

                    // Send SIGWINCH to TUI to force a redraw with the correct size
                    // This ensures the client sees the current screen at the right dimensions
                    if let Some(pid) = self.child_pid {
                        let _ = kill(pid, Signal::SIGWINCH);
                    }
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
                Err(e) => {
                    tracing::error!("Accept error: {}", e);
                }
            }

            // Proxy I/O between client and PTY
            if let Some(ref mut stream) = client {
                // Read from client, write to PTY
                match stream.read(&mut buf) {
                    Ok(0) => {
                        daemon_log(&self.session_id, "Client disconnected (read 0)");
                        // Send terminal cleanup sequences before disconnecting
                        Self::send_terminal_cleanup(stream, &self.session_id);
                        client = None;
                    }
                    Ok(n) => {
                        daemon_log(&self.session_id, &format!("Client -> PTY: {} bytes", n));

                        // Process buffer, extracting window size messages
                        // Window size format: ESC ] W S [cols:u16] [rows:u16] (8 bytes)
                        let data = &buf[..n];
                        let mut i = 0;
                        while i < data.len() {
                            // Look for window size escape sequence
                            if i + 8 <= data.len()
                                && data[i] == 0x1b
                                && data[i + 1] == 0x5d
                                && data[i + 2] == 0x57
                                && data[i + 3] == 0x53
                            {
                                let cols = u16::from_be_bytes([data[i + 4], data[i + 5]]);
                                let rows = u16::from_be_bytes([data[i + 6], data[i + 7]]);
                                daemon_log(
                                    &self.session_id,
                                    &format!("Window size update: {}x{}", cols, rows),
                                );

                                // Update PTY size
                                let mut ws: libc::winsize = unsafe { std::mem::zeroed() };
                                ws.ws_col = cols;
                                ws.ws_row = rows;
                                ws.ws_xpixel = 0;
                                ws.ws_ypixel = 0;
                                unsafe {
                                    libc::ioctl(master_fd, libc::TIOCSWINSZ, &ws);
                                }

                                // Send SIGWINCH to TUI to trigger redraw
                                if let Some(pid) = self.child_pid {
                                    let _ = kill(pid, Signal::SIGWINCH);
                                    daemon_log(&self.session_id, "Sent SIGWINCH to TUI");
                                }

                                i += 8; // Skip past window size message
                                continue;
                            }

                            // Find next potential escape sequence or end of data
                            let mut end = i + 1;
                            while end < data.len() && data[end] != 0x1b {
                                end += 1;
                            }

                            // Write non-escape data to PTY
                            let chunk = &data[i..end];
                            if !chunk.is_empty() {
                                let written = unsafe {
                                    libc::write(
                                        master_fd,
                                        chunk.as_ptr() as *const libc::c_void,
                                        chunk.len(),
                                    )
                                };
                                daemon_log(
                                    &self.session_id,
                                    &format!("Wrote {} bytes to PTY", written),
                                );
                            }
                            i = end;
                        }
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
                    Err(e) => {
                        daemon_log(&self.session_id, &format!("Client read error: {}", e));
                        client = None;
                    }
                }
            }

            // Read from PTY, write to client
            let mut pty_buf = [0u8; 4096];
            let n = unsafe {
                libc::read(
                    master_fd,
                    pty_buf.as_mut_ptr() as *mut libc::c_void,
                    pty_buf.len(),
                )
            };

            if n > 0 {
                let n = n as usize;
                daemon_log(&self.session_id, &format!("PTY -> Client: {} bytes", n));
                if let Some(ref mut stream) = client {
                    if let Err(e) = stream.write_all(&pty_buf[..n])
                        && e.kind() != std::io::ErrorKind::WouldBlock
                    {
                        daemon_log(&self.session_id, &format!("Client write error: {}", e));
                        client = None;
                    }
                } else {
                    // No client connected - output is discarded
                    // This is fine - the TUI keeps running, client can attach later
                }
            } else if n < 0 {
                let err = std::io::Error::last_os_error();
                if err.kind() != std::io::ErrorKind::WouldBlock
                    && err.raw_os_error() != Some(libc::EAGAIN)
                {
                    daemon_log(
                        &self.session_id,
                        &format!("PTY read error: {} (closing)", err),
                    );
                    // PTY closed - TUI exited
                    break;
                }
            }

            // Small sleep to avoid busy loop
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        // Cleanup
        let _ = std::fs::remove_file(&self.socket_path);

        // Wait for child and get exit code
        if let Some(pid) = self.child_pid {
            match waitpid(pid, None) {
                Ok(WaitStatus::Exited(_, code)) => Ok(code),
                Ok(WaitStatus::Signaled(_, sig, _)) => Ok(128 + sig as i32),
                _ => Ok(1),
            }
        } else {
            Ok(0)
        }
    }

    /// Get the session ID
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Get the socket path
    pub fn socket_path(&self) -> &PathBuf {
        &self.socket_path
    }
}

impl Drop for SessionServer {
    fn drop(&mut self) {
        self.running.store(false, Ordering::SeqCst);

        // Kill child if still running
        if let Some(pid) = self.child_pid {
            let _ = kill(pid, Signal::SIGTERM);
        }

        // Remove socket
        let _ = std::fs::remove_file(&self.socket_path);
    }
}

/// Spawn a session server in the background
///
/// This forks a daemon process that runs the session server.
/// Returns the session ID and socket path.
pub fn spawn_session(
    session_id: Option<String>,
    tui_args: Vec<String>,
) -> Result<(String, PathBuf)> {
    let session_id = session_id.unwrap_or_else(super::generate_session_id);
    let socket_path = super::get_session_socket_path(&session_id)?;

    // Ensure parent directory exists
    if let Some(parent) = socket_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    match unsafe { fork() }.context("Failed to fork daemon")? {
        ForkResult::Child => {
            // Daemon process
            setsid().ok(); // New session

            // Double fork to fully daemonize
            match unsafe { fork() } {
                Ok(ForkResult::Child) => {
                    // Grandchild - actual daemon
                    daemon_log(&session_id, "Daemon grandchild started");

                    // Close standard fds
                    close_fd(0);
                    close_fd(1);
                    close_fd(2);

                    // Redirect to /dev/null
                    let null = std::fs::OpenOptions::new()
                        .read(true)
                        .write(true)
                        .open("/dev/null")
                        .expect("Failed to open /dev/null");
                    let null_fd = null.as_raw_fd();
                    let _ = dup2_fd(null_fd, 0);
                    let _ = dup2_fd(null_fd, 1);
                    let _ = dup2_fd(null_fd, 2);

                    daemon_log(&session_id, "Standard FDs redirected to /dev/null");

                    // Run the session server
                    daemon_log(&session_id, "Creating SessionServer...");
                    let mut server = match SessionServer::new(session_id.clone()) {
                        Ok(s) => s,
                        Err(e) => {
                            daemon_log(&session_id, &format!("Failed to create server: {}", e));
                            std::process::exit(1);
                        }
                    };
                    daemon_log(
                        &session_id,
                        &format!("SessionServer created, running with args: {:?}", tui_args),
                    );
                    let exit_code = match server.run(tui_args) {
                        Ok(code) => {
                            daemon_log(
                                &session_id,
                                &format!("Server run completed with code {}", code),
                            );
                            code
                        }
                        Err(e) => {
                            daemon_log(&session_id, &format!("Server run failed: {}", e));
                            1
                        }
                    };
                    std::process::exit(exit_code);
                }
                Ok(ForkResult::Parent { .. }) => {
                    // First child exits immediately
                    std::process::exit(0);
                }
                Err(_) => {
                    std::process::exit(1);
                }
            }
        }
        ForkResult::Parent { child } => {
            // Wait for first child to exit (it exits immediately)
            let _ = waitpid(child, None);

            // Wait for socket to appear
            for _ in 0..50 {
                if socket_path.exists() {
                    return Ok((session_id, socket_path));
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }

            bail!("Session server failed to start (socket not created)");
        }
    }
}
