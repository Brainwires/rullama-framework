pub mod crypto;
pub mod discovery;
pub mod protocol;
pub mod socket;

pub use crypto::*;
pub use discovery::{
    cleanup_session, cleanup_stale_sockets, delete_agent_metadata, format_agent_tree,
    get_agent_depth, get_child_agents, get_root_agents, list_agent_sessions,
    list_agent_sessions_with_metadata, read_agent_metadata, update_agent_metadata,
    write_agent_metadata,
};
pub use protocol::{
    AgentConfig, AgentMessage, AgentMetadata, ChildNotifyAction, DisplayMessage, Handshake,
    HandshakeResponse, LockChangeType, LockInfo, ParentSignalType, ResourceLockType, ViewerMessage,
};
pub use socket::{
    EncryptedIpcConnection, EncryptedIpcReader, EncryptedIpcWriter, IpcConnection, IpcReader,
    IpcWriter,
};
