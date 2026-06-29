//! Agent operations — supervisor, health monitoring, attention, parallel coordination,
//! hibernation data types, and autonomous training.

pub mod health;
pub mod hibernate;

#[cfg(feature = "supervisor")]
pub mod supervisor;

#[cfg(feature = "attention")]
pub mod attention;

#[cfg(feature = "parallel")]
pub mod parallel;

#[cfg(feature = "training")]
pub mod training_loop;

pub use health::{DegradationSignal, HealthMonitor, HealthStatus, PerformanceMetrics};
pub use hibernate::{HibernateManifest, HibernatedSession};

#[cfg(feature = "supervisor")]
pub use supervisor::AgentSupervisor;

#[cfg(feature = "attention")]
pub use attention::AttentionMechanism;

#[cfg(feature = "parallel")]
pub use parallel::ParallelCoordinator;

#[cfg(feature = "training")]
pub use training_loop::AutonomousTrainingLoop;
