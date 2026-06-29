//! ProxyService — the assembled proxy with its run loop.

use crate::config::ProxyConfig;
use crate::convert::ConversionRegistry;
use crate::error::{ProxyError, ProxyResult};
use crate::inspector::{EventBroadcaster, EventStore};
use crate::middleware::MiddlewareStack;
use crate::transport::{InboundConnection, TransportConnector};
use std::sync::Arc;
use tokio::sync::{mpsc, watch};

const CONNECTION_CHANNEL_CAPACITY: usize = 256;

/// A boxed listener factory that produces a future accepting inbound connections.
pub(crate) type ListenerFactory = Box<
    dyn Fn(
            mpsc::Sender<InboundConnection>,
            watch::Receiver<bool>,
        ) -> futures::future::BoxFuture<'static, ProxyResult<()>>
        + Send
        + Sync,
>;

/// The assembled proxy service. Call [`run()`](ProxyService::run) to start.
pub struct ProxyService {
    pub(crate) config: ProxyConfig,
    pub(crate) middleware: MiddlewareStack,
    pub(crate) connector: Box<dyn TransportConnector>,
    pub(crate) listener_factory: ListenerFactory,
    pub(crate) conversions: ConversionRegistry,
    pub(crate) event_store: Arc<EventStore>,
    pub(crate) event_broadcaster: Arc<EventBroadcaster>,
}

impl ProxyService {
    /// Run the proxy until shutdown.
    pub async fn run(self) -> ProxyResult<()> {
        let (shutdown_tx, shutdown_rx) = watch::channel(false);
        let (conn_tx, mut conn_rx) =
            mpsc::channel::<InboundConnection>(CONNECTION_CHANNEL_CAPACITY);

        // Spawn the inspector API if configured
        #[cfg(feature = "inspector-api")]
        if let Some(api_addr) = self.config.inspector.api_addr {
            let store = self.event_store.clone();
            let broadcaster = self.event_broadcaster.clone();
            let api_shutdown = shutdown_rx.clone();
            tokio::spawn(async move {
                if let Err(e) = crate::inspector::api::run_inspector_api(
                    api_addr,
                    store,
                    broadcaster,
                    api_shutdown,
                )
                .await
                {
                    tracing::error!(error = %e, "Inspector API failed");
                }
            });
        }

        // Spawn the listener
        let listener_shutdown = shutdown_rx.clone();
        let listener_fut = (self.listener_factory)(conn_tx, listener_shutdown);
        let listener_handle = tokio::spawn(async move {
            if let Err(e) = listener_fut.await {
                tracing::error!(error = %e, "Listener failed");
            }
        });

        // Process connections
        let connector = Arc::new(self.connector);
        let middleware = Arc::new(self.middleware);
        let timeout = self.config.timeout;

        tracing::info!("Proxy service started");

        while let Some((request, resp_tx)) = conn_rx.recv().await {
            let connector = connector.clone();
            let middleware = middleware.clone();

            tokio::spawn(async move {
                let result = tokio::time::timeout(timeout, async {
                    // Run request through middleware
                    let (request, depth) = match middleware.process_request(request).await {
                        Ok(Ok((req, depth))) => (req, depth),
                        Ok(Err(response)) => {
                            // Middleware short-circuited
                            let _ = resp_tx.send(response);
                            return Ok(());
                        }
                        Err(e) => return Err(e),
                    };

                    // Forward to upstream
                    let response = connector.forward(request).await?;

                    // Run response through middleware (reverse)
                    let response = middleware.process_response(response, depth).await?;

                    let _ = resp_tx.send(response);
                    Ok::<(), ProxyError>(())
                })
                .await;

                match result {
                    Ok(Ok(())) => {}
                    Ok(Err(e)) => {
                        tracing::error!(error = %e, "Request processing failed");
                    }
                    Err(_) => {
                        tracing::warn!("Request timed out");
                    }
                }
            });
        }

        // Shutdown
        let _ = shutdown_tx.send(true);
        listener_handle.await.ok();

        tracing::info!("Proxy service stopped");
        Ok(())
    }

    /// Access the conversion registry.
    pub fn conversions(&self) -> &ConversionRegistry {
        &self.conversions
    }

    /// Access the event store for querying captured traffic.
    pub fn event_store(&self) -> &Arc<EventStore> {
        &self.event_store
    }

    /// Access the broadcaster for subscribing to live events.
    pub fn event_broadcaster(&self) -> &Arc<EventBroadcaster> {
        &self.event_broadcaster
    }
}
