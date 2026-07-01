use std::time::Duration;

use nusb::{
    Interface,
    transfer::{ControlIn, ControlOut, ControlType, Recipient, RequestBuffer},
};
use tracing::debug;

use super::device::find_device;
use super::types::UsbError;

/// An open USB device handle with a claimed interface, ready for transfers.
///
/// All transfer methods are async and use `nusb`'s native async API — no
/// blocking thread pool needed.
///
/// The interface is released when this handle is dropped.
pub struct UsbHandle {
    interface: Interface,
    endpoint_in: Option<u8>,
    endpoint_out: Option<u8>,
}

impl UsbHandle {
    /// Open the first device matching `vendor_id:product_id` and claim
    /// `interface_number`.
    pub async fn open(
        vendor_id: u16,
        product_id: u16,
        interface_number: u8,
    ) -> Result<Self, UsbError> {
        let info = find_device(vendor_id, product_id)?;
        let device = info
            .open()
            .map_err(|e| UsbError::OpenFailed(e.to_string()))?;

        let interface = device
            .claim_interface(interface_number)
            .map_err(|e| UsbError::ClaimFailed(interface_number, e.to_string()))?;

        // Discover first bulk IN and OUT endpoints from the active alternate setting
        let (endpoint_in, endpoint_out) = discover_bulk_endpoints(&interface);

        debug!(
            "Opened {vendor_id:04x}:{product_id:04x} iface={interface_number} \
             bulk_in={endpoint_in:?} bulk_out={endpoint_out:?}"
        );

        Ok(Self {
            interface,
            endpoint_in,
            endpoint_out,
        })
    }

    // ── Control Transfers ─────────────────────────────────────────────────────

    /// Issue a control IN transfer (host ← device).
    pub async fn control_in(
        &self,
        request_type: ControlType,
        recipient: Recipient,
        request: u8,
        value: u16,
        index: u16,
        length: u16,
    ) -> Result<Vec<u8>, UsbError> {
        let ctrl = ControlIn {
            control_type: request_type,
            recipient,
            request,
            value,
            index,
            length,
        };
        self.interface
            .control_in(ctrl)
            .await
            .into_result()
            .map_err(|e| UsbError::TransferFailed {
                endpoint: 0x00,
                reason: e.to_string(),
            })
    }

    /// Issue a control OUT transfer (host → device).
    pub async fn control_out(
        &self,
        request_type: ControlType,
        recipient: Recipient,
        request: u8,
        value: u16,
        index: u16,
        data: Vec<u8>,
    ) -> Result<(), UsbError> {
        let ctrl = ControlOut {
            control_type: request_type,
            recipient,
            request,
            value,
            index,
            data: &data,
        };
        self.interface
            .control_out(ctrl)
            .await
            .into_result()
            .map_err(|e| UsbError::TransferFailed {
                endpoint: 0x00,
                reason: e.to_string(),
            })?;
        Ok(())
    }

    // ── Bulk Transfers ────────────────────────────────────────────────────────

    /// Bulk read from `endpoint` (or the auto-discovered bulk IN endpoint).
    pub async fn bulk_read(
        &self,
        endpoint: Option<u8>,
        length: usize,
        _timeout: Duration,
    ) -> Result<Vec<u8>, UsbError> {
        let ep = endpoint
            .or(self.endpoint_in)
            .ok_or_else(|| UsbError::Other("no bulk IN endpoint available".into()))?;

        self.interface
            .bulk_in(ep, RequestBuffer::new(length))
            .await
            .into_result()
            .map_err(|e| UsbError::TransferFailed {
                endpoint: ep,
                reason: e.to_string(),
            })
    }

    /// Bulk write to `endpoint` (or the auto-discovered bulk OUT endpoint).
    pub async fn bulk_write(
        &self,
        endpoint: Option<u8>,
        data: Vec<u8>,
        _timeout: Duration,
    ) -> Result<usize, UsbError> {
        let ep = endpoint
            .or(self.endpoint_out)
            .ok_or_else(|| UsbError::Other("no bulk OUT endpoint available".into()))?;

        let len = data.len();
        self.interface
            .bulk_out(ep, data)
            .await
            .into_result()
            .map_err(|e| UsbError::TransferFailed {
                endpoint: ep,
                reason: e.to_string(),
            })?;
        Ok(len)
    }

    // ── Interrupt Transfers ───────────────────────────────────────────────────

    /// Read from an interrupt IN endpoint (e.g. HID reports).
    pub async fn interrupt_read(
        &self,
        endpoint: u8,
        length: usize,
        _timeout: Duration,
    ) -> Result<Vec<u8>, UsbError> {
        self.interface
            .interrupt_in(endpoint, RequestBuffer::new(length))
            .await
            .into_result()
            .map_err(|e| UsbError::TransferFailed {
                endpoint,
                reason: e.to_string(),
            })
    }

    /// Write to an interrupt OUT endpoint.
    pub async fn interrupt_write(
        &self,
        endpoint: u8,
        data: Vec<u8>,
        _timeout: Duration,
    ) -> Result<usize, UsbError> {
        let len = data.len();
        self.interface
            .interrupt_out(endpoint, data)
            .await
            .into_result()
            .map_err(|e| UsbError::TransferFailed {
                endpoint,
                reason: e.to_string(),
            })?;
        Ok(len)
    }

    /// The auto-discovered bulk IN endpoint address, if any.
    pub fn bulk_in_endpoint(&self) -> Option<u8> {
        self.endpoint_in
    }

    /// The auto-discovered bulk OUT endpoint address, if any.
    pub fn bulk_out_endpoint(&self) -> Option<u8> {
        self.endpoint_out
    }
}

/// Scan the first alternate setting of `interface` for bulk endpoint addresses.
fn discover_bulk_endpoints(interface: &Interface) -> (Option<u8>, Option<u8>) {
    let mut ep_in: Option<u8> = None;
    let mut ep_out: Option<u8> = None;

    // Only inspect the first alternate setting
    if let Some(alt) = interface.descriptors().next() {
        for ep in alt.endpoints() {
            use nusb::transfer::EndpointType;
            if ep.transfer_type() == EndpointType::Bulk {
                if ep.direction() == nusb::transfer::Direction::In {
                    ep_in.get_or_insert(ep.address());
                } else {
                    ep_out.get_or_insert(ep.address());
                }
            }
        }
    }

    (ep_in, ep_out)
}
