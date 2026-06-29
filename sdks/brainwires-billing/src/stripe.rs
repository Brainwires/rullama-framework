//! Stripe REST client for agent billing workflows.
//!
//! Covers metered usage reporting, payment link creation, and customer balance
//! queries. Use a restricted API key (`rk_…`) — never a secret key.

use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::BillingImplError;

const STRIPE_API_BASE: &str = "https://api.stripe.com/v1";

pub struct StripeClient {
    api_key: String,
    http: Client,
}

impl StripeClient {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            http: Client::new(),
        }
    }

    /// Report metered usage against a Stripe billing meter.
    pub async fn report_usage(
        &self,
        event_name: &str,
        stripe_customer_id: &str,
        quantity: u64,
    ) -> Result<MeterEventResponse, BillingImplError> {
        let now = chrono::Utc::now().timestamp();
        let params = [
            ("event_name", event_name.to_string()),
            (
                "payload[stripe_customer_id]",
                stripe_customer_id.to_string(),
            ),
            ("payload[value]", quantity.to_string()),
            ("timestamp", now.to_string()),
        ];

        let resp = self
            .http
            .post(format!("{STRIPE_API_BASE}/billing/meter_events"))
            .basic_auth(&self.api_key, Some(""))
            .form(&params)
            .send()
            .await
            .map_err(|e| BillingImplError::Stripe(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(BillingImplError::Stripe(format!(
                "Stripe meter_events returned {status}: {body}"
            )));
        }

        resp.json()
            .await
            .map_err(|e| BillingImplError::Stripe(e.to_string()))
    }

    /// Create a Stripe payment link for a one-time purchase.
    pub async fn create_payment_link(
        &self,
        price_id: &str,
        quantity: u64,
    ) -> Result<PaymentLinkResponse, BillingImplError> {
        let params = [
            ("line_items[0][price]", price_id.to_string()),
            ("line_items[0][quantity]", quantity.to_string()),
        ];

        let resp = self
            .http
            .post(format!("{STRIPE_API_BASE}/payment_links"))
            .basic_auth(&self.api_key, Some(""))
            .form(&params)
            .send()
            .await
            .map_err(|e| BillingImplError::Stripe(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(BillingImplError::Stripe(format!(
                "Stripe payment_links returned {status}: {body}"
            )));
        }

        resp.json()
            .await
            .map_err(|e| BillingImplError::Stripe(e.to_string()))
    }

    /// Retrieve a customer's current balance in cents (negative = credit).
    pub async fn customer_balance(&self, customer_id: &str) -> Result<i64, BillingImplError> {
        let resp = self
            .http
            .get(format!("{STRIPE_API_BASE}/customers/{customer_id}"))
            .basic_auth(&self.api_key, Some(""))
            .send()
            .await
            .map_err(|e| BillingImplError::Stripe(e.to_string()))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(BillingImplError::Stripe(format!(
                "Stripe customers/{customer_id} returned {status}: {body}"
            )));
        }

        #[derive(Deserialize)]
        struct CustomerResp {
            balance: i64,
        }
        let c: CustomerResp = resp
            .json()
            .await
            .map_err(|e| BillingImplError::Stripe(e.to_string()))?;
        Ok(c.balance)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeterEventResponse {
    pub identifier: String,
    pub event_name: String,
    pub timestamp: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaymentLinkResponse {
    pub id: String,
    pub url: String,
    pub active: bool,
}
