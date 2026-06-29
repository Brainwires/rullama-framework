//! Google Calendar API v3 client.

use anyhow::{Context, Result};
use reqwest::Client;
use serde::Deserialize;
use serde_json::Value;

use super::types::{
    Attendee, AttendeeStatus, BusyStatus, CalendarEvent, CalendarInfo, FreeBusySlot,
};

const CALENDAR_API_BASE: &str = "https://www.googleapis.com/calendar/v3";
const TOKEN_URL: &str = "https://oauth2.googleapis.com/token";

/// Google Calendar API client using OAuth2 credentials.
pub struct GoogleCalendarClient {
    client: Client,
    access_token: String,
    client_id: String,
    client_secret: String,
    refresh_token: String,
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
    #[allow(dead_code)]
    expires_in: u64,
}

impl GoogleCalendarClient {
    /// Create a new client and obtain an access token via refresh.
    pub async fn new(client_id: &str, client_secret: &str, refresh_token: &str) -> Result<Self> {
        let client = Client::new();
        let access_token =
            Self::refresh_access_token(&client, client_id, client_secret, refresh_token).await?;

        Ok(Self {
            client,
            access_token,
            client_id: client_id.to_string(),
            client_secret: client_secret.to_string(),
            refresh_token: refresh_token.to_string(),
        })
    }

    /// Refresh the OAuth2 access token.
    async fn refresh_access_token(
        client: &Client,
        client_id: &str,
        client_secret: &str,
        refresh_token: &str,
    ) -> Result<String> {
        let resp: TokenResponse = client
            .post(TOKEN_URL)
            .form(&[
                ("client_id", client_id),
                ("client_secret", client_secret),
                ("refresh_token", refresh_token),
                ("grant_type", "refresh_token"),
            ])
            .send()
            .await
            .context("Failed to refresh Google OAuth2 token")?
            .json()
            .await
            .context("Failed to parse token response")?;

        Ok(resp.access_token)
    }

    /// Re-authenticate if the current token has expired.
    pub async fn ensure_token(&mut self) -> Result<()> {
        // Simple strategy: always refresh. A production implementation would
        // check expiry time and only refresh when needed.
        self.access_token = Self::refresh_access_token(
            &self.client,
            &self.client_id,
            &self.client_secret,
            &self.refresh_token,
        )
        .await?;
        Ok(())
    }

    /// List events from a calendar within a time range.
    pub async fn list_events(
        &self,
        calendar_id: &str,
        time_min: Option<&str>,
        time_max: Option<&str>,
        max_results: u32,
    ) -> Result<Vec<CalendarEvent>> {
        let mut url = format!(
            "{}/calendars/{}/events?maxResults={}&singleEvents=true&orderBy=startTime",
            CALENDAR_API_BASE,
            urlencoding::encode(calendar_id),
            max_results,
        );
        if let Some(min) = time_min {
            url.push_str(&format!("&timeMin={}", urlencoding::encode(min)));
        }
        if let Some(max) = time_max {
            url.push_str(&format!("&timeMax={}", urlencoding::encode(max)));
        }

        let resp: Value = self
            .client
            .get(&url)
            .bearer_auth(&self.access_token)
            .send()
            .await
            .context("Failed to list Google Calendar events")?
            .json()
            .await?;

        let items = resp
            .get("items")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        let events = items.iter().filter_map(Self::parse_event).collect();
        Ok(events)
    }

    /// Create a new calendar event.
    pub async fn create_event(
        &self,
        calendar_id: &str,
        event: &CalendarEvent,
    ) -> Result<CalendarEvent> {
        let url = format!(
            "{}/calendars/{}/events",
            CALENDAR_API_BASE,
            urlencoding::encode(calendar_id),
        );

        let body = Self::event_to_google_json(event);

        let resp: Value = self
            .client
            .post(&url)
            .bearer_auth(&self.access_token)
            .json(&body)
            .send()
            .await
            .context("Failed to create Google Calendar event")?
            .json()
            .await?;

        Self::parse_event(&resp)
            .ok_or_else(|| anyhow::anyhow!("Failed to parse created event response"))
    }

    /// Update an existing calendar event.
    pub async fn update_event(
        &self,
        calendar_id: &str,
        event_id: &str,
        event: &CalendarEvent,
    ) -> Result<CalendarEvent> {
        let url = format!(
            "{}/calendars/{}/events/{}",
            CALENDAR_API_BASE,
            urlencoding::encode(calendar_id),
            urlencoding::encode(event_id),
        );

        let body = Self::event_to_google_json(event);

        let resp: Value = self
            .client
            .put(&url)
            .bearer_auth(&self.access_token)
            .json(&body)
            .send()
            .await
            .context("Failed to update Google Calendar event")?
            .json()
            .await?;

        Self::parse_event(&resp)
            .ok_or_else(|| anyhow::anyhow!("Failed to parse updated event response"))
    }

    /// Delete a calendar event.
    pub async fn delete_event(&self, calendar_id: &str, event_id: &str) -> Result<()> {
        let url = format!(
            "{}/calendars/{}/events/{}",
            CALENDAR_API_BASE,
            urlencoding::encode(calendar_id),
            urlencoding::encode(event_id),
        );

        let resp = self
            .client
            .delete(&url)
            .bearer_auth(&self.access_token)
            .send()
            .await
            .context("Failed to delete Google Calendar event")?;

        if resp.status().is_success() || resp.status().as_u16() == 204 {
            Ok(())
        } else {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Delete failed: {}", body)
        }
    }

    /// Query free/busy information.
    pub async fn free_busy(
        &self,
        calendar_ids: &[String],
        time_min: &str,
        time_max: &str,
    ) -> Result<Vec<FreeBusySlot>> {
        let url = format!("{}/freeBusy", CALENDAR_API_BASE);

        let items: Vec<Value> = calendar_ids
            .iter()
            .map(|id| serde_json::json!({"id": id}))
            .collect();

        let body = serde_json::json!({
            "timeMin": time_min,
            "timeMax": time_max,
            "items": items,
        });

        let resp: Value = self
            .client
            .post(&url)
            .bearer_auth(&self.access_token)
            .json(&body)
            .send()
            .await
            .context("Failed to query free/busy")?
            .json()
            .await?;

        let mut slots = Vec::new();
        if let Some(calendars) = resp.get("calendars").and_then(|c| c.as_object()) {
            for (_cal_id, cal_data) in calendars {
                if let Some(busy_arr) = cal_data.get("busy").and_then(|b| b.as_array()) {
                    for slot in busy_arr {
                        let start = slot
                            .get("start")
                            .and_then(|s| s.as_str())
                            .unwrap_or_default();
                        let end = slot.get("end").and_then(|e| e.as_str()).unwrap_or_default();
                        slots.push(FreeBusySlot {
                            start: start.to_string(),
                            end: end.to_string(),
                            status: BusyStatus::Busy,
                        });
                    }
                }
            }
        }
        Ok(slots)
    }

    /// List available calendars.
    pub async fn list_calendars(&self) -> Result<Vec<CalendarInfo>> {
        let url = format!("{}/users/me/calendarList", CALENDAR_API_BASE);

        let resp: Value = self
            .client
            .get(&url)
            .bearer_auth(&self.access_token)
            .send()
            .await
            .context("Failed to list calendars")?
            .json()
            .await?;

        let items = resp
            .get("items")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        let calendars = items
            .iter()
            .map(|item| CalendarInfo {
                id: item
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string(),
                name: item
                    .get("summary")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default()
                    .to_string(),
                color: item
                    .get("backgroundColor")
                    .and_then(|v| v.as_str())
                    .map(String::from),
                primary: item
                    .get("primary")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false),
            })
            .collect();

        Ok(calendars)
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn parse_event(item: &Value) -> Option<CalendarEvent> {
        let id = item.get("id")?.as_str()?.to_string();
        let title = item
            .get("summary")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();

        let (start, all_day) = if let Some(s) = item.get("start") {
            if let Some(dt) = s.get("dateTime").and_then(|v| v.as_str()) {
                (dt.to_string(), false)
            } else if let Some(d) = s.get("date").and_then(|v| v.as_str()) {
                (d.to_string(), true)
            } else {
                return None;
            }
        } else {
            return None;
        };

        let end = if let Some(e) = item.get("end") {
            if let Some(dt) = e.get("dateTime").and_then(|v| v.as_str()) {
                dt.to_string()
            } else if let Some(d) = e.get("date").and_then(|v| v.as_str()) {
                d.to_string()
            } else {
                start.clone()
            }
        } else {
            start.clone()
        };

        let attendees = item
            .get("attendees")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|a| {
                        let email = a.get("email")?.as_str()?.to_string();
                        let name = a
                            .get("displayName")
                            .and_then(|v| v.as_str())
                            .map(String::from);
                        let status = match a
                            .get("responseStatus")
                            .and_then(|v| v.as_str())
                            .unwrap_or("needsAction")
                        {
                            "accepted" => AttendeeStatus::Accepted,
                            "declined" => AttendeeStatus::Declined,
                            "tentative" => AttendeeStatus::Tentative,
                            _ => AttendeeStatus::NeedsAction,
                        };
                        Some(Attendee {
                            email,
                            name,
                            status,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        Some(CalendarEvent {
            id,
            title,
            description: item
                .get("description")
                .and_then(|v| v.as_str())
                .map(String::from),
            location: item
                .get("location")
                .and_then(|v| v.as_str())
                .map(String::from),
            start,
            end,
            all_day,
            attendees,
            recurrence: None,
            reminders: vec![],
            calendar_id: None,
        })
    }

    fn event_to_google_json(event: &CalendarEvent) -> Value {
        let start = if event.all_day {
            serde_json::json!({"date": event.start})
        } else {
            serde_json::json!({"dateTime": event.start})
        };
        let end = if event.all_day {
            serde_json::json!({"date": event.end})
        } else {
            serde_json::json!({"dateTime": event.end})
        };

        let mut body = serde_json::json!({
            "summary": event.title,
            "start": start,
            "end": end,
        });

        if let Some(ref desc) = event.description {
            body["description"] = serde_json::json!(desc);
        }
        if let Some(ref loc) = event.location {
            body["location"] = serde_json::json!(loc);
        }
        if !event.attendees.is_empty() {
            let attendees: Vec<Value> = event
                .attendees
                .iter()
                .map(|a| {
                    let mut obj = serde_json::json!({"email": a.email});
                    if let Some(ref name) = a.name {
                        obj["displayName"] = serde_json::json!(name);
                    }
                    obj
                })
                .collect();
            body["attendees"] = serde_json::json!(attendees);
        }

        body
    }
}

/// Encode a string for use in URL paths (re-exported for convenience).
mod urlencoding {
    pub fn encode(s: &str) -> String {
        url_encode(s)
    }

    fn url_encode(s: &str) -> String {
        let mut result = String::with_capacity(s.len());
        for c in s.chars() {
            match c {
                'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => result.push(c),
                _ => {
                    for b in c.to_string().as_bytes() {
                        result.push_str(&format!("%{:02X}", b));
                    }
                }
            }
        }
        result
    }
}
