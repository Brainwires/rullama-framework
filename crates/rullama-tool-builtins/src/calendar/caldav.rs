//! CalDAV protocol client for calendar CRUD operations.

use anyhow::{Context, Result};
use reqwest::Client;

use super::types::{CalendarEvent, CalendarInfo};

/// CalDAV client for interacting with CalDAV-compliant servers.
pub struct CalDavClient {
    client: Client,
    base_url: String,
    username: String,
    password: String,
}

impl CalDavClient {
    /// Create a new CalDAV client.
    pub fn new(base_url: &str, username: &str, password: &str) -> Self {
        Self {
            client: Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            username: username.to_string(),
            password: password.to_string(),
        }
    }

    /// Discover available calendars via PROPFIND.
    pub async fn list_calendars(&self) -> Result<Vec<CalendarInfo>> {
        let url = format!("{}/", self.base_url);

        let body = r#"<?xml version="1.0" encoding="utf-8" ?>
<D:propfind xmlns:D="DAV:" xmlns:C="urn:ietf:params:xml:ns:caldav">
  <D:prop>
    <D:displayname/>
    <D:resourcetype/>
    <C:calendar-color/>
  </D:prop>
</D:propfind>"#;

        let resp = self
            .client
            .request(reqwest::Method::from_bytes(b"PROPFIND").unwrap(), &url)
            .header("Depth", "1")
            .header("Content-Type", "application/xml")
            .basic_auth(&self.username, Some(&self.password))
            .body(body)
            .send()
            .await
            .context("CalDAV PROPFIND failed")?;

        let text = resp.text().await?;
        // Simple XML parsing — a production implementation would use a proper XML parser.
        let calendars = Self::parse_propfind_calendars(&text);
        Ok(calendars)
    }

    /// List events from a calendar within a time range via REPORT.
    pub async fn list_events(
        &self,
        calendar_path: &str,
        time_min: Option<&str>,
        time_max: Option<&str>,
    ) -> Result<Vec<CalendarEvent>> {
        let url = format!(
            "{}/{}",
            self.base_url,
            calendar_path.trim_start_matches('/')
        );

        let time_range = match (time_min, time_max) {
            (Some(min), Some(max)) => {
                format!(r#"<C:time-range start="{}" end="{}"/>"#, min, max)
            }
            (Some(min), None) => format!(r#"<C:time-range start="{}"/>"#, min),
            (None, Some(max)) => format!(r#"<C:time-range end="{}"/>"#, max),
            (None, None) => String::new(),
        };

        let body = format!(
            r#"<?xml version="1.0" encoding="utf-8" ?>
<C:calendar-query xmlns:D="DAV:" xmlns:C="urn:ietf:params:xml:ns:caldav">
  <D:prop>
    <D:getetag/>
    <C:calendar-data/>
  </D:prop>
  <C:filter>
    <C:comp-filter name="VCALENDAR">
      <C:comp-filter name="VEVENT">
        {}
      </C:comp-filter>
    </C:comp-filter>
  </C:filter>
</C:calendar-query>"#,
            time_range
        );

        let resp = self
            .client
            .request(reqwest::Method::from_bytes(b"REPORT").unwrap(), &url)
            .header("Depth", "1")
            .header("Content-Type", "application/xml")
            .basic_auth(&self.username, Some(&self.password))
            .body(body)
            .send()
            .await
            .context("CalDAV REPORT failed")?;

        let text = resp.text().await?;
        let events = Self::parse_ical_events(&text);
        Ok(events)
    }

    /// Create a new event via PUT with iCalendar format.
    pub async fn create_event(&self, calendar_path: &str, event: &CalendarEvent) -> Result<()> {
        let event_path = format!(
            "{}/{}/{}.ics",
            self.base_url,
            calendar_path.trim_start_matches('/'),
            &event.id
        );

        let ical = Self::event_to_ical(event);

        let resp = self
            .client
            .put(&event_path)
            .header("Content-Type", "text/calendar; charset=utf-8")
            .basic_auth(&self.username, Some(&self.password))
            .body(ical)
            .send()
            .await
            .context("CalDAV PUT (create) failed")?;

        let status = resp.status();
        if status.is_success() || status.as_u16() == 201 {
            Ok(())
        } else {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("CalDAV create failed ({}): {}", status, body)
        }
    }

    /// Update an existing event via PUT.
    pub async fn update_event(&self, calendar_path: &str, event: &CalendarEvent) -> Result<()> {
        // CalDAV uses the same PUT method for create and update
        self.create_event(calendar_path, event).await
    }

    /// Delete an event via DELETE.
    pub async fn delete_event(&self, calendar_path: &str, event_id: &str) -> Result<()> {
        let event_path = format!(
            "{}/{}/{}.ics",
            self.base_url,
            calendar_path.trim_start_matches('/'),
            event_id
        );

        let resp = self
            .client
            .delete(&event_path)
            .basic_auth(&self.username, Some(&self.password))
            .send()
            .await
            .context("CalDAV DELETE failed")?;

        let status = resp.status();
        if status.is_success() || status.as_u16() == 204 {
            Ok(())
        } else {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("CalDAV delete failed ({}): {}", status, body)
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Convert a `CalendarEvent` to iCalendar format string.
    fn event_to_ical(event: &CalendarEvent) -> String {
        let mut lines = vec![
            "BEGIN:VCALENDAR".to_string(),
            "VERSION:2.0".to_string(),
            "PRODID:-//Brainwires//CalDAV Client//EN".to_string(),
            "BEGIN:VEVENT".to_string(),
            format!("UID:{}", event.id),
            format!("SUMMARY:{}", event.title),
        ];

        if event.all_day {
            // All-day events use DATE values
            let start_date = event.start.split('T').next().unwrap_or(&event.start);
            let end_date = event.end.split('T').next().unwrap_or(&event.end);
            lines.push(format!(
                "DTSTART;VALUE=DATE:{}",
                start_date.replace('-', "")
            ));
            lines.push(format!("DTEND;VALUE=DATE:{}", end_date.replace('-', "")));
        } else {
            lines.push(format!("DTSTART:{}", Self::rfc3339_to_ical(&event.start)));
            lines.push(format!("DTEND:{}", Self::rfc3339_to_ical(&event.end)));
        }

        if let Some(ref desc) = event.description {
            lines.push(format!("DESCRIPTION:{}", desc));
        }
        if let Some(ref loc) = event.location {
            lines.push(format!("LOCATION:{}", loc));
        }
        for attendee in &event.attendees {
            let name_param = attendee
                .name
                .as_ref()
                .map(|n| format!(";CN={}", n))
                .unwrap_or_default();
            lines.push(format!("ATTENDEE{}:mailto:{}", name_param, attendee.email));
        }

        lines.push("END:VEVENT".to_string());
        lines.push("END:VCALENDAR".to_string());

        lines.join("\r\n")
    }

    /// Convert RFC-3339 datetime to iCalendar datetime format.
    fn rfc3339_to_ical(dt: &str) -> String {
        // "2025-06-01T10:00:00Z" -> "20250601T100000Z"
        dt.replace(['-', ':'], "")
    }

    /// Parse PROPFIND response to extract calendar info (simplified).
    fn parse_propfind_calendars(xml: &str) -> Vec<CalendarInfo> {
        // Simplified parser — looks for displayname elements.
        // A production implementation would use quick-xml or roxmltree.
        let mut calendars = Vec::new();
        let mut idx = 0u32;
        for line in xml.lines() {
            let trimmed = line.trim();
            if trimmed.contains("<displayname>") || trimmed.contains("<D:displayname>") {
                let name = trimmed
                    .replace("<displayname>", "")
                    .replace("</displayname>", "")
                    .replace("<D:displayname>", "")
                    .replace("</D:displayname>", "")
                    .trim()
                    .to_string();
                if !name.is_empty() {
                    calendars.push(CalendarInfo {
                        id: format!("calendar-{}", idx),
                        name,
                        color: None,
                        primary: idx == 0,
                    });
                    idx += 1;
                }
            }
        }
        calendars
    }

    /// Parse CalDAV REPORT response containing iCalendar data (simplified).
    fn parse_ical_events(xml: &str) -> Vec<CalendarEvent> {
        // Extract VCALENDAR data from XML CDATA and parse iCalendar.
        // This is a simplified implementation; production code would use
        // proper XML + iCalendar parsing.
        let mut events = Vec::new();

        // Find all VEVENT blocks in the response
        let mut remaining = xml;
        while let Some(start_pos) = remaining.find("BEGIN:VEVENT") {
            if let Some(end_pos) = remaining[start_pos..].find("END:VEVENT") {
                let vevent = &remaining[start_pos..start_pos + end_pos + "END:VEVENT".len()];
                if let Some(event) = Self::parse_single_vevent(vevent) {
                    events.push(event);
                }
                remaining = &remaining[start_pos + end_pos + "END:VEVENT".len()..];
            } else {
                break;
            }
        }

        events
    }

    fn parse_single_vevent(vevent: &str) -> Option<CalendarEvent> {
        let get_prop = |name: &str| -> Option<String> {
            for line in vevent.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with(name) {
                    let value = if let Some(idx) = trimmed.find(':') {
                        &trimmed[idx + 1..]
                    } else {
                        continue;
                    };
                    return Some(value.to_string());
                }
            }
            None
        };

        let uid = get_prop("UID")?;
        let summary = get_prop("SUMMARY").unwrap_or_default();
        let start = get_prop("DTSTART").unwrap_or_default();
        let end = get_prop("DTEND").unwrap_or(start.clone());
        let description = get_prop("DESCRIPTION");
        let location = get_prop("LOCATION");

        let all_day = vevent.contains("VALUE=DATE");

        Some(CalendarEvent {
            id: uid,
            title: summary,
            description,
            location,
            start,
            end,
            all_day,
            attendees: vec![],
            recurrence: None,
            reminders: vec![],
            calendar_id: None,
        })
    }
}
