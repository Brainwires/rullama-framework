//! Calendar tools: list, create, update, delete events, and find free time.

pub mod caldav;
pub mod google;
pub mod types;

use std::collections::HashMap;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use brainwires_core::{Tool, ToolContext, ToolInputSchema, ToolResult};

use self::caldav::CalDavClient;
use self::google::GoogleCalendarClient;
use self::types::CalendarEvent;

/// Calendar provider configuration variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CalendarProvider {
    /// Google Calendar via OAuth2.
    GoogleCalendar {
        /// OAuth2 client ID.
        client_id: String,
        /// OAuth2 client secret.
        client_secret: String,
        /// OAuth2 refresh token.
        refresh_token: String,
    },
    /// CalDAV-compliant server.
    CalDav {
        /// CalDAV server URL.
        url: String,
        /// Authentication username.
        username: String,
        /// Authentication password.
        password: String,
    },
}

/// Configuration for the calendar tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalendarConfig {
    /// Calendar provider settings.
    pub provider: CalendarProvider,
    /// Default calendar ID to operate on.
    #[serde(default = "default_calendar_id")]
    pub default_calendar_id: String,
}

fn default_calendar_id() -> String {
    "primary".to_string()
}

/// Calendar tool implementation providing event CRUD and free/busy queries.
pub struct CalendarTool;

impl CalendarTool {
    /// Return tool definitions for calendar operations.
    pub fn get_tools() -> Vec<Tool> {
        vec![
            Self::list_events_tool(),
            Self::create_event_tool(),
            Self::update_event_tool(),
            Self::delete_event_tool(),
            Self::find_free_time_tool(),
        ]
    }

    fn list_events_tool() -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "calendar_id".to_string(),
            json!({"type": "string", "description": "Calendar ID (default: primary)"}),
        );
        properties.insert(
            "time_min".to_string(),
            json!({"type": "string", "description": "Start of time range (RFC-3339)"}),
        );
        properties.insert(
            "time_max".to_string(),
            json!({"type": "string", "description": "End of time range (RFC-3339)"}),
        );
        properties.insert(
            "max_results".to_string(),
            json!({"type": "integer", "description": "Maximum number of events (default: 25)"}),
        );
        Tool {
            name: "calendar_list_events".to_string(),
            description: "List calendar events within a time range.".to_string(),
            input_schema: ToolInputSchema::object(properties, vec![]),
            requires_approval: false,
            ..Default::default()
        }
    }

    fn create_event_tool() -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "title".to_string(),
            json!({"type": "string", "description": "Event title"}),
        );
        properties.insert(
            "start".to_string(),
            json!({"type": "string", "description": "Start date-time (RFC-3339)"}),
        );
        properties.insert(
            "end".to_string(),
            json!({"type": "string", "description": "End date-time (RFC-3339)"}),
        );
        properties.insert(
            "description".to_string(),
            json!({"type": "string", "description": "Event description"}),
        );
        properties.insert(
            "location".to_string(),
            json!({"type": "string", "description": "Event location"}),
        );
        properties.insert(
            "all_day".to_string(),
            json!({"type": "boolean", "description": "Whether this is an all-day event"}),
        );
        properties.insert(
            "attendees".to_string(),
            json!({"type": "array", "items": {"type": "string"}, "description": "Attendee email addresses"}),
        );
        properties.insert(
            "calendar_id".to_string(),
            json!({"type": "string", "description": "Calendar ID (default: primary)"}),
        );
        Tool {
            name: "calendar_create_event".to_string(),
            description: "Create a new calendar event.".to_string(),
            input_schema: ToolInputSchema::object(
                properties,
                vec!["title".to_string(), "start".to_string(), "end".to_string()],
            ),
            requires_approval: true,
            ..Default::default()
        }
    }

    fn update_event_tool() -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "event_id".to_string(),
            json!({"type": "string", "description": "Event ID to update"}),
        );
        properties.insert(
            "title".to_string(),
            json!({"type": "string", "description": "New event title"}),
        );
        properties.insert(
            "start".to_string(),
            json!({"type": "string", "description": "New start date-time (RFC-3339)"}),
        );
        properties.insert(
            "end".to_string(),
            json!({"type": "string", "description": "New end date-time (RFC-3339)"}),
        );
        properties.insert(
            "description".to_string(),
            json!({"type": "string", "description": "New event description"}),
        );
        properties.insert(
            "location".to_string(),
            json!({"type": "string", "description": "New event location"}),
        );
        properties.insert(
            "calendar_id".to_string(),
            json!({"type": "string", "description": "Calendar ID (default: primary)"}),
        );
        Tool {
            name: "calendar_update_event".to_string(),
            description: "Update an existing calendar event.".to_string(),
            input_schema: ToolInputSchema::object(properties, vec!["event_id".to_string()]),
            requires_approval: true,
            ..Default::default()
        }
    }

    fn delete_event_tool() -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "event_id".to_string(),
            json!({"type": "string", "description": "Event ID to delete"}),
        );
        properties.insert(
            "calendar_id".to_string(),
            json!({"type": "string", "description": "Calendar ID (default: primary)"}),
        );
        Tool {
            name: "calendar_delete_event".to_string(),
            description: "Delete a calendar event.".to_string(),
            input_schema: ToolInputSchema::object(properties, vec!["event_id".to_string()]),
            requires_approval: true,
            ..Default::default()
        }
    }

    fn find_free_time_tool() -> Tool {
        let mut properties = HashMap::new();
        properties.insert(
            "time_min".to_string(),
            json!({"type": "string", "description": "Start of search range (RFC-3339)"}),
        );
        properties.insert(
            "time_max".to_string(),
            json!({"type": "string", "description": "End of search range (RFC-3339)"}),
        );
        properties.insert(
            "calendar_ids".to_string(),
            json!({"type": "array", "items": {"type": "string"}, "description": "Calendar IDs to check (default: primary)"}),
        );
        Tool {
            name: "calendar_find_free_time".to_string(),
            description: "Find free time slots across one or more calendars.".to_string(),
            input_schema: ToolInputSchema::object(
                properties,
                vec!["time_min".to_string(), "time_max".to_string()],
            ),
            requires_approval: false,
            ..Default::default()
        }
    }

    /// Execute a calendar tool by name.
    #[tracing::instrument(name = "tool.execute", skip(input, context), fields(tool_name))]
    pub async fn execute(
        tool_use_id: &str,
        tool_name: &str,
        input: &Value,
        context: &ToolContext,
    ) -> ToolResult {
        let result = match tool_name {
            "calendar_list_events" => Self::handle_list_events(input, context).await,
            "calendar_create_event" => Self::handle_create_event(input, context).await,
            "calendar_update_event" => Self::handle_update_event(input, context).await,
            "calendar_delete_event" => Self::handle_delete_event(input, context).await,
            "calendar_find_free_time" => Self::handle_find_free_time(input, context).await,
            _ => Err(anyhow::anyhow!("Unknown calendar tool: {}", tool_name)),
        };
        match result {
            Ok(output) => ToolResult::success(tool_use_id.to_string(), output),
            Err(e) => ToolResult::error(
                tool_use_id.to_string(),
                format!("Calendar operation failed: {}", e),
            ),
        }
    }

    // ── Handler implementations ─────────────────────────────────────────────

    async fn handle_list_events(input: &Value, context: &ToolContext) -> Result<String> {
        let config = Self::get_config(context)?;
        let calendar_id = input
            .get("calendar_id")
            .and_then(|v| v.as_str())
            .unwrap_or(&config.default_calendar_id);
        let time_min = input.get("time_min").and_then(|v| v.as_str());
        let time_max = input.get("time_max").and_then(|v| v.as_str());
        let max_results = input
            .get("max_results")
            .and_then(|v| v.as_u64())
            .unwrap_or(25) as u32;

        match &config.provider {
            CalendarProvider::GoogleCalendar {
                client_id,
                client_secret,
                refresh_token,
            } => {
                let client =
                    GoogleCalendarClient::new(client_id, client_secret, refresh_token).await?;
                let events = client
                    .list_events(calendar_id, time_min, time_max, max_results)
                    .await?;
                Ok(serde_json::to_string_pretty(&events)?)
            }
            CalendarProvider::CalDav {
                url,
                username,
                password,
            } => {
                let client = CalDavClient::new(url, username, password);
                let events = client.list_events(calendar_id, time_min, time_max).await?;
                Ok(serde_json::to_string_pretty(&events)?)
            }
        }
    }

    async fn handle_create_event(input: &Value, context: &ToolContext) -> Result<String> {
        let config = Self::get_config(context)?;
        let calendar_id = input
            .get("calendar_id")
            .and_then(|v| v.as_str())
            .unwrap_or(&config.default_calendar_id);

        let event = Self::parse_event_input(input)?;

        match &config.provider {
            CalendarProvider::GoogleCalendar {
                client_id,
                client_secret,
                refresh_token,
            } => {
                let client =
                    GoogleCalendarClient::new(client_id, client_secret, refresh_token).await?;
                let created = client.create_event(calendar_id, &event).await?;
                Ok(serde_json::to_string_pretty(&created)?)
            }
            CalendarProvider::CalDav {
                url,
                username,
                password,
            } => {
                let client = CalDavClient::new(url, username, password);
                client.create_event(calendar_id, &event).await?;
                Ok(serde_json::to_string_pretty(&event)?)
            }
        }
    }

    async fn handle_update_event(input: &Value, context: &ToolContext) -> Result<String> {
        let config = Self::get_config(context)?;
        let calendar_id = input
            .get("calendar_id")
            .and_then(|v| v.as_str())
            .unwrap_or(&config.default_calendar_id);
        let event_id = input
            .get("event_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("'event_id' is required"))?;

        let mut event = Self::parse_event_input(input)?;
        event.id = event_id.to_string();

        match &config.provider {
            CalendarProvider::GoogleCalendar {
                client_id,
                client_secret,
                refresh_token,
            } => {
                let client =
                    GoogleCalendarClient::new(client_id, client_secret, refresh_token).await?;
                let updated = client.update_event(calendar_id, event_id, &event).await?;
                Ok(serde_json::to_string_pretty(&updated)?)
            }
            CalendarProvider::CalDav {
                url,
                username,
                password,
            } => {
                let client = CalDavClient::new(url, username, password);
                client.update_event(calendar_id, &event).await?;
                Ok(serde_json::to_string_pretty(&event)?)
            }
        }
    }

    async fn handle_delete_event(input: &Value, context: &ToolContext) -> Result<String> {
        let config = Self::get_config(context)?;
        let calendar_id = input
            .get("calendar_id")
            .and_then(|v| v.as_str())
            .unwrap_or(&config.default_calendar_id);
        let event_id = input
            .get("event_id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("'event_id' is required"))?;

        match &config.provider {
            CalendarProvider::GoogleCalendar {
                client_id,
                client_secret,
                refresh_token,
            } => {
                let client =
                    GoogleCalendarClient::new(client_id, client_secret, refresh_token).await?;
                client.delete_event(calendar_id, event_id).await?;
                Ok(format!("Event '{}' deleted successfully", event_id))
            }
            CalendarProvider::CalDav {
                url,
                username,
                password,
            } => {
                let client = CalDavClient::new(url, username, password);
                client.delete_event(calendar_id, event_id).await?;
                Ok(format!("Event '{}' deleted successfully", event_id))
            }
        }
    }

    async fn handle_find_free_time(input: &Value, context: &ToolContext) -> Result<String> {
        let config = Self::get_config(context)?;
        let time_min = input
            .get("time_min")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("'time_min' is required"))?;
        let time_max = input
            .get("time_max")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("'time_max' is required"))?;

        let calendar_ids: Vec<String> = input
            .get("calendar_ids")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_else(|| vec![config.default_calendar_id.clone()]);

        match &config.provider {
            CalendarProvider::GoogleCalendar {
                client_id,
                client_secret,
                refresh_token,
            } => {
                let client =
                    GoogleCalendarClient::new(client_id, client_secret, refresh_token).await?;
                let slots = client.free_busy(&calendar_ids, time_min, time_max).await?;
                Ok(serde_json::to_string_pretty(&slots)?)
            }
            CalendarProvider::CalDav { .. } => {
                anyhow::bail!("Free/busy queries are not yet supported for CalDAV")
            }
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn get_config(context: &ToolContext) -> Result<CalendarConfig> {
        let config_json = context.metadata.get("calendar_config").ok_or_else(|| {
            anyhow::anyhow!(
                "Calendar configuration not found. Set 'calendar_config' in ToolContext.metadata."
            )
        })?;
        let config: CalendarConfig = serde_json::from_str(config_json)?;
        Ok(config)
    }

    fn parse_event_input(input: &Value) -> Result<CalendarEvent> {
        let title = input
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("Untitled Event")
            .to_string();
        let start = input
            .get("start")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let end = input
            .get("end")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string();
        let all_day = input
            .get("all_day")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let attendees = input
            .get("attendees")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| {
                        v.as_str().map(|email| types::Attendee {
                            email: email.to_string(),
                            name: None,
                            status: types::AttendeeStatus::NeedsAction,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(CalendarEvent {
            id: uuid::Uuid::new_v4().to_string(),
            title,
            description: input
                .get("description")
                .and_then(|v| v.as_str())
                .map(String::from),
            location: input
                .get("location")
                .and_then(|v| v.as_str())
                .map(String::from),
            start,
            end,
            all_day,
            attendees,
            recurrence: None,
            reminders: vec![],
            calendar_id: input
                .get("calendar_id")
                .and_then(|v| v.as_str())
                .map(String::from),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_tools() {
        let tools = CalendarTool::get_tools();
        assert_eq!(tools.len(), 5);

        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"calendar_list_events"));
        assert!(names.contains(&"calendar_create_event"));
        assert!(names.contains(&"calendar_update_event"));
        assert!(names.contains(&"calendar_delete_event"));
        assert!(names.contains(&"calendar_find_free_time"));
    }

    #[test]
    fn test_create_event_requires_approval() {
        let tools = CalendarTool::get_tools();
        let create = tools
            .iter()
            .find(|t| t.name == "calendar_create_event")
            .unwrap();
        assert!(create.requires_approval);
    }

    #[test]
    fn test_create_event_required_fields() {
        let tools = CalendarTool::get_tools();
        let create = tools
            .iter()
            .find(|t| t.name == "calendar_create_event")
            .unwrap();
        let required = create.input_schema.required.as_ref().unwrap();
        assert!(required.contains(&"title".to_string()));
        assert!(required.contains(&"start".to_string()));
        assert!(required.contains(&"end".to_string()));
    }

    #[test]
    fn test_delete_event_requires_approval() {
        let tools = CalendarTool::get_tools();
        let delete = tools
            .iter()
            .find(|t| t.name == "calendar_delete_event")
            .unwrap();
        assert!(delete.requires_approval);
    }

    #[test]
    fn test_find_free_time_required_fields() {
        let tools = CalendarTool::get_tools();
        let fft = tools
            .iter()
            .find(|t| t.name == "calendar_find_free_time")
            .unwrap();
        let required = fft.input_schema.required.as_ref().unwrap();
        assert!(required.contains(&"time_min".to_string()));
        assert!(required.contains(&"time_max".to_string()));
    }

    #[test]
    fn test_calendar_config_serde_roundtrip() {
        let config = CalendarConfig {
            provider: CalendarProvider::GoogleCalendar {
                client_id: "id".to_string(),
                client_secret: "secret".to_string(),
                refresh_token: "token".to_string(),
            },
            default_calendar_id: "primary".to_string(),
        };
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: CalendarConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.default_calendar_id, "primary");
    }

    #[tokio::test]
    async fn test_execute_unknown_tool() {
        let context = ToolContext {
            working_directory: ".".to_string(),
            ..Default::default()
        };
        let input = json!({});
        let result = CalendarTool::execute("1", "unknown_calendar_tool", &input, &context).await;
        assert!(result.is_error);
    }
}
