//! Shared calendar types used by Google Calendar and CalDAV clients.

use serde::{Deserialize, Serialize};

/// A calendar event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalendarEvent {
    /// Unique event identifier.
    pub id: String,
    /// Event title / summary.
    pub title: String,
    /// Event description.
    #[serde(default)]
    pub description: Option<String>,
    /// Event location.
    #[serde(default)]
    pub location: Option<String>,
    /// Start date-time (RFC-3339).
    pub start: String,
    /// End date-time (RFC-3339).
    pub end: String,
    /// Whether this is an all-day event.
    #[serde(default)]
    pub all_day: bool,
    /// Event attendees.
    #[serde(default)]
    pub attendees: Vec<Attendee>,
    /// Recurrence rule, if any.
    #[serde(default)]
    pub recurrence: Option<Recurrence>,
    /// Reminder minutes before the event.
    #[serde(default)]
    pub reminders: Vec<u32>,
    /// Calendar this event belongs to.
    #[serde(default)]
    pub calendar_id: Option<String>,
}

/// An event attendee.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attendee {
    /// Attendee email address.
    pub email: String,
    /// Attendee display name.
    #[serde(default)]
    pub name: Option<String>,
    /// Attendance response status.
    #[serde(default)]
    pub status: AttendeeStatus,
}

/// Attendance response status.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum AttendeeStatus {
    /// Attendee accepted the invitation.
    Accepted,
    /// Attendee declined the invitation.
    Declined,
    /// Attendee tentatively accepted.
    Tentative,
    /// No response yet.
    #[default]
    NeedsAction,
}

/// Recurrence rule for repeating events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recurrence {
    /// Recurrence frequency.
    pub frequency: RecurrenceFreq,
    /// Interval between recurrences (e.g. every 2 weeks).
    #[serde(default = "default_interval")]
    pub interval: u32,
    /// Maximum number of occurrences.
    #[serde(default)]
    pub count: Option<u32>,
    /// End date for recurrence (RFC-3339).
    #[serde(default)]
    pub until: Option<String>,
}

fn default_interval() -> u32 {
    1
}

/// Recurrence frequency.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum RecurrenceFreq {
    /// Every day.
    Daily,
    /// Every week.
    Weekly,
    /// Every month.
    Monthly,
    /// Every year.
    Yearly,
}

/// Information about a calendar.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalendarInfo {
    /// Calendar identifier.
    pub id: String,
    /// Calendar display name.
    pub name: String,
    /// Calendar color (hex string).
    #[serde(default)]
    pub color: Option<String>,
    /// Whether this is the user's primary calendar.
    #[serde(default)]
    pub primary: bool,
}

/// A free/busy time slot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FreeBusySlot {
    /// Slot start (RFC-3339).
    pub start: String,
    /// Slot end (RFC-3339).
    pub end: String,
    /// Busy status.
    pub status: BusyStatus,
}

/// Free/busy status.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum BusyStatus {
    /// Time slot is free.
    Free,
    /// Time slot is busy.
    Busy,
    /// Time slot is tentatively busy.
    Tentative,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calendar_event_serde_roundtrip() {
        let event = CalendarEvent {
            id: "evt-1".to_string(),
            title: "Team Meeting".to_string(),
            description: Some("Weekly sync".to_string()),
            location: Some("Room 101".to_string()),
            start: "2025-06-01T10:00:00Z".to_string(),
            end: "2025-06-01T11:00:00Z".to_string(),
            all_day: false,
            attendees: vec![Attendee {
                email: "bob@example.com".to_string(),
                name: Some("Bob".to_string()),
                status: AttendeeStatus::Accepted,
            }],
            recurrence: Some(Recurrence {
                frequency: RecurrenceFreq::Weekly,
                interval: 1,
                count: Some(10),
                until: None,
            }),
            reminders: vec![15, 5],
            calendar_id: Some("primary".to_string()),
        };
        let json = serde_json::to_string(&event).unwrap();
        let deserialized: CalendarEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, "evt-1");
        assert_eq!(deserialized.title, "Team Meeting");
        assert_eq!(deserialized.attendees.len(), 1);
        assert_eq!(deserialized.attendees[0].status, AttendeeStatus::Accepted);
    }

    #[test]
    fn test_attendee_status_default() {
        let status = AttendeeStatus::default();
        assert_eq!(status, AttendeeStatus::NeedsAction);
    }

    #[test]
    fn test_recurrence_serde_roundtrip() {
        let rec = Recurrence {
            frequency: RecurrenceFreq::Monthly,
            interval: 2,
            count: None,
            until: Some("2026-12-31T00:00:00Z".to_string()),
        };
        let json = serde_json::to_string(&rec).unwrap();
        let deserialized: Recurrence = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.frequency, RecurrenceFreq::Monthly);
        assert_eq!(deserialized.interval, 2);
    }

    #[test]
    fn test_calendar_info_serde_roundtrip() {
        let info = CalendarInfo {
            id: "cal-1".to_string(),
            name: "Work".to_string(),
            color: Some("#4285f4".to_string()),
            primary: true,
        };
        let json = serde_json::to_string(&info).unwrap();
        let deserialized: CalendarInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.name, "Work");
        assert!(deserialized.primary);
    }

    #[test]
    fn test_free_busy_slot_serde_roundtrip() {
        let slot = FreeBusySlot {
            start: "2025-06-01T09:00:00Z".to_string(),
            end: "2025-06-01T10:00:00Z".to_string(),
            status: BusyStatus::Busy,
        };
        let json = serde_json::to_string(&slot).unwrap();
        let deserialized: FreeBusySlot = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.status, BusyStatus::Busy);
    }
}
