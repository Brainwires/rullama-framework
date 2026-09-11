/**
 * Shared calendar types used by Google Calendar and CalDAV clients.
 *
 * Equivalent to Rust's `rullama_tools::calendar::types` module.
 */

/** Attendance response status. */
export type AttendeeStatus =
  | "accepted"
  | "declined"
  | "tentative"
  | "needsAction";

/** Recurrence frequency. */
export type RecurrenceFreq = "daily" | "weekly" | "monthly" | "yearly";

/** Free/busy status. */
export type BusyStatus = "free" | "busy" | "tentative";

/** An event attendee. */
export interface Attendee {
  /** Attendee e-mail address. */
  email: string;
  /** Display name, if known. */
  name: string | null;
  /** RSVP status. */
  status: AttendeeStatus;
}

/** Recurrence rule for repeating events. */
export interface Recurrence {
  /** How often the event repeats. */
  frequency: RecurrenceFreq;
  /** Interval between recurrences (e.g. every 2 weeks). */
  interval: number;
  /** Maximum number of occurrences. */
  count: number | null;
  /** End date for recurrence (RFC-3339). */
  until: string | null;
}

/** A calendar event. */
export interface CalendarEvent {
  /** Provider-assigned event ID. */
  id: string;
  /** Event title. */
  title: string;
  /** Free-form description. */
  description: string | null;
  /** Location text. */
  location: string | null;
  /** Start date-time (RFC-3339). */
  start: string;
  /** End date-time (RFC-3339). */
  end: string;
  /** Whether the event spans whole days. */
  all_day: boolean;
  /** Invited attendees. */
  attendees: Attendee[];
  /** Recurrence rule, if the event repeats. */
  recurrence: Recurrence | null;
  /** Reminder minutes before the event. */
  reminders: number[];
  /** Calendar the event belongs to. */
  calendar_id: string | null;
}

/** Information about a calendar. */
export interface CalendarInfo {
  /** Calendar ID. */
  id: string;
  /** Calendar display name. */
  name: string;
  /** Display colour, if set. */
  color: string | null;
  /** Whether this is the account's primary calendar. */
  primary: boolean;
}

/** A free/busy time slot. */
export interface FreeBusySlot {
  /** Slot start (RFC-3339). */
  start: string;
  /** Slot end (RFC-3339). */
  end: string;
  /** Busy / free / tentative. */
  status: BusyStatus;
}

/** Build a default attendee with status "needsAction". */
export function newAttendee(email: string, name?: string): Attendee {
  return { email, name: name ?? null, status: "needsAction" };
}
