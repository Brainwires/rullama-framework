/**
 * Shared calendar types used by Google Calendar and CalDAV clients.
 *
 * Equivalent to Rust's `brainwires_tools::calendar::types` module.
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
  email: string;
  name: string | null;
  status: AttendeeStatus;
}

/** Recurrence rule for repeating events. */
export interface Recurrence {
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
  id: string;
  title: string;
  description: string | null;
  location: string | null;
  /** Start date-time (RFC-3339). */
  start: string;
  /** End date-time (RFC-3339). */
  end: string;
  all_day: boolean;
  attendees: Attendee[];
  recurrence: Recurrence | null;
  /** Reminder minutes before the event. */
  reminders: number[];
  calendar_id: string | null;
}

/** Information about a calendar. */
export interface CalendarInfo {
  id: string;
  name: string;
  color: string | null;
  primary: boolean;
}

/** A free/busy time slot. */
export interface FreeBusySlot {
  /** Slot start (RFC-3339). */
  start: string;
  /** Slot end (RFC-3339). */
  end: string;
  status: BusyStatus;
}

/** Build a default attendee with status "needsAction". */
export function newAttendee(email: string, name?: string): Attendee {
  return { email, name: name ?? null, status: "needsAction" };
}
