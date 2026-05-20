/**
 * CalDAV protocol client for calendar CRUD operations.
 *
 * Equivalent to Rust's `brainwires_tools::calendar::caldav` module.
 */

import type { CalendarEvent, CalendarInfo } from "./types.ts";

function basicAuth(username: string, password: string): string {
  return "Basic " + btoa(`${username}:${password}`);
}

/** CalDAV client for interacting with CalDAV-compliant servers. */
export class CalDavClient {
  readonly base_url: string;
  readonly username: string;
  readonly password: string;

  constructor(base_url: string, username: string, password: string) {
    this.base_url = base_url.replace(/\/+$/, "");
    this.username = username;
    this.password = password;
  }

  /** Discover available calendars via PROPFIND. */
  async listCalendars(): Promise<CalendarInfo[]> {
    const url = `${this.base_url}/`;
    const body = `<?xml version="1.0" encoding="utf-8" ?>
<D:propfind xmlns:D="DAV:" xmlns:C="urn:ietf:params:xml:ns:caldav">
  <D:prop>
    <D:displayname/>
    <D:resourcetype/>
    <C:calendar-color/>
  </D:prop>
</D:propfind>`;
    const resp = await fetch(url, {
      method: "PROPFIND",
      headers: {
        Depth: "1",
        "Content-Type": "application/xml",
        Authorization: basicAuth(this.username, this.password),
      },
      body,
    });
    const text = await resp.text();
    return CalDavClient.parsePropfindCalendars(text);
  }

  /** List events from a calendar within a time range via REPORT. */
  async listEvents(
    calendar_path: string,
    time_min: string | null | undefined,
    time_max: string | null | undefined,
  ): Promise<CalendarEvent[]> {
    const url = `${this.base_url}/${calendar_path.replace(/^\/+/, "")}`;

    let timeRange: string;
    if (time_min && time_max) {
      timeRange = `<C:time-range start="${time_min}" end="${time_max}"/>`;
    } else if (time_min) {
      timeRange = `<C:time-range start="${time_min}"/>`;
    } else if (time_max) {
      timeRange = `<C:time-range end="${time_max}"/>`;
    } else {
      timeRange = "";
    }

    const body = `<?xml version="1.0" encoding="utf-8" ?>
<C:calendar-query xmlns:D="DAV:" xmlns:C="urn:ietf:params:xml:ns:caldav">
  <D:prop>
    <D:getetag/>
    <C:calendar-data/>
  </D:prop>
  <C:filter>
    <C:comp-filter name="VCALENDAR">
      <C:comp-filter name="VEVENT">
        ${timeRange}
      </C:comp-filter>
    </C:comp-filter>
  </C:filter>
</C:calendar-query>`;

    const resp = await fetch(url, {
      method: "REPORT",
      headers: {
        Depth: "1",
        "Content-Type": "application/xml",
        Authorization: basicAuth(this.username, this.password),
      },
      body,
    });
    const text = await resp.text();
    return CalDavClient.parseIcalEvents(text);
  }

  /** Create a new event via PUT with iCalendar format. */
  async createEvent(
    calendar_path: string,
    event: CalendarEvent,
  ): Promise<void> {
    const eventPath = `${this.base_url}/${
      calendar_path.replace(/^\/+/, "")
    }/${event.id}.ics`;
    const ical = CalDavClient.eventToIcal(event);
    const resp = await fetch(eventPath, {
      method: "PUT",
      headers: {
        "Content-Type": "text/calendar; charset=utf-8",
        Authorization: basicAuth(this.username, this.password),
      },
      body: ical,
    });
    if (!resp.ok && resp.status !== 201) {
      const body = await resp.text().catch(() => "");
      throw new Error(`CalDAV create failed (${resp.status}): ${body}`);
    }
  }

  /** Update an existing event via PUT. */
  updateEvent(calendar_path: string, event: CalendarEvent): Promise<void> {
    return this.createEvent(calendar_path, event);
  }

  /** Delete an event via DELETE. */
  async deleteEvent(calendar_path: string, event_id: string): Promise<void> {
    const eventPath = `${this.base_url}/${
      calendar_path.replace(/^\/+/, "")
    }/${event_id}.ics`;
    const resp = await fetch(eventPath, {
      method: "DELETE",
      headers: { Authorization: basicAuth(this.username, this.password) },
    });
    if (!resp.ok && resp.status !== 204) {
      const body = await resp.text().catch(() => "");
      throw new Error(`CalDAV delete failed (${resp.status}): ${body}`);
    }
  }

  // ── Helpers ────────────────────────────────────────────────────────────────

  /** Convert a CalendarEvent to iCalendar format string. */
  static eventToIcal(event: CalendarEvent): string {
    const lines: string[] = [
      "BEGIN:VCALENDAR",
      "VERSION:2.0",
      "PRODID:-//Brainwires//CalDAV Client//EN",
      "BEGIN:VEVENT",
      `UID:${event.id}`,
      `SUMMARY:${event.title}`,
    ];

    if (event.all_day) {
      const startDate = event.start.split("T")[0] ?? event.start;
      const endDate = event.end.split("T")[0] ?? event.end;
      lines.push(`DTSTART;VALUE=DATE:${startDate.replaceAll("-", "")}`);
      lines.push(`DTEND;VALUE=DATE:${endDate.replaceAll("-", "")}`);
    } else {
      lines.push(`DTSTART:${CalDavClient.rfc3339ToIcal(event.start)}`);
      lines.push(`DTEND:${CalDavClient.rfc3339ToIcal(event.end)}`);
    }

    if (event.description) lines.push(`DESCRIPTION:${event.description}`);
    if (event.location) lines.push(`LOCATION:${event.location}`);
    for (const a of event.attendees) {
      const nameParam = a.name ? `;CN=${a.name}` : "";
      lines.push(`ATTENDEE${nameParam}:mailto:${a.email}`);
    }

    lines.push("END:VEVENT", "END:VCALENDAR");
    return lines.join("\r\n");
  }

  /** Convert RFC-3339 datetime to iCalendar datetime format. */
  static rfc3339ToIcal(dt: string): string {
    return dt.replaceAll("-", "").replaceAll(":", "");
  }

  /** Parse PROPFIND response to extract calendar info (simplified). */
  static parsePropfindCalendars(xml: string): CalendarInfo[] {
    const calendars: CalendarInfo[] = [];
    let idx = 0;
    for (const line of xml.split("\n")) {
      const trimmed = line.trim();
      if (
        trimmed.includes("<displayname>") ||
        trimmed.includes("<D:displayname>")
      ) {
        const name = trimmed
          .replaceAll("<displayname>", "")
          .replaceAll("</displayname>", "")
          .replaceAll("<D:displayname>", "")
          .replaceAll("</D:displayname>", "")
          .trim();
        if (name.length > 0) {
          calendars.push({
            id: `calendar-${idx}`,
            name,
            color: null,
            primary: idx === 0,
          });
          idx += 1;
        }
      }
    }
    return calendars;
  }

  /** Parse CalDAV REPORT response containing iCalendar data (simplified). */
  static parseIcalEvents(xml: string): CalendarEvent[] {
    const events: CalendarEvent[] = [];
    let remaining = xml;
    let startIdx = remaining.indexOf("BEGIN:VEVENT");
    while (startIdx !== -1) {
      const endIdx = remaining.slice(startIdx).indexOf("END:VEVENT");
      if (endIdx === -1) break;
      const vevent = remaining.slice(
        startIdx,
        startIdx + endIdx + "END:VEVENT".length,
      );
      const parsed = CalDavClient.parseSingleVevent(vevent);
      if (parsed) events.push(parsed);
      remaining = remaining.slice(startIdx + endIdx + "END:VEVENT".length);
      startIdx = remaining.indexOf("BEGIN:VEVENT");
    }
    return events;
  }

  private static parseSingleVevent(vevent: string): CalendarEvent | null {
    const getProp = (name: string): string | null => {
      for (const line of vevent.split("\n")) {
        const trimmed = line.trim();
        if (trimmed.startsWith(name)) {
          const idx = trimmed.indexOf(":");
          if (idx === -1) continue;
          return trimmed.slice(idx + 1);
        }
      }
      return null;
    };

    const uid = getProp("UID");
    if (!uid) return null;
    const summary = getProp("SUMMARY") ?? "";
    const start = getProp("DTSTART") ?? "";
    const end = getProp("DTEND") ?? start;
    const description = getProp("DESCRIPTION");
    const location = getProp("LOCATION");
    const all_day = vevent.includes("VALUE=DATE");

    return {
      id: uid,
      title: summary,
      description,
      location,
      start,
      end,
      all_day,
      attendees: [],
      recurrence: null,
      reminders: [],
      calendar_id: null,
    };
  }
}
