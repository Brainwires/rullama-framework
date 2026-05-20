/**
 * Google Calendar API v3 client.
 *
 * Equivalent to Rust's `brainwires_tools::calendar::google` module.
 */

import type {
  Attendee,
  AttendeeStatus,
  CalendarEvent,
  CalendarInfo,
  FreeBusySlot,
} from "./types.ts";

const CALENDAR_API_BASE = "https://www.googleapis.com/calendar/v3";
const TOKEN_URL = "https://oauth2.googleapis.com/token";

function urlEncode(s: string): string {
  let out = "";
  for (const c of s) {
    if (/[A-Za-z0-9\-_.~]/.test(c)) {
      out += c;
    } else {
      for (const b of new TextEncoder().encode(c)) {
        out += "%" + b.toString(16).padStart(2, "0").toUpperCase();
      }
    }
  }
  return out;
}

interface GoogleTokenResponse {
  access_token: string;
  expires_in?: number;
}

/** Google Calendar API client using OAuth2 credentials. */
export class GoogleCalendarClient {
  private access_token: string;
  private readonly client_id: string;
  private readonly client_secret: string;
  private readonly refresh_token: string;

  private constructor(
    access_token: string,
    client_id: string,
    client_secret: string,
    refresh_token: string,
  ) {
    this.access_token = access_token;
    this.client_id = client_id;
    this.client_secret = client_secret;
    this.refresh_token = refresh_token;
  }

  /** Create a new client and obtain an access token via refresh. */
  static async create(
    client_id: string,
    client_secret: string,
    refresh_token: string,
  ): Promise<GoogleCalendarClient> {
    const token = await GoogleCalendarClient.refreshAccessToken(
      client_id,
      client_secret,
      refresh_token,
    );
    return new GoogleCalendarClient(
      token,
      client_id,
      client_secret,
      refresh_token,
    );
  }

  private static async refreshAccessToken(
    client_id: string,
    client_secret: string,
    refresh_token: string,
  ): Promise<string> {
    const body = new URLSearchParams({
      client_id,
      client_secret,
      refresh_token,
      grant_type: "refresh_token",
    }).toString();
    const resp = await fetch(TOKEN_URL, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body,
    });
    if (!resp.ok) {
      throw new Error(
        `Failed to refresh Google OAuth2 token: ${resp.status}`,
      );
    }
    const json = await resp.json() as GoogleTokenResponse;
    return json.access_token;
  }

  /** Re-authenticate and update the access token. */
  async ensureToken(): Promise<void> {
    this.access_token = await GoogleCalendarClient.refreshAccessToken(
      this.client_id,
      this.client_secret,
      this.refresh_token,
    );
  }

  /** List events from a calendar within a time range. */
  async listEvents(
    calendar_id: string,
    time_min: string | null | undefined,
    time_max: string | null | undefined,
    max_results: number,
  ): Promise<CalendarEvent[]> {
    let url = `${CALENDAR_API_BASE}/calendars/${urlEncode(calendar_id)}/events` +
      `?maxResults=${max_results}&singleEvents=true&orderBy=startTime`;
    if (time_min) url += `&timeMin=${urlEncode(time_min)}`;
    if (time_max) url += `&timeMax=${urlEncode(time_max)}`;

    const resp = await fetch(url, {
      headers: { Authorization: `Bearer ${this.access_token}` },
    });
    if (!resp.ok) {
      throw new Error(`Failed to list Google Calendar events: ${resp.status}`);
    }
    const body = await resp.json() as { items?: unknown[] };
    const items = Array.isArray(body.items) ? body.items : [];
    const events: CalendarEvent[] = [];
    for (const item of items) {
      const e = GoogleCalendarClient.parseEvent(item);
      if (e) events.push(e);
    }
    return events;
  }

  /** Create a new calendar event. */
  async createEvent(
    calendar_id: string,
    event: CalendarEvent,
  ): Promise<CalendarEvent> {
    const url =
      `${CALENDAR_API_BASE}/calendars/${urlEncode(calendar_id)}/events`;
    const body = GoogleCalendarClient.eventToGoogleJson(event);
    const resp = await fetch(url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.access_token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      throw new Error(
        `Failed to create Google Calendar event: ${resp.status}`,
      );
    }
    const resbody = await resp.json();
    const created = GoogleCalendarClient.parseEvent(resbody);
    if (!created) throw new Error("Failed to parse created event response");
    return created;
  }

  /** Update an existing calendar event. */
  async updateEvent(
    calendar_id: string,
    event_id: string,
    event: CalendarEvent,
  ): Promise<CalendarEvent> {
    const url = `${CALENDAR_API_BASE}/calendars/${urlEncode(calendar_id)}` +
      `/events/${urlEncode(event_id)}`;
    const body = GoogleCalendarClient.eventToGoogleJson(event);
    const resp = await fetch(url, {
      method: "PUT",
      headers: {
        Authorization: `Bearer ${this.access_token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      throw new Error(
        `Failed to update Google Calendar event: ${resp.status}`,
      );
    }
    const resbody = await resp.json();
    const updated = GoogleCalendarClient.parseEvent(resbody);
    if (!updated) throw new Error("Failed to parse updated event response");
    return updated;
  }

  /** Delete a calendar event. */
  async deleteEvent(calendar_id: string, event_id: string): Promise<void> {
    const url = `${CALENDAR_API_BASE}/calendars/${urlEncode(calendar_id)}` +
      `/events/${urlEncode(event_id)}`;
    const resp = await fetch(url, {
      method: "DELETE",
      headers: { Authorization: `Bearer ${this.access_token}` },
    });
    if (!resp.ok && resp.status !== 204) {
      const body = await resp.text().catch(() => "");
      throw new Error(`Delete failed: ${body}`);
    }
  }

  /** Query free/busy information. */
  async freeBusy(
    calendar_ids: readonly string[],
    time_min: string,
    time_max: string,
  ): Promise<FreeBusySlot[]> {
    const url = `${CALENDAR_API_BASE}/freeBusy`;
    const items = calendar_ids.map((id) => ({ id }));
    const body = {
      timeMin: time_min,
      timeMax: time_max,
      items,
    };
    const resp = await fetch(url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.access_token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      throw new Error(`Failed to query free/busy: ${resp.status}`);
    }
    const res = await resp.json() as {
      calendars?: Record<string, { busy?: Array<{ start?: string; end?: string }> }>;
    };
    const slots: FreeBusySlot[] = [];
    if (res.calendars) {
      for (const cal of Object.values(res.calendars)) {
        for (const busy of cal.busy ?? []) {
          slots.push({
            start: busy.start ?? "",
            end: busy.end ?? "",
            status: "busy",
          });
        }
      }
    }
    return slots;
  }

  /** List available calendars. */
  async listCalendars(): Promise<CalendarInfo[]> {
    const url = `${CALENDAR_API_BASE}/users/me/calendarList`;
    const resp = await fetch(url, {
      headers: { Authorization: `Bearer ${this.access_token}` },
    });
    if (!resp.ok) throw new Error(`Failed to list calendars: ${resp.status}`);
    const body = await resp.json() as { items?: unknown[] };
    const items = Array.isArray(body.items) ? body.items : [];
    return items.map((item) => {
      const raw = item as Record<string, unknown>;
      return {
        id: (raw.id as string | undefined) ?? "",
        name: (raw.summary as string | undefined) ?? "",
        color: (raw.backgroundColor as string | undefined) ?? null,
        primary: raw.primary === true,
      };
    });
  }

  // ── Helpers ────────────────────────────────────────────────────────────────

  /** Parse a Google Calendar event object into a CalendarEvent. */
  static parseEvent(item: unknown): CalendarEvent | null {
    if (!item || typeof item !== "object") return null;
    const raw = item as Record<string, unknown>;
    const id = typeof raw.id === "string" ? raw.id : null;
    if (!id) return null;
    const title = typeof raw.summary === "string" ? raw.summary : "";

    const startObj = raw.start as Record<string, unknown> | undefined;
    let start = "";
    let all_day = false;
    if (startObj) {
      if (typeof startObj.dateTime === "string") {
        start = startObj.dateTime;
      } else if (typeof startObj.date === "string") {
        start = startObj.date;
        all_day = true;
      } else {
        return null;
      }
    } else {
      return null;
    }

    const endObj = raw.end as Record<string, unknown> | undefined;
    let end = start;
    if (endObj) {
      if (typeof endObj.dateTime === "string") {
        end = endObj.dateTime;
      } else if (typeof endObj.date === "string") {
        end = endObj.date;
      }
    }

    const attendeesRaw = Array.isArray(raw.attendees) ? raw.attendees : [];
    const attendees: Attendee[] = [];
    for (const a of attendeesRaw) {
      if (!a || typeof a !== "object") continue;
      const ar = a as Record<string, unknown>;
      const email = typeof ar.email === "string" ? ar.email : null;
      if (!email) continue;
      const name = typeof ar.displayName === "string" ? ar.displayName : null;
      const rs = typeof ar.responseStatus === "string"
        ? ar.responseStatus
        : "needsAction";
      let status: AttendeeStatus;
      switch (rs) {
        case "accepted":
          status = "accepted";
          break;
        case "declined":
          status = "declined";
          break;
        case "tentative":
          status = "tentative";
          break;
        default:
          status = "needsAction";
      }
      attendees.push({ email, name, status });
    }

    return {
      id,
      title,
      description: typeof raw.description === "string" ? raw.description : null,
      location: typeof raw.location === "string" ? raw.location : null,
      start,
      end,
      all_day,
      attendees,
      recurrence: null,
      reminders: [],
      calendar_id: null,
    };
  }

  /** Convert a CalendarEvent into the Google Calendar JSON shape. */
  static eventToGoogleJson(event: CalendarEvent): Record<string, unknown> {
    const startObj = event.all_day
      ? { date: event.start }
      : { dateTime: event.start };
    const endObj = event.all_day
      ? { date: event.end }
      : { dateTime: event.end };

    const body: Record<string, unknown> = {
      summary: event.title,
      start: startObj,
      end: endObj,
    };
    if (event.description) body.description = event.description;
    if (event.location) body.location = event.location;
    if (event.attendees.length > 0) {
      body.attendees = event.attendees.map((a) => {
        const obj: Record<string, unknown> = { email: a.email };
        if (a.name) obj.displayName = a.name;
        return obj;
      });
    }
    return body;
  }
}
