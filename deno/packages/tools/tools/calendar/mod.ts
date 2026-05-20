/**
 * Calendar tools: list, create, update, delete events, and find free time.
 *
 * Equivalent to Rust's `brainwires_tools::calendar` module.
 */

import {
  objectSchema,
  type Tool,
  type ToolContext,
  ToolResult,
} from "@brainwires/core";

import { CalDavClient } from "./caldav.ts";
import { GoogleCalendarClient } from "./google.ts";
import type { Attendee, CalendarEvent } from "./types.ts";

export { CalDavClient } from "./caldav.ts";
export { GoogleCalendarClient } from "./google.ts";
export type {
  Attendee,
  AttendeeStatus,
  BusyStatus,
  CalendarEvent,
  CalendarInfo,
  FreeBusySlot,
  Recurrence,
  RecurrenceFreq,
} from "./types.ts";
export { newAttendee } from "./types.ts";

/** Calendar provider configuration variants. */
export type CalendarProvider =
  | {
    type: "google_calendar";
    client_id: string;
    client_secret: string;
    refresh_token: string;
  }
  | {
    type: "cal_dav";
    url: string;
    username: string;
    password: string;
  };

/** Configuration for the calendar tool. */
export interface CalendarConfig {
  provider: CalendarProvider;
  /** Default calendar ID to operate on (default: "primary"). */
  default_calendar_id: string;
}

/** Calendar tool — event CRUD + free/busy. */
export class CalendarTool {
  /** Return tool definitions for calendar operations. */
  static getTools(): Tool[] {
    return [
      CalendarTool.listEventsTool(),
      CalendarTool.createEventTool(),
      CalendarTool.updateEventTool(),
      CalendarTool.deleteEventTool(),
      CalendarTool.findFreeTimeTool(),
    ];
  }

  private static listEventsTool(): Tool {
    return {
      name: "calendar_list_events",
      description: "List calendar events within a time range.",
      input_schema: objectSchema({
        calendar_id: {
          type: "string",
          description: "Calendar ID (default: primary)",
        },
        time_min: {
          type: "string",
          description: "Start of time range (RFC-3339)",
        },
        time_max: {
          type: "string",
          description: "End of time range (RFC-3339)",
        },
        max_results: {
          type: "integer",
          description: "Maximum number of events (default: 25)",
        },
      }, []),
      requires_approval: false,
    };
  }

  private static createEventTool(): Tool {
    return {
      name: "calendar_create_event",
      description: "Create a new calendar event.",
      input_schema: objectSchema({
        title: { type: "string", description: "Event title" },
        start: {
          type: "string",
          description: "Start date-time (RFC-3339)",
        },
        end: { type: "string", description: "End date-time (RFC-3339)" },
        description: { type: "string", description: "Event description" },
        location: { type: "string", description: "Event location" },
        all_day: {
          type: "boolean",
          description: "Whether this is an all-day event",
        },
        attendees: {
          type: "array",
          items: { type: "string" },
          description: "Attendee email addresses",
        },
        calendar_id: {
          type: "string",
          description: "Calendar ID (default: primary)",
        },
      }, ["title", "start", "end"]),
      requires_approval: true,
    };
  }

  private static updateEventTool(): Tool {
    return {
      name: "calendar_update_event",
      description: "Update an existing calendar event.",
      input_schema: objectSchema({
        event_id: { type: "string", description: "Event ID to update" },
        title: { type: "string", description: "New event title" },
        start: {
          type: "string",
          description: "New start date-time (RFC-3339)",
        },
        end: {
          type: "string",
          description: "New end date-time (RFC-3339)",
        },
        description: {
          type: "string",
          description: "New event description",
        },
        location: { type: "string", description: "New event location" },
        calendar_id: {
          type: "string",
          description: "Calendar ID (default: primary)",
        },
      }, ["event_id"]),
      requires_approval: true,
    };
  }

  private static deleteEventTool(): Tool {
    return {
      name: "calendar_delete_event",
      description: "Delete a calendar event.",
      input_schema: objectSchema({
        event_id: { type: "string", description: "Event ID to delete" },
        calendar_id: {
          type: "string",
          description: "Calendar ID (default: primary)",
        },
      }, ["event_id"]),
      requires_approval: true,
    };
  }

  private static findFreeTimeTool(): Tool {
    return {
      name: "calendar_find_free_time",
      description: "Find free time slots across one or more calendars.",
      input_schema: objectSchema({
        time_min: {
          type: "string",
          description: "Start of search range (RFC-3339)",
        },
        time_max: {
          type: "string",
          description: "End of search range (RFC-3339)",
        },
        calendar_ids: {
          type: "array",
          items: { type: "string" },
          description: "Calendar IDs to check (default: primary)",
        },
      }, ["time_min", "time_max"]),
      requires_approval: false,
    };
  }

  /** Execute a calendar tool by name. */
  static async execute(
    tool_use_id: string,
    tool_name: string,
    input: Record<string, unknown>,
    context: ToolContext,
  ): Promise<ToolResult> {
    try {
      const output = await CalendarTool.dispatch(tool_name, input, context);
      return ToolResult.success(tool_use_id, output);
    } catch (e) {
      return ToolResult.error(
        tool_use_id,
        `Calendar operation failed: ${(e as Error).message}`,
      );
    }
  }

  private static async dispatch(
    tool_name: string,
    input: Record<string, unknown>,
    context: ToolContext,
  ): Promise<string> {
    switch (tool_name) {
      case "calendar_list_events":
        return CalendarTool.handleListEvents(input, context);
      case "calendar_create_event":
        return CalendarTool.handleCreateEvent(input, context);
      case "calendar_update_event":
        return CalendarTool.handleUpdateEvent(input, context);
      case "calendar_delete_event":
        return CalendarTool.handleDeleteEvent(input, context);
      case "calendar_find_free_time":
        return CalendarTool.handleFindFreeTime(input, context);
      default:
        throw new Error(`Unknown calendar tool: ${tool_name}`);
    }
  }

  // ── Handler implementations ───────────────────────────────────────────────

  private static async handleListEvents(
    input: Record<string, unknown>,
    context: ToolContext,
  ): Promise<string> {
    const config = CalendarTool.getConfig(context);
    const calendar_id = (input.calendar_id as string | undefined) ??
      config.default_calendar_id;
    const time_min = input.time_min as string | undefined;
    const time_max = input.time_max as string | undefined;
    const max_results = (input.max_results as number | undefined) ?? 25;

    if (config.provider.type === "google_calendar") {
      const { client_id, client_secret, refresh_token } = config.provider;
      const client = await GoogleCalendarClient.create(
        client_id,
        client_secret,
        refresh_token,
      );
      const events = await client.listEvents(
        calendar_id,
        time_min,
        time_max,
        max_results,
      );
      return JSON.stringify(events, null, 2);
    }
    const { url, username, password } = config.provider;
    const client = new CalDavClient(url, username, password);
    const events = await client.listEvents(calendar_id, time_min, time_max);
    return JSON.stringify(events, null, 2);
  }

  private static async handleCreateEvent(
    input: Record<string, unknown>,
    context: ToolContext,
  ): Promise<string> {
    const config = CalendarTool.getConfig(context);
    const calendar_id = (input.calendar_id as string | undefined) ??
      config.default_calendar_id;
    const event = CalendarTool.parseEventInput(input);

    if (config.provider.type === "google_calendar") {
      const { client_id, client_secret, refresh_token } = config.provider;
      const client = await GoogleCalendarClient.create(
        client_id,
        client_secret,
        refresh_token,
      );
      const created = await client.createEvent(calendar_id, event);
      return JSON.stringify(created, null, 2);
    }
    const { url, username, password } = config.provider;
    const client = new CalDavClient(url, username, password);
    await client.createEvent(calendar_id, event);
    return JSON.stringify(event, null, 2);
  }

  private static async handleUpdateEvent(
    input: Record<string, unknown>,
    context: ToolContext,
  ): Promise<string> {
    const config = CalendarTool.getConfig(context);
    const calendar_id = (input.calendar_id as string | undefined) ??
      config.default_calendar_id;
    const event_id = input.event_id as string | undefined;
    if (!event_id) throw new Error("'event_id' is required");

    const event = CalendarTool.parseEventInput(input);
    event.id = event_id;

    if (config.provider.type === "google_calendar") {
      const { client_id, client_secret, refresh_token } = config.provider;
      const client = await GoogleCalendarClient.create(
        client_id,
        client_secret,
        refresh_token,
      );
      const updated = await client.updateEvent(calendar_id, event_id, event);
      return JSON.stringify(updated, null, 2);
    }
    const { url, username, password } = config.provider;
    const client = new CalDavClient(url, username, password);
    await client.updateEvent(calendar_id, event);
    return JSON.stringify(event, null, 2);
  }

  private static async handleDeleteEvent(
    input: Record<string, unknown>,
    context: ToolContext,
  ): Promise<string> {
    const config = CalendarTool.getConfig(context);
    const calendar_id = (input.calendar_id as string | undefined) ??
      config.default_calendar_id;
    const event_id = input.event_id as string | undefined;
    if (!event_id) throw new Error("'event_id' is required");

    if (config.provider.type === "google_calendar") {
      const { client_id, client_secret, refresh_token } = config.provider;
      const client = await GoogleCalendarClient.create(
        client_id,
        client_secret,
        refresh_token,
      );
      await client.deleteEvent(calendar_id, event_id);
    } else {
      const { url, username, password } = config.provider;
      const client = new CalDavClient(url, username, password);
      await client.deleteEvent(calendar_id, event_id);
    }
    return `Event '${event_id}' deleted successfully`;
  }

  private static async handleFindFreeTime(
    input: Record<string, unknown>,
    context: ToolContext,
  ): Promise<string> {
    const config = CalendarTool.getConfig(context);
    const time_min = input.time_min as string | undefined;
    const time_max = input.time_max as string | undefined;
    if (!time_min) throw new Error("'time_min' is required");
    if (!time_max) throw new Error("'time_max' is required");

    const raw_ids = input.calendar_ids;
    const calendar_ids = Array.isArray(raw_ids)
      ? raw_ids.filter((x): x is string => typeof x === "string")
      : [config.default_calendar_id];

    if (config.provider.type === "google_calendar") {
      const { client_id, client_secret, refresh_token } = config.provider;
      const client = await GoogleCalendarClient.create(
        client_id,
        client_secret,
        refresh_token,
      );
      const slots = await client.freeBusy(calendar_ids, time_min, time_max);
      return JSON.stringify(slots, null, 2);
    }
    throw new Error("Free/busy queries are not yet supported for CalDAV");
  }

  // ── Helpers ────────────────────────────────────────────────────────────────

  private static getConfig(context: ToolContext): CalendarConfig {
    const raw = context.metadata["calendar_config"];
    if (!raw) {
      throw new Error(
        "Calendar configuration not found. Set 'calendar_config' in ToolContext.metadata.",
      );
    }
    return JSON.parse(raw) as CalendarConfig;
  }

  private static parseEventInput(
    input: Record<string, unknown>,
  ): CalendarEvent {
    const title = (input.title as string | undefined) ?? "Untitled Event";
    const start = (input.start as string | undefined) ?? "";
    const end = (input.end as string | undefined) ?? "";
    const all_day = (input.all_day as boolean | undefined) ?? false;

    const rawAttendees = input.attendees;
    const attendees: Attendee[] = Array.isArray(rawAttendees)
      ? rawAttendees
        .filter((x): x is string => typeof x === "string")
        .map((email): Attendee => ({
          email,
          name: null,
          status: "needsAction",
        }))
      : [];

    return {
      id: crypto.randomUUID(),
      title,
      description: (input.description as string | undefined) ?? null,
      location: (input.location as string | undefined) ?? null,
      start,
      end,
      all_day,
      attendees,
      recurrence: null,
      reminders: [],
      calendar_id: (input.calendar_id as string | undefined) ?? null,
    };
  }
}
