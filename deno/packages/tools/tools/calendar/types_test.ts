import { assert, assertEquals } from "@std/assert";
import type {
  CalendarEvent,
  CalendarInfo,
  FreeBusySlot,
  Recurrence,
} from "./types.ts";
import { newAttendee } from "./types.ts";

Deno.test("CalendarEvent JSON round-trip", () => {
  const event: CalendarEvent = {
    id: "evt-1",
    title: "Team Meeting",
    description: "Weekly sync",
    location: "Room 101",
    start: "2025-06-01T10:00:00Z",
    end: "2025-06-01T11:00:00Z",
    all_day: false,
    attendees: [
      { email: "bob@example.com", name: "Bob", status: "accepted" },
    ],
    recurrence: {
      frequency: "weekly",
      interval: 1,
      count: 10,
      until: null,
    },
    reminders: [15, 5],
    calendar_id: "primary",
  };
  const round = JSON.parse(JSON.stringify(event)) as CalendarEvent;
  assertEquals(round.id, "evt-1");
  assertEquals(round.title, "Team Meeting");
  assertEquals(round.attendees.length, 1);
  assertEquals(round.attendees[0].status, "accepted");
});

Deno.test("Attendee default status is needsAction", () => {
  assertEquals(newAttendee("x@example.com").status, "needsAction");
});

Deno.test("Recurrence JSON round-trip", () => {
  const rec: Recurrence = {
    frequency: "monthly",
    interval: 2,
    count: null,
    until: "2026-12-31T00:00:00Z",
  };
  const round = JSON.parse(JSON.stringify(rec)) as Recurrence;
  assertEquals(round.frequency, "monthly");
  assertEquals(round.interval, 2);
});

Deno.test("CalendarInfo JSON round-trip", () => {
  const info: CalendarInfo = {
    id: "cal-1",
    name: "Work",
    color: "#4285f4",
    primary: true,
  };
  const round = JSON.parse(JSON.stringify(info)) as CalendarInfo;
  assertEquals(round.name, "Work");
  assert(round.primary);
});

Deno.test("FreeBusySlot JSON round-trip", () => {
  const slot: FreeBusySlot = {
    start: "2025-06-01T09:00:00Z",
    end: "2025-06-01T10:00:00Z",
    status: "busy",
  };
  const round = JSON.parse(JSON.stringify(slot)) as FreeBusySlot;
  assertEquals(round.status, "busy");
});
