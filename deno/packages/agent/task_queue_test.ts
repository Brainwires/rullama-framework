import { assertEquals, assertThrows } from "@std/assert";
import { Task } from "@rullama/core";
import { TaskQueue } from "./task_queue.ts";

Deno.test("enqueue and dequeue", () => {
  const queue = new TaskQueue(10);
  const task = new Task("test-1", "Test task");

  queue.enqueue(task, "normal");
  assertEquals(queue.size(), 1);

  const dequeued = queue.dequeue();
  assertEquals(dequeued?.task.id, "test-1");
  assertEquals(queue.size(), 0);
});

Deno.test("priority ordering", () => {
  const queue = new TaskQueue(10);

  queue.enqueue(new Task("low-1", "Low priority"), "low");
  queue.enqueue(new Task("normal-1", "Normal priority"), "normal");
  queue.enqueue(new Task("high-1", "High priority"), "high");
  queue.enqueue(new Task("urgent-1", "Urgent priority"), "urgent");

  assertEquals(queue.dequeue()!.task.id, "urgent-1");
  assertEquals(queue.dequeue()!.task.id, "high-1");
  assertEquals(queue.dequeue()!.task.id, "normal-1");
  assertEquals(queue.dequeue()!.task.id, "low-1");
});

Deno.test("max size enforcement", () => {
  const queue = new TaskQueue(2);

  queue.enqueue(new Task("1", "Task 1"), "normal");
  queue.enqueue(new Task("2", "Task 2"), "normal");

  assertThrows(() => queue.enqueue(new Task("3", "Task 3"), "normal"));
});

Deno.test("remove by id", () => {
  const queue = new TaskQueue(10);

  queue.enqueue(new Task("1", "Task 1"), "normal");
  queue.enqueue(new Task("2", "Task 2"), "high");

  assertEquals(queue.size(), 2);

  const removed = queue.removeById("1");
  assertEquals(removed?.task.id, "1");
  assertEquals(queue.size(), 1);
});

Deno.test("assign to worker", () => {
  const queue = new TaskQueue(10);
  queue.enqueue(new Task("test-1", "Test task"), "normal");

  const dequeued = queue.dequeueAndAssign("worker-1");
  assertEquals(dequeued?.assignedTo, "worker-1");
});

Deno.test("peek does not remove", () => {
  const queue = new TaskQueue(10);
  queue.enqueue(new Task("test-1", "Test task"), "high");

  const peeked = queue.peek();
  assertEquals(peeked?.task.id, "test-1");
  assertEquals(queue.size(), 1);
});

Deno.test("is empty and is full", () => {
  const queue = new TaskQueue(2);

  assertEquals(queue.isEmpty(), true);
  assertEquals(queue.isFull(), false);

  queue.enqueue(new Task("1", "Task 1"), "normal");
  assertEquals(queue.isEmpty(), false);
  assertEquals(queue.isFull(), false);

  queue.enqueue(new Task("2", "Task 2"), "normal");
  assertEquals(queue.isEmpty(), false);
  assertEquals(queue.isFull(), true);
});

Deno.test("clear", () => {
  const queue = new TaskQueue(10);
  queue.enqueue(new Task("1", "Task 1"), "normal");
  queue.enqueue(new Task("2", "Task 2"), "high");

  assertEquals(queue.size(), 2);
  queue.clear();
  assertEquals(queue.size(), 0);
  assertEquals(queue.isEmpty(), true);
});

Deno.test("size by priority", () => {
  const queue = new TaskQueue(10);

  queue.enqueue(new Task("1", "T1"), "urgent");
  queue.enqueue(new Task("2", "T2"), "high");
  queue.enqueue(new Task("3", "T3"), "high");
  queue.enqueue(new Task("4", "T4"), "normal");

  const sizes = queue.sizeByPriority();
  assertEquals(sizes.urgent, 1);
  assertEquals(sizes.high, 2);
  assertEquals(sizes.normal, 1);
  assertEquals(sizes.low, 0);
});

Deno.test("default queue size", () => {
  const queue = new TaskQueue();
  assertEquals(queue.maxSize, 100);
});
