// Example: MessageStore — conversation messages with search
// Demonstrates creating an InMemoryMessageStore, adding messages, searching,
// and listing messages by conversation.
// Run: deno run deno/examples/storage/message_store.ts

import {
  InMemoryMessageStore,
  type MessageMetadata,
} from "@rullama/storage";

async function main() {
  console.log("=== Message Store Example ===\n");

  // 1. Create an in-memory message store
  const store = new InMemoryMessageStore();
  await store.ensureTable();
  console.log("InMemoryMessageStore initialized.\n");

  // 2. Insert some demo messages
  const conversationId = "conv-demo-1";
  const now = Math.floor(Date.now() / 1000);

  const messages: MessageMetadata[] = [
    {
      messageId: "msg-1",
      conversationId,
      role: "user",
      content: "How do I implement a binary search tree in TypeScript?",
      tokenCount: 12,
      createdAt: now,
    },
    {
      messageId: "msg-2",
      conversationId,
      role: "assistant",
      content:
        "You can implement a BST using a class with left and right child references. Each node stores a key and optional value.",
      tokenCount: 28,
      modelId: "gpt-4o",
      createdAt: now + 1,
    },
    {
      messageId: "msg-3",
      conversationId,
      role: "user",
      content: "What about balancing? Should I use a red-black tree?",
      tokenCount: 11,
      createdAt: now + 2,
    },
    {
      messageId: "msg-4",
      conversationId: "conv-demo-2",
      role: "user",
      content: "Explain the difference between TCP and UDP protocols.",
      tokenCount: 10,
      createdAt: now + 3,
    },
  ];

  // Add messages one at a time
  for (const msg of messages) {
    await store.add(msg);
  }
  console.log(`Inserted ${messages.length} messages across 2 conversations.\n`);

  // 3. Search across all conversations (in-memory store returns all matches)
  console.log("--- Search (all conversations) ---");
  const results = await store.search("tree data structures", 3, 0.0);
  console.log(`Search results: ${results.length} messages`);
  for (const [msg, score] of results) {
    console.log(
      `  [${score.toFixed(3)}] (${msg.messageId}) ${msg.role}: ${msg.content}`,
    );
  }
  console.log();

  // 4. Search within a single conversation
  console.log("--- Search (within conversation) ---");
  const convResults = await store.searchConversation(
    conversationId,
    "balancing algorithms",
    2,
    0.0,
  );
  console.log(
    `Search within "${conversationId}": ${convResults.length} results`,
  );
  for (const [msg, score] of convResults) {
    console.log(`  [${score.toFixed(3)}] ${msg.content}`);
  }
  console.log();

  // 5. List all messages for a conversation
  console.log("--- List by Conversation ---");
  const convMessages = await store.getByConversation(conversationId);
  console.log(
    `Conversation "${conversationId}" has ${convMessages.length} messages:`,
  );
  for (const msg of convMessages) {
    console.log(`  ${msg.messageId} (${msg.role}): ${msg.content}`);
  }
  console.log();

  // 6. Retrieve a single message
  console.log("--- Get Single Message ---");
  const single = await store.get("msg-2");
  if (single) {
    console.log(`  Found: ${single.messageId} by ${single.role}`);
    console.log(`  Model: ${single.modelId ?? "unknown"}`);
    console.log(`  Tokens: ${single.tokenCount ?? "unknown"}`);
  }
  console.log();

  // 7. Delete a message and verify
  console.log("--- Delete Operations ---");
  await store.delete("msg-4");
  const afterDelete = await store.getByConversation("conv-demo-2");
  console.log(
    `After deleting msg-4, conv-demo-2 has ${afterDelete.length} messages`,
  );

  // 8. Batch add
  console.log("\n--- Batch Add ---");
  const batchMessages: MessageMetadata[] = [
    {
      messageId: "msg-5",
      conversationId: "conv-demo-3",
      role: "user",
      content: "What is the time complexity of quicksort?",
      createdAt: now + 10,
    },
    {
      messageId: "msg-6",
      conversationId: "conv-demo-3",
      role: "assistant",
      content: "Quicksort has O(n log n) average and O(n^2) worst case.",
      createdAt: now + 11,
    },
  ];
  await store.addBatch(batchMessages);
  const batchConv = await store.getByConversation("conv-demo-3");
  console.log(
    `Batch added ${batchMessages.length} messages to conv-demo-3 (total: ${batchConv.length})`,
  );

  console.log("\nDone.");
}

await main();
