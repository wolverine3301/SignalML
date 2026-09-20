---
description: Pick the highest-priority unblocked item from the agent queue and work it
---

Work an item from the agent queue without being told which one.

1. Read `docs/AGENT_QUEUE.md` in full. It is the source of truth for priority — it
   outranks MIGRATION_PLAN.md and DECISION_POINTS.md on the question of *what to do next*.
2. Check current state before choosing: `git branch -a`, `git log --oneline -10`, and the
   Review queue table. An item another branch already covers is not available.
3. Pick the **lowest-tier unblocked item** (Tier A before B before C). If the user passed
   an argument ($ARGUMENTS), treat it as the item id or a topic and pick that instead —
   but say so if it is not the top priority, then do it anyway; their call overrides the
   queue.
4. Follow the Session protocol in that file exactly: branch from `main`, one item, never
   push, never merge, dev loop green before commit, update the Review queue in the same
   commit.
5. End with a short digest in chat: what you did, what is on the branch, what needs a
   human verdict, and what you deliberately did not do.

If every Tier A and B item is blocked, say so plainly and stop. Do not fall through to
Tier C filler — that failure is exactly what this queue exists to prevent.
