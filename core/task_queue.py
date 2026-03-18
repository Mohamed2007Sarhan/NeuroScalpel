"""
task_queue.py
=============
Multi-task queue for the NeuroScalpel pipeline.

When the user types an order containing multiple distinct wrong facts,
Phase 1 (DeepSeek) returns a JSON list of tasks. This module parses that
list, stores tasks, and drives sequential execution one task at a time.

Each EditTask contains everything needed for 1 full Phase1->Phase5 cycle.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Callable

logger = logging.getLogger("NeuroScalpel.TaskQueue")


@dataclass
class EditTask:
    """A single knowledge-edit task extracted from the user's order."""
    index: int                   # 0-based position in queue
    analysis: str                # Brief explanation of what's wrong
    trick_prompt: str            # Exact prompt to feed target model
    subject: str                 # Entity being corrected (e.g. "Egypt")
    wrong_value: str             # What the model currently says
    correct_value: str           # What it should say instead
    target_layer: Optional[int] = None    # Filled in after Phase 3
    target_point: Optional[int] = None   # Filled in after Phase 3
    status: str = "queued"       # queued | running | done | failed


class TaskQueue:
    """
    Manages a list of EditTask objects and tracks current execution position.

    Usage
    -----
        queue = TaskQueue()
        queue.parse_from_phase1_response(json_string)

        while queue.has_next():
            task = queue.pop_next()
            # run pipeline for task ...
            queue.mark_done(task.index)
    """

    def __init__(self):
        self._tasks: List[EditTask] = []
        self._current_index: int = 0
        self.on_task_started: Optional[Callable[[EditTask, int, int], None]] = None
        self.on_task_finished: Optional[Callable[[EditTask], None]] = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def parse_from_phase1_response(self, response_text: str) -> List[EditTask]:
        """
        Parses Phase 1 DeepSeek JSON output into EditTask objects.

        Expects one of two formats:
          - Single task: {"analysis": "...", "trick_prompt": "...", "subject": "...",
                          "wrong_value": "...", "correct_value": "..."}
          - Multi task:  [{"analysis": ...}, {"analysis": ...}, ...]

        Falls back gracefully for partial or malformed JSON.
        """
        self._tasks.clear()
        self._current_index = 0

        # Strip markdown code fences if present
        clean = response_text.strip()
        for fence in ["```json", "```"]:
            if clean.startswith(fence):
                clean = clean[len(fence):]
        clean = clean.rstrip("`").strip()

        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError as e:
            logger.warning(f"Phase 1 JSON parse failed: {e}. Treating as single task.")
            # Create a minimal single task from whatever we got
            self._tasks.append(EditTask(
                index=0,
                analysis="Could not parse structured response.",
                trick_prompt=response_text[:500],
                subject="unknown",
                wrong_value="unknown",
                correct_value="unknown"
            ))
            return self._tasks

        # Normalize to list
        if isinstance(parsed, dict):
            parsed = [parsed]

        for i, item in enumerate(parsed):
            self._tasks.append(EditTask(
                index=i,
                analysis=item.get("analysis", ""),
                trick_prompt=item.get("trick_prompt", item.get("prompt", "")),
                subject=item.get("subject", "unknown"),
                wrong_value=item.get("wrong_value", item.get("wrong", "")),
                correct_value=item.get("correct_value", item.get("correct", ""))
            ))

        logger.info(f"TaskQueue loaded with {len(self._tasks)} task(s).")
        return self._tasks

    def load_single(self, analysis: str, trick_prompt: str,
                    subject: str = "", wrong_value: str = "",
                    correct_value: str = ""):
        """Convenience: bypass JSON parsing and load exactly one task."""
        self._tasks.clear()
        self._current_index = 0
        self._tasks.append(EditTask(
            index=0, analysis=analysis,
            trick_prompt=trick_prompt,
            subject=subject,
            wrong_value=wrong_value,
            correct_value=correct_value
        ))

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def has_next(self) -> bool:
        return self._current_index < len(self._tasks)

    def peek_next(self) -> Optional[EditTask]:
        if self.has_next():
            return self._tasks[self._current_index]
        return None

    def pop_next(self) -> Optional[EditTask]:
        """Returns the next task and marks it as 'running'."""
        if not self.has_next():
            return None
        task = self._tasks[self._current_index]
        task.status = "running"
        total = len(self._tasks)
        current = self._current_index + 1
        logger.info(f"Starting task {current}/{total}: {task.subject}")
        if self.on_task_started:
            self.on_task_started(task, current, total)
        return task

    def advance(self):
        """Move internal pointer forward (call after popping)."""
        self._current_index += 1

    def mark_done(self, task_index: int, success: bool = True):
        """Marks a task as done or failed."""
        for t in self._tasks:
            if t.index == task_index:
                t.status = "done" if success else "failed"
                if self.on_task_finished:
                    self.on_task_finished(t)
                break

    def set_target(self, task_index: int, layer: int, point: int):
        """Called by Phase 3 to store the locked target on the task."""
        for t in self._tasks:
            if t.index == task_index:
                t.target_layer = layer
                t.target_point = point
                break

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    @property
    def all_tasks(self) -> List[EditTask]:
        return list(self._tasks)

    @property
    def total(self) -> int:
        return len(self._tasks)

    @property
    def completed(self) -> int:
        return sum(1 for t in self._tasks if t.status in ("done", "failed"))

    @property
    def current_position(self) -> int:
        """1-based position of the task currently running."""
        return self._current_index + 1

    def reset(self):
        """Full reset for a new pipeline run."""
        self._tasks.clear()
        self._current_index = 0

    def summary_text(self) -> str:
        lines = []
        for t in self._tasks:
            icon = {"queued": "⏳", "running": "⚙️", "done": "✅", "failed": "❌"}.get(t.status, "?")
            lines.append(f"{icon} [{t.index + 1}] {t.subject}: {t.wrong_value} -> {t.correct_value}")
        return "\n".join(lines)
