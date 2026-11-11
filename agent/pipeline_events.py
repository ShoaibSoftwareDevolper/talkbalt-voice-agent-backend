"""Live call signals for the browser's pipeline view and the logs: VAD state, per-turn latency and
the latest LLM request's token usage. Only states and numbers are sent, never what was said."""

import asyncio
import json
import logging

from livekit import rtc
from livekit.agents import ConversationItemAddedEvent, UserStateChangedEvent
from livekit.agents.llm import ChatMessage
from livekit.agents.metrics import LLMMetrics

from call_history import turn_metrics

logger = logging.getLogger("talkbalt-agent")

PIPELINE_TOPIC = "talkbalt.pipeline"


class PipelinePublisher:
    """One small JSON message per event, on a text stream with topic PIPELINE_TOPIC."""

    def __init__(self, room: rtc.Room) -> None:
        self._room = room
        self._tasks: set[asyncio.Task[rtc.TextStreamInfo]] = set()

    def on_user_state(self, ev: UserStateChangedEvent) -> None:
        self._send({"type": "user_state", "state": ev.new_state})

    def on_llm_metrics(self, m: LLMMetrics) -> None:
        self._send(
            {
                "type": "llm_usage",
                "prompt_tokens": m.prompt_tokens,
                "cached_tokens": m.prompt_cached_tokens,
                "completion_tokens": m.completion_tokens,
                "reasoning_tokens": m.reasoning_tokens,
            }
        )

    def on_item_added(self, ev: ConversationItemAddedEvent) -> None:
        if isinstance(ev.item, ChatMessage) and ev.item.role in ("user", "assistant"):
            role = "user" if ev.item.role == "user" else "agent"
            metrics = turn_metrics(ev.item)
            logger.info("latency %s turn: %s", role, " ".join(f"{k}={v if v is not None else 'n/a'}" for k, v in metrics.items()))
            self._send({"type": "latency", "role": role, **metrics})

    def _send(self, payload: dict[str, object]) -> None:
        if not self._room.isconnected():
            return
        task = asyncio.create_task(self._room.local_participant.send_text(json.dumps(payload), topic=PIPELINE_TOPIC))
        self._tasks.add(task)
        task.add_done_callback(self._on_sent)

    def _on_sent(self, task: asyncio.Task[rtc.TextStreamInfo]) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.debug("pipeline event not sent: %s", task.exception())
