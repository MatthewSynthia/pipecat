#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Parallel sentence aggregation across TTS, LLM, and user-facing text channels."""

from collections.abc import AsyncIterator
from dataclasses import dataclass

from pipecat.utils.text.base_text_aggregator import AggregationType
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator


@dataclass
class ParallelAggregation:
    """One completed sentence, in each of the three parallel text channels.

    Parameters:
        tts_text: The sentence as sent to the TTS service (post-filter/transform).
        llm_text: The sentence in the original LLM text (with any pattern delimiters).
        user_facing_text: The sentence as shown to the user (no TTS tags/transforms).
    """

    tts_text: str
    llm_text: str
    user_facing_text: str


class ParallelTextAggregator:
    """Groups streamed tokens back into sentences across three parallel channels.

    Used by :class:`~pipecat.utils.context.aggregated_frame_sequencer.AggregatedFrameSequencer`
    when a TTS service streams tokens individually (``TextAggregationMode.TOKEN``)
    but still needs whole-sentence units for word-timestamp tracking and RTVI
    progress. Each token contributes one part to each of the three channels
    (tts / llm / user-facing); sentence completion is driven by the **TTS text**
    and completes all three together.

    Boundary timing: a sentence-ending boundary is only *confirmed* by lookahead —
    the first non-whitespace character of the *next* sentence, which arrives as
    the next token. So the token that makes the underlying
    :class:`SimpleTextAggregator` yield "Hi there!" is " I'm" (the next sentence's
    first token). This aggregator therefore emits the text accumulated **before**
    that triggering token and lets the triggering token begin the next sentence's
    buffer. Because LLMs emit punctuation as its own token and the following word
    as the next token, this slices cleanly at token boundaries and never inside a
    token, so the three channels stay aligned even when a transform makes them
    differ in length.

    Example::

        agg = ParallelTextAggregator()
        async for _ in agg.aggregate("Hi", "Hi", "Hi"): ...          # nothing yet
        async for _ in agg.aggregate("!", "!", "!"): ...             # nothing yet (needs lookahead)
        async for sentence in agg.aggregate(" How", " How", " How"): # yields "Hi!"
            ...  # sentence.tts_text == "Hi!"
    """

    def __init__(self):
        """Initialize the aggregator with empty channels."""
        # A plain SENTENCE-mode aggregator drives boundary detection on the TTS
        # text. The TTS text is already post-transform, so tag/pattern-aware
        # boundary rules are not needed here.
        self._aggregator = SimpleTextAggregator(aggregation_type=AggregationType.SENTENCE)
        self._reset()

    def _reset(self):
        # Tokens accumulated since the last emitted sentence, per channel.
        self._tts = ""
        self._llm = ""
        self._user = ""

    async def aggregate(
        self, tts_text: str, llm_text: str, user_facing_text: str
    ) -> AsyncIterator[ParallelAggregation]:
        """Feed one token (all three channels) and yield any completed sentence.

        Args:
            tts_text: The token as sent to the TTS service.
            llm_text: The token in the original LLM text.
            user_facing_text: The token as shown to the user.

        Yields:
            A :class:`ParallelAggregation` for each sentence completed by this
            token (at most one). The yielded sentence is the text accumulated
            *before* this token; this token starts the next sentence's buffer.
        """
        boundary = False
        async for _ in self._aggregator.aggregate(tts_text):
            boundary = True

        if boundary and self._tts:
            yield ParallelAggregation(self._tts, self._llm, self._user)
            self._tts = self._llm = self._user = ""

        # Plain concatenation: LLM tokens already carry their own spacing.
        self._tts += tts_text
        self._llm += llm_text
        self._user += user_facing_text

    async def flush(self) -> ParallelAggregation | None:
        """Emit any trailing partial sentence at end of turn.

        Returns:
            A :class:`ParallelAggregation` for the accumulated-but-unemitted text,
            or ``None`` when nothing substantive is buffered.
        """
        await self._aggregator.flush()
        if self._user.strip():
            result = ParallelAggregation(self._tts, self._llm, self._user)
            self._reset()
            return result
        return None

    async def handle_interruption(self):
        """Discard all buffered text (called on interruption/reset)."""
        await self._aggregator.handle_interruption()
        self._reset()
