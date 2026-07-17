#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Tests for ParallelTextAggregator.

Verifies that streamed tokens are grouped back into sentences across the three
parallel channels (tts / llm / user-facing), that a sentence is emitted only
once the *next* sentence's first token supplies the lookahead (no over-grouping),
and that flush/interruption behave correctly.
"""

import unittest

from pipecat.utils.text.parallel_text_aggregator import ParallelTextAggregator


async def _feed(agg, *tokens, mirror=True):
    """Feed same-text tokens through the aggregator, collecting emitted sentences.

    When ``mirror`` is True each token is used for all three channels. Returns the
    list of (tts, llm, user) tuples emitted across the whole run.
    """
    out = []
    for t in tokens:
        async for s in agg.aggregate(t, t, t):
            out.append((s.tts_text, s.llm_text, s.user_facing_text))
    return out


class TestParallelTextAggregator(unittest.IsolatedAsyncioTestCase):
    async def test_no_boundary_yields_nothing(self):
        agg = ParallelTextAggregator()
        out = await _feed(agg, "Hi", " there")
        self.assertEqual(out, [])

    async def test_terminal_token_alone_does_not_emit(self):
        # A sentence-ending token needs lookahead (the next sentence's first
        # token) before it is confirmed, so "!" alone emits nothing.
        agg = ParallelTextAggregator()
        out = await _feed(agg, "Hi", " there", "!")
        self.assertEqual(out, [])

    async def test_next_sentence_token_confirms_first_sentence_only(self):
        # The lookahead token (" How") confirms "Hi there!" — but must NOT be
        # folded into it (that was the over-grouping bug that broke progress).
        agg = ParallelTextAggregator()
        out = await _feed(agg, "Hi", " there", "!", " How")
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][0], "Hi there!")

    async def test_multi_sentence_stream_yields_each_cleanly(self):
        agg = ParallelTextAggregator()
        tokens = ["Hi", " there", "!", " How", " are", " you", "?"]
        emitted = await _feed(agg, *tokens)
        # Only the first sentence is confirmed mid-stream; the second waits.
        self.assertEqual([e[0] for e in emitted], ["Hi there!"])
        # The trailing sentence comes out on flush.
        f = await agg.flush()
        self.assertIsNotNone(f)
        self.assertEqual(f.tts_text, " How are you?")

    async def test_flush_emits_trailing_partial_sentence(self):
        # A response ending with no terminal punctuation still flushes what remains.
        agg = ParallelTextAggregator()
        await _feed(agg, "Just", " a", " fragment")
        f = await agg.flush()
        self.assertIsNotNone(f)
        self.assertEqual(f.user_facing_text, "Just a fragment")

    async def test_flush_returns_none_when_empty(self):
        agg = ParallelTextAggregator()
        self.assertIsNone(await agg.flush())

    async def test_flush_returns_none_for_whitespace_only(self):
        agg = ParallelTextAggregator()
        await _feed(agg, "   ")
        self.assertIsNone(await agg.flush())

    async def test_handle_interruption_resets(self):
        agg = ParallelTextAggregator()
        await _feed(agg, "Hi", " there")
        await agg.handle_interruption()
        self.assertIsNone(await agg.flush())
        # Behaves fresh afterwards.
        out = await _feed(agg, "Bye", "!", " Next")
        self.assertEqual([e[0] for e in out], ["Bye!"])

    async def test_channels_stay_token_aligned_under_transform(self):
        # tts_text differs from user-facing/llm (a simulated transform), but
        # because emission is token-aligned (whole tokens only), the three
        # channels correspond to the same sentence span.
        agg = ParallelTextAggregator()
        out = []
        # (tts, llm, user) per token: "$5" is spoken as "five dollars".
        triples = [
            ("five dollars", "$5", "$5"),
            (".", ".", "."),
            (" Thanks", " Thanks", " Thanks"),
        ]
        for tts, llm, user in triples:
            async for s in agg.aggregate(tts, llm, user):
                out.append((s.tts_text, s.llm_text, s.user_facing_text))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0], ("five dollars.", "$5.", "$5."))


if __name__ == "__main__":
    unittest.main()
