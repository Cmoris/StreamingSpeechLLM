import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import torch

@dataclass
class Token:
    t_start: float
    t_end:   float
    text:    str
    speaker: str          # "A" or "B"
    audio_path: str

@dataclass
class StreamEvent:
    start:   float
    end:     float
    speaker: str          # "A" | "B" | "SYSTEM"
    token:   str          # surface text or special token
    kind:    str          # "asr" | "ts" | "te" | "bc" | "silence" | "yield"

# ── Parsing ─────────────────────────────────────────────────────────────────

def parse_jsonl(raw_lines: list[str]) -> list[Token]:
    """
    Parse raw JSONL lines → flat, time-sorted list of Token objects.
    Speaker is inferred from the audio path (_a / _b).
    """
    tokens: list[Token] = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        user_content = record[0]["content"]

        # detect speaker from audio path
        audio_path = next(c["audio"] for c in user_content if c["type"] == "audio")
        speaker = "A" if "_a.wav" in audio_path else "B"

        # extract text_stream from assistant turn
        text_stream = record[1]["content"][0]["text_stream"]
        for (t_start, t_end, char) in text_stream:
            tokens.append(Token(t_start, t_end, char, speaker, audio_path))

    tokens.sort(key=lambda t: t.t_start)
    return tokens


# ── Segment grouping ─────────────────────────────────────────────────────────

def group_into_utterances(
    tokens: list[Token],
    pause_threshold: float = 1.0
) -> list[list[Token]]:
    """
    Group consecutive same-speaker tokens into utterances.
    A new utterance starts when:
      - the speaker changes, OR
      - the gap between tokens exceeds pause_threshold seconds.
    """
    if not tokens:
        return []

    utterances: list[list[Token]] = []
    current: list[Token] = [tokens[0]]

    for tok in tokens[1:]:
        prev = current[-1]
        gap  = tok.t_start - prev.t_end
        if tok.speaker != prev.speaker or gap > pause_threshold:
            utterances.append(current)
            current = [tok]
        else:
            current.append(tok)

    utterances.append(current)
    return utterances


# ── Turn-taking / turn-end detection ─────────────────────────────────────────

def detect_overlap(utt_a: list[Token], utt_b: list[Token]) -> bool:
    """True when two utterances temporally overlap (= turn-taking situation)."""
    a_start, a_end = utt_a[0].t_start,  utt_a[-1].t_end
    b_start, b_end = utt_b[0].t_start,  utt_b[-1].t_end
    return a_start < b_end and b_start < a_end