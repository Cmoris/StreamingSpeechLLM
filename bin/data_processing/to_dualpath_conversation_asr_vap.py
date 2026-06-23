import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import torch

SWITCH_GAP = 0.7       # Different speaker response max gap (seconds) to be considered turn switch
RESUME_WINDOW = 1.5    # Original speaker resume window (seconds) to be considered holding the floor

# ── Special tokens ──────────────────────────────────────────────────────────
TS_TOKEN = "<ts>"   # turn-start / turn-taking
TE_TOKEN = "<te>"   # turn-end

# ── Data structures ─────────────────────────────────────────────────────────

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
    kind:    str          # "asr" | "ts" | "te"

# ── Parsing ─────────────────────────────────────────────────────────────────

def parse_jsonl(raw_lines: list[str]) -> list[Token]:
    """
    Parse raw JSONL lines → flat, time-sorted list of Token objects.
    Speaker is inferred from the audio path (_a / _b).
    """
    utterances = []
    for line in raw_lines:
        tokens: list[Token] = []
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
        utterances.append(tokens)
    
    return utterances


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


# ── Stream event generation ──────────────────────────────────────────────────

# def build_stream(
#     utterances: list[list[Token]],
# ) -> list[StreamEvent]:

#     events: list[StreamEvent] = []
#     interrupted_at: list[Optional[float]] = [None] * len(utterances)
#     turn_end_at: list[Optional[float]] = [None] * len(utterances)
#     for i in range(len(utterances) - 1):
#         curr_utt = utterances[i]
#         next_utt = utterances[i + 1]
#         if (next_utt[0].speaker != curr_utt[0].speaker):
#             # 打断时刻 = 下一speaker开始说话的时刻
#             interrupted_at[i] = next_utt[0].t_start
            
#         elif (next_utt[0].speaker == curr_utt[0].speaker
#                 and not detect_overlap(curr_utt, next_utt)
#                 and next_utt[0].t_start - curr_utt[-1].t_end > TURN_GAP_THRESHOLD):
#             # 同人连续说话但有停顿
#             turn_end_at[i] = curr_utt[-1].t_end
                

#     for i, utt in enumerate(utterances):
#         spk       = utt[0].speaker
#         interrupt = interrupted_at[i]
       
#         if interrupt is not None:
#             # ── 被打断：emit打断时刻之前的字符，然后插<ts> ────────────────
#             for tok in utt:
#                 if tok.t_start < interrupt:
#                     events.append(StreamEvent(
#                         start=tok.t_start, end=tok.t_end,
#                         speaker=spk, token=tok.text, kind="asr"
#                     ))
#             # <ts> 插在打断时刻
#             events.append(StreamEvent(
#                 start=interrupt, end=interrupt,
#                 speaker=spk, token=TS_TOKEN, kind="ts"
#             ))
#         elif turn_end_at[i] is not None:
#             # ── 正常结束：emit所有字符，句尾插<te> ───────────────────────
#             for tok in utt:
#                 events.append(StreamEvent(
#                     start=tok.t_start, end=tok.t_end,
#                     speaker=spk, token=tok.text, kind="asr"
#                 ))
#             t_end = utt[-1].t_end
#             events.append(StreamEvent(
#                 start=t_end, end=t_end,
#                 speaker=spk, token=TE_TOKEN, kind="te"
#             ))

#     # 全局按时间排序；同时刻：asr < ts/te
#     # kind_order = {"asr": 0, "ts": 1, "te": 1}
#     events.sort(key=lambda e: e.start)
#     return events

def build_stream(
    utterances: list[list[Token]],
) -> list[StreamEvent]:

    events: list[StreamEvent] = []

    for i, utt in enumerate(utterances):

        spk = utt[0].speaker
        curr_end = utt[-1].t_end

        # --------------------------------------------------
        # emit all ASR tokens
        # --------------------------------------------------
        for tok in utt:
            events.append(StreamEvent(
                start=tok.t_start,
                end=tok.t_end,
                speaker=spk,
                token=tok.text,
                kind="asr"
            ))

        # --------------------------------------------------
        # determine boundary token
        # --------------------------------------------------

        boundary_token = TE_TOKEN

        next_other = None
        next_same = None

        # 找未来最近的不同speaker / 同speaker utterance
        for j in range(i + 1, len(utterances)):

            future_utt = utterances[j]
            future_spk = future_utt[0].speaker

            if future_spk != spk and next_other is None:
                next_other = future_utt

            if future_spk == spk and next_same is None:
                next_same = future_utt

            if next_other is not None and next_same is not None:
                break

        # --------------------------------------------------
        # TURN SWITCH 判定
        # --------------------------------------------------

        if next_other is not None:

            other_start = next_other[0].t_start
            other_gap = other_start - curr_end

            # 不同speaker很快接话
            if 0 <= other_gap <= SWITCH_GAP:

                resumed = False

                # 原speaker是否很快resume
                if next_same is not None:

                    same_start = next_same[0].t_start

                    # 原speaker很快恢复说话
                    if same_start - curr_end <= RESUME_WINDOW:
                        resumed = True

                # 原speaker没有resume
                if not resumed:
                    boundary_token = TS_TOKEN

        # --------------------------------------------------
        # emit boundary token
        # --------------------------------------------------

        events.append(StreamEvent(
            start=curr_end,
            end=curr_end,
            speaker=spk,
            token=boundary_token,
            kind="ts" if boundary_token == TS_TOKEN else "te"
        ))

    # ------------------------------------------------------
    # global sort
    # ------------------------------------------------------

    events.sort(key=lambda e: e.start)

    return events


# ── Dataset record builder ───────────────────────────────────────────────────

def build_dataset_record(
    conversation_id: str,
    raw_lines: list[str],
) -> dict:
    """
    Full pipeline: raw JSONL lines → one dataset record.

    Output format
    -------------
    {
      "id": "000-02",
      "utterances": [                   # grouped utterances with metadata
        { "speaker": "A", "start": 7.3, "end": 8.8,
          "text": "よろしくおねがいします。",
          "is_turn_taking": false },
        ...
      ],
      "stream": [                        # dual-channel ordered stream
        { "time": 7.304, "speaker": "A", "token": "<ts>", "kind": "ts" },
        { "time": 7.304, "speaker": "A", "token": "よ",   "kind": "asr" },
        ...
        { "time": 8.508, "speaker": "A", "token": "<te>", "kind": "te" },
        ...
      ]
    }
    """
    utterances     = parse_jsonl(raw_lines)
    stream     = build_stream(utterances)

    # annotate turn-taking on utterances
    turn_taking: set[int] = set()
    for i in range(1, len(utterances)):
        prev, curr = utterances[i - 1], utterances[i]
        if curr[0].speaker != prev[0].speaker and detect_overlap(prev, curr):
            turn_taking.add(i)

    utt_records = []
    for i, utt in enumerate(utterances):
        utt_records.append({
            "speaker":        utt[0].speaker,
            "audio":          utt[0].audio_path,
            "start":          utt[0].t_start,
            "end":            utt[-1].t_end,
            "text":           "".join(t.text for t in utt),
            "is_turn_taking": i in turn_taking,
        })
    
    user_msg = {
        "role": "user",
        "content": [
            {"type":"utterances", "utterances": utt_records},
            {"type":"id", "id":conversation_id}
        ]
    }
    
    assistant_msg = {
        "role": "assistant",
        "content": [
            {"type": "text_stream", "text_stream": [vars(e) for e in stream]}
        ]
    }
    
    conversation = [user_msg, assistant_msg]

    return conversation


# ── Training sample formatter ────────────────────────────────────────────────

def to_training_sequence(record: dict) -> str:
    """
    Flatten the dual-channel stream into a single training sequence string.

    Format:  [A] <ts> よ ろ … <te>  [B] <ts> よ … <te>  …
    Overlapping speech is interleaved by real time.
    """
    parts = []
    for ev in record[1]["content"][0]["text_stream"]:
        prefix = f"[{ev['speaker']}]" if ev["kind"] == "ts" else ""
        parts.append(f"{prefix}{ev['token']}")
    return " ".join(p for p in parts if p)


def read_jsonl(jsonl_file: str, length: int):
    conversations = []
    with open(jsonl_file, "r") as f:
        lines = f.readlines()
        for i in range(length, len(lines), length):

            conversations.append(lines[i-length:i])
            
    return conversations


def save_conversations(conversations:list, output_path):
    with open(output_path, 'w') as f:
        lengths = []
        for conversation in conversations:
            line = json.dumps(conversation, ensure_ascii=True) + '\n'
            lengths.append(len(line))
            f.write(line)
        seeks = [0] + torch.tensor(lengths).cumsum(dim=-1).tolist()[:-1]
        f.write(json.dumps(seeks))
    
    print(f"Saved to: {output_path}")


def generate_dataset(jsonl_dir:str, output_dir:str, length:int):
    output = Path(output_dir)
    output.mkdir(exist_ok=True, parents=True)
    jsonl_files = [x for x in Path(jsonl_dir).glob("*.jsonl")]
    
    for jsonl_file in jsonl_files:
        conversations = read_jsonl(str(jsonl_file), length)
        conversation_id = jsonl_file.stem
        results = []
        for conv in conversations:
            record = build_dataset_record(conversation_id, conv)
            results.append(record)

        output_path = output / jsonl_file.name
        save_conversations(results, output_path)
            
            
def record_display(record: dict):
    print("=== Utterances ===")
    for u in record[0]["content"]:
        flag = " ← TURN-TAKING" if u["is_turn_taking"] else ""
        print(f"  [{u['speaker']}] {u['start']:.2f}-{u['end']:.2f}  {u['text']}{flag}")

    print("\n=== Stream (first 20 events) ===")
    for ev in record[1]["content"][0]["text_stream"][:20]:
        print(f"  {ev['start']:.3f} {ev['end']:.3f}  [{ev['speaker']}]  {ev['token']:6s}  ({ev['kind']})")
    breakpoint()
    print("\n=== Training sequence ===")
    print(to_training_sequence(record))

            
    

if __name__ == "__main__":
    # RAW = [
    #     '[{"role":"user","content":[{"type":"audio","audio":"/n/work1/muyun/Dataset/zoom2025/audios/A_all/000-02_a.wav"},{"type":"text","text":"transcrip it"}]},{"role":"assistant","content":[{"type":"text_stream","text_stream":[[7.304,7.324,"よ"],[7.324,7.345,"ろ"],[7.345,7.365,"し"],[7.365,7.386,"く"],[7.386,7.406,"お"],[7.406,7.426,"願"],[7.426,7.447,"い"],[7.447,7.467,"し"],[7.467,7.488,"ま"],[7.488,7.508,"す"],[7.508,8.793,"。"]]}]}]',
    #     '[{"role":"user","content":[{"type":"audio","audio":"/n/work1/muyun/Dataset/zoom2025/audios/B_all/000-02_b.wav"},{"type":"text","text":"transcrip it"}]},{"role":"assistant","content":[{"type":"text_stream","text_stream":[[8.763,8.783,"よ"],[8.783,8.804,"ろ"],[8.804,8.824,"し"],[8.824,8.845,"く"],[8.845,8.865,"お"],[8.865,8.886,"願"],[8.886,8.906,"い"],[8.906,8.926,"し"],[8.926,8.947,"ま"],[8.947,8.967,"す"],[8.967,10.315,"。"]]}]}]',
    #     '[{"role":"user","content":[{"type":"audio","audio":"/n/work1/muyun/Dataset/zoom2025/audios/A_all/000-02_a.wav"},{"type":"text","text":"transcrip it"}]},{"role":"assistant","content":[{"type":"text_stream","text_stream":[[12.693,12.714,"最"],[12.714,12.735,"近"],[12.735,13.223,"。"]]}]}]',
    #     '[{"role":"user","content":[{"type":"audio","audio":"/n/work1/muyun/Dataset/zoom2025/audios/A_all/000-02_a.wav"},{"type":"text","text":"transcrip it"}]},{"role":"assistant","content":[{"type":"text_stream","text_stream":[[13.624,13.645,"た"],[13.645,13.665,"お"],[13.665,13.686,"店"],[13.686,13.707,"と"],[13.707,13.727,"か"],[13.727,13.748,"あ"],[13.748,13.769,"り"],[13.769,13.789,"ま"],[13.789,13.81,"す"],[13.81,14.762,"？"]]}]}]',
    #     '[{"role":"user","content":[{"type":"audio","audio":"/n/work1/muyun/Dataset/zoom2025/audios/B_all/000-02_b.wav"},{"type":"text","text":"transcrip it"}]},{"role":"assistant","content":[{"type":"text_stream","text_stream":[[16.141,16.162,"最"],[16.162,16.183,"近"],[16.183,16.204,"は"],[16.204,16.942,"。"]]}]}]',
    #     '[{"role":"user","content":[{"type":"audio","audio":"/n/work1/muyun/Dataset/zoom2025/audios/A_all/000-02_a.wav"},{"type":"text","text":"transcrip it"}]},{"role":"assistant","content":[{"type":"text_stream","text_stream":[[17.161,17.187,"う"],[17.187,17.213,"ん"],[17.213,17.342,"。"]]}]}]',
    # ]

    # record = build_dataset_record("000-02", RAW, pause_threshold=2.0)

    # print("=== Utterances ===")
    # for u in record[0]["content"]:
    #     flag = " ← TURN-TAKING" if u["is_turn_taking"] else ""
    #     print(f"  [{u['speaker']}] {u['start']:.2f}-{u['end']:.2f}  {u['text']}{flag}")

    # print("\n=== Stream (first 20 events) ===")
    # for ev in record[1]["content"][0]["text_stream"][:20]:
    #     print(f"  {ev['start']:.3f} {ev['end']:.3f}  [{ev['speaker']}]  {ev['token']:6s}  ({ev['kind']})")

    # print("\n=== Training sequence ===")
    # print(to_training_sequence(record))

    # # save
    # with open("dataset_000-02.json", "w", encoding="utf-8") as f:
    #     json.dump(record, f, ensure_ascii=False, indent=2)
    # print("\nSaved → dataset_000-02.json")
    length = 2
    split = "test"
    jsonl_dir = f"/n/work6/yizhang/Moris/zoom2025/finetune_labels/{split}"
    output_dir = f"/n/work6/yizhang/Moris/zoom2025/finetune_labels/l{length}_{split}"
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    generate_dataset(jsonl_dir=jsonl_dir, output_dir=output_dir, length=length)