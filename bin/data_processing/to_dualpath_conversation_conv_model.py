import json
from pathlib import Path
from dataclasses import dataclass, field

import torch

from build_conversational_state import (find_next_other_speaker,
                                        find_next_same_speaker,
                                        is_backchannel,
                                        is_pause,
                                        is_turn_switch,
                                        is_silence,
                                        is_turn_end,
                                        build_stream,
                                        utt_start, utt_end  )

# ── Data structures ─────────────────────────────────────────────────────────

@dataclass
class Token:
    t_start: float
    t_end:   float
    text:    str
    bc_label: str
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
        backchannel_label = record[1]["content"][0].get("backchannel_label", "non_backchannel")
        for (t_start, t_end, char) in text_stream:
            tokens.append(Token(t_start, t_end, char, backchannel_label, speaker, audio_path))
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
    utterances =    parse_jsonl(raw_lines)
    stream     =    build_stream(utterances)
    # annotate turn-taking on utterances
    turn_taking: set[int] = set()
    turn_ending: set[int] = set()
    pause : set[int] = set()
    silence: set[int] = set()
    bc: set[int] = set()
    
    for i in range(len(utterances)):
        next_same = find_next_same_speaker(
            utterances,
            i,
        )

        next_other = find_next_other_speaker(
            utterances,
            i,
        )
        
        turn_taking.add(i) if is_turn_switch(
            curr_utt=utterances[i],
            next_other=next_other,
            next_same=next_same
        ) else None

        next_same = find_next_same_speaker(
            utterances,
            i,
        )
        
        pause.add(i) if is_pause(
            curr_utt=utterances[i],
            next_same=next_same
        ) else None
     
        silence.add(i) if is_silence(
            curr_utt=utterances[i],
            utterances=utterances,
            curr_idx=i
        ) else None
        
        if next_other is not None:
            bc.add(i) if is_backchannel(
                curr_utt=utterances[i],
                next_utt=next_other,
                next_same=next_same
            ) else None
            
        turn_ending.add(i) if is_turn_end(
            curr_utt=utterances[i],
            next_same=next_same,
            next_other=next_other
        ) else None
    
    utt_records = []
    for i, utt in enumerate(utterances):
        utt_records.append({
            "speaker":        utt[0].speaker,
            "audio":          utt[0].audio_path,
            "start":          utt[0].t_start,
            "end":            utt[-1].t_end,
            "text":           "".join(t.text for t in utt),
            "is_turn_taking": i in turn_taking,
            "is_pause":       i in pause,
            "is_silence":     i in silence,
            "is_back_channel": i in bc,
            "is_turn_ending": i in turn_ending
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


def read_jsonl(jsonl_file: str, length: int):
    conversations = []
    with open(jsonl_file, "r") as f:
        lines = f.readlines()
        for i in range(length, len(lines)-1, length):

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
            # record_display(record)
            results.append(record)

        output_path = output / jsonl_file.name
        save_conversations(results, output_path)
            
            
def record_display(record: dict):
    print("=== Utterances ===")
    for u in record[0]["content"][0]["utterances"]:
        flag = " ← TURN-TAKING" if u["is_turn_taking"] else ""
        print(f"  [{u['speaker']}] {u['start']:.2f}-{u['end']:.2f}  {u['text']}{flag}")

    print("\n=== Stream (first 20 events) ===")
    for ev in record[1]["content"][0]["text_stream"]:
        print(f"  {ev['start']:.3f} {ev['end']:.3f}  [{ev['speaker']}]  {ev['token']:6s}  ({ev['kind']})")


def generate_train_dataset_for_pretrain(length: int=2):
    split = "train_with_backchannel"
    jsonl_dir = f"/ctd/Works/m-wu/Datasets/zoom2025/pretrain_labels/{split}"
    output_dir = f"/ctd/Works/m-wu/Datasets/zoom2025/pretrain_labels/l{length}_conv_{split}"
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    generate_dataset(jsonl_dir=jsonl_dir, output_dir=output_dir, length=length)
        
            
def generate_test_dataset_for_pretrain(length: int=2):
    split = "test_with_backchannel"
    jsonl_dir = f"/ctd/Works/m-wu/Datasets/zoom2025/pretrain_labels/{split}"
    output_dir = f"/ctd/Works/m-wu/Datasets/zoom2025/pretrain_labels/l{length}_conv_{split}"
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    generate_dataset(jsonl_dir=jsonl_dir, output_dir=output_dir, length=length)
    
def generate_train_dataset_for_finetune(length: int=2):
    split = "train_with_backchannel"
    jsonl_dir = f"/ctd/Works/m-wu/Datasets/zoom2025/finetune_labels/{split}"
    output_dir = f"/ctd/Works/m-wu/Datasets/zoom2025/finetune_labels/l{length}_conv_{split}"
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    generate_dataset(jsonl_dir=jsonl_dir, output_dir=output_dir, length=length)
    
def generate_test_dataset_for_finetune(length: int=2):
    split = "test_with_backchannel"
    jsonl_dir = f"/ctd/Works/m-wu/Datasets/zoom2025/finetune_labels/{split}"
    output_dir = f"/ctd/Works/m-wu/Datasets/zoom2025/finetune_labels/l{length}_conv_{split}"
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    generate_dataset(jsonl_dir=jsonl_dir, output_dir=output_dir, length=length)

if __name__ == "__main__":
    length = 10
    # generate_test_dataset_for_pretrain(length)
    # generate_train_dataset_for_pretrain(length)
    generate_train_dataset_for_finetune(length)
    generate_test_dataset_for_finetune(length)
