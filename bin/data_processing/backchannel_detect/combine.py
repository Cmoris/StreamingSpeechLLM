import json
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm


def read_last_line(path: Path) -> str:
    """
    Read the last line of a file safely.
    """
    with open(path, "rb") as f:
        try:
            f.seek(-2, 2)

            while f.tell() > 0:
                if f.read(1) == b"\n":
                    break
                f.seek(-2, 1)

            return f.readline().decode("utf-8").strip()

        except OSError:
            # single line file
            f.seek(0)
            return f.readline().decode("utf-8").strip()


def load_conversation(annotation_path: Path, seek: int):
    """
    Load one jsonl sample from seek position.
    """
    with open(annotation_path, "r", encoding="utf-8") as f:
        f.seek(seek)
        line = f.readline()

    return json.loads(line)


def extract_text(sample) -> str:
    """
    Extract concatenated text from text_stream.
    """
    try:
        text_stream = sample[1]["content"][0]["text_stream"]
        return "".join(x[2] for x in text_stream).strip()

    except Exception:
        return ""


# =========================
# paths
# =========================

annotation_dir = Path(
    "/n/work6/yizhang/Moris/zoom2025/pretrain_labels/train"
)

merged_file = Path(
    "/home/yizhang/Moris/StreamingSpeechLLM/bin/data_processing/backchannel_detect/merged_sorted.json"
)

output_path = Path(
    "/home/yizhang/Moris/StreamingSpeechLLM/bin/data_processing/backchannel_detect/combined.jsonl"
)


# =========================
# build handles
# =========================

jsonl_files = sorted(annotation_dir.glob("*.jsonl"))

handles: List[Tuple[Path, int]] = []

for file in jsonl_files:
    seeks = json.loads(read_last_line(file))

    handles.extend(
        [(file, seek) for seek in seeks]
    )

print(f"Loaded {len(handles)} handles")


# =========================
# load merged labels
# =========================

with open(merged_file, "r", encoding="utf-8") as f:
    merged_data = json.load(f)

# id -> item
merged_map = {
    item["id"]: item
    for item in merged_data
}


# =========================
# merge
# =========================

results = []

for i, (annotation_path, seek) in tqdm(enumerate(handles)):

    sample = load_conversation(annotation_path, seek)

    text = extract_text(sample)

    merged_item = merged_map.get(i)

    # default label
    backchannel_label = "non_backchannel"

    if merged_item is not None:

        merged_text = merged_item["text"].strip()

        # safety check
        if merged_text == text:
            backchannel_label = merged_item["label"]

        else:
            print(
                f"[WARNING] text mismatch at id={i}\n"
                f"merged : {merged_text}\n"
                f"sample : {text}\n"
            )

    # append annotation
    sample[1]["content"].append({
        "type": "backchannel",
        "backchannel": backchannel_label,
    })

    results.append(sample)

    # if i % 1000 == 0:
    #     print(f"Processed {i}/{len(handles)}")


# =========================
# save
# =========================

with open(output_path, "w", encoding="utf-8") as f:
    for result in results:
        f.write(
            json.dumps(result, ensure_ascii=False) + "\n"
        )

print(f"Saved to: {output_path}")