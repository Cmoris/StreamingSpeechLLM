import json
import argparse
from pathlib import Path


def make_relative_from_marker(path: str, marker: str = "zoom2025") -> str:
    """
    Keep path from marker onward.

    Example:
        /n/work6/yizhang/Moris/zoom2025/audios/A_gd/2_a.wav
    ->  zoom2025/audios/A_gd/2_a.wav
    """
    if not isinstance(path, str):
        return path

    idx = path.find(marker)
    if idx == -1:
        return path

    return path[idx:]


def process_record(record, marker: str = "zoom2025"):
    """
    record is expected to be a list like:
    [
      {"role": "user", "content": [...]},
      {"role": "assistant", "content": [...]}
    ]

    Only modify:
      user/content/type=utterances/utterances/*/audio
    """
    if not isinstance(record, list):
        return record

    for msg in record:
        if not isinstance(msg, dict):
            continue

        content = msg.get("content", [])
        if not isinstance(content, list):
            continue

        for item in content:
            if not isinstance(item, dict):
                continue

            if item.get("type") != "utterances":
                continue

            utterances = item.get("utterances", [])
            if not isinstance(utterances, list):
                continue

            for utt in utterances:
                if not isinstance(utt, dict):
                    continue

                if "audio" in utt:
                    utt["audio"] = make_relative_from_marker(
                        utt["audio"],
                        marker=marker,
                    )

    return record


def process_jsonl_file(input_path: Path, output_path: Path, marker: str = "zoom2025"):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_lines = 0
    num_failed = 0

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:

        for line_idx, line in enumerate(fin, start=1):
            line = line.rstrip("\n")

            if not line.strip():
                fout.write("\n")
                continue

            try:
                record = json.loads(line)
                record = process_record(record, marker=marker)

                fout.write(
                    json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                num_lines += 1

            except Exception as e:
                num_failed += 1
                print(f"[WARN] Failed to process {input_path} line {line_idx}: {e}")

                # 保守起见，坏行原样写回
                fout.write(line + "\n")

    return num_lines, num_failed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory containing jsonl files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save processed jsonl files.",
    )
    parser.add_argument(
        "--marker",
        type=str,
        default="zoom2025",
        help="Keep audio path from this marker onward.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.jsonl",
        help="File pattern, default: *.jsonl",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    jsonl_files = sorted(input_dir.rglob(args.pattern))

    print(f"Found {len(jsonl_files)} jsonl files.")

    total_lines = 0
    total_failed = 0

    for input_path in jsonl_files:
        rel_path = input_path.relative_to(input_dir)
        output_path = output_dir / rel_path

        num_lines, num_failed = process_jsonl_file(
            input_path=input_path,
            output_path=output_path,
            marker=args.marker,
        )

        total_lines += num_lines
        total_failed += num_failed

        print(
            f"[OK] {input_path} -> {output_path} "
            f"lines={num_lines}, failed={num_failed}"
        )

    print("Done.")
    print(f"Total lines processed: {total_lines}")
    print(f"Total failed lines: {total_failed}")


if __name__ == "__main__":
    main()