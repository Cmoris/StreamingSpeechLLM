import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from infer_utils import (
    SPEAKERS,
    SPECIAL_TOKENS,
    TOKEN_PATTERN,
    collect_special_tokens,
    special_token_f1_sequence,
    speaker_cer,
)


DELIMITED_REPEAT_PATTERN = re.compile(r"([^、,，]+)([、,，]?)")


@dataclass
class Aggregate:
    turns: int = 0
    samples: set[Any] = field(default_factory=set)
    speaker_edits: Counter = field(default_factory=Counter)
    speaker_ref_chars: Counter = field(default_factory=Counter)
    speaker_cer_sum: Counter = field(default_factory=Counter)
    special_tp: int = 0
    special_fp: int = 0
    special_fn: int = 0
    special_precision_sum: float = 0.0
    special_recall_sum: float = 0.0
    special_f1_sum: float = 0.0
    pred_special_counts: Counter = field(default_factory=Counter)
    ref_special_counts: Counter = field(default_factory=Counter)
    matched_special_counts: Counter = field(default_factory=Counter)
    cleaned_turns: int = 0
    pred_chars_before_cleanup: int = 0
    pred_chars_after_cleanup: int = 0

    def add(self, record: dict[str, Any], metrics: dict[str, Any]) -> None:
        self.turns += 1
        if "sample_id" in record:
            self.samples.add(record["sample_id"])
        if record.get("_cleanup_changed", False):
            self.cleaned_turns += 1
        self.pred_chars_before_cleanup += record.get("_pred_chars_before_cleanup", 0)
        self.pred_chars_after_cleanup += record.get("_pred_chars_after_cleanup", 0)

        cer = metrics["speaker_cer"]
        for speaker in SPEAKERS:
            self.speaker_edits[speaker] += cer[speaker]["edits"]
            self.speaker_ref_chars[speaker] += cer[speaker]["ref_chars"]
            self.speaker_cer_sum[speaker] += cer[speaker]["cer"]

        special = metrics["special_token_f1"]
        self.special_tp += special["tp_lcs"]
        self.special_fp += special["fp"]
        self.special_fn += special["fn"]
        self.special_precision_sum += special["precision"]
        self.special_recall_sum += special["recall"]
        self.special_f1_sum += special["f1"]

        pred_counts = Counter(collect_special_tokens(record.get("pred", "")))
        ref_counts = Counter(collect_special_tokens(record.get("ref", "")))
        self.pred_special_counts.update(pred_counts)
        self.ref_special_counts.update(ref_counts)
        self.matched_special_counts.update(pred_counts & ref_counts)

    def summary(self) -> dict[str, Any]:
        total_edits = sum(self.speaker_edits.values())
        total_ref_chars = sum(self.speaker_ref_chars.values())
        special_pred = self.special_tp + self.special_fp
        special_ref = self.special_tp + self.special_fn
        precision = self.special_tp / max(special_pred, 1)
        recall = self.special_tp / max(special_ref, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        speaker_summary = {}
        for speaker in SPEAKERS:
            edits = self.speaker_edits[speaker]
            ref_chars = self.speaker_ref_chars[speaker]
            speaker_summary[speaker] = {
                "edits": edits,
                "ref_chars": ref_chars,
                "micro_cer": edits / max(ref_chars, 1),
                "macro_cer": self.speaker_cer_sum[speaker] / max(self.turns, 1),
            }

        token_summary = {}
        for token in SPECIAL_TOKENS:
            tp = self.matched_special_counts[token]
            pred = self.pred_special_counts[token]
            ref = self.ref_special_counts[token]
            token_precision = tp / max(pred, 1)
            token_recall = tp / max(ref, 1)
            token_f1 = (
                2 * token_precision * token_recall / max(token_precision + token_recall, 1e-8)
            )
            token_summary[token] = {
                "pred": pred,
                "ref": ref,
                "matched_count": tp,
                "precision": token_precision,
                "recall": token_recall,
                "f1": token_f1,
            }

        return {
            "turns": self.turns,
            "samples": len(self.samples),
            "repeat_cleanup": {
                "changed_turns": self.cleaned_turns,
                "pred_chars_before": self.pred_chars_before_cleanup,
                "pred_chars_after": self.pred_chars_after_cleanup,
                "pred_chars_removed": (
                    self.pred_chars_before_cleanup - self.pred_chars_after_cleanup
                ),
            },
            "speaker_cer": speaker_summary,
            "micro_avg_speaker_cer": {
                "edits": total_edits,
                "ref_chars": total_ref_chars,
                "cer": total_edits / max(total_ref_chars, 1),
            },
            "special_token_f1": {
                "tp_lcs": self.special_tp,
                "fp": self.special_fp,
                "fn": self.special_fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "macro_precision": self.special_precision_sum / max(self.turns, 1),
                "macro_recall": self.special_recall_sum / max(self.turns, 1),
                "macro_f1": self.special_f1_sum / max(self.turns, 1),
            },
            "special_token_counts": token_summary,
        }


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc


def evaluate_record(record: dict[str, Any]) -> dict[str, Any]:
    pred = record.get("pred")
    ref = record.get("ref")
    if not isinstance(pred, str) or not isinstance(ref, str):
        raise ValueError("Each record must contain string fields: pred, ref")
    return {
        "speaker_cer": speaker_cer(pred=pred, ref=ref),
        "special_token_f1": special_token_f1_sequence(pred=pred, ref=ref),
    }


def collapse_repeated_substrings(
    text: str,
    min_repeats: int,
    max_unit_chars: int,
    keep_repeats: int,
) -> str:
    if min_repeats <= keep_repeats or len(text) < min_repeats:
        return text

    output = []
    i = 0
    text_len = len(text)

    while i < text_len:
        best_unit = ""
        best_count = 0
        best_saved_chars = 0
        max_unit = min(max_unit_chars, (text_len - i) // min_repeats)

        for unit_len in range(1, max_unit + 1):
            unit = text[i : i + unit_len]
            count = 1
            pos = i + unit_len
            while text.startswith(unit, pos):
                count += 1
                pos += unit_len

            if count < min_repeats:
                continue

            saved_chars = unit_len * (count - keep_repeats)
            if saved_chars > best_saved_chars:
                best_unit = unit
                best_count = count
                best_saved_chars = saved_chars

        if best_count:
            output.append(best_unit * keep_repeats)
            i += len(best_unit) * best_count
        else:
            output.append(text[i])
            i += 1

    return "".join(output)


def collapse_delimited_repetitions(text: str, min_repeats: int) -> str:
    items = []
    pos = 0
    for match in DELIMITED_REPEAT_PATTERN.finditer(text):
        if match.start() != pos:
            return text
        items.append([match.group(1), match.group(2)])
        pos = match.end()

    if pos != len(text) or not items:
        return text

    output = []
    i = 0
    while i < len(items):
        phrase = items[i][0].strip()
        count = 1
        j = i + 1
        while j < len(items) and items[j][0].strip() == phrase:
            count += 1
            j += 1

        if phrase and count >= min_repeats:
            delimiter = items[j - 1][1] if j < len(items) else ""
            output.append(items[i][0] + delimiter)
        else:
            output.extend(segment + delimiter for segment, delimiter in items[i:j])
        i = j

    return "".join(output)


def clean_repeated_prediction(
    pred: str,
    min_repeats: int,
    max_unit_chars: int,
    keep_repeats: int,
) -> str:
    parts = TOKEN_PATTERN.split(pred)
    cleaned_parts = []

    for part in parts:
        if not part:
            continue
        if TOKEN_PATTERN.fullmatch(part):
            cleaned_parts.append(part)
        else:
            cleaned = collapse_delimited_repetitions(part, min_repeats=min_repeats)
            cleaned = collapse_repeated_substrings(
                cleaned,
                min_repeats=min_repeats,
                max_unit_chars=max_unit_chars,
                keep_repeats=keep_repeats,
            )
            cleaned_parts.append(cleaned)

    return "".join(cleaned_parts)


def prepare_record(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    pred = record.get("pred")
    if not isinstance(pred, str):
        return record

    if args.no_repeat_cleanup:
        cleaned_pred = pred
    else:
        cleaned_pred = clean_repeated_prediction(
            pred,
            min_repeats=args.repeat_min_repeats,
            max_unit_chars=args.repeat_max_unit_chars,
            keep_repeats=args.repeat_keep_repeats,
        )

    record["pred_raw"] = pred
    record["pred"] = cleaned_pred
    record["_cleanup_changed"] = cleaned_pred != pred
    record["_pred_chars_before_cleanup"] = len(pred)
    record["_pred_chars_after_cleanup"] = len(cleaned_pred)
    return record


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... <truncated {len(text) - max_chars} chars>"


def print_summary(title: str, summary: dict[str, Any]) -> None:
    print(f"\n[{title}]")
    print(f"turns={summary['turns']} samples={summary['samples']}")
    cleanup = summary["repeat_cleanup"]
    print(
        "repeat_cleanup="
        f"changed_turns={cleanup['changed_turns']} "
        f"pred_chars_removed={cleanup['pred_chars_removed']} "
        f"before={cleanup['pred_chars_before']} after={cleanup['pred_chars_after']}"
    )

    cer = summary["micro_avg_speaker_cer"]
    print(
        "speaker_cer_micro="
        f"{format_pct(cer['cer'])} edits={cer['edits']} ref_chars={cer['ref_chars']}"
    )
    for speaker in SPEAKERS:
        item = summary["speaker_cer"][speaker]
        print(
            f"  {speaker}: micro={format_pct(item['micro_cer'])} "
            f"macro={format_pct(item['macro_cer'])} "
            f"edits={item['edits']} ref_chars={item['ref_chars']}"
        )

    special = summary["special_token_f1"]
    print(
        "special_token_f1_micro="
        f"{format_pct(special['f1'])} "
        f"precision={format_pct(special['precision'])} "
        f"recall={format_pct(special['recall'])} "
        f"tp={special['tp_lcs']} fp={special['fp']} fn={special['fn']}"
    )
    print(
        "special_token_f1_macro="
        f"{format_pct(special['macro_f1'])} "
        f"precision={format_pct(special['macro_precision'])} "
        f"recall={format_pct(special['macro_recall'])}"
    )

    print("special_token_counts:")
    for token, item in summary["special_token_counts"].items():
        print(
            f"  {token}: pred={item['pred']} ref={item['ref']} "
            f"matched={item['matched_count']} "
            f"f1={format_pct(item['f1'])}"
        )


def collect_worst_examples(records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    ranked = []
    for record in records:
        metrics = evaluate_record(record)
        cer = metrics["speaker_cer"]["micro_avg"]["cer"]
        special_f1 = metrics["special_token_f1"]["f1"]
        ranked.append(
            {
                "sample_id": record.get("sample_id"),
                "turn_id": record.get("turn_id"),
                "rank": record.get("rank"),
                "source_file": record.get("_source_file"),
                "speaker_cer": cer,
                "special_f1": special_f1,
                "pred": record.get("pred", ""),
                "pred_raw": record.get("pred_raw", record.get("pred", "")),
                "cleanup_changed": record.get("_cleanup_changed", False),
                "ref": record.get("ref", ""),
            }
        )
    return sorted(ranked, key=lambda x: (x["speaker_cer"], 1.0 - x["special_f1"]), reverse=True)[
        :top_k
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze QwenAudio2 nonstreaming JSONL results."
    )
    parser.add_argument(
        "results",
        nargs="*",
        default=None,
        help="JSONL files or directories. Defaults to bin/QwenAudio2/nonstreaming_results.",
    )
    parser.add_argument(
        "--pattern",
        default="*.jsonl",
        help="Glob pattern used when a result path is a directory.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of worst examples to print. Set 0 to disable.",
    )
    parser.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write the machine-readable summary JSON.",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=240,
        help="Max pred/ref chars printed for each worst example. Set 0 for full text.",
    )
    parser.add_argument(
        "--no-repeat-cleanup",
        action="store_true",
        help="Disable repeated substring cleanup before metric calculation.",
    )
    parser.add_argument(
        "--repeat-min-repeats",
        type=int,
        default=3,
        help="Minimum consecutive repeats needed before a substring is collapsed.",
    )
    parser.add_argument(
        "--repeat-max-unit-chars",
        type=int,
        default=20,
        help="Maximum repeated substring length considered during cleanup.",
    )
    parser.add_argument(
        "--repeat-keep-repeats",
        type=int,
        default=1,
        help="Number of repeated units kept after cleanup.",
    )
    return parser.parse_args()


def resolve_result_paths(paths: list[str] | None, pattern: str) -> list[Path]:
    if not paths:
        paths = [str(Path(__file__).resolve().parent / "nonstreaming_results")]

    result_paths: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            result_paths.extend(sorted(path.glob(pattern)))
        else:
            result_paths.append(path)

    result_paths = [path for path in result_paths if path.is_file()]
    if not result_paths:
        raise FileNotFoundError("No result JSONL files found.")
    return sorted(result_paths)


def main() -> None:
    args = parse_args()
    paths = resolve_result_paths(args.results, args.pattern)

    overall = Aggregate()
    by_file: dict[str, Aggregate] = defaultdict(Aggregate)
    by_rank: dict[str, Aggregate] = defaultdict(Aggregate)
    records: list[dict[str, Any]] = []

    for path in paths:
        for _, record in iter_jsonl(path):
            record["_source_file"] = str(path)
            record = prepare_record(record, args)
            metrics = evaluate_record(record)
            rank = str(record.get("rank", "unknown"))

            overall.add(record, metrics)
            by_file[str(path)].add(record, metrics)
            by_rank[rank].add(record, metrics)
            records.append(record)

    output = {
        "files": [str(path) for path in paths],
        "overall": overall.summary(),
        "by_file": {name: agg.summary() for name, agg in by_file.items()},
        "by_rank": {rank: agg.summary() for rank, agg in sorted(by_rank.items())},
    }

    print_summary("overall", output["overall"])
    for name, summary in output["by_file"].items():
        print_summary(f"file: {name}", summary)
    for rank, summary in output["by_rank"].items():
        print_summary(f"rank: {rank}", summary)

    if args.top_k > 0:
        worst_examples = collect_worst_examples(records, args.top_k)
        output["worst_examples"] = worst_examples
        print(f"\n[worst_examples top_k={args.top_k}]")
        for idx, item in enumerate(worst_examples, start=1):
            print(
                f"{idx}. file={item['source_file']} sample={item['sample_id']} "
                f"turn={item['turn_id']} rank={item['rank']} "
                f"speaker_cer={format_pct(item['speaker_cer'])} "
                f"special_f1={format_pct(item['special_f1'])} "
                f"cleanup_changed={item['cleanup_changed']}"
            )
            print(f"   pred: {truncate_text(item['pred'], args.max_text_chars)}")
            if item["cleanup_changed"]:
                print(f"   raw : {truncate_text(item['pred_raw'], args.max_text_chars)}")
            print(f"   ref : {truncate_text(item['ref'], args.max_text_chars)}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nWrote JSON summary to {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
