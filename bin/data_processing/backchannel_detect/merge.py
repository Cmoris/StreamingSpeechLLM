import json
from pathlib import Path
from typing import List, Dict, TypedDict, Optional

from back_channel_detect_llm import UtterancesDataset

# ============================================================
# Merge labels back (新增函数)
# ============================================================

def merge_labels_back(
    dataset: UtterancesDataset,
    results: List[Dict],
    output_dir: str,
):
    """
    将分类结果合并回原始 conversation，
    在 content[0] 里追加 backchannel_label 字段。
    """

    # id -> label 的映射
    id_to_label = {r["id"]: r["label"] for r in results}

    # 按原始 jsonl 文件分组，避免重复开关文件
    # handles[i] = (file_path, seek)
    from collections import defaultdict
    file_to_updates: Dict[Path, List[tuple]] = defaultdict(list)

    for idx, label in id_to_label.items():
        file_path, seek = dataset.handles[idx]
        file_to_updates[file_path].append((seek, idx, label))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for file_path, updates in file_to_updates.items():

        # 读取该文件所有行
        lines = file_path.read_text(encoding="utf-8").splitlines()

        # seek -> label (seek 是字节偏移，需要先建立 seek->行号 映射)
        # 重新建立 seek 映射
        seek_to_lineno = {}
        byte_offset = 0
        for lineno, line in enumerate(lines):
            seek_to_lineno[byte_offset] = lineno
            byte_offset += len(line.encode("utf-8")) + 1  # +1 for \n

        # 解析所有行为 dict（懒加载，只改需要的）
        parsed_lines = [json.loads(l) for l in lines]

        for seek, idx, label in updates:
            lineno = seek_to_lineno.get(seek)
            if lineno is None:
                print(f"[WARN] seek={seek} not found in {file_path}")
                continue

            record = parsed_lines[lineno]

            # 将 label 注入到 content[0] 中
            try:
                record[1]["content"][0]["backchannel_label"] = label
            except (IndexError, KeyError, TypeError) as e:
                print(f"[WARN] Failed to inject label at line {lineno}: {e}")
                continue

            parsed_lines[lineno] = record

        # 写出到新文件（不覆盖原始数据）
        out_file = output_path / file_path.name
        with open(out_file, "w", encoding="utf-8") as f:
            for record in parsed_lines:
                f.write(json.dumps(record, ensure_ascii=True) + "\n")

        print(f"Written: {out_file}")

def merge():
    json_dir = "/home/yizhang/Moris/StreamingSpeechLLM/bin/data_processing/backchannel_detect/finetune_train"
    json_files = [x for x in Path(json_dir).glob("*.json")]

    all_results = []

    for file in json_files:
        with open(file) as f:
            all_results.extend(json.load(f))

    seen = set()
    deduped_results = []
    for r in all_results:
        if r["id"] not in seen:
            seen.add(r["id"])
            deduped_results.append(r)

    print(f"Total before dedup: {len(all_results)}, after: {len(deduped_results)}")
    model_name = "Qwen/Qwen3-8B"
    annotation_dir = "/n/work6/yizhang/Moris/zoom2025/finetune_labels/train"
    output_dir = "/n/work6/yizhang/Moris/zoom2025/finetune_labels/train_with_backchannel"
    
    dataset = UtterancesDataset(
        model_name=model_name,
        annotation_dir=annotation_dir
    )
    
    merge_labels_back(
        dataset=dataset,
        results=deduped_results,
        output_dir=output_dir,
    )

if __name__ == "__main__":
    merge()