import json
from pathlib import Path
from collections import Counter

def count_backchannel_labels(jsonl_path: str):
    
    counter = Counter()
    total = 0
    
    with open(jsonl_path, encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines[:-1]:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            
            label = record[1]["content"][0].get("backchannel_label", "non_backchannel")
            counter[label] += 1
            total += 1
    
    print(f"Total: {total}\n")
    for label, count in counter.most_common():
        print(f"  {label}: {count} ({count / total * 100:.1f}%)")

if __name__ == "__main__":
    dir = "/home/yizhang/Moris/StreamingSpeechLLM/bin/data_processing/backchannel_detect/pretrain_train"
    jsonl_files = [x for x in Path(dir).glob("*.jsonl")]
    for file in jsonl_files:
        count_backchannel_labels(str(file))