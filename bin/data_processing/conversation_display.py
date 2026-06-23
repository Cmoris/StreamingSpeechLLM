import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

logging.basicConfig(
        filename="display.log",
        filemode="w",  # 每次运行覆盖；改成 "a" 是追加
        level=logging.INFO,
        format="%(asctime)s - %(message)s"
    )

def readlastline(path: str):
    with open(path, "rb") as f:
        f.seek(-2, 2) # avoid last \n
        while f.read(1) != b"\n":  
            f.seek(-2, 1)
        return f.readline()

def record_display(record: dict):
    print("=== Utterances ===")
    logging.info("=== Utterances ===")
    for u in record[0]["content"][0]["utterances"]:
        turn_flag = " ← TURN-TAKING" if u["is_turn_taking"] else ""
        bc_flag = " ← BACKCHANNEL" if u["is_back_channel"] else ""
        pause_flag = " ← PAUSE" if u["is_pause"] else ""
        silence_flag = " ← SILENCE" if u["is_silence"] else ""
        print(f"  [{u['speaker']}] {u['start']:.2f}-{u['end']:.2f}  {u['text']}{turn_flag}{bc_flag}{pause_flag}{silence_flag}")
        logging.info(f"  [{u['speaker']}] {u['start']:.2f}-{u['end']:.2f}  {u['text']}{turn_flag}{bc_flag}{pause_flag}{silence_flag}")

    # print("\n=== Stream (first 20 events) ===")
    # for ev in record[1]["content"][0]["text_stream"]:
    #     print(f"  {ev['start']:.3f} {ev['end']:.3f}  [{ev['speaker']}]  {ev['token']:6s}  ({ev['kind']})")
    # print("\n=== Training sequence ===")
    
def load_conversations(handles, index):
    annotation_path, seek = handles[index]
    with open(annotation_path) as f:
        f.seek(seek)
        line = f.readline()
    line = json.loads(line)
    return line
    
if __name__ == "__main__":
    dir = "/n/work6/yizhang/Moris/zoom2025/finetune_labels/l3_conv_train_with_backchannel"
    jsonl_files = [x for x in Path(dir).glob("*.jsonl")]
    handles = []
    for file in jsonl_files:
        seeks = json.loads(readlastline(file))
        handles.extend(zip([file] * len(seeks), seeks))
        
    for i in range(len(handles)):
        record = load_conversations(handles, index=i)
        record_display(record)