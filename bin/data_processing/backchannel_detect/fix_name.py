import shutil
from pathlib import Path

dir = "/n/work6/yizhang/Moris/zoom2025/finetune_labels/test_with_backchannel"
output_dir = "/n/work6/yizhang/Moris/zoom2025/pretrain_labels/test_with_backchannel"
Path(output_dir).mkdir(parents=True, exist_ok=True)
files = [x for x in Path(dir).glob("*.jsonl")]
for old_file in files:
    old_name = old_file.name
    new_name = "_".join(old_name.split("_")[1:])
    new_file = Path(output_dir) / new_name
    shutil.copy2(old_file, new_file)