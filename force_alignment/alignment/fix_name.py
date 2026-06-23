import shutil
from pathlib import Path

dir = "/home/yizhang/Moris/StreamingSpeechLLM/force_alignment/results_all"
output_dir = "/home/yizhang/Moris/StreamingSpeechLLM/force_alignment/results_all_fixed_name"
Path(output_dir).mkdir(parents=True, exist_ok=True)
files = [x for x in Path(dir).glob("*.jsonl")]
for old_file in files:
    old_name = old_file.name
    new_name = "_".join(old_name.split("_")[2:])
    new_file = Path(output_dir) / new_name
    shutil.copy2(old_file, new_file)