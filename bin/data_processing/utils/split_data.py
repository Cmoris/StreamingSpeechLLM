import json
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

def copy_files(src_files: list, dst_dir: str):
    Path(dst_dir).mkdir(parents=True, exist_ok=True)
    for src_file in src_files:
        dst_file = Path(dst_dir) / src_file.name
        shutil.copy2(src_file, dst_file)
    
if __name__ == "__main__":
    # all_lables_dir = "/n/work6/yizhang/Moris/zoom2025/alignment_all_labels"
    labels_dir = "/n/work6/yizhang/Moris/zoom2025/alignment_labels"

    # all_labels = [x for x in Path(all_lables_dir).glob("*.jsonl")]
    labels = [x for x in Path(labels_dir).glob("*.jsonl")]

    # pretrain_labels = [x for x in all_labels if (Path(labels_dir) / x.name) not in labels]
    finetune_labels = labels
    
    # pretrain_train, pretrain_test = train_test_split(pretrain_labels, test_size=0.2, random_state=42)
    finetune_train, finetune_test = train_test_split(finetune_labels, test_size=0.2, random_state=42)
    
    # copy_files(pretrain_train, "/n/work6/yizhang/Moris/zoom2025/pretrain_labels/train")
    # copy_files(pretrain_test, "/n/work6/yizhang/Moris/zoom2025/pretrain_labels/test")
    copy_files(finetune_train, "/n/work6/yizhang/Moris/zoom2025/finetune_labels/train")
    copy_files(finetune_test, "/n/work6/yizhang/Moris/zoom2025/finetune_labels/test")