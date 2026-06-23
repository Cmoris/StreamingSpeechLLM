import json
from pathlib import Path
from tqdm import tqdm

def txt_to_jsonl(input_dir: str, output_dir: str):
    Path(output_dir).mkdir(exist_ok=True)
    input_dir = Path(input_dir)

    txt_files = sorted(input_dir.glob("*.txt"))

    for txt_file in tqdm(txt_files, desc="Processing txt files"):
        audio_path = txt_file.with_suffix(".wav")

        if not audio_path.exists():
            print(f"Missing audio: {audio_path}")
            continue
        
        output_jsonl = Path(output_dir) / txt_file.with_suffix(".jsonl").name
        
        with open(output_jsonl, "w", encoding="utf-8") as fout:
            audio_path = txt_file.with_suffix(".wav")

            if not audio_path.exists():
                print(f"Missing audio: {audio_path}")
                continue

            with open(txt_file, "r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()

                    if not line:
                        continue

                    parts = line.split("\t")

                    if len(parts) != 3:
                        continue

                    start, end, text = parts

                    if text.strip() == "●":
                        continue
                    
                    if text.strip() == "〓":
                        continue
                    
                    text = text.replace("●", "")
                    text = text.replace("〓", "")
                    text = text.replace("。", " ")
                    
                    # 跳过空转写
                    if text.strip() == "":
                        continue
                    
                    duration = float(end) - float(start)

                    item = {
                        "audio_path": str(audio_path),
                        "start": float(start),
                        "end": float(end),
                        "duration": duration,
                        "text": text.strip(),
                    }

                    fout.write(
                        json.dumps(
                            item,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )


if __name__ == "__main__":
    txt_to_jsonl(
        input_dir="/n/work6/yizhang/Moris/zoom2025/audios/B_gd",
        output_dir="/n/work6/yizhang/Moris/zoom2025/asr_labels",
    )