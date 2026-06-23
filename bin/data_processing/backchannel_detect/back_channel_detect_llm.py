from typing import List, Dict, TypedDict, Optional
from collections import Counter
import os
import re
import json
from pathlib import Path
import argparse

import torch
import torch.distributed as dist

from torch.utils.data import (
    Dataset,
    DataLoader,
    DistributedSampler,
)

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from tqdm import tqdm


# ============================================================
# Segment
# ============================================================

class SingleSegment(TypedDict):
    index: int
    start: float
    end: float
    text: str


# ============================================================
# Heuristic Filter
# ============================================================

class BackchannelFilter:

    DEFAULT_BACKCHANNEL_TOKENS = {
        "うん", "うーん", "うんうん",
        "はい", "はいはい",
        "え", "ええ", "へえ",
        "あ", "ああ", "あー",
        "ほう", "ふーん",
        "そう", "そうそう",
        "なるほど", "そっか",
    }

    def __init__(
        self,
        max_duration: float = 1.0,
        max_chars: int = 8,
        backchannel_tokens: Optional[set] = None,
    ):
        self.max_duration = max_duration
        self.max_chars = max_chars

        self.backchannel_tokens = (
            backchannel_tokens
            if backchannel_tokens is not None
            else self.DEFAULT_BACKCHANNEL_TOKENS
        )

    def simple_tokenize(self, text: str) -> List[str]:

        text = re.sub(r"[。、！？…]", " ", text)
        text = text.strip()

        tokens = []

        for chunk in text.split():

            if chunk in self.backchannel_tokens:
                tokens.append(chunk)

            elif "うん" in chunk:
                tokens.extend(["うん"] * chunk.count("うん"))

            elif "そう" in chunk:
                tokens.extend(["そう"] * chunk.count("そう"))

            elif "はい" in chunk:
                tokens.extend(["はい"] * chunk.count("はい"))

            else:
                tokens.append(chunk)

        return tokens

    def low_information_density(self, text: str) -> bool:

        tokens = self.simple_tokenize(text)

        if not tokens:
            return False

        total = len(tokens)
        unique = len(set(tokens))
        counts = Counter(tokens)

        backchannel_count = sum(
            c for t, c in counts.items()
            if t in self.backchannel_tokens
        )

        repeated_count = sum(
            c for _, c in counts.items()
            if c > 1
        )

        backchannel_ratio = backchannel_count / total
        unique_ratio = unique / total
        repetition_ratio = repeated_count / total

        return (
            backchannel_ratio >= 0.6
            and unique_ratio <= 0.6
            and repetition_ratio >= 0.3
        )

    def heuristic_backchannel(
        self,
        segment: SingleSegment,
    ) -> bool:

        text = segment["text"].strip().lower()
        duration = segment["end"] - segment["start"]

        if not text:
            return False

        if (
            text in self.backchannel_tokens
            and duration <= self.max_duration
        ):
            return True

        if duration <= 0.8 and len(text) <= 4:
            return True

        if (
            duration <= self.max_duration
            and len(text) <= self.max_chars
        ):
            return True

        return False


# ============================================================
# Utils
# ============================================================

def readlastline(path: str):
    with open(path, "rb") as f:
        f.seek(-2, 2) # avoid last \n
        while f.read(1) != b"\n":  
            f.seek(-2, 1)
        return f.readline()


def parse_jsonl(raw_lines: List[str]) -> List[SingleSegment]:

    segments = []

    for line in raw_lines:

        record = json.loads(line)

        text_stream = record[1]["content"][0]["text_stream"]

        text = "".join([x[2] for x in text_stream])

        segment = SingleSegment(
            start=text_stream[0][0],
            end=text_stream[-1][1],
            text=text,
        )

        segments.append(segment)

    return segments


# ============================================================
# Dataset
# ============================================================

class UtterancesDataset(Dataset):

    def __init__(
        self,
        annotation_dir: str,
        model_name: str = "Qwen/Qwen3.6-35B-A3B",
    ):
        self.prompt = """You are classifying short Japanese listener responses in dialogue.

    Choose exactly one label:

    continuer:
    A short response that mainly signals "I am listening / please continue".
    Examples: うん, はい, ええ, うんうん, はいはい, そうですね when it only acknowledges.

    assessment:
    A short response that expresses evaluation, surprise, understanding, emotion, or stance.
    Examples: へえ, なるほど, すごい, そうなんですね, そっか, ああそうですか.

    non_backchannel:
    Not a listener backchannel. This includes self-introduction, greeting, question, full answer, or content-bearing utterance.
    Examples: こんばんは, よろしくお願いします, ジュンです, 何年ぐらいですか.

    Output only one label from:
    continuer
    assessment
    non_backchannel"""
        self.folder_path = Path(annotation_dir)
    
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
        )
        
        self.backchannel_filter = BackchannelFilter()
        self.handles = []
        self.samples = []

        jsonl_files = list(self.folder_path.glob("*.jsonl"))
        
        for file in jsonl_files:
            seeks = json.loads(readlastline(file))
            self.handles.extend(zip([file] * len(seeks), seeks))
            
        self._load()
        
    def load_conversation(self, index):
        annotation_path, seek = self.handles[index]
        with open(annotation_path) as f:
            f.seek(seek)
            line = f.readline()
        line = json.loads(line)
        return line

    def _load(self):
        for i in range(len(self.handles)):
            conversation = self.load_conversation(i)
            text_stream = conversation[1]["content"][0]["text_stream"]

            text = "".join([x[2] for x in text_stream])

            segment = SingleSegment(
                index=i,
                start=text_stream[0][0],
                end=text_stream[-1][1],
                text=text,
            )

            if not segment["text"].strip():
                continue

            if (
                self.backchannel_filter.heuristic_backchannel(segment)
                or self.backchannel_filter.low_information_density(segment["text"].strip())
            ):
                self.samples.append(segment)

    def __getitem__(self, idx):
        seg = self.samples[idx]  
        return {
            "id": seg["index"],
            "text": seg["text"],
            "system_prompt": self.prompt,
            "user_prompt": f'Utterance: {seg["text"]}',
        }
        
    def __len__(self):
        return len(self.samples)
    
    def collate_fn(self, batch):

        prompts = []

        for x in batch:

            messages = [
                {
                    "role": "system",
                    "content": x["system_prompt"]  
                },
                {
                    "role": "user",
                    "content": x["user_prompt"],
                }
            ]

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )

            prompts.append(text)

        model_inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "ids": [x["id"] for x in batch],
            "texts": [x["text"] for x in batch],
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
        }

# ============================================================
# Classifier
# ============================================================

def parse_label(resp: str) -> str:
    resp = resp.strip().lower()

    first_line = resp.splitlines()[0].strip()
    first_line = first_line.replace(".", "").replace(":", "")

    valid = {"continuer", "assessment", "non_backchannel"}

    if first_line in valid:
        return first_line

    # fallback：按完整词匹配
    for label in ["non_backchannel", "assessment", "continuer"]:
        if re.search(rf"\b{re.escape(label)}\b", resp):
            return label

    return "non_backchannel"

class BackchannelLLMClassifier:

    LABELS = {
        "continuer",
        "assessment",
        "non_backchannel",
    }

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.6-35B-A3B",
        device: str = None,
        bf16: bool = True,
    ):

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding_side="left",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if bf16 else torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
        ).to(device)

        self.model.eval()


    @torch.inference_mode()
    def predict_batch(
        self,
        batch,
        max_new_tokens: int = 256,
    ):

        input_ids = batch["input_ids"].to(self.model.device)
        attention_mask = batch["attention_mask"].to(self.model.device)

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        batch_thinking_contents = []
        batch_contents = []
        
        prompt_len = input_ids.shape[1]
        
        for i in range(outputs.shape[0]):
            output_ids = outputs[i, prompt_len:].tolist()

            try:
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0

            thinking_content = self.tokenizer.decode(
                output_ids[:index],
                skip_special_tokens=True
            ).strip()

            content = self.tokenizer.decode(
                output_ids[index:],
                skip_special_tokens=True
            ).strip()
           
            batch_thinking_contents.append(thinking_content)
            batch_contents.append(content)
        
        labels = []
        
        for resp in batch_contents:

            resp = resp.strip().lower()
            print(f"Response: {resp}")
            
            pred = parse_label(resp)

            labels.append(pred)

        return labels


# ============================================================
# Distributed
# ============================================================

def setup_distributed():

    if "RANK" not in os.environ:
        return False

    dist.init_process_group("nccl")

    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    return True


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--model_name", type=str, default="Qwen/Qwen3.6-35B-A3B")
    args.add_argument("--annotation_dir", type=str, default="")
    args.add_argument("--output_dir", type=str, default="")
    args.add_argument("--barch_size", type=int, default=4)
    args = args.parse_args()
    model_name = args.model_name
    annotation_dir = args.annotation_dir
    output_dir = args.output_dir
    Path(output_dir).mkdir(exist_ok=True)
    safe_name = model_name.split('/')[-1]
    
    dist.init_process_group("nccl")

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    
    classifier = BackchannelLLMClassifier(model_name=model_name, device=device)

    dataset = UtterancesDataset(
        model_name=model_name,
        annotation_dir=annotation_dir
    )
    
    sampler = DistributedSampler(
                dataset, 
                num_replicas=world_size,
                rank=local_rank,
                shuffle=False,
                drop_last=False,
            )
  
    dataloader = DataLoader(
        dataset,
        batch_size=args.barch_size,
        sampler=sampler,
        num_workers=8,
        pin_memory=False,
        collate_fn=dataset.collate_fn,
    )
    results = []

    for batch in tqdm(dataloader):

        labels = classifier.predict_batch(batch)

        for text, label, id_ in zip(batch["texts"], labels, batch["ids"]):

            results.append({
                "id": id_,
                "text": text,
                "label": label,
            })

    output_path = Path(args.output_dir) / (safe_name + f"_results_rank{local_rank}.json")
    
    with open(output_path, "w") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=2,
        )
        
    dist.barrier()  # 等所有 rank 写完

    if local_rank == 0:

        all_results = []

        for rank in range(world_size):
            rank_path = Path(args.output_dir) / (safe_name + f"_results_rank{rank}.json")
            with open(rank_path) as f:
                all_results.extend(json.load(f))

        seen = set()
        deduped_results = []
        for r in all_results:
            if r["id"] not in seen:
                seen.add(r["id"])
                deduped_results.append(r)

        print(f"Total before dedup: {len(all_results)}, after: {len(deduped_results)}")
        
if __name__ == "__main__":
    main()
    