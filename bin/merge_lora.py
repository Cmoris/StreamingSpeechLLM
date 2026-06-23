from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, Qwen2AudioForConditionalGeneration, Qwen2_5OmniThinkerForConditionalGeneration
from peft import PeftModel

# ===== 路径配置 =====
base_model_path = "Qwen/Qwen2-Audio-7B-Instruct"
lora_path = "/n/work6/yizhang/Moris/Models/StreamingSpeechLLM/ASR/qwen2audio_finetunel2_chunk1s_lora16/checkpoint-3000"
save_path = "/n/work6/yizhang/Moris/Models/StreamingSpeechLLM/ASR/qwen2audio_finetunel2_chunk1s_lora16/model"

Path(save_path).mkdir(parents=True, exist_ok=True)


# ===== 加载 base model =====
base_model = Qwen2AudioForConditionalGeneration.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(base_model_path)

# ===== 加载 LoRA =====
model = PeftModel.from_pretrained(
    base_model,
    lora_path
)

# ===== merge LoRA 权重 =====
model = model.merge_and_unload()

# ===== 保存 merge 后模型 =====
model.save_pretrained(
    save_path,
    safe_serialization=True
)

processor.save_pretrained(save_path)
processor.tokenizer.save_pretrained(save_path)

print(f"Merged model saved to: {save_path}")