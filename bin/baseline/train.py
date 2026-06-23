import re
import os
import random
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Literal
from pathlib import Path
import torch
import transformers

from model import SpeechLLMs, SpeechLLMConfigs
from dataset import DualChannelConvDataset, DualChannelASRDataset, SingleChannelASRDataset
from speechllm_trainer import SpeechLLMTrainer, get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, safe_save_model_for_hf_trainer
from constants import DEFAULT_CONTEXT_LENGTH, DEFAULT_SAMPLE_RATE, TS_TOKEN, TE_TOKEN, SPEAKER_TOKENS, BC_TOKEN, PAUSE_TOKEN, SILENCE_TOKEN

# NOTE: fast tokenizer warning issue: https://github.com/huggingface/transformers/issues/5486
os.environ["TOKENIZERS_PARALLELISM"] = "true"

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def set_seed(seed: int = 42):
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Model / Data / Training argument dataclasses ──────────────────────────────

@dataclass
class ModelArguments:
    model_type:  Optional[str] = field(default="SpeechQwen3")
    model_path:  Optional[str] = field(default="Qwen/Qwen3-0.6B")
    # Connector
    mm_projector_a_type:        Optional[str] = field(default="linear")
    # Audio tower
    audio_tower:            Optional[str] = field(default=None)
    # Qformer
    use_qformer:          bool  = field(default=True)
    pretrain_qformer:     Optional[str] = field(default=None)
    window_level_Qformer: bool  = field(default=False)
    qformer_model:        str   = field(default="bert-large-uncased")
    qformer_layers:       int   = field(default=2)
    qformer_dim:          int   = field(default=1024)
    queries_per_sec:      int   = field(default=3)
    second_per_window:    float = field(default=0.333333)
    second_stride:        float = field(default=0.333333)   
    # Projector
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    # Audio config forwarded to model config
    audio_frame_rate:     int   = field(default=25)


@dataclass
class DataArguments:
    annotation_dir: str   = field(default="")
    audio_root_a:   str   = field(default="")
    audio_root_b:   str   = field(default="")
    context_length: int   = field(default=DEFAULT_CONTEXT_LENGTH)
    sample_rate:    int   = field(default=DEFAULT_SAMPLE_RATE)
    query:          Optional[str] = field(default=None)
    data_version:   Literal[
        "dual_channel_asr",
        "single_channel_asr",
        "dual_channel_conv"
    ] = field(
        default="dual_channel_conv",
        metadata={
            "help": (
                "Dataset format. "
                "'dual_channel_conv' for original A/B dual-channel conversation data; "
                "'single_channel_asr' for one-line one-audio ASR jsonl data."
            )
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")
    mm_projector_lr:        Optional[float] = field(default=None)
    freeze_mm_mlp_adapter:  bool            = field(default=False)
    freeze_backbone:        bool            = field(default=False,metadata={"help": "Whether to freeze the LLM backbone."})
    tune_mm_mlp_adapter:    bool            = field(default=False)
    tune_qformer:           bool            = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."}
    )
    # Quantisation
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    # LoRA
    lora_enable:      bool  = field(default=False)
    lora_r:           int   = field(default=16)
    lora_alpha:       int   = field(default=32)
    lora_dropout:     float = field(default=0.05)
    lora_weight_path: str   = field(default="")
    lora_bias:        str   = field(default="none")
    target_modules:   str   = field(default="q_proj.k_proj.v_proj.o_proj")


# ── Data module ────────────────────────────────────────────────────────────────

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    processor: transformers.AutoProcessor,
    data_args: DataArguments,
) -> Dict:
    annotation_paths = [
        str(x) for x in Path(data_args.annotation_dir).glob("*.jsonl")
    ]
    if not annotation_paths:
        raise FileNotFoundError(
            f"No .jsonl files found in annotation_dir={data_args.annotation_dir!r}"
        )
    if data_args.data_version == "dual_channel_conv":
        dataset = DualChannelConvDataset(
            annotation_paths=annotation_paths,
            processor=processor,
            tokenizer=tokenizer,
            audio_root_a=data_args.audio_root_a,
            audio_root_b=data_args.audio_root_b,
            context_length=data_args.context_length,
            sample_rate=data_args.sample_rate,
            query=data_args.query,
        )
    elif data_args.data_version == "single_channel_asr":
        dataset = SingleChannelASRDataset(
            annotation_paths=annotation_paths,
            processor=processor,
            tokenizer=tokenizer,
            sample_rate=data_args.sample_rate,
            query=data_args.query,
        )
    elif data_args.data_version == "dual_channel_asr":
        dataset = DualChannelASRDataset(
            annotation_paths=annotation_paths,
            processor=processor,
            tokenizer=tokenizer,
            audio_root_a=data_args.audio_root_a,
            audio_root_b=data_args.audio_root_b,
            context_length=data_args.context_length,
            sample_rate=data_args.sample_rate,
            query=data_args.query,
        )

    return dict(
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=dataset.data_collator,
    )


# ── Main training entry point ──────────────────────────────────────────────────

def train(attn_implementation: Optional[str] = None):
    global local_rank
    set_seed(42)

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    # ── Quantisation config ────────────────────────────────────────────────────
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=training_args.double_quant,
            bnb_4bit_quant_type=training_args.quant_type,
            bnb_4bit_quant_storage=compute_dtype,
        )

    # ── Model type validation ──────────────────────────────────────────────────
    if model_args.model_type not in SpeechLLMs:
        raise ValueError(
            f"Unknown model_type={model_args.model_type!r}. "
            f"Supported: {list(SpeechLLMs.keys())}"
        )
    if model_args.audio_tower is None:
        raise ValueError("audio_tower must be specified.")

    # ── Load config ────────────────────────────────────────────────────────────
    config = SpeechLLMConfigs[model_args.model_type].from_pretrained(
        model_args.model_path, trust_remote_code=True
    )
    config._attn_implementation = attn_implementation

    # ── Load model ─────────────────────────────────────────────────────────────
    device_map = (
        {"": training_args.local_rank}
        if training_args.local_rank != -1
        else "auto"
    )
    model = SpeechLLMs[model_args.model_type].from_pretrained(
        model_args.model_path,
        config=config,
        torch_dtype=compute_dtype,
        device_map=device_map,
        **bnb_model_from_pretrained_args,
    )

    # ── k-bit preparation ──────────────────────────────────────────────────────
    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training

        model.config.dtype = (
            torch.float32
            if training_args.fp16
            else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
        )

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # ── LoRA ───────────────────────────────────────────────────────────────────
    # assert training_args.lora_enable != training_args.freeze_backbone and 
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=training_args.target_modules.split(","),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    elif training_args.freeze_backbone:
        rank0_print("Freezing backbone...")
        model.requires_grad_(False)
    else:
        rank0_print("Training all parameters...")
        model.requires_grad_(True)

    # ── Tokenizer ──────────────────────────────────────────────────────────────
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_path,
        model_max_length=training_args.model_max_length,
    )
    new_tokens = [
        TE_TOKEN,
        TS_TOKEN,
        BC_TOKEN,
        PAUSE_TOKEN,
        SILENCE_TOKEN,
        SPEAKER_TOKENS["A"][0],
        SPEAKER_TOKENS["A"][1],
        SPEAKER_TOKENS["B"][0],
        SPEAKER_TOKENS["B"][1],
    ]
    tokenizer.add_tokens(new_tokens, special_tokens=False)

    
    # ── Audio tower + projector initialisation ─────────────────────────────────
    model.get_model().initialize_audio_modules(
        model_args=model_args,
        fsdp=training_args.fsdp,
    )

    audio_tower = model.get_audio_tower()
    audio_tower.to(
        dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        device=training_args.device,
    )

    for p in model.get_model().mm_projector_a.parameters():
        p.requires_grad = True

    if training_args.bits in [4, 8]:
        model.get_model().mm_projector_a.to(
            dtype=compute_dtype, device=training_args.device
        )

    # ── Qformer initialisation ─────────────────────────────────────────────────
    if model_args.use_qformer and model_args.qformer_model:
        model.get_model().initialize_qformer(
            model_args=model_args,
            fsdp=training_args.fsdp,
        )
        model.get_qformer().to(
            dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
            device=training_args.device,
        )

        model.get_model().query_tokens.to(
            dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
            device=training_args.device,
        )
        model.get_model().ln_head.to(device=training_args.device)
        
        if training_args.tune_qformer:
            for param in model.get_model().Qformer.parameters():
                param.requires_grad = True
            
            model.get_model().query_tokens.requires_grad_(True)
        else:
            for param in model.get_model().Qformer.parameters():
                param.requires_grad = False
            model.get_model().query_tokens.requires_grad_(False)

    # ── Final projector initialisation ────────────────────────────────────────
    model.get_model().initialize_projector(
        model_args=model_args,
        fsdp=training_args.fsdp,
    )
    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter
    if training_args.tune_mm_mlp_adapter:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True
    else:
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = False

    if training_args.bits in [4, 8]:
        model.get_model().mm_projector.to(
            dtype=compute_dtype, device=training_args.device
        )

    # ── k-bit dtype fixups ─────────────────────────────────────────────────────
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer) and training_args.bf16:
                module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if ("lm_head" in name or "embed_tokens" in name) and hasattr(module, "weight"):
                if training_args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    # rank0_print("Current model:", model)

    # ── Data module ────────────────────────────────────────────────────────────
    processor = model.get_audio_tower().audio_processor

    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        processor=processor,
        data_args=data_args,
    )

    # Move model to device (non-quantised path only)
    if training_args.bits not in [4, 8]:
        model.to(training_args.device)

    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    # ── Trainer ────────────────────────────────────────────────────────────────
    trainer = SpeechLLMTrainer(
        model=model,
        args=training_args,
        **data_module,
    )

    checkpoint_dir = pathlib.Path(training_args.output_dir)
    if list(checkpoint_dir.glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()
    model.config.use_cache = True

    # ── Save weights ───────────────────────────────────────────────────────────
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank in (0, -1):
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(
                non_lora_state_dict,
                os.path.join(training_args.output_dir, "non_lora_trainables.bin"),
            )
    else:
        safe_save_model_for_hf_trainer(
            trainer=trainer, output_dir=training_args.output_dir
        )


if __name__ == "__main__":
    train()