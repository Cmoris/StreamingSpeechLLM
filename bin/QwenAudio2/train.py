from dataclasses import asdict, dataclass, field
from typing import List, Optional, Literal
from pathlib import Path
import torch
import transformers
from transformers import (
    Trainer, 
    AutoProcessor,
    HfArgumentParser, 
    logging, 
    AutoConfig,
    Qwen2AudioForConditionalGeneration
)

from constants import DEFAULT_SAMPLE_RATE, DEFAULT_CHUNK_SECS, DEFAULT_CONTEXT_LENGTH, TE_TOKEN, TS_TOKEN, SPEAKER_TOKENS, BC_TOKEN, PAUSE_TOKEN, SILENCE_TOKEN, QUERY
from data import DualChannelStreamingConvDataset, DualChannelStreamingASRDataset, DualChannelStreamingASRVAPDataset, SingleChannelStreamingASRDataset
from nonstreaming_data import SingleChannelASRDataset, DualChannelASRDataset, DualChannelConvDataset

logger = logging.get_logger(__name__)
local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def set_seed(seed=42):
    """
    Set the random seed for reproducible results.

    :param seed: An integer value to be used as the random seed.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
@dataclass
class DataArguments:
    annotation_dir: str = field(default="")
    audio_root_a: str = field(default="")
    audio_root_b: str = field(default="")
    chunk_secs: float = field(default=DEFAULT_CHUNK_SECS)
    sample_rate: int = field(default=DEFAULT_SAMPLE_RATE)
    context_length: int = field(default=DEFAULT_CONTEXT_LENGTH)
    query: str = field(default=QUERY)
    data_version:  Literal[
        "dual_channel_asr",
        "single_channel_asr",
        "dual_channel_conv",
        "dual_channel_streaming_conv",
        "dual_channel_streaming_asr",
        "single_channel_streaming_asr"
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
class ModelArguments:
    pretrained_model_name_or_path: str = field(default='Qwen/Qwen2-Audio-7B-Instruct')
    freeze_modules: Optional[List[str]] = field(default=None)  

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = field(default="./output/qwen2_audio_finetuned")  
    overwrite_output_dir: bool = field(default=True)  
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(default=None)
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_weight_path: str = field(default="")
    lora_bias: str = field(default="none")
    target_modules: str = field(default="q_proj,k_proj,v_proj,o_proj")  # 修正默认值格式

    
def train():
    global local_rank
    set_seed(42)
    
    # 解析参数
    parser = HfArgumentParser((TrainingArguments, ModelArguments, DataArguments))
    training_args, model_args, data_args = parser.parse_args_into_dataclasses()
    
    local_rank = training_args.local_rank
    
    # 设置计算数据类型
    compute_dtype = (
        torch.float16 if training_args.fp16 
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    # 量化配置
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device} if training_args.device else "auto",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
                bnb_4bit_quant_storage=compute_dtype,
            )
        ))

    # 加载配置和模型 - 修正为正确的Qwen2Audio类
    rank0_print(f"Loading model from {model_args.pretrained_model_name_or_path}")
    
    config = AutoConfig.from_pretrained(model_args.pretrained_model_name_or_path)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_args.pretrained_model_name_or_path, 
        config=config,
        dtype=compute_dtype,
        **bnb_model_from_pretrained_args
    )
    
    # 准备量化训练
    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.dtype = (
            torch.float32 if training_args.fp16 
            else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )
        model = prepare_model_for_kbit_training(
            model, 
            use_gradient_checkpointing=training_args.gradient_checkpointing
        )
    
    # LoRA配置
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=training_args.target_modules.split(','),  # 修正分隔符
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
        
    # 冻结指定模块

    if model_args.freeze_modules is not None:
        if training_args.lora_enable:
            for m in model_args.freeze_modules:
                for name, param in model.base_model.model.named_parameters():
                    if name.startswith(m):
                        param.requires_grad = False
                        # rank0_print(f"FROZEN: {name}")
        else:
            for m in model_args.freeze_modules:
                for name, param in model.named_parameters():
                    if name.startswith(m):
                        param.requires_grad = False
                        # rank0_print(f"FROZEN: {name}")
    
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    # 加载处理器
    rank0_print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        model_args.pretrained_model_name_or_path, 
        padding_side='right'
    )
    
    # 设置tokenizer的pad_token
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        
    rank0_print("Add tokens...")

    processor.tokenizer.add_tokens([TE_TOKEN, TS_TOKEN, BC_TOKEN, PAUSE_TOKEN, SILENCE_TOKEN, SPEAKER_TOKENS['A'][0], SPEAKER_TOKENS['A'][1], SPEAKER_TOKENS['B'][0], SPEAKER_TOKENS['B'][1]], special_tokens=False)
    
    # print("tokenizer:", len(processor.tokenizer.get_vocab()))
    # print("embedding:", model.get_input_embeddings().weight.shape[0])
    # 准备数据集
    rank0_print("Preparing dataset...")
    annotation_paths = [str(x) for x in Path(data_args.annotation_dir).glob("*.jsonl")]
    
    if data_args.data_version == "dual_channel_streaming_asr":
        dataset = DualChannelStreamingASRDataset(
            annotation_paths=annotation_paths,
            processor=processor,
            audio_root_a=data_args.audio_root_a,
            audio_root_b=data_args.audio_root_b,
            chunk_secs=data_args.chunk_secs,
            sample_rate=data_args.sample_rate,
            query=data_args.query,
        )
    elif data_args.data_version == "dual_channel_streaming_conv":
        dataset = DualChannelStreamingConvDataset(
            annotation_paths=annotation_paths,
            processor=processor,
            audio_root_a=data_args.audio_root_a,
            audio_root_b=data_args.audio_root_b,
            chunk_secs=data_args.chunk_secs,
            sample_rate=data_args.sample_rate,
            query=data_args.query,
        )
    elif data_args.data_version == "single_channel_streaming_asr":
        dataset = SingleChannelStreamingASRDataset(
            annotation_paths=annotation_paths,
            processor=processor,
            sample_rate=data_args.sample_rate,
            query=data_args.query,
        )
    elif data_args.data_version == "dual_channel_conv":
        dataset = DualChannelConvDataset(
            annotation_paths=annotation_paths,
            processor=processor,
            audio_root_a=data_args.audio_root_a,
            audio_root_b=data_args.audio_root_b,
            chunk_secs=data_args.chunk_secs,
            sample_rate=data_args.sample_rate,
            query=data_args.query,
        )
        
    elif data_args.data_version == "dual_channel_asr":
        dataset = DualChannelASRDataset(
            annotation_paths=annotation_paths,
            processor=processor,
            audio_root_a=data_args.audio_root_a,
            audio_root_b=data_args.audio_root_b,
            chunk_secs=data_args.chunk_secs,
            sample_rate=data_args.sample_rate,
            query=data_args.query,
        )
        
    elif data_args.data_version == "single_channel_asr":
        dataset = SingleChannelASRDataset(
            annotation_paths=annotation_paths,
            processor=processor,
            sample_rate=data_args.sample_rate,
            query=data_args.query,
        )
        
    else:
        raise ValueError(f"Unsupported data_version: {data_args.data_version}")
    
    # 创建Trainer并开始训练
    rank0_print("Starting training...")
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=dataset,
        data_collator=dataset.data_collator,
    )
    
    # 开始训练
    checkpoint_dir = Path(training_args.output_dir)
    if list(checkpoint_dir.glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    
    # 保存最终模型
    rank0_print("Saving final model...")
    trainer.save_model()
    processor.save_pretrained(training_args.output_dir)
    
if __name__ == "__main__":
    train()