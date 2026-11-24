import torch
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
LOAD_4BIT = False  # set True for the 8B model


def load_model(model_id=MODEL_ID, load_4bit=LOAD_4BIT):
    kwargs = {
        "device_map": "cuda:0",
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
    }
    if load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        kwargs["dtype"] = torch.float16
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **kwargs).eval()
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor
