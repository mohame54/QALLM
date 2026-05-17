from transformers import AutoTokenizer
from unsloth import FastLanguageModel # FastLanguageModel for LLMs
import torch


def load_unsloth_model(model_id, dtype="float16"):
    dtype = getattr(torch, dtype)
    transformers_tokenizer = AutoTokenizer.from_pretrained(model_id)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_id,
        load_in_4bit = False, # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context,
        dtype=dtype,
    )
    return model, tokenizer, transformers_tokenizer



def get_peft_model(
    model_id: str,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=None,
    lora_rank=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_gradient_checkpointing=True,
    use_rslora=False,
    loftq_config=None,
    target_modules=None,
    dtype="bfloat16",
    adapter_path=None
):
    model, tokenizer, hf_tokenizer = load_unsloth_model(model_id, dtype=dtype)
    if adapter_path is None:
        peft_kwargs = {
            "finetune_language_layers": finetune_language_layers,
            "finetune_attention_modules": finetune_attention_modules,
            "finetune_mlp_modules": finetune_mlp_modules,
            "r": r if r is not None else lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "bias": bias,
            "random_state": random_state,
            "use_gradient_checkpointing": use_gradient_checkpointing,
            "use_rslora": use_rslora,
            "loftq_config": loftq_config,
        }
        if target_modules is not None:
            peft_kwargs["target_modules"] = target_modules

        model = FastLanguageModel.get_peft_model(model, **peft_kwargs)
    else:
        try:
            _parts = adapter_path.split("/")
            if len(_parts) > 2:
                _adapter_repo = "/".join(_parts[:2])
                _adapter_subfolder = "/".join(_parts[2:])
                model = FastLanguageModel.get_peft_model(
                    model, _adapter_repo, subfolder=_adapter_subfolder
                )
            else:
                model = FastLanguageModel.get_peft_model(model, adapter_path)
        except ImportError as exc:
            raise RuntimeError(
                "peft is required when --adapter_path is set. Install peft or omit adapter_path."
            ) from exc
    return model, tokenizer, hf_tokenizer