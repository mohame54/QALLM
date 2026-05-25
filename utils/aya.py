import gc
import torch
import torch.nn as nn
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from peft import PeftModel


def load_lora_extended_model(
    base_model_id: str,
    checkpoint_dir: str,
    load_in_4bit=False,
    max_seq_length=2048,
):
    # ── 1. Detect best supported dtype ────────────────────────────────────────
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        print("✅ GPU supports bfloat16 — using bfloat16.")
    else:
        dtype = torch.float16
        print("⚠️  GPU does not support bfloat16 — falling back to float16.")

    # ── 2. Load the extended tokenizer from the checkpoint ───────────────────
    print(f"\n📑 Loading tokenizer from checkpoint: {checkpoint_dir} …")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    extended_vocab_size = len(tokenizer)
    print(f"📊 Target vocabulary size detected: {extended_vocab_size}")

    # ── 3. Load the vanilla BASE model ───────────────────────────────────────
    print(f"\n📦 Loading vanilla base model: {base_model_id} …")
    base_model, _ = FastLanguageModel.from_pretrained(
        model_name = base_model_id,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    # ── 4. Resize the base model layers ──────────────────────────────────────
    print(f"🔄 Resizing base model layers to {extended_vocab_size} …")
    base_model.resize_token_embeddings(extended_vocab_size)

    # ── 4b. MANUALLY FORCE RESIZE LM_HEAD TO FIX METADATA MISMATCH ───────────
    output_embeddings = base_model.get_output_embeddings()
    if output_embeddings is not None and output_embeddings.out_features != extended_vocab_size:
        print(f"🔧 Correcting lm_head structure: {output_embeddings.out_features} -> {extended_vocab_size}")
        
        new_lm_head = nn.Linear(
            in_features=output_embeddings.in_features,
            out_features=extended_vocab_size,
            bias=output_embeddings.bias is not None,
            dtype=output_embeddings.weight.dtype,
            device=output_embeddings.weight.device
        )
        
        # Handle copying depending on whether weight-tying already expanded the tensor
        with torch.no_grad():
            if output_embeddings.weight.shape[0] == extended_vocab_size:
                new_lm_head.weight.copy_(output_embeddings.weight)
            else:
                old_num_tokens = output_embeddings.weight.shape[0]
                new_lm_head.weight[:old_num_tokens, :] = output_embeddings.weight
            
            if output_embeddings.bias is not None:
                if output_embeddings.bias.shape[0] == extended_vocab_size:
                    new_lm_head.bias.copy_(output_embeddings.bias)
                else:
                    old_num_tokens = output_embeddings.bias.shape[0]
                    new_lm_head.bias[:old_num_tokens] = output_embeddings.bias
        
        # Re-bind the fixed layer back into all structural variants of the model
        base_model.set_output_embeddings(new_lm_head)
        if hasattr(base_model, "lm_head"):
            base_model.lm_head = new_lm_head
        if hasattr(base_model, "model") and hasattr(base_model.model, "lm_head"):
            base_model.model.lm_head = new_lm_head

        # ── 4c. Clean up temporary embedding references to free GPU memory ────
        print("🗑️  Deleting temporary layer references and clearing CUDA cache …")
        del output_embeddings, new_lm_head
        gc.collect()
        torch.cuda.empty_cache()

    # ── 5. Apply the LoRA adapters and updated embeddings ────────────────────
    print(f"\n🧬 Injecting LoRA adapters and loading trained weights …")
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)

    # ── 6. Enable native Unsloth inference speedups ──────────────────────────
    print("🚀 Activating Unsloth inference optimizations …")
    FastLanguageModel.for_inference(model)

    print("\n✅ Done! Model shapes matched, VRAM cleaned, and everything loaded successfully.")
    return model, tokenizer