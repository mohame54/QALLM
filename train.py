import os
from unsloth import FastLanguageModel
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
_orig_env_setitem = os.environ.__class__.__setitem__
def _guarded_env_setitem(self, key, value):
    if key == "UNSLOTH_RETURN_LOGITS" and value != "1":
        return
    _orig_env_setitem(self, key, value)
os.environ.__class__.__setitem__ = _guarded_env_setitem

import argparse
import pandas as pd
from utils.datasets import QADataset, QADPODataset
from utils.model import get_peft_model
from utils.common import download_gdown_file, load_json
from trainers.sft import Qwen3_5Collator, Mytrainer
from trainers.dpo import Qwen3_5DPOTrainer
from trainers.configs import Qwen3_5SFTConfig, Qwen3_5DPOConfig


CUR_DIR = os.path.dirname(os.path.abspath(__file__))


def _sample_stratified(df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Return up to n rows stratified by language (subset column prefix)."""
    if "subset" not in df.columns:
        return df.iloc[: min(n, len(df))].copy()
    lang_col = df["subset"].str.split("_", n=1).str[0]
    langs = list(lang_col.unique())
    per_lang, remainder = divmod(n, len(langs))
    parts: list[pd.DataFrame] = []
    for i, lang in enumerate(langs):
        lang_df = df[lang_col == lang]
        k = min(per_lang + (1 if i < remainder else 0), len(lang_df))
        parts.append(lang_df.sample(n=k, random_state=seed))
    return pd.concat(parts).reset_index(drop=True)


def run_training(args: argparse.Namespace) -> None:
    # ── resolve file paths ──────────────────────────────────────────────────
    train_file = args.train_file
    if args.gdrive_train_file_id:
        train_file = download_gdown_file(args.gdrive_train_file_id, args.train_file)

    val_file: str | None = args.val_file
    if args.gdrive_val_file_id:
        default_val_path = os.path.join(CUR_DIR, "data", "val_download.csv")
        val_file = download_gdown_file(
            args.gdrive_val_file_id, args.val_file or default_val_path
        )

    # ── save CLI values before args is shadowed by a config dataclass ────────
    mode = args.mode
    model_id = args.model_id
    peft_kwargs_path = args.peft_kwargs_path or os.path.join(CUR_DIR, "configs", "peft.json")
    trainer_kwargs_path = args.trainer_kwargs_path or os.path.join(
        CUR_DIR, "configs", "dpo_trainer.json" if mode == "dpo" else "sft_trainer.json"
    )
    sampling_alpha = args.sampling_alpha
    max_seq_length = args.max_seq_length
    val_sample_size = args.val_sample_size
    eval_steps = args.eval_steps
    sft_adapter = args.sft_adapter  # DPO only: load SFT LoRA weights, no trainer state
    _ckpt_arg = args.resume_from_checkpoint
    resume_from_checkpoint: bool | str | None = (
        True if isinstance(_ckpt_arg, str) and _ckpt_arg.lower() == "true"
        else _ckpt_arg
    )

    # ── model ────────────────────────────────────────────────────────────────
    peft_kwargs = load_json(peft_kwargs_path) if peft_kwargs_path else {}
    if mode == "dpo" and sft_adapter:
        # Load an SFT LoRA adapter as the DPO starting policy.
        # Do NOT pass to trainer.train() — there is no DPO optimizer state to resume.
        print(f"[train] Loading SFT adapter as DPO init policy: {sft_adapter}")
        peft_kwargs["adapter_path"] = sft_adapter
    elif resume_from_checkpoint:
        # Resume a checkpoint of the same mode (optimizer + scheduler state intact).
        print(f"[train] Resuming from checkpoint: {resume_from_checkpoint}")
        peft_kwargs["adapter_path"] = resume_from_checkpoint
    model, tokenizer, hf_tokenizer = get_peft_model(model_id, **peft_kwargs)

    FastLanguageModel.for_training(model)

    # When an SFT adapter was used as DPO init, there is no DPO checkpoint to resume.
    dpo_resume = resume_from_checkpoint if not sft_adapter else None

    if mode == "dpo":
        _run_dpo(
            args,
            model=model,
            tokenizer=tokenizer,
            hf_tokenizer=hf_tokenizer,
            train_file=train_file,
            val_file=val_file,
            trainer_kwargs_path=trainer_kwargs_path,
            val_sample_size=val_sample_size,
            eval_steps=eval_steps,
            resume_from_checkpoint=dpo_resume,
        )
    else:
        _run_sft(
            args,
            model=model,
            tokenizer=tokenizer,
            hf_tokenizer=hf_tokenizer,
            train_file=train_file,
            val_file=val_file,
            trainer_kwargs_path=trainer_kwargs_path,
            sampling_alpha=sampling_alpha,
            max_seq_length=max_seq_length,
            val_sample_size=val_sample_size,
            eval_steps=eval_steps,
            resume_from_checkpoint=resume_from_checkpoint,
        )


def _run_sft(
    model,
    tokenizer,
    hf_tokenizer,
    train_file: str,
    val_file: str | None,
    trainer_kwargs_path: str,
    sampling_alpha: float,
    max_seq_length: int,
    val_sample_size: int | None,
    eval_steps: int | None,
    resume_from_checkpoint,
) -> None:
    # ── training dataset ─────────────────────────────────────────────────────
    train_dataset = QADataset.from_csv(train_file, hf_tokenizer)
    train_dataset.mode("train")
    train_dataset = train_dataset.to_hf_dataset()

    # ── validation dataset (optional) ────────────────────────────────────────
    val_dataset = None
    if val_file:
        val_ds = QADataset.from_csv(val_file, hf_tokenizer)
        val_ds.mode("val")
        if val_sample_size and val_sample_size > 0:
            val_ds.df = _sample_stratified(val_ds.df, val_sample_size)
            print(
                f"[train] Validation set: {len(val_ds.df)} samples "
                f"(stratified from {val_file})"
            )
        val_dataset = val_ds.to_hf_dataset()

    my_custom_collator = Qwen3_5Collator(hf_tokenizer, max_seq_length=max_seq_length)

    sft_config = Qwen3_5SFTConfig.from_json(trainer_kwargs_path)

    if val_dataset is not None and eval_steps and eval_steps > 0:
        sft_config.eval_strategy = "steps"
        sft_config.eval_steps = eval_steps
    else:
        sft_config.eval_strategy = "no"

    trainer = Mytrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=my_custom_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=sft_config,
        sampling_alpha=sampling_alpha,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


def _run_dpo(
    args,
    model,
    tokenizer,
    hf_tokenizer,
    train_file: str,
    val_file: str | None,
    trainer_kwargs_path: str,
    val_sample_size: int | None,
    eval_steps: int | None,
    resume_from_checkpoint,
) -> None:
    # ── training dataset ─────────────────────────────────────────────────────
    train_ds = QADPODataset.from_csv(train_file, hf_tokenizer)
    train_dataset = train_ds.to_hf_dataset()

    # ── validation dataset (optional) ────────────────────────────────────────
    val_dataset = None
    if val_file:
        val_ds = QADPODataset.from_csv(val_file, hf_tokenizer)
        if val_sample_size and val_sample_size > 0:
            val_ds.df = _sample_stratified(val_ds.df, val_sample_size)
            print(
                f"[train] DPO val set: {len(val_ds.df)} samples "
                f"(stratified from {val_file})"
            )
        val_dataset = val_ds.to_hf_dataset()

    # ── reference model ───────────────────────────────────────────────────────
    # Default: ref_model=None + precompute_ref_log_probs=True (set in the config).
    # TRL snapshots the log-probs of the currently loaded model (your SFT LoRA)
    # in one upfront pass, then discards the computation — only one model in VRAM.
    #
    # --ref_model overrides this with an explicit second model instance (2× VRAM).
    # When using --ref_model, set precompute_ref_log_probs=false in the config.
    ref_model = None
    if args.ref_model:
        import torch
        from transformers import AutoModelForCausalLM
        from peft import PeftModel
        parts = args.ref_model.split("/", 2)
        ref_id = "/".join(parts[:2])
        ref_adapter_path = parts[2] if len(parts) > 2 else None
        print(f"[train] Loading explicit DPO reference model: {args.ref_model}")
        ref_model = AutoModelForCausalLM.from_pretrained(ref_id, torch_dtype=torch.bfloat16)
        if ref_adapter_path:
            ref_model = PeftModel.from_pretrained(ref_model, ref_adapter_path)
        for p in ref_model.parameters():
            p.requires_grad_(False)
        ref_model.eval()

    dpo_config = Qwen3_5DPOConfig.from_json(trainer_kwargs_path)

    if val_dataset is not None and eval_steps and eval_steps > 0:
        dpo_config.eval_strategy = "steps"
        dpo_config.eval_steps = eval_steps
    else:
        dpo_config.eval_strategy = "no"

    trainer = Qwen3_5DPOTrainer(
        model=model,
        ref_model=ref_model,
        processing_class=hf_tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=dpo_config,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ── training data ────────────────────────────────────────────────────────
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--gdrive_train_file_id", type=str, default=None)

    # ── validation data ──────────────────────────────────────────────────────
    parser.add_argument(
        "--val_file",
        type=str,
        default=None,
        help="Path to validation CSV (same format as train). Optional.",
    )
    parser.add_argument(
        "--gdrive_val_file_id",
        type=str,
        default=None,
        help="Google Drive file ID for the validation CSV.",
    )
    parser.add_argument(
        "--val_sample_size",
        type=int,
        default=None,
        help="Number of validation samples (stratified by language). "
             "Uses full val set if not set.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=None,
        help="Run validation every N optimizer steps. "
             "Requires --val_file or --gdrive_val_file_id.",
    )

    # ── model & training ─────────────────────────────────────────────────────
    parser.add_argument(
        "--mode",
        type=str,
        default="sft",
        choices=["sft", "dpo"],
        help="Training mode: 'sft' (default) or 'dpo'.",
    )
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory to resume training from, or 'true' to "
             "auto-resume from the latest checkpoint in output_dir.",
    )
    parser.add_argument("--peft_kwargs_path", type=str, default=None)
    parser.add_argument(
        "--sft_adapter",
        type=str,
        default=None,
        help="DPO mode only. Path or Hub ID of a pre-trained SFT LoRA adapter to use as "
             "the DPO starting policy. Loads weights only — no optimizer state is resumed. "
             "Cannot be combined with --resume_from_checkpoint.",
    )
    parser.add_argument(
        "--ref_model",
        type=str,
        default=None,
        help="DPO mode only. Path or Hub ID of an explicit reference model. "
             "If omitted, precompute_ref_log_probs=true in the config is used to snapshot "
             "the loaded policy's log-probs before training (recommended, no extra VRAM).",
    )
    parser.add_argument(
        "--trainer_kwargs_path",
        type=str,
        default=None,
        help="Path to trainer config JSON. Defaults to configs/sft_trainer.json (sft) "
             "or configs/dpo_trainer.json (dpo).",
    )
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument(
        "--sampling_alpha",
        type=float,
        default=0.5,
        help="Temperature sampling alpha for SFT (0=balanced, 1=natural). Default 0.5.",
    )

    args = parser.parse_args()
    run_training(args)
