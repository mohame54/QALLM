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
from utils.datasets import QADataset
from utils.model import get_peft_model
from utils.common import download_gdown_file, load_json
from trainers.sft import Qwen3_5Collator, Mytrainer
from trainers.configs import Qwen3_5SFTConfig



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

    # ── save CLI values before args is shadowed by SFTConfig ────────────────
    model_id = args.model_id
    peft_kwargs_path = args.peft_kwargs_path or os.path.join(CUR_DIR, "configs", "peft.json")
    sft_trainer_kwargs_path = args.sft_trainer_kwargs_path or os.path.join(
        CUR_DIR, "configs", "sft_trainer.json"
    )
    sampling_alpha = args.sampling_alpha
    max_seq_length = args.max_seq_length
    val_sample_size = args.val_sample_size
    eval_steps = args.eval_steps
    _ckpt_arg = args.resume_from_checkpoint
    resume_from_checkpoint: bool | str | None = (
        True if isinstance(_ckpt_arg, str) and _ckpt_arg.lower() == "true"
        else _ckpt_arg
    )

    # ── model ────────────────────────────────────────────────────────────────
    peft_kwargs = load_json(peft_kwargs_path) if peft_kwargs_path else {}
    if resume_from_checkpoint:
        print(f"[train] Resuming from checkpoint: {resume_from_checkpoint}")
        peft_kwargs["adapter_path"] = resume_from_checkpoint
    model, tokenizer, hf_tokenizer = get_peft_model(model_id, **peft_kwargs)

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

    # ── collator & trainer config ─────────────────────────────────────────────
    my_custom_collator = Qwen3_5Collator(hf_tokenizer, max_seq_length=max_seq_length)
    FastLanguageModel.for_training(model)

    sft_config = Qwen3_5SFTConfig.from_json(sft_trainer_kwargs_path)

    # Enable step-level validation when a val set and eval_steps are provided.
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
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint directory to resume training from, or 'true' to "
             "auto-resume from the latest checkpoint in output_dir.",
    )
    parser.add_argument("--peft_kwargs_path", type=str, default=None)
    parser.add_argument("--sft_trainer_kwargs_path", type=str, default=None)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument(
        "--sampling_alpha",
        type=float,
        default=0.5,
        help="Temperature sampling alpha (0=balanced, 1=natural). Default 0.5.",
    )

    args = parser.parse_args()
    run_training(args)
