import argparse
import inspect
import json
import os
import traceback
from typing import Any

import pandas as pd
import torch
import tqdm
from unsloth import FastLanguageModel

from utils.common import download_gdown_file
from utils.datasets import process_single_row
from utils.instructions import AyaChatPromptTemplate, strip_qwen_thinking_tokens
from utils.metrics import calculate_rouge_score
from utils.model import get_peft_model, load_unsloth_model

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GEN_KWARGS_PATH = os.path.join(CUR_DIR, "configs", "gen.json")


def _detect_model_family(model_id: str) -> str:
    """Return 'aya' for Cohere/Aya models, 'qwen' otherwise."""
    mid = model_id.lower()
    if "aya" in mid:
        return "aya"
    if "qwen" in mid:
        return "qwen"
    return "qwen"


def _compute_exact_match(pred: str, ref: str) -> bool:
    return pred.strip().lower() == ref.strip().lower()


def _load_gen_kwargs(path: str | None) -> dict[str, Any]:
    if not path or not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            return {}
        return json.loads(raw)
    except (json.JSONDecodeError, OSError):
        return {}


def _filter_generate_kwargs(model: Any, user_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep only kwargs accepted by this model's `generate` (avoids TypeError).

    If the generate signature contains a **kwargs catch-all, all user kwargs are
    forwarded — Unsloth's inference wrapper does this, so we must not filter them.
    """
    gen_fn = getattr(model, "generate", None)
    if gen_fn is None:
        return {}
    try:
        sig = inspect.signature(gen_fn)
    except (TypeError, ValueError):
        return dict(user_kwargs)
    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )
    if has_var_keyword:
        return dict(user_kwargs)
    allowed = set(sig.parameters)
    return {k: v for k, v in user_kwargs.items() if k in allowed}


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_pad_token(hf_tokenizer: Any) -> None:
    if hf_tokenizer.pad_token_id is None:
        hf_tokenizer.pad_token = hf_tokenizer.eos_token
        hf_tokenizer.pad_token_id = hf_tokenizer.eos_token_id



@torch.inference_mode()
def generate_batch(
    rows: list[pd.Series],
    model: Any,
    hf_tokenizer: Any,
    start_answer_txt: str,
    gen_kwargs: dict[str, Any],
    model_family: str = "qwen",
    debug: bool = False,
    debug_n: int = 0,
) -> list[str]:
    _ensure_pad_token(hf_tokenizer)

    aya_template = AyaChatPromptTemplate(hf_tokenizer) if model_family == "aya" else None

    prompt_ids: list[list[int]] = []
    prompt_texts: list[str] = []  # raw strings kept for debug printing
    max_len = 0
    input_text = ""
    try:
        for row in rows:
            st_answer_txt = None if model_family == "aya" else start_answer_txt
            messages = process_single_row(
                row,
                hf_tokenizer,
                add_answer=False,
                start_answer_txt=st_answer_txt,
            )
            if model_family == "aya":
                # AyaChatPromptTemplate handles the system prompt internally and
                # opens the assistant turn with generation=True; drop any assistant
                # placeholder that process_single_row may have appended.
                user_messages = [m for m in messages if m["role"] != "assistant"]
                input_text = aya_template.apply_chat_prompt_template(
                    user_messages, generation=True
                )
                inputs = hf_tokenizer.encode(input_text, add_special_tokens=False)
            else:
                input_text = hf_tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
                input_text = input_text[
                    : input_text.index(start_answer_txt) + len(start_answer_txt)
                ]
                inputs = hf_tokenizer.encode(input_text, add_special_tokens=False)
            max_len = max(max_len, len(inputs))
            prompt_ids.append(inputs)
            prompt_texts.append(input_text)
    except Exception as e:
        print(f"Error processing row: {e} within {input_text}")
        traceback.print_exc()
        raise

    if not prompt_ids:
        return []
    # Left-pad so all real tokens end at position max_len (decoder-only requirement).
    padded_input: list[list[int]] = []
    attention_mask: list[list[int]] = []
    pad_id = hf_tokenizer.pad_token_id
    for ids in prompt_ids:
        pad_amt = max_len - len(ids)
        if hf_tokenizer.padding_side == "left":
            padded_input.append([pad_id] * pad_amt + ids)
            attention_mask.append([0] * pad_amt + [1] * len(ids))
        else:
            padded_input.append(ids + [pad_id] * pad_amt)
            attention_mask.append([1] * len(ids) + [0] * pad_amt)

    batch: dict = {
        "input_ids": torch.tensor(padded_input, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }
    dev = _model_device(model)
    input_ids = batch["input_ids"].to(dev)
    attention_mask = batch["attention_mask"].to(dev)

    filtered = _filter_generate_kwargs(model, gen_kwargs)
    filtered["eos_token_id"] = hf_tokenizer.eos_token_id
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **filtered,
    )  # (B, prompt_len + new_tokens)

    results: list[str] = []
    for i in range(len(rows)):
        decoded = hf_tokenizer.decode(
            output_ids[i][max_len:],
            skip_special_tokens=True,
        )
        question = rows[i]["input"]
        results.append(strip_qwen_thinking_tokens(decoded, question))

    if debug:
        n_to_print = len(rows) if debug_n <= 0 else min(debug_n, len(rows))
        sep = "─" * 60
        for i in range(n_to_print):
            raw_output = hf_tokenizer.decode(
                output_ids[i][max_len:], skip_special_tokens=False
            )
            print(f"\n{sep}")
            print(f"[DEBUG] Sample {i} — INPUT (special tokens kept):")
            print(prompt_texts[i])
            print(f"[DEBUG] Sample {i} — OUTPUT (special tokens kept):")
            print(raw_output)
        print(sep + "\n")

    return results


def _prepare_df(df: pd.DataFrame, sample_size: int | None) -> pd.DataFrame:
    if sample_size is not None and sample_size > 0:
        if "subset" not in df.columns:
            return df.iloc[: min(sample_size, len(df))].copy()
        lang_col = df["subset"].str.split("_", n=1).str[0]
        langs = list(lang_col.unique())
        n_langs = len(langs)
        per_lang, remainder = divmod(sample_size, n_langs)
        parts: list[pd.DataFrame] = []
        for i, lang in enumerate(langs):
            lang_df = df[lang_col == lang]
            n = min(per_lang + (1 if i < remainder else 0), len(lang_df))
            parts.append(lang_df.sample(n=n, random_state=42))
        return pd.concat(parts).reset_index(drop=True)
    return df


def _print_eval_summary(rows: list[dict[str, Any]]) -> None:
    if not rows or "rouge1_f1" not in rows[0]:
        return
    pdf = pd.DataFrame(rows)
    metric_cols = [
        c for c in ("exact_match", "rouge1_f1", "rougeL_f1", "score") if c in pdf.columns
    ]

    print("\n" + "=" * 60)
    print("EVAL SUMMARY")
    print("=" * 60)
    for col in metric_cols:
        print(f"  {col}: {pdf[col].mean():.4f}")

    if "expected_lang" in pdf.columns:
        print("\nPER-LANGUAGE BREAKDOWN")
        print("-" * 60)
        for lang, grp in pdf.groupby("expected_lang"):
            print(f"  [{lang}]  (n={len(grp)})")
            for col in metric_cols:
                print(f"    {col}: {grp[col].mean():.4f}")

    print("=" * 60 + "\n")


def run_eval(args: argparse.Namespace) -> None:
    if args.gdrive_dataset_id:
        out_path = args.data_csv_path or os.path.join(CUR_DIR, "data", "eval_download.csv")
        dataset_path = download_gdown_file(args.gdrive_dataset_id, out_path)
    else:
        if not args.data_csv_path:
            raise ValueError("Provide --data_csv_path or --gdrive_dataset_id.")
        dataset_path = args.data_csv_path

    df = pd.read_csv(dataset_path)
    df = _prepare_df(df, args.sample_size)

    if args.mode == "eval" and "output" not in df.columns:
        raise ValueError("eval mode requires an 'output' column in the CSV.")

    gen_path = args.gen_kwargs_path or DEFAULT_GEN_KWARGS_PATH
    gen_from_file = _load_gen_kwargs(gen_path)
    # CLI max_new_tokens always wins over the JSON file value.
    gen_kwargs = {**gen_from_file, "max_new_tokens": args.max_new_tokens}

    model_family = _detect_model_family(args.model_id)
    print(f"[eval] Detected model family: '{model_family}'")

    if args.adapter_path:
        model, _, hf_tokenizer = get_peft_model(
            args.model_id, dtype=args.dtype, adapter_path=args.adapter_path, add_peft_kwargs=False
        )
    else:
        # Base model — no LoRA adapter; load directly without any PEFT wrapping.
        model, _, hf_tokenizer = load_unsloth_model(args.model_id, dtype=args.dtype)
    FastLanguageModel.for_inference(model)
    model.eval()

    filtered_once = _filter_generate_kwargs(model, gen_kwargs)
    if set(gen_kwargs) - set(filtered_once):
        dropped = set(gen_kwargs) - set(filtered_once)
        print(f"[eval] Ignored unknown generate kwargs (not supported by model): {sorted(dropped)}")

    rows: list[dict[str, Any]] = []
    labels = ["TargetRLF1", "TargetR1F1", "TargetLLM"]
    batch_size = args.batch_size
    n_batches = (len(df) + batch_size - 1) // batch_size
    debug_samples_remaining = args.debug_samples if args.debug else 0

    for batch_idx in tqdm.tqdm(range(n_batches), desc=args.mode):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(df))
        batch_rows = [df.iloc[i] for i in range(start, end)]
        global_indices = list(range(start, end))

        try:
            gen_answers = generate_batch(
                batch_rows,
                model,
                hf_tokenizer,
                args.start_answer_txt,
                filtered_once,
                model_family=model_family,
                debug=debug_samples_remaining > 0,
                debug_n=debug_samples_remaining,
            )
        except Exception as exc:
            tqdm.tqdm.write(f"[WARNING] batch {batch_idx} (rows {start}-{end-1}) failed: {exc}")
            gen_answers = [""] * len(batch_rows)
            print(traceback.format_exc())
        debug_samples_remaining = max(0, debug_samples_remaining - len(batch_rows))

        for row, idx, gen_answer in zip(batch_rows, global_indices, gen_answers):
            if not gen_answer.strip():
                gen_answer = (
                    "I am unable to provide an answer to this question at this time."
                )

            if args.mode == "eval":
                gold = row["output"]
                rouge = calculate_rouge_score(str(gold), gen_answer)
                lang, country = "", ""
                try:
                    parts = str(row["subset"]).split("_", 1)
                    lang = parts[0] if parts else ""
                    country = parts[1] if len(parts) > 1 else ""
                except Exception:
                    pass
                rec: dict[str, Any] = {
                    "ID": row.get("ID", idx),
                    "input": row.get("input", ""),
                    "gold_answer": gold,
                    "gen_answer": gen_answer,
                    "expected_lang": lang,
                    "expected_country": country,
                    "exact_match": int(_compute_exact_match(gen_answer, str(gold))),
                }
                rec.update(rouge)
                rows.append(rec)
            else:
                rec = {
                    "ID": row.get("ID", idx),
                    "expected_lang": "",
                    "expected_country": "",
                }
                try:
                    parts = str(row["subset"]).split("_", 1)
                    rec["expected_lang"] = parts[0] if parts else ""
                    rec["expected_country"] = parts[1] if len(parts) > 1 else ""
                except Exception:
                    pass
                for lbl in labels:
                    rec[lbl] = gen_answer
                rows.append(rec)

        if args.to_save_path and rows:
            save_df = pd.DataFrame(rows)
            if args.mode == "test":
                save_df = save_df[["ID"] + labels]
            os.makedirs(
                os.path.dirname(os.path.abspath(args.to_save_path)) or ".",
                exist_ok=True,
            )
            save_df.to_csv(args.to_save_path, index=False)

    if args.mode == "eval":
        _print_eval_summary(rows)

    if args.to_save_path:
        print(f"Results saved to: {args.to_save_path}")
    elif not rows:
        print("No rows processed.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Unsloth inference on eval/test CSV (same prompts as training)."
    )
    p.add_argument(
        "--data_csv_path",
        type=str,
        default=None,
        help="CSV with columns: subset, input, ID; and output for --mode eval.",
    )
    p.add_argument(
        "--gdrive_dataset_id",
        type=str,
        default=None,
        help="Optional Google Drive file id; file is saved to --data_csv_path or data/eval_download.csv.",
    )
    p.add_argument("--model_id", type=str, required=True, help="Base model HF id or local path.")
    p.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Optional PEFT adapter directory to load on top of base model.",
    )
    p.add_argument(
        "--dtype",
        type=str,
        default="float16",
        help="Torch dtype name for weights (e.g. float16 for T4, bfloat16 if supported).",
    )
    p.add_argument(
        "--mode",
        choices=("eval", "test"),
        default="eval",
        help="eval: ROUGE vs output; test: submission columns TargetRLF1, TargetR1F1, TargetLLM.",
    )
    p.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="If set, sample exactly N rows stratified by language (floor(N/num_langs) per language, remainder distributed round-robin).",
    )
    p.add_argument(
        "--gen_kwargs_path",
        type=str,
        default=None,
        help=f"JSON with model.generate kwargs. Default: {DEFAULT_GEN_KWARGS_PATH}",
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Max new tokens (also forwarded to generate; overrides JSON if both set).",
    )
    p.add_argument(
        "--start_answer_txt",
        type=str,
        default="##Answer:",
        help="Assistant prefix marker (must match training).",
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of rows per generate call. Higher = better GPU utilisation; lower if OOM.",
    )
    p.add_argument(
        "--to_save_path",
        type=str,
        default=None,
        help="Write incremental CSV results here (recommended).",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Print raw inputs and outputs (special tokens kept) for the first "
             "--debug_samples samples. Useful for verifying the prompt format.",
    )
    p.add_argument(
        "--debug_samples",
        type=int,
        default=5,
        help="Number of samples to print when --debug is set (default: 5).",
    )
    return p.parse_args()


if __name__ == "__main__":
    run_eval(parse_args())