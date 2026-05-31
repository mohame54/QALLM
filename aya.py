import argparse
import os
import warnings
import pandas as pd
import tqdm
from utils.hf import download_checkpoint_from_hf
from utils.aya import load_lora_extended_model
from utils.common import download_gdown_file, load_json
from utils.instructions import AyaChatPromptTemplate, process_single_row
from utils.text import remove_repeated_ngrams

warnings.filterwarnings("ignore")

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

MAX_LENS_SUBSET = {
    'Aka_Gha': 512,
    'Amh_Eth': 128,
    'Eng_Eth': 64,
    'Eng_Gha': 180,
    'Eng_Ken': 205,
    'Eng_Uga': 312,
    'Lug_Uga': 511,
    'Swa_Ken': 347
}

MAX_LENS_SUBSET_2 = {
    "Swa_Ken": 500,
    "Lug_Uga": 650,
    "Aka_Gha": 500,
    "Amh_Eth": 200,
    "Eng_Eth": 80,
    "Eng_Uga": 420,
    "Eng_Gha": 300,
    "Eng_Ken": 312,
}


def main(args):
    adapter_path = args.adapter_path
    if args.hf_checkpoint:
        if not args.repo_id:
            raise ValueError("repo_id must be provided when hf_checkpoint is specified.")
        adapter_path = download_checkpoint_from_hf(
            repo_id=args.repo_id,
            checkpoint_dir=args.hf_checkpoint,
            local_dir=".",
        )

    if not adapter_path:
        raise ValueError("Either hf_checkpoint or adapter_path must be provided.")

    model, tokenizer = load_lora_extended_model(
        base_model_id=args.base_model_id,
        checkpoint_dir=adapter_path,
        load_in_4bit=False,
    )

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    aya_chat_template = AyaChatPromptTemplate(tokenizer)

    if args.gdrive_id:
        test_df_path = download_gdown_file(args.gdrive_id, args.test_data_path)
    else:
        test_df_path = args.test_data_path

    if not os.path.exists(test_df_path):
        raise FileNotFoundError(
            f"Test data file not found at {test_df_path}. Please check the path or gdrive_id."
        )

    test_df = pd.read_csv(test_df_path)

    # ── Split rows across GPUs if running in parallel mode ───────────────────
    if args.gpu_id is not None:
        total_rows = len(test_df)
        # Assign each row to a GPU via round-robin so subsets stay balanced
        # across GPUs (avoids one GPU getting all of a slow subset).
        indices = list(range(args.gpu_id, total_rows, args.num_gpus))
        test_df = test_df.iloc[indices].reset_index(drop=True)
        print(f"🔀 GPU {args.gpu_id}/{args.num_gpus}: processing {len(test_df)} / {total_rows} rows")

    results = []
    global_gen_kwargs = load_json(os.path.join(CUR_DIR, "configs", "gen.json"))
    for subset in test_df['subset'].unique():
        sub_df = test_df[test_df['subset'] == subset]
        gen_kwargs = global_gen_kwargs.copy()
        gen_kwargs.update({'max_new_tokens': MAX_LENS_SUBSET[subset]})
        print(f"Generating for subset {subset} with kwargs: {gen_kwargs}")
        results.append(
            generate_with_subset(
                sub_df=sub_df,
                tokenizer=tokenizer,
                aya_chat_template=aya_chat_template,
                gen_kwargs=gen_kwargs,
                batch_sz=args.batch_sz,
                model=model,
                min_n_grams=args.min_n_grams,
                max_n_grams=args.max_n_grams,
                max_passes=args.max_passes,
            )
        )

    results_df = pd.concat(results)
    results_df.to_csv(args.save_file_path, index=False)
    print(f"✅ Saved {len(results_df)} rows to {args.save_file_path}")


def generate_with_subset(
    sub_df,
    tokenizer,
    aya_chat_template,
    gen_kwargs,
    batch_sz,
    model,
    min_n_grams=3,
    max_n_grams=8,
    max_passes=10,
):
    all_messages = []
    all_gens = []
    all_ids = []
    for _, row in sub_df.iterrows():
        all_messages.append(process_single_row(row, tokenizer, add_answer=False))
        all_ids.append(row["ID"])

    max_len = 0
    pgbar = tqdm.tqdm(range(0, len(all_messages), batch_sz), desc="Generating the data")
    for i in pgbar:
        batch_messages = all_messages[i: i + batch_sz]
        input_texts = [
            aya_chat_template.apply_chat_prompt(messages, generation=True)
            for messages in batch_messages
        ]
        inputs = tokenizer(
            text=input_texts,
            images=None,
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(next(model.parameters()).device)

        prompt_length = inputs["input_ids"].shape[1]
        ids = model.generate(
            **inputs,
            **gen_kwargs,
            eos_token_id=aya_chat_template.stop_token_ids(),
        )
        gen_ids = ids[:, prompt_length:]
        max_len = max(gen_ids.shape[1], max_len)
        gens = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        gens = [
            gens[j].replace(sub_df.iloc[i + j]["input"], "").strip()
            for j in range(len(gens))
        ]
        gens = [
            remove_repeated_ngrams(
                tokenizer,
                gen,
                min_n=min_n_grams,
                max_n=max_n_grams,
                max_passes=max_passes,
            )
            for gen in gens
        ]
        all_gens.extend(gens)
        pgbar.set_postfix({"Latest Gen": gens[0], "Max Gen Len": max_len})

    data = {"ID": all_ids}
    labels = ["TargetRLF1", "TargetR1F1", "TargetLLM"]
    data.update({l: all_gens for l in labels})
    return pd.DataFrame(data)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Aya inference on a test dataset."
    )
    parser.add_argument("--hf_checkpoint", default=None, type=str)
    parser.add_argument(
        "--base_model_id",
        type=str,
        default="CohereLabs/tiny-aya-global",
        help="Base model HF id or local path.",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Path to the LoRA adapter directory.",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="data/test_public.csv",
        help="Path to the test data CSV file.",
    )
    parser.add_argument(
        "--gdrive_id",
        type=str,
        default=None,
        help="Optional Google Drive file ID for test data.",
    )
    parser.add_argument(
        "--save_file_path",
        type=str,
        default="submission-Finetuned_Aya.csv",
        help="CSV path where results will be saved.",
    )
    parser.add_argument(
        "--batch_sz",
        type=int,
        default=8,
        help="Batch size for generating data.",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="Muhammed164/SDFT",
        help="Hugging Face repo ID for the checkpoint.",
    )
    parser.add_argument(
        "--min_n_grams",
        type=int,
        default=3,
        help="Minimum n-gram length to remove.",
    )
    parser.add_argument(
        "--max_n_grams",
        type=int,
        default=8,
        help="Maximum n-gram length to remove.",
    )
    parser.add_argument(
        "--max_passes",
        type=int,
        default=10,
        help="Maximum number of passes to remove n-grams.",
    )
    # ── Internal use only — set by run_parallel.sh, not by the user directly ─
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=None,
        help="GPU index for this process (0..num_gpus-1). None = single GPU mode.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=2,
        help="Total number of parallel GPU processes (used with --gpu_id).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())