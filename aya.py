import argparse
import os
import warnings
import pandas as pd
import tqdm
from utils.hf import download_checkpoint_from_hf
from utils.aya import load_lora_extended_model
from utils.common import download_gdown_file, load_json
from utils.instructions import AyaChatPromptTemplate, process_single_row

warnings.filterwarnings("ignore")

CUR_DIR = os.path.dirname(os.path.abspath(__file__))


def main(args):
    adapter_path = args.adapter_path
    if  args.hf_checkpoint:
        if not args.repo_id:
            raise ValueError("repo_id must be provided when hf_checkpoint is specified.")
        download_checkpoint_from_hf(
            repo_id=args.repo_id,
            checkpoint_dir=args.hf_checkpoint,
            local_dir=".",
        )
        adapter_path = args.hf_checkpoint
    
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

    all_messages = []
    all_gens = []
    all_ids = []
    batch_sz = args.batch_sz
    max_len = 0
    if args.gdrive_id:
        test_df_path = download_gdown_file(args.gdrive_id, args.test_data_path)
    else:
        test_df_path = args.test_data_path
    
    if not os.path.exists(test_df_path):
       raise FileNotFoundError(f"Test data file not found at {test_df_path}. Please check the path or gdrive_id.")
    test_df = pd.read_csv(test_df_path)
    for _, row in test_df.iterrows():
        all_messages.append(process_single_row(row, tokenizer, add_answer=False))
        all_ids.append(row["ID"])

    gen_kwargs = load_json(os.path.join(CUR_DIR, "configs", "gen.json"))
    pgbar = tqdm.tqdm(range(0, len(all_messages), batch_sz), desc="Generating the data")
    for i in pgbar:
        batch_messages = all_messages[i : i + batch_sz]
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
        ).to("cuda")

        prompt_length = inputs["input_ids"].shape[1]
        ids = model.generate(
            **inputs,
            **gen_kwargs,
            eos_token_id=aya_chat_template.stop_token_ids(),
        )
        gen_ids = ids[:, :prompt_length]
        max_len = max(gen_ids.shape[1], max_len)
        gens = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        gens = [
            gens[j].replace(test_df.iloc[i + j]["input"], "").strip()
            for j in range(len(gens))
        ]
        all_gens.extend(gens)
        pgbar.set_postfix({"Latest Gen": gens[0], "Max Gen Len": max_len})

    data = {
        "ID": all_ids,
    }
    labels = ["TargetRLF1", "TargetR1F1", "TargetLLM"]
    data.update({l: all_gens for l in labels})
    results_df = pd.DataFrame(data)
    results_df.to_csv(args.save_file_path, index=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Aya inference on a test dataset."
    )
    parser.add_argument(
        "--hf_checkpoint",
        default=None,
        type=str
    )
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
        help="Hugging Face repo ID for the checkpoint (required if --hf_checkpoint is specified).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())