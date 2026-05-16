import os

# Must be set before unsloth is imported; unsloth's for_training() resets this
# flag to "0" internally, so we also patch os.environ to prevent that.
os.environ["UNSLOTH_RETURN_LOGITS"] = "1"
_orig_env_setitem = os.environ.__class__.__setitem__
def _guarded_env_setitem(self, key, value):
    if key == "UNSLOTH_RETURN_LOGITS" and value != "1":
        return
    _orig_env_setitem(self, key, value)
os.environ.__class__.__setitem__ = _guarded_env_setitem

import argparse
from utils.datasets import QADataset
from utils.model import get_peft_model
from utils.common import download_gdown_file, load_json
from trainers.sft import Qwen3_5Collator, Mytrainer
from trainers.configs import Qwen3_5SFTConfig
from unsloth import FastLanguageModel


CUR_DIR = os.path.dirname(os.path.abspath(__file__))


def run_training(
    args: argparse.Namespace,
):
    train_file = args.train_file
    if args.gdrive_train_file_id:
        train_file = download_gdown_file(args.gdrive_train_file_id, args.train_file)

    model_id = args.model_id
    peft_kwargs_path = args.peft_kwargs_path or os.path.join(CUR_DIR, "configs", "peft.json")
    sft_trainer_kwargs_path = args.sft_trainer_kwargs_path or os.path.join(CUR_DIR, "configs", "sft_trainer.json")
    if peft_kwargs_path:
        peft_kwargs = load_json(peft_kwargs_path)
    else:
        peft_kwargs = {}
    model, tokenizer, hf_tokenizer = get_peft_model(model_id, **peft_kwargs)
    sampling_alpha = args.sampling_alpha
    train_dataset = QADataset.from_csv(train_file, hf_tokenizer)
    train_dataset.mode("train")
    train_dataset = train_dataset.to_hf_dataset()
    my_custom_collator = Qwen3_5Collator(hf_tokenizer, max_seq_length=args.max_seq_length)
    FastLanguageModel.for_training(model)
    args = Qwen3_5SFTConfig.from_json(sft_trainer_kwargs_path)
    trainer = Mytrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=my_custom_collator,
        train_dataset=train_dataset,
        args=args,
        sampling_alpha=sampling_alpha,
    )
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--gdrive_train_file_id", type=str, default=None)
    parser.add_argument("--model_id", type=str, required=True)
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