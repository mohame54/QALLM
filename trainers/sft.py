import torch
from typing import Any
from trl import SFTTrainer
from utils.instructions import create_tokens_labels
from trainers.configs import Qwen3_5SFTConfig
from torch.utils.data import DataLoader
from utils.datasets import LanguageStratifiedBatchSampler


class Qwen3_5Collator:
    def __init__(
        self,
        tokenizer: Any,
        max_seq_length: int,
        to_replace="<|im_start|>assistant\n<think>\n\n</think>\n\n",
        answer_start_text="##Answer:",
        chat_template_kwargs=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.to_replace = to_replace
        self.answer_start_text = answer_start_text

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch_input_ids: list[list[int]] = []
        batch_labels: list[list[int]] = []
        max_len = 0
        pad_id = self.tokenizer.pad_token_id
        for feat in features:
            messages = feat["messages"]

            full_ids, _, labels = create_tokens_labels(
                self.tokenizer,
                messages,
                answer_start_txt=self.answer_start_text,
                tokenize=True
            )
            batch_input_ids.append(full_ids)
            batch_labels.append(labels)
            max_len = min(max(max_len, len(full_ids)), self.max_seq_length)
           
        padded_input: list[list[int]] = []
        padded_labels: list[list[int]] = []
        attention_mask: list[list[int]] = []

        for ids, lab in zip(batch_input_ids, batch_labels, strict=True):
            pad_amt = max_len - len(ids)
            padded_input.append(ids + [pad_id] * pad_amt)
            padded_labels.append(lab + [-100] * pad_amt)
            attention_mask.append([1] * len(ids) + [0] * pad_amt)

        batch: dict = {
            "input_ids": torch.tensor(padded_input, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
        }
        return batch





class Mytrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        # 1. Save your custom collator from kwargs before initializing the parent
        custom_collator = kwargs.get("data_collator")
        self.sampling_alpha = kwargs.pop("sampling_alpha", 0.5)

        super().__init__(*args, **kwargs)

        # 2. Force SFTTrainer to use your custom collator, overriding its fallback
        if custom_collator is not None:
            self.data_collator = custom_collator

    def get_train_dataloader(self):
       

        langs = [row["subset"].split("_")[0] for row in self.train_dataset]
        batch_sampler = LanguageStratifiedBatchSampler(
            langs=langs,
            micro_batch_size=self.args.per_device_train_batch_size,
            grad_accum_steps=self.args.gradient_accumulation_steps,
            alpha=self.sampling_alpha,
            seed=self.args.seed,
        )
        return DataLoader(
            self.train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _prepare_dataset(
        self,
        dataset: Any,
        processing_class: Any,
        args: Any,
        packing: bool,
        formatting_func: Any,
        dataset_name: str,
    ) -> Any:
        """Skip TRL's automatic chat-template tokenization."""
        return dataset
