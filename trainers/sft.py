import torch
from collections import defaultdict
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
        subsets: list[str] = []
        max_len = 0
        pad_id = self.tokenizer.pad_token_id
        for feat in features:
            messages = feat["messages"]
            subsets.append(feat.get("subset", ""))

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
            "subset": subsets,
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

        # Ensure TRL's _metrics dict exists (guaranteed by SFTTrainer, but guard anyway)
        if not hasattr(self, "_metrics"):
            self._metrics = defaultdict(list)

        # Accumulate raw token counts across all micro-batches (grad-accum steps)
        # and flush them to ratios only at logging time, so every token has equal weight.
        self._overall_metrics: dict[str, int] = {"correct": 0, "total": 0}
        self._lang_metrics: dict[str, dict[str, int]] = defaultdict(
            lambda: {"correct": 0, "total": 0}
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        subsets: list[str] | None = inputs.pop("subset", None)

        labels = inputs.get("labels")
        outputs = model(**inputs)
        loss = outputs.loss / self.args.gradient_accumulation_steps

        # Token-prediction accuracy over the answer spans (non-masked positions).
        # Wrapped in try/except because unsloth may return EmptyLogits when
        # UNSLOTH_RETURN_LOGITS is not honoured; in that case we skip accuracy
        # silently rather than crashing the training run.
        try:
            if labels is not None and outputs.logits is not None:
                logits = outputs.logits          # (B, T, V)
                # Causal-LM shift: logit at t predicts label at t+1
                shift_logits = logits[..., :-1, :].contiguous()   # (B, T-1, V)
                shift_labels = labels[..., 1:].contiguous()        # (B, T-1)

                mask = shift_labels != -100
                total_tokens = mask.sum().item()

                if total_tokens > 0:
                    preds = shift_logits.argmax(dim=-1)
                    correct = (preds == shift_labels) & mask

                    # Accumulate raw counts so every token has equal weight across
                    # all micro-batches in a gradient-accumulation window.
                    self._overall_metrics["correct"] += correct.sum().item()
                    self._overall_metrics["total"] += total_tokens

                    # Accumulate per-language raw counts
                    if subsets is not None:
                        for i, subset in enumerate(subsets):
                            lang = subset.split("_")[0] if subset else "unknown"
                            n_tokens = mask[i].sum().item()
                            n_correct = correct[i].sum().item()
                            if n_tokens > 0:
                                self._lang_metrics[lang]["correct"] += n_correct
                                self._lang_metrics[lang]["total"] += n_tokens
        except (NotImplementedError, AttributeError):
            pass  # logits unavailable (unsloth EmptyLogits); skip accuracy

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict, **kwargs):
        if self._overall_metrics["total"] > 0:
            self._metrics["token_accuracy"].append(
                self._overall_metrics["correct"] / self._overall_metrics["total"]
            )
        self._overall_metrics = {"correct": 0, "total": 0}

        for lang, stats in self._lang_metrics.items():
            if stats["total"] > 0:
                self._metrics[f"token_accuracy_{lang}"].append(
                    stats["correct"] / stats["total"]
                )
        self._lang_metrics.clear()
        super().log(logs, **kwargs)

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
