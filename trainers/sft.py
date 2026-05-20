import torch
from collections import defaultdict
from typing import Any
import pandas as pd
from trl import SFTTrainer
from utils.instructions import (
    AyaChatPromptTemplate,
    create_tokens_labels,
    STUDENT_TEMPLATE,
    SYSTEM_PROMPT,
)
from trainers.configs import Qwen3_5SFTConfig
from torch.utils.data import DataLoader
from utils.datasets import LanguageStratifiedBatchSampler, COUNTRY_MAP
from utils.metrics import calculate_rouge_score


class AyaCollator:
    def __init__(
        self,
        tokenizer: Any,
        max_seq_length: int,
        chat_template_kwargs=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.chat_template_kwargs = dict(chat_template_kwargs or {})

        self.chat_template = AyaChatPromptTemplate(tokenizer)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch_input_ids: list[list[int]] = []
        batch_labels: list[list[int]] = []
        subsets: list[str] = []
        max_len = 0
        pad_id = self.tokenizer.pad_token_id
        for feat in features:
            messages = feat["messages"]
            subsets.append(feat.get("subset", ""))
            full_ids, _, labels = self.chat_template.create_token_labels(
                messages,
            )
            batch_input_ids.append(full_ids)
            batch_labels.append(labels)
            max_len = min(max(max_len, len(full_ids)), self.max_seq_length)

        padded_input: list[list[int]] = []
        padded_labels: list[list[int]] = []
        attention_mask: list[list[int]] = []
        for ids, lab in zip(batch_input_ids, batch_labels, strict=True):
            pad_amt = max_len - len(ids)
            attn_mask = [1] * len(ids)
            if self.chat_template.padding_side == "left":
                ids = [pad_id] * pad_amt + ids
                lab = [-100] * pad_amt + lab
                attn_mask = [0] * pad_amt + attn_mask
            else:
                ids = ids + [pad_id] * pad_amt
                lab = lab + [-100] * pad_amt
                attn_mask = attn_mask + [0] * pad_amt
            assert len(ids) == len(lab) == len(attn_mask)
            padded_input.append(ids)
            padded_labels.append(lab)
            attention_mask.append(attn_mask)
        batch: dict = {
            "input_ids": torch.tensor(padded_input, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "subset": subsets,
        }
        return batch


class Qwen3_5Collator:
    def __init__(
        self,
        tokenizer: Any,
        max_seq_length: int,
        answer_start_text="##Answer:",
        chat_template_kwargs=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
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
                tokenize=True,
            )
            batch_input_ids.append(full_ids)
            batch_labels.append(labels)
            max_len = min(max(max_len, len(full_ids)), self.max_seq_length)
           
        padded_input: list[list[int]] = []
        padded_labels: list[list[int]] = []
        attention_mask: list[list[int]] = []

        for ids, lab in zip(batch_input_ids, batch_labels, strict=True):
            pad_amt = max_len - len(ids)
            attn_mask = [1] * len(ids)
            if self.tokenizer.padding_side == "left":
                ids = [pad_id] * pad_amt + ids
                lab = [-100] * pad_amt + lab
                attn_mask = [0] * pad_amt + attn_mask
            else:
                ids = ids + [pad_id] * pad_amt
                lab = lab + [-100] * pad_amt
                attn_mask = attn_mask + [0] * pad_amt
            assert len(ids) == len(lab) == len(attn_mask)
            padded_input.append(ids)
            padded_labels.append(lab)
            attention_mask.append(attn_mask)

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
        # Generation-eval settings: sample M examples from val_df every eval pass.
        self.gen_eval_samples: int = kwargs.pop("gen_eval_samples", 0)
        self.val_df: pd.DataFrame | None = kwargs.pop("val_df", None)
        self.model_family: str = kwargs.pop("model_family", "qwen")
        self.max_gen_length: int = kwargs.pop("max_gen_length", 256)

        super().__init__(*args, **kwargs)

        # 2. Force SFTTrainer to use your custom collator, overriding its fallback
        if custom_collator is not None:
            self.data_collator = custom_collator

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

        # Scale loss only during training; eval loss is already a batch mean.
        if model.training:
            loss = outputs.loss / self.args.gradient_accumulation_steps
        else:
            loss = outputs.loss
        if model.training:
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
                print("Logits unavailable (unsloth EmptyLogits); skipping accuracy")

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict, start_time: float | None = None, **kwargs):
        # Flush accumulated raw counts → ratios and merge directly into logs.
        # SFTTrainer does NOT override log(), so _metrics is never consumed
        # by the parent chain — we must inject the values into logs ourselves.
        if self._overall_metrics["total"] > 0:
            logs["token_accuracy"] = (
                self._overall_metrics["correct"] / self._overall_metrics["total"]
            )
        self._overall_metrics = {"correct": 0, "total": 0}

        for lang, stats in self._lang_metrics.items():
            if stats["total"] > 0:
                logs[f"token_accuracy_{lang}"] = stats["correct"] / stats["total"]
        self._lang_metrics.clear()

        if start_time is not None:
            super().log(logs, start_time, **kwargs)
        else:
            super().log(logs, **kwargs)

    def get_eval_dataloader(self, eval_dataset=None):
        dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        if dataset is None:
            return None
        return DataLoader(
            dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    @torch.no_grad()
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        logs: dict[str, float] = {}
        # ── generation-based evaluation (exact-match, token-F1) ─────────────
        if self.gen_eval_samples > 0 and self.val_df is not None:
            try:
                gen_logs = self._run_generation_eval(
                    self.gen_eval_samples, metric_key_prefix
                )
                logs.update(gen_logs)
            except Exception as exc:
                print(f"[eval] Generation eval failed: {exc}")

        # Print a readable summary
        step = self.state.global_step
        print(f"\n{'='*60}")
        print(f"VALIDATION @ step {step}")
        print(f"{'='*60}")
        for k, v in logs.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        print(f"{'='*60}\n")
        super(Mytrainer, self).log(logs)
        return logs

    @torch.no_grad()
    def _run_generation_eval(
        self, n_samples: int, metric_key_prefix: str = "eval"
    ) -> dict[str, float]:
        hf_tokenizer = self.data_collator.tokenizer
        dev = next(self.model.parameters()).device

        sample_df = self.val_df.sample(
            n=min(n_samples, len(self.val_df)),
            random_state=self.state.global_step,
        ).reset_index(drop=True)

        aya_template = AyaChatPromptTemplate(hf_tokenizer) if self.model_family == "aya" else None

        predictions: list[str] = []
        references: list[str] = []
        subsets_list: list[str] = []

        for _, row in sample_df.iterrows():
            subset = str(row.get("subset", ""))
            ref_answer = str(row.get("output", "")).strip()

            try:
                lang, ctry = subset.split("_", 1)
            except ValueError:
                lang, ctry = subset, ""
            if ctry in COUNTRY_MAP:
                ctry = COUNTRY_MAP[ctry]

            content = (
                STUDENT_TEMPLATE.format(question=row["input"], language=lang, country=ctry)
                if (lang or ctry)
                else str(row["input"])
            )

            if self.model_family == "aya":
                # Build user-only messages; apply_chat_prompt_template adds system +
                # opens the assistant turn via generation=True.
                messages = [{"role": "user", "content": content}]
                prompt_text = aya_template.apply_chat_prompt_template(
                    messages, generation=True
                )
                input_ids = hf_tokenizer.encode(
                    prompt_text, add_special_tokens=False, return_tensors="pt"
                ).to(dev)
            else:
                # Qwen: build prefix up to (and including) the ##Answer: marker so
                # the model continues the answer rather than regenerating the marker.
                prefix_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ]
                prefix_ids = hf_tokenizer.apply_chat_template(
                    prefix_messages,
                    enable_thinking=False,
                    add_generation_prompt=True,
                    tokenize=True,
                )
                answer_start_ids = hf_tokenizer.encode(
                    "##Answer:", add_special_tokens=False
                )
                input_ids = torch.tensor(
                    [prefix_ids + answer_start_ids], dtype=torch.long
                ).to(dev)
            out_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_gen_length,
                do_sample=False,
                pad_token_id=hf_tokenizer.pad_token_id or hf_tokenizer.eos_token_id,
                eos_token_id=hf_tokenizer.eos_token_id,
            )

            gen_ids = out_ids[0][input_ids.shape[-1]:]
            pred = hf_tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            predictions.append(pred)
            references.append(ref_answer)
            subsets_list.append(subset)

        return self._compute_gen_metrics(
            predictions, references, subsets_list, metric_key_prefix
        )

    def _compute_gen_metrics(
        self,
        predictions: list[str],
        references: list[str],
        subsets: list[str],
        prefix: str,
    ) -> dict[str, float]:
        def normalize(s: str) -> str:
            return s.strip().lower()

        n = len(predictions)
        exact_matches: list[bool] = []
        rouge1_scores: list[float] = []
        rougeL_scores: list[float] = []
        combined_scores: list[float] = []

        for pred, ref in zip(predictions, references):
            exact_matches.append(normalize(pred) == normalize(ref))
            rouge = calculate_rouge_score(ref, pred)
            rouge1_scores.append(rouge["rouge1_f1"])
            rougeL_scores.append(rouge["rougeL_f1"])
            combined_scores.append(rouge["score"])

        def mean(vals: list) -> float:
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        logs: dict[str, float] = {
            f"{prefix}_gen_exact_match": mean(exact_matches),
            f"{prefix}_gen_rouge1": mean(rouge1_scores),
            f"{prefix}_gen_rougeL": mean(rougeL_scores),
            f"{prefix}_gen_score": mean(combined_scores),
            f"{prefix}_gen_n_samples": float(n),
        }

        lang_em: dict[str, list] = defaultdict(list)
        lang_r1: dict[str, list] = defaultdict(list)
        lang_rl: dict[str, list] = defaultdict(list)
        lang_sc: dict[str, list] = defaultdict(list)

        for em, r1, rl, sc, subset in zip(
            exact_matches, rouge1_scores, rougeL_scores, combined_scores, subsets
        ):
            lang = subset.split("_")[0] if subset else "unknown"
            lang_em[lang].append(em)
            lang_r1[lang].append(r1)
            lang_rl[lang].append(rl)
            lang_sc[lang].append(sc)

        for lang in lang_em:
            logs[f"{prefix}_gen_exact_match_{lang}"] = mean(lang_em[lang])
            logs[f"{prefix}_gen_rouge1_{lang}"] = mean(lang_r1[lang])
            logs[f"{prefix}_gen_rougeL_{lang}"] = mean(lang_rl[lang])
            logs[f"{prefix}_gen_score_{lang}"] = mean(lang_sc[lang])

        return logs

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
