from collections import defaultdict
import math
import random

import numpy as np
import pandas as pd
from datasets import Dataset
from torch.utils.data import Sampler
from typing import Literal

from utils.instructions import create_question, create_message_instance

COUNTRY_MAP: dict[str, str] = {
    "Uga": "Uganda",
    "Gha": "Ghana",
    "Eth": "Ethiopia",
    "Ken": "Kenya",
}


def _compute_language_allocations(
    lang_counts: dict[str, int],
    effective_batch_size: int,
    alpha: float,
) -> dict[str, int]:
    langs = sorted(lang_counts.keys())
    counts = np.array([lang_counts[lang] for lang in langs], dtype=np.float64)
    weights = counts ** alpha
    probs = weights / weights.sum()
    raw = probs * effective_batch_size
    floors = np.floor(raw).astype(int)

    # Guarantee every language appears in each effective batch.
    for i, lang in enumerate(langs):
        if floors[i] == 0:
            floors[i] = 1

    total = int(floors.sum())
    if total > effective_batch_size:
        excess = total - effective_batch_size
        order = np.argsort(-counts)
        for lang_idx in order:
            if excess == 0:
                break
            if floors[lang_idx] > 1:
                floors[lang_idx] -= 1
                excess -= 1
    elif total < effective_batch_size:
        remainder = effective_batch_size - total
        frac = raw - np.floor(raw)
        order = np.argsort(-frac)
        for i in range(remainder):
            floors[order[i % len(langs)]] += 1

    return {lang: int(floors[i]) for i, lang in enumerate(langs)}


class LanguageStratifiedBatchSampler(Sampler):
    """
    Builds effective batches with temperature-weighted language slots, then yields
    micro-batches for gradient accumulation. Each language pool cycles with reshuffle
    when exhausted (no within-cycle duplicates).
    """

    def __init__(
        self,
        langs: list[str],
        micro_batch_size: int,
        grad_accum_steps: int,
        alpha: float = 0.5,
        seed: int = 42,
    ):
        self.micro_batch_size = micro_batch_size
        self.grad_accum_steps = grad_accum_steps
        self.effective_batch_size = micro_batch_size * grad_accum_steps
        self.alpha = alpha
        self.rng = random.Random(seed)

        lang_indices: dict[str, list[int]] = defaultdict(list)
        for idx, lang in enumerate(langs):
            lang_indices[lang].append(idx)

        self.lang_indices = dict(lang_indices)
        lang_counts = {lang: len(idxs) for lang, idxs in self.lang_indices.items()}
        self.allocs = _compute_language_allocations(
            lang_counts, self.effective_batch_size, alpha
        )

        max_pool = max(lang_counts.values())
        max_alloc = max(self.allocs.values())
        self._n_effective_batches = math.ceil(max_pool / max_alloc)

    def _next_index(self, lang: str, pools: dict[str, list[int]], ptrs: dict[str, int]) -> int:
        pool = pools[lang]
        ptr = ptrs[lang]
        if ptr >= len(pool):
            self.rng.shuffle(pool)
            ptrs[lang] = 0
            ptr = 0
        idx = pool[ptr]
        ptrs[lang] = ptr + 1
        return idx

    def __iter__(self):
        pools = {
            lang: self.rng.sample(indices, len(indices))
            for lang, indices in self.lang_indices.items()
        }
        ptrs = {lang: 0 for lang in self.lang_indices}

        for _ in range(self._n_effective_batches):
            effective_batch: list[int] = []
            for lang, n_slots in self.allocs.items():
                for _ in range(n_slots):
                    effective_batch.append(self._next_index(lang, pools, ptrs))
            self.rng.shuffle(effective_batch)

            for start in range(0, len(effective_batch), self.micro_batch_size):
                yield effective_batch[start : start + self.micro_batch_size]

    def __len__(self) -> int:
        return self._n_effective_batches * self.grad_accum_steps


class QADataset:
    def __init__(
        self,
        df: pd.DataFrame,
        transformers_tokenizer,
        add_answer=False,
        start_answer_txt="##Answer:",
    ):
        self.df = df
        self.transformers_tokenizer = transformers_tokenizer
        self.add_answer = add_answer
        self.start_answer_txt = start_answer_txt

    def mode(self, mode: Literal["train", "val", "test"]):
        self._mode = mode
        if self._mode in ["train", "val"]:
            self.add_answer = True
        elif self._mode == "test":
            self.add_answer = False

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        answer = row["output"] if self.add_answer else None
        messages = process_single_row(
            row, self.transformers_tokenizer, self.add_answer, self.start_answer_txt
        )
        return messages, answer

    def __iter__(self):
        return iter(self.df)

    @classmethod
    def from_csv(cls, path: str, transformers_tokenizer):
        df = pd.read_csv(path).dropna(subset=["subset"]).reset_index(drop=True)
        return cls(df, transformers_tokenizer)

    def to_hf_dataset(self):
        df = self.df.copy(deep=True)
        old_columns = df.columns.tolist()
        df["messages"] = df.apply(
            lambda row: process_single_row(
                row,
                self.transformers_tokenizer,
                self.add_answer,
                self.start_answer_txt,
            ),
            axis=1,
        )
        df.drop(columns=[c for c in old_columns if c != "subset"], inplace=True)
        return Dataset.from_pandas(df)


class QADPODataset(QADataset):
    def to_hf_dataset(self):
        df = self.df.copy(deep=True)
        old_columns = df.columns.tolist()

        df["prompt"] = df.apply(
            lambda row: create_question(
                row["input"],
                self.transformers_tokenizer,
                None,
                language=row["expected_lang"],
                country=row["expected_country"],
                apply_chat_template=False,
            ), axis=1)
        df['chosen'] = df['gold_answer'].apply(
            lambda x: [{"role": "assistant", "content": self.start_answer_txt + " " + x}]
        )
        df['rejected'] = df['gen_answer'].apply(
            lambda x: [{"role": "assistant", "content": self.start_answer_txt + " " + x}]
        )
        df.drop(columns=old_columns, inplace=True)
        return Dataset.from_pandas(df)



def process_single_row(
    row,
    tokenizer,
    add_answer=False,
    start_answer_txt="##Answer:",
    q_column_name="input",
    a_column_name="output",
):
    try:
        lang, ctry = row["subset"].split("_")
    except Exception:
        print(row["subset"])
        raise
    if ctry in COUNTRY_MAP:
        ctry = COUNTRY_MAP[ctry]
    if add_answer:
        messages = create_message_instance(
            tokenizer,
            row[q_column_name],
            row[a_column_name],
            answer_start=start_answer_txt,
            language=lang,
            country=ctry,
        )
    else:
        messages = create_question(
            row[q_column_name],
            tokenizer,
            answer_start=start_answer_txt,
            language=lang,
            country=ctry,
            apply_chat_template=False,
        )
    return messages
