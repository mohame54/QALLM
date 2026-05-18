from trl import DPOTrainer


class Qwen3_5DPOTrainer(DPOTrainer):
    # Unsloth's Qwen3_5 patch incorrectly sets is_vision_model=True for
    # text-only Qwen3.5 models. Override _prepare_dataset to clear the flag
    # before it branches into the vision processing path.
    def _prepare_dataset(self, dataset, processing_class, args, split_name, **kwargs):
        self.is_vision_model = False
        return super()._prepare_dataset(dataset, processing_class, args, split_name, **kwargs)

    def log(self, logs: dict, start_time: float | None = None, **kwargs):
        if start_time is not None:
            super().log(logs, start_time, **kwargs)
        else:
            super().log(logs, **kwargs)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        step = self.state.global_step
        print(f"\n{'='*60}")
        print(f"VALIDATION @ step {step}")
        print(f"{'='*60}")
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        print(f"{'='*60}\n")

        return metrics
