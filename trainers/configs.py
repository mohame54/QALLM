from dataclasses import fields, is_dataclass
import json
from typing import Any

from trl import SFTConfig


class Qwen3_5SFTConfig(SFTConfig):
    def __init__(self, **kwargs):
        accepted_keys = self._accepted_config_keys()
        sft_kwargs = {key: value for key, value in kwargs.items() if key in accepted_keys}
        extra_kwargs = {key: value for key, value in kwargs.items() if key not in accepted_keys}

        super().__init__(**sft_kwargs)

        # Keep newer TRL config values available even if the installed version
        # does not accept them in SFTConfig.__init__ yet.
        for key, value in extra_kwargs.items():
            setattr(self, key, value)

    @staticmethod
    def _accepted_config_keys() -> set[str]:
        if not is_dataclass(SFTConfig):
            return set()
        return {field.name for field in fields(SFTConfig) if field.init}

    @classmethod
    def from_json(cls, path: str):
        with open(path, "r") as f:
            config: dict[str, Any] = json.load(f)
        return cls(**config)