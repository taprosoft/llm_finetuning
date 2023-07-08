import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from exllama_lib.lora import ExLlamaLora
from exllama_lib.model import ExLlama, ExLlamaCache, ExLlamaConfig
from transformers import (
    GenerationConfig,
    LlamaTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast


class ExllamaHF(PreTrainedModel):
    def __init__(self, config: ExLlamaConfig):
        super().__init__(PretrainedConfig())
        self.ex_config = config
        self.ex_model = ExLlama(self.ex_config)
        self.generation_config = GenerationConfig()
        self.lora = None

    def _validate_model_class(self):
        pass

    def _validate_model_kwargs(self, model_kwargs: Dict[str, Any]):
        pass

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids, **kwargs}

    @property
    def device(self) -> torch.device:
        return torch.device(0)

    def __call__(self, *args, **kwargs):
        # TODO: Some decoding methods (such as Contrastive Search) may not work at this time
        assert len(args) == 0, "no *args should be passed to forward"
        use_cache = kwargs.get("use_cache", True)
        seq = kwargs["input_ids"][0].tolist()
        cache = kwargs["past_key_values"] if "past_key_values" in kwargs else None
        if cache is None:
            cache = ExLlamaCache(self.ex_model)
            self.ex_model.forward(
                torch.tensor([seq[:-1]], dtype=torch.long),
                cache,
                preprocess_only=True,
                lora=self.lora,
            )

        logits = self.ex_model.forward(
            torch.tensor([seq[-1:]], dtype=torch.long), cache, lora=self.lora
        ).to(kwargs["input_ids"].device)

        return CausalLMOutputWithPast(
            logits=logits, past_key_values=cache if use_cache else None
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        **kwargs,
    ):
        assert (
            len(model_args) == 0 and len(kwargs) == 0
        ), "extra args is currently not supported"
        if isinstance(pretrained_model_name_or_path, str):
            pretrained_model_name_or_path = Path(pretrained_model_name_or_path)

        config = ExLlamaConfig(pretrained_model_name_or_path / "config.json")

        # from 'oobabooga/text-generation-webui/modules/exllama.py'
        weight_path = None
        for ext in [".safetensors", ".pt", ".bin"]:
            found = list(pretrained_model_name_or_path.glob(f"*{ext}"))
            if len(found) > 0:
                weight_path = found[-1]
                break
        assert (
            weight_path is not None
        ), f'could not find weight in "{pretrained_model_name_or_path}"'

        config.model_path = str(weight_path)
        config.max_seq_len = 2048
        config.compress_pos_emb = 1

        if torch.version.hip:
            config.rmsnorm_no_half2 = True
            config.rope_no_half2 = True
            config.matmul_no_half2 = True
            config.silu_no_half2 = True

        # This slowes down a bit but align better with autogptq generation.
        # TODO: Should give user choice to tune the exllama config
        # config.fused_attn = False
        # config.fused_mlp_thd = 0

        return ExllamaHF(config)


def load_lora(model, lora_path):
    lora_config_path = os.path.join(lora_path, "adapter_config.json")
    lora_path = os.path.join(lora_path, "adapter_model.bin")
    lora = ExLlamaLora(model.ex_model, lora_config_path, lora_path)
    model.lora = lora

    return model


def load_model_exllama(base, delta):
    model = ExllamaHF.from_pretrained(base)

    # load normal HF tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(base, use_fast=False)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # setup PEFT model
    if os.path.exists(delta):
        model = load_lora(model, delta)

    return model, tokenizer
