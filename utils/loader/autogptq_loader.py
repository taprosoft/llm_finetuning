from pathlib import Path

from auto_gptq import AutoGPTQForCausalLM, get_gptq_peft_model
from auto_gptq.utils.peft_utils import GPTQLoraConfig
from peft import TaskType
from transformers import LlamaTokenizer


def load_model_autogptq(
    model_id, lora_path=None, load_lora=False, lora_trainable=False, backend="cuda"
):
    model_name_or_path = Path(model_id)
    pt_path = None
    use_triton = backend == "triton"
    # Find the model checkpoint
    for ext in [".safetensors", ".pt", ".bin"]:
        found = list(model_name_or_path.glob(f"*{ext}"))
        if len(found) > 0:
            if len(found) > 1:
                print(
                    f"More than one {ext} model has been found. The last one will be selected. It could be wrong."
                )

            pt_path = found[-1]
            break

    # load model and prepare for kbit training
    model = AutoGPTQForCausalLM.from_quantized(
        model_id,
        use_safetensors=True,
        model_basename=pt_path.stem,
        use_triton=use_triton,
        warmup_triton=False,
        trainable=lora_trainable,
        inject_fused_attention=False,
        inject_fused_mlp=False,
    )

    tokenizer = LlamaTokenizer.from_pretrained(model_id, use_fast=False)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    if load_lora:
        # creating model
        peft_config = GPTQLoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=[
                "q_proj",
                "v_proj",
            ],
            task_type=TaskType.CAUSAL_LM,
            inference_mode=not lora_trainable,
        )

        print("Loading LoRA", lora_path)
        model = get_gptq_peft_model(
            model,
            peft_config=peft_config,
            auto_find_all_linears=False,
            train_mode=lora_trainable,
            model_id=lora_path,
        )

    return model, tokenizer
