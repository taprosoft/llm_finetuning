from pathlib import Path

from alpaca_lora_4bit import autograd_4bit
from alpaca_lora_4bit.autograd_4bit import load_llama_model_4bit_low_ram
from alpaca_lora_4bit.gradient_checkpointing import apply_gradient_checkpointing
from alpaca_lora_4bit.monkeypatch.llama_attn_hijack_xformers import (
    hijack_llama_attention,
)
from alpaca_lora_4bit.monkeypatch.peft_tuners_lora_monkey_patch import (
    replace_peft_model_with_int4_lora_model,
)
from peft import PeftModel, get_peft_model

replace_peft_model_with_int4_lora_model()
hijack_llama_attention()


def find_pt_checkpoint(model_id):
    model_name_or_path = Path(model_id)
    pt_path = None
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

    return str(pt_path)


def load_model_gptq(
    model_id,
    lora_config,
    group_size=128,
    gradient_checkpointing=False,
    lora_path=None,
    load_lora=True,
    lora_trainable=False,
    backend="cuda",
    device_map="auto",
):
    if backend == "triton":
        autograd_4bit.switch_backend_to("triton")
    else:
        autograd_4bit.switch_backend_to("cuda")

    config_path = model_id
    model_path = find_pt_checkpoint(model_id)
    model, tokenizer = load_llama_model_4bit_low_ram(
        config_path,
        model_path,
        groupsize=group_size,
        device_map=device_map,
        is_v1_model=False,
    )

    if load_lora:
        if lora_path:
            model = PeftModel.from_pretrained(
                model, lora_path, is_trainable=lora_trainable
            )
        else:
            model = get_peft_model(model, lora_config)

    # Scales to half
    print("Fitting 4bit scales and zeros to half")
    for _, m in model.named_modules():
        if "Autograd4bitQuantLinear" in str(type(m)) or "Linear4bitLt" in str(type(m)):
            if hasattr(m, "is_v1_model") and m.is_v1_model:
                m.zeros = m.zeros.half()
            m.scales = m.scales.half()

    if gradient_checkpointing:
        # Use gradient checkpointing
        print("Applying gradient checkpointing ...")
        apply_gradient_checkpointing(model, checkpoint_ratio=1.0)

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer
