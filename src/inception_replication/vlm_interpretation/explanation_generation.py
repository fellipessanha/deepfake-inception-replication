import json
from pathlib import Path

from PIL.Image import Image


def _image(x: Path | Image) -> str | Image:
    "Qwen's processor takes a file path or a PIL image."
    return str(x.absolute()) if isinstance(x, Path) else x


def make_prompt_message(
    frame: Path | Image, aux: Path | Image, system_prompt: str, record: dict | None = None
):
    user_content: list[dict] = [
        {"type": "image", "image": _image(frame)},
        {"type": "image", "image": _image(aux)},
    ]
    if record is not None:
        # the "object with the model output" the prompts refer to
        user_content.append({"type": "text", "text": json.dumps(record)})
    messages = [
        {
            "role": "system",
            "type": "text",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {"role": "user", "content": user_content},
    ]
    return messages


def generate_llm_output(
    model,
    processor,
    frame: Path | Image,
    overlay: Path | Image,
    system_prompt: str,
    record: dict | None = None,
) -> str:
    messages = make_prompt_message(frame, overlay, system_prompt, record)

    # Preparation for inference
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(output_text[0])
    return output_text[0]
