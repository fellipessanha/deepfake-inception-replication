import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from inception_replication.classification.inception_model import load_model
from inception_replication.saliency.vlm_input import _save, vlm_input
from inception_replication.vlm_interpretation.explanation_generation import (
    generate_llm_output,
)
from inception_replication.vlm_interpretation.prompts import SystemPrompts


def explain_face(
    inception,
    vlm,
    processor,
    image: Image.Image,
    system_prompt: str = SystemPrompts.Specifics.value,
) -> tuple[np.ndarray, np.ndarray, dict, str]:
    """One tightly cropped face (as in the FaceForensics extracts) ->
    GradCAM spotlight, GradCAM overlay, prediction record, VLM explanation."""
    lit, overlay, record = vlm_input(inception, image)
    explanation = generate_llm_output(
        vlm,
        processor,
        image.convert("RGB"),
        Image.fromarray((overlay.clip(0, 1) * 255).astype(np.uint8)),
        system_prompt,
        record,
    )
    return lit, overlay, record, explanation


def _main() -> None:
    # transformers is heavy and only the CLI needs the real VLM
    from inception_replication.vlm_interpretation.model_loading import (
        load_model as load_vlm,
    )

    parser = argparse.ArgumentParser(
        description="Classify one cropped face, run GradCAM and have a VLM explain it."
    )
    parser.add_argument("path", help="a single tightly cropped face image")
    parser.add_argument("-o", "--output-dir", default="vlm_output")
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument(
        "--prompt",
        choices=[p.name for p in SystemPrompts],
        default=SystemPrompts.Specifics.name,
        help="one of the built-in system prompts",
    )
    prompt.add_argument(
        "--system-prompt",
        help='your own system prompt text; from a file: --system-prompt "$(cat prompt.txt)"',
    )
    args = parser.parse_args()
    system_prompt = args.system_prompt or SystemPrompts[args.prompt].value

    vlm, processor = load_vlm()
    lit, overlay, record, explanation = explain_face(
        load_model(), vlm, processor, Image.open(args.path), system_prompt
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.path).stem
    _save(lit, out / f"{stem}_spotlight.png")
    _save(overlay, out / f"{stem}_gradcam.png")
    record |= {"image": str(args.path), "explanation": explanation}
    (out / f"{stem}.json").write_text(json.dumps(record, indent=2))
    print(f"\n{record['predicted']}: written to {out}/{stem}_{{spotlight,gradcam}}.png and {stem}.json")


if __name__ == "__main__":
    _main()
