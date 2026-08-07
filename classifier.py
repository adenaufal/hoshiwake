from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from config import MODEL_ID


def _resolve_local_or_hf_file(model_id: str, filename: str) -> str:
    candidate = Path(model_id) / filename
    if candidate.exists():
        return str(candidate)

    from huggingface_hub import hf_hub_download

    return hf_hub_download(model_id, filename=filename)


def _load_transformers_model(model_id: str, device: str):
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    model = model.to(device)
    model.eval()
    model._hoshiwake_backend = "transformers"
    return processor, model


def _load_caveduck_timm_model(model_id: str, device: str):
    import timm
    from torchvision import transforms

    config_path = _resolve_local_or_hf_file(model_id, "config.json")
    ckpt_path = _resolve_local_or_hf_file(model_id, "pytorch_model.pt")

    with open(config_path, "r", encoding="utf-8") as handle:
        model_config = json.load(handle)

    input_size = int(model_config.get("input_size", 224))
    normalization = model_config.get("normalization", {})
    mean = normalization.get("mean", [0.485, 0.456, 0.406])
    std = normalization.get("std", [0.229, 0.224, 0.225])

    model = timm.create_model("convnext_tiny.fb_in22k_ft_in1k", pretrained=False, num_classes=2)
    trust_checkpoint = os.environ.get("HOSHIWAKE_TRUST_CHECKPOINT") == "1"
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=not trust_checkpoint)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load '{ckpt_path}' in safe mode ({exc}). The checkpoint contains "
            f"non-tensor objects; re-save it as a plain state_dict, or re-run with "
            f"HOSHIWAKE_TRUST_CHECKPOINT=1 only if you trust the file's origin."
        ) from exc
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    model._hoshiwake_backend = "timm_caveduck"

    class_names = model_config.get("class_names", [])
    if isinstance(class_names, list) and len(class_names) == 2:
        model._hoshiwake_id2label = {0: str(class_names[0]), 1: str(class_names[1])}
    else:
        model._hoshiwake_id2label = {0: "label_0", 1: "label_1"}

    processor = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return processor, model


def load_model(device: str, model_id: str = MODEL_ID):
    """Load processor and model once."""
    errors: list[str] = []

    try:
        return _load_transformers_model(model_id, device)
    except Exception as exc:
        errors.append(f"transformers backend failed: {exc}")

    model_id_lower = model_id.lower()
    looks_like_caveduck = "caveduckai/nsfw-classifier" in model_id_lower or (
        bool(model_id) and (Path(model_id) / "pytorch_model.pt").exists()
    )
    if looks_like_caveduck:
        try:
            return _load_caveduck_timm_model(model_id, device)
        except Exception as exc:
            errors.append(f"timm backend failed: {exc}")

    raise RuntimeError(
        f"Unable to load model '{model_id}'. Tried available backends.\n"
        + "\n".join(errors)
    )


def _label_for_index(id2label: dict[Any, str], index: int) -> str:
    if index in id2label:
        return id2label[index]
    string_index = str(index)
    if string_index in id2label:
        return id2label[string_index]
    # "label_<n>" (not the bare index) so downstream keyword matching in
    # sorter.score_groups can still bucket the class via "label_0"/"label_1".
    return f"label_{index}"


def _build_result(probabilities: torch.Tensor, id2label: dict[Any, str]) -> dict[str, Any]:
    all_scores: dict[str, float] = {}
    for index, score in enumerate(probabilities.tolist()):
        all_scores[_label_for_index(id2label, index)] = float(score)

    top_index = int(torch.argmax(probabilities).item())
    top_label = _label_for_index(id2label, top_index)
    top_score = float(probabilities[top_index].item())

    return {
        "label": top_label,
        "label_index": top_index,
        "score": top_score,
        "all_scores": all_scores,
    }


def _error_result(exc: Exception) -> dict[str, Any]:
    return {
        "label": "",
        "label_index": -1,
        "score": 0.0,
        "all_scores": {},
        "error": str(exc),
    }


def _classify_transformers(images: list[Image.Image], processor, model, device: str):
    inputs = processor(images=images, return_tensors="pt")
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().cpu()
    return [_build_result(prob_vector, model.config.id2label) for prob_vector in probabilities]


def _classify_caveduck(images: list[Image.Image], processor, model, device: str):
    tensors = [processor(image) for image in images]
    batch = torch.stack(tensors, dim=0).to(device)

    with torch.no_grad():
        logits = model(batch)

    probabilities = torch.nn.functional.softmax(logits, dim=-1).detach().cpu()
    id2label = getattr(model, "_hoshiwake_id2label", {0: "label_0", 1: "label_1"})
    return [_build_result(prob_vector, id2label) for prob_vector in probabilities]


def classify_batch(
    images: list[Image.Image], processor, model, device: str
) -> list[dict[str, Any]]:
    """Classify a batch of PIL images.

    Always returns one result per input image. Images that fail even the
    single-image retry get an error result ({"error": ..., "all_scores": {}})
    instead of aborting the whole batch.
    """
    if not images:
        return []

    rgb_images = [image if image.mode == "RGB" else image.convert("RGB") for image in images]
    backend = getattr(model, "_hoshiwake_backend", "transformers")
    runner = _classify_caveduck if backend == "timm_caveduck" else _classify_transformers

    try:
        return runner(rgb_images, processor, model, device)
    except Exception as batch_exc:
        print(f"[WARN] Batch classification failed ({batch_exc}); retrying images individually.")
        results = []
        for image in rgb_images:
            try:
                results.append(runner([image], processor, model, device)[0])
            except Exception as exc:
                print(f"[WARN] Failed to classify an image: {exc}")
                results.append(_error_result(exc))
        return results


def classify_single(image: Image.Image, processor, model, device: str) -> dict[str, Any]:
    """Convenience wrapper around classify_batch for one image."""
    return classify_batch([image], processor, model, device)[0]
