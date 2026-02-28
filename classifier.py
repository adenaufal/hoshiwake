from __future__ import annotations

from typing import Any

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

from config import MODEL_ID


def load_model(device: str):
    """Load image processor and model once."""
    try:
        processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
        model = model.to(device)
        model.eval()
        return processor, model
    except Exception as exc:
        raise RuntimeError(
            f"Unable to load model '{MODEL_ID}'. Check network access and model availability."
        ) from exc


def _label_for_index(id2label: dict[Any, str], index: int) -> str:
    if index in id2label:
        return id2label[index]
    string_index = str(index)
    if string_index in id2label:
        return id2label[string_index]
    return string_index


def _build_result(probabilities: torch.Tensor, id2label: dict[Any, str]) -> dict[str, Any]:
    all_scores: dict[str, float] = {}
    for index, score in enumerate(probabilities.tolist()):
        all_scores[_label_for_index(id2label, index)] = float(score)

    top_index = int(torch.argmax(probabilities).item())
    top_label = _label_for_index(id2label, top_index)
    top_score = float(probabilities[top_index].item())

    return {
        "label": top_label,
        "score": top_score,
        "all_scores": all_scores,
    }


def _classify_without_fallback(images: list[Image.Image], processor, model, device: str):
    inputs = processor(images=images, return_tensors="pt")
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1).detach().cpu()
    return [_build_result(prob_vector, model.config.id2label) for prob_vector in probabilities]


def classify_batch(
    images: list[Image.Image], processor, model, device: str
) -> list[dict[str, Any]]:
    """Classify a batch of PIL images."""
    if not images:
        return []

    rgb_images = [image.convert("RGB") for image in images]

    try:
        return _classify_without_fallback(rgb_images, processor, model, device)
    except Exception:
        results = []
        for image in rgb_images:
            single_result = _classify_without_fallback([image], processor, model, device)[0]
            results.append(single_result)
        return results


def classify_single(image: Image.Image, processor, model, device: str) -> dict[str, Any]:
    """Convenience wrapper around classify_batch for one image."""
    return classify_batch([image], processor, model, device)[0]
