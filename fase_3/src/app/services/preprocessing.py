"""Text preprocessing.

Deliberately minimal. TF-IDF vectorization is part of the ONNX graph
itself (the sklearn Pipeline was converted whole — see
ml/convert_to_onnx.py), so this module only does generic cleanup that
should happen before text reaches the model, not feature extraction.
"""


def clean_text(text: str) -> str:
    """Collapses runs of whitespace/newlines into single spaces and trims
    the ends. Casing is left alone — the vectorizer inside the ONNX graph
    already lowercases by default, doing it twice would be redundant.
    """
    return " ".join(text.split())


def clean_texts(texts: list[str]) -> list[str]:
    return [clean_text(text) for text in texts]
