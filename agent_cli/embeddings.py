from __future__ import annotations

from collections.abc import Sequence


class HuggingFaceEmbedder:
    def __init__(self, model_name: str, token: str | None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            from huggingface_hub import login
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face embedding dependencies are missing. "
                "Install the project dependencies first."
            ) from exc

        if token:
            login(token=token, add_to_git_credential=False)
        self.model_name = model_name
        self._model = SentenceTransformer(model_name, token=token or None)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
        return [[float(value) for value in row] for row in vectors]
