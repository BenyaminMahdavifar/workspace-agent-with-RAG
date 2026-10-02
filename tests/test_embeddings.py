import sys
from types import ModuleType

from agent_cli.embeddings import HuggingFaceEmbedder


def test_hugging_face_token_logs_in_and_loads_selected_model(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def login(*, token: str, add_to_git_credential: bool) -> None:
        calls["login"] = (token, add_to_git_credential)

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, token: str | None) -> None:
            calls["model"] = (model_name, token)

        def encode(self, texts, convert_to_numpy, normalize_embeddings):
            calls["encode"] = (texts, convert_to_numpy, normalize_embeddings)
            return [[0.25, 0.75] for _ in texts]

    huggingface_hub = ModuleType("huggingface_hub")
    huggingface_hub.login = login
    sentence_transformers = ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "huggingface_hub", huggingface_hub)
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)

    embedder = HuggingFaceEmbedder("org/test-embedding", "hf-secret")

    assert calls["login"] == ("hf-secret", False)
    assert calls["model"] == ("org/test-embedding", "hf-secret")
    assert embedder.encode(["one"]) == [[0.25, 0.75]]
