from pathlib import Path

from agent_cli.knowledge import ErrorKnowledgeBase


class FakeEmbedder:
    model_name = "fake-model"

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [
            [
                float(text.lower().count("timeout") + 1),
                float(text.lower().count("network") + 1),
                float(len(text) % 13 + 1),
            ]
            for text in texts
        ]


def test_error_is_persisted_then_resolved_and_searchable(tmp_path: Path) -> None:
    knowledge = ErrorKnowledgeBase(tmp_path, FakeEmbedder())
    error_id = knowledge.record_error("TimeoutError: request timed out", "network request")
    entry = tmp_path / "knowledge" / f"{error_id}.md"

    initial = entry.read_text(encoding="utf-8")
    assert "- Status: unresolved" in initial
    assert "TimeoutError: request timed out" in initial

    knowledge.resolve_error(error_id, "Retry with a bounded timeout.")
    matches = knowledge.search("network timeout")

    updated = entry.read_text(encoding="utf-8")
    assert "- Status: resolved" in updated
    assert "Retry with a bounded timeout." in updated
    assert matches
    assert "Retry with a bounded timeout." in matches[0].content


def test_refresh_reindexes_changed_markdown_without_duplicate_chunks(
    tmp_path: Path,
) -> None:
    embedder = FakeEmbedder()
    knowledge = ErrorKnowledgeBase(tmp_path, embedder)
    error_id = knowledge.record_error("Network error", "first context")
    path = tmp_path / "knowledge" / f"{error_id}.md"
    path.write_text("# Updated\n\n" + ("updated context " * 150), encoding="utf-8")

    knowledge.refresh_index()
    knowledge.refresh_index()
    matches = knowledge.search("updated context")
    with knowledge._connect() as connection:
        sources = [
            row[0]
            for row in connection.execute(
                "SELECT source FROM documents WHERE source LIKE ?",
                (f"{path.name}#chunk-%",),
            )
        ]

    assert sources
    assert len(sources) == len(set(sources))
    assert len({match.source for match in matches}) == len(matches)
