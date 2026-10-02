from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol


class Embedder(Protocol):
    model_name: str

    def encode(self, texts: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class KnowledgeMatch:
    content: str
    score: float
    source: str


class ErrorKnowledgeBase:
    def __init__(self, root: Path, embedder: Embedder) -> None:
        self.root = root
        self.documents_dir = root / "knowledge"
        self.database_path = root / "knowledge.sqlite3"
        self.embedder = embedder
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    source TEXT PRIMARY KEY,
                    content_hash TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    model_name TEXT NOT NULL
                )
                """
            )

    def record_error(self, error: str, context: str) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        error_id = f"{timestamp}-{hashlib.sha256(error.encode('utf-8')).hexdigest()[:8]}"
        safe_error = self._safe_markdown(error)
        safe_context = self._safe_markdown(context)
        path = self.documents_dir / f"{error_id}.md"
        content = (
            f"# Error: {error_id}\n\n"
            f"- Recorded: {datetime.now(timezone.utc).isoformat()}\n"
            "- Status: unresolved\n\n"
            "## Error\n\n"
            f"{safe_error}\n\n"
            "## Context\n\n"
            f"{safe_context}\n"
        )
        path.write_text(content, encoding="utf-8")
        self._index_document(path, content)
        return error_id

    def resolve_error(self, error_id: str, solution: str) -> None:
        path = self.documents_dir / f"{error_id}.md"
        if not path.is_file():
            raise FileNotFoundError(f"Knowledge entry {error_id!r} does not exist.")
        content = path.read_text(encoding="utf-8")
        content = content.replace("- Status: unresolved", "- Status: resolved", 1)
        content += f"\n## Resolution\n\n{self._safe_markdown(solution)}\n"
        path.write_text(content, encoding="utf-8")
        self._index_document(path, content)

    def search(self, query: str, limit: int = 3) -> list[KnowledgeMatch]:
        self.refresh_index()
        query_vector = self.embedder.encode([query])[0]
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT source, content, embedding FROM documents WHERE model_name = ?",
                (self.embedder.model_name,),
            ).fetchall()

        best_by_document: dict[str, KnowledgeMatch] = {}
        for source, content, serialized_vector in rows:
            vector = json.loads(serialized_vector)
            score = self._cosine_similarity(query_vector, vector)
            document = source.split("#chunk-", 1)[0]
            current = best_by_document.get(document)
            if current is None or score > current.score:
                best_by_document[document] = KnowledgeMatch(
                    content=content,
                    score=score,
                    source=document,
                )
        matches = list(best_by_document.values())
        matches.sort(key=lambda match: match.score, reverse=True)
        return matches[: max(0, limit)]

    def refresh_index(self) -> None:
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM documents WHERE model_name != ?",
                (self.embedder.model_name,),
            )
            indexed = {
                row[0].split("#chunk-", 1)[0]: row[1]
                for row in connection.execute(
                    "SELECT source, content_hash FROM documents WHERE model_name = ?",
                    (self.embedder.model_name,),
                )
            }
        current_paths = {path.name for path in self.documents_dir.glob("*.md")}
        for stale in set(indexed) - current_paths:
            with self._connect() as connection:
                connection.execute(
                    "DELETE FROM documents WHERE source LIKE ?",
                    (f"{stale}#chunk-%",),
                )

        for path in self.documents_dir.glob("*.md"):
            content = path.read_text(encoding="utf-8")
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if indexed.get(path.name) != content_hash:
                self._index_document(path, content)

    def _index_document(self, path: Path, content: str) -> None:
        chunks = self._chunks(content)
        vectors = self.embedder.encode(chunks)
        if len(vectors) != len(chunks):
            raise RuntimeError("Embedding model returned an unexpected number of vectors.")
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM documents WHERE source LIKE ?",
                (f"{path.name}#chunk-%",),
            )
            connection.executemany(
                """
                INSERT INTO documents (source, content_hash, content, embedding, model_name)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        f"{path.name}#chunk-{index}",
                        content_hash,
                        chunk,
                        json.dumps(vector),
                        self.embedder.model_name,
                    )
                    for index, (chunk, vector) in enumerate(zip(chunks, vectors))
                ],
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path)

    @staticmethod
    def _chunks(content: str, max_chars: int = 1200) -> list[str]:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", content) if part.strip()]
        chunks: list[str] = []
        current = ""
        for paragraph in paragraphs:
            while len(paragraph) > max_chars:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(paragraph[:max_chars])
                paragraph = paragraph[max_chars:]
            candidate = f"{current}\n\n{paragraph}".strip()
            if len(candidate) > max_chars and current:
                chunks.append(current)
                current = paragraph
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks or [content[:max_chars]]

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        if len(left) != len(right):
            raise ValueError("Embedding dimensions do not match.")
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)

    @staticmethod
    def _safe_markdown(value: str) -> str:
        return value.replace("\x00", "").strip() or "(empty)"
